#!/usr/bin/env python3
"""Fine-tuned image benchmark on the P-disjoint splits.

One invocation trains one model for one fold/repeat/seed, writes per-epoch
validation metrics, saves the best checkpoint, and writes test-set probabilities
and logits for later pooling.

Example smoke test:
    python scripts/train.py --folds 0 --repeats 0 --seeds 42 --epochs 1 \
        --image-size 128 --limit-train 64 --batch-size 16 --workers 0 \
        --run-name smoke
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

REPO = Path(__file__).resolve().parents[1]
CLASSES = ("Dermatitis", "Eczema", "Scabies", "Tinea Ringworm", "Vitiligo")
CLASS_INDEX = {name: index for index, name in enumerate(CLASSES)}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SplitDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, data_dir: Path, transform):
        self.frame = frame.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(self.data_dir / row["filepath"]).convert("RGB")
        return self.transform(image), CLASS_INDEX[row["class_label"]], index


def make_transforms(image_size: int):
    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, scale=(0.65, 1.0), ratio=(0.75, 1.333)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train, eval_transform


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["class_label"].value_counts().to_dict()
    return {name: int(counts.get(name, 0)) for name in CLASSES}


def make_sampler(frame: pd.DataFrame) -> WeightedRandomSampler:
    counts = class_counts(frame)
    weights = frame["class_label"].map(lambda name: 1.0 / counts[name]).to_numpy()
    weights = np.ascontiguousarray(weights, dtype=np.float64).copy()
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def class_weights(frame: pd.DataFrame) -> torch.Tensor:
    counts = class_counts(frame)
    total = sum(counts.values())
    weights = [total / (len(CLASSES) * max(counts[name], 1)) for name in CLASSES]
    return torch.as_tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimiser, device, scaler, amp):
    model.train()
    total_loss = 0.0
    total = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimiser.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        else:
            loss.backward()
            optimiser.step()
        total_loss += loss.item() * len(labels)
        total += len(labels)
    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, frame, amp):
    model.eval()
    total_loss = 0.0
    total = 0
    labels_all = []
    preds_all = []
    probs_all = []
    logits_all = []
    indices_all = []
    for images, labels, indices in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
        logits = logits.float()
        probs = torch.softmax(logits, dim=1)
        total_loss += loss.item() * len(labels)
        total += len(labels)
        labels_all.extend(labels.cpu().tolist())
        preds_all.extend(probs.argmax(dim=1).cpu().tolist())
        probs_all.extend(probs.cpu().numpy())
        logits_all.extend(logits.cpu().numpy())
        indices_all.extend(indices.tolist())
    labels_all = np.asarray(labels_all)
    preds_all = np.asarray(preds_all)
    probs_all = np.asarray(probs_all)
    logits_all = np.asarray(logits_all)
    indices_all = np.asarray(indices_all)
    metrics = {
        "loss": total_loss / max(total, 1),
        "accuracy": float((labels_all == preds_all).mean()),
        "macro_f1": float(
            f1_score(labels_all, preds_all, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(labels_all, preds_all)),
        "per_class_recall": {
            cls: float(value)
            for cls, value in zip(
                CLASSES,
                recall_score(
                    labels_all,
                    preds_all,
                    labels=list(range(len(CLASSES))),
                    average=None,
                    zero_division=0,
                ),
            )
        },
        "confusion_matrix": confusion_matrix(
            labels_all, preds_all, labels=list(range(len(CLASSES)))
        ).tolist(),
    }
    rows = []
    for position, index in enumerate(indices_all):
        row = frame.iloc[index]
        record = {
            "filepath": row["filepath"],
            "group_id": row["group_id"],
            "y_true": row["class_label"],
            "y_pred": CLASSES[int(preds_all[position])],
        }
        for class_index, cls in enumerate(CLASSES):
            record[f"prob_{cls}"] = float(probs_all[position, class_index])
            record[f"logit_{cls}"] = float(logits_all[position, class_index])
        rows.append(record)
    return metrics, pd.DataFrame(rows)


def load_split(splits_dir: Path, fold: int, repeat: int, split: str) -> pd.DataFrame:
    path = splits_dir / f"fold{fold}_rep{repeat}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def run_one(args, fold: int, repeat: int, seed: int) -> dict:
    seed_everything(seed)
    train_frame = load_split(args.splits, fold, repeat, "train")
    val_frame = load_split(args.splits, fold, repeat, "val")
    test_frame = load_split(args.splits, fold, repeat, "test")
    if args.limit_train:
        train_frame = train_frame.groupby("class_label", group_keys=False).head(
            max(args.limit_train // len(CLASSES), 1)
        )
    train_transform, eval_transform = make_transforms(args.image_size)
    sampler = make_sampler(train_frame) if args.class_balanced else None
    train_loader = DataLoader(
        SplitDataset(train_frame, args.data_dir, train_transform),
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=args.device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        SplitDataset(val_frame, args.data_dir, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device.type == "cuda",
    )
    test_loader = DataLoader(
        SplitDataset(test_frame, args.data_dir, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device.type == "cuda",
    )

    model = timm.create_model(
        args.model,
        pretrained=not args.no_pretrained,
        num_classes=len(CLASSES),
        img_size=args.image_size,
    ).to(args.device)
    loss_weights = class_weights(train_frame).to(args.device) if args.class_weighted_loss else None
    criterion = nn.CrossEntropyLoss(
        weight=loss_weights, label_smoothing=args.label_smoothing
    )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(args.epochs, 1), eta_min=args.lr * 0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.device.type == "cuda")
    amp = args.amp and args.device.type == "cuda"

    tag = f"fold{fold}_rep{repeat}_seed{seed}"
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoints / f"{args.run_name}_{tag}.pt"
    history = []
    best = {"epoch": 0, "macro_f1": -1.0}
    stale = 0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimiser, args.device, scaler, amp
        )
        val_metrics, _ = evaluate(
            model, val_loader, criterion, args.device, val_frame, amp
        )
        scheduler.step()
        elapsed = time.time() - started
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_loss": val_metrics["loss"],
            "seconds": round(elapsed, 2),
        }
        history.append(entry)
        print(
            f"{tag} epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} ({elapsed:.1f}s)"
        )
        if val_metrics["macro_f1"] > best["macro_f1"]:
            best = {"epoch": epoch, "macro_f1": val_metrics["macro_f1"]}
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": val_metrics["macro_f1"],
                    "config": {
                        "model": args.model,
                        "fold": fold,
                        "repeat": repeat,
                        "seed": seed,
                        "image_size": args.image_size,
                    },
                },
                checkpoint_path,
            )
            stale = 0
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                print(f"{tag} early stopping at epoch {epoch}")
                break

    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics, test_rows = evaluate(
        model, test_loader, criterion, args.device, test_frame, amp
    )
    print(
        f"{tag} test macro_f1={test_metrics['macro_f1']:.4f} "
        f"balanced_acc={test_metrics['balanced_accuracy']:.4f}"
    )

    args.predictions.mkdir(parents=True, exist_ok=True)
    test_rows.insert(0, "seed", seed)
    test_rows.insert(0, "split", "test")
    test_path = args.predictions / f"finetuned_{args.run_name}_{tag}_test.csv"
    test_rows.to_csv(test_path, index=False)

    result = {
        "config": {
            "run_name": args.run_name,
            "model": args.model,
            "fold": fold,
            "repeat": repeat,
            "seed": seed,
            "epochs": args.epochs,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "class_balanced": args.class_balanced,
            "class_weighted_loss": args.class_weighted_loss,
            "amp": amp,
            "n_train": int(len(train_frame)),
            "n_val": int(len(val_frame)),
            "n_test": int(len(test_frame)),
            "train_class_counts": class_counts(train_frame),
            "pretrained": not args.no_pretrained,
            "checkpoint": str(checkpoint_path),
        },
        "best_epoch": best["epoch"],
        "best_val_macro_f1": best["macro_f1"],
        "history": history,
        "test": test_metrics,
        "test_predictions": str(test_path),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    result_path = args.out / f"finetuned_{args.run_name}_{tag}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def is_complete(args, fold: int, repeat: int, seed: int) -> bool:
    tag = f"fold{fold}_rep{repeat}_seed{seed}"
    result = args.out / f"finetuned_{args.run_name}_{tag}.json"
    predictions = args.predictions / f"finetuned_{args.run_name}_{tag}_test.csv"
    checkpoint = args.checkpoints / f"{args.run_name}_{tag}.pt"
    return result.is_file() and predictions.is_file() and checkpoint.is_file()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="resnet50.a1_in1k", help="timm model name")
    ap.add_argument("--data-dir", type=Path, default=REPO / "data" / "dataset")
    ap.add_argument("--splits", type=Path, default=REPO / "splits" / "disjoint")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "finetuned")
    ap.add_argument(
        "--checkpoints", type=Path, default=REPO / "results" / "checkpoints"
    )
    ap.add_argument(
        "--predictions",
        type=Path,
        default=REPO / "results" / "predictions",
    )
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--repeats", default="0,1,2")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--class-balanced", action="store_true")
    ap.add_argument("--class-weighted-loss", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.run_name is None:
        args.run_name = args.model.replace(".", "_").replace("/", "_")
    args.device = torch.device(args.device)

    folds = parse_int_list(args.folds)
    repeats = parse_int_list(args.repeats)
    seeds = parse_int_list(args.seeds)
    for fold in folds:
        for repeat in repeats:
            for seed in seeds:
                if args.resume and is_complete(args, fold, repeat, seed):
                    print(f"skip fold{fold}_rep{repeat}_seed{seed}: already complete")
                    continue
                run_one(args, fold, repeat, seed)


if __name__ == "__main__":
    main()
