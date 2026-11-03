#!/usr/bin/env python3
"""Fine-tuned image benchmark on the P-disjoint splits.

One invocation trains one model for one fold/repeat (or a list of them), writes
per-epoch validation metrics, saves the best checkpoint, and writes test-set
probabilities for later pooling.

Example smoke test:
    python scripts/train.py --folds 0 --repeats 0 --epochs 1 --image-size 128 \
        --limit-train 64 --batch-size 16 --workers 0 --run-name smoke
"""
from __future__ import annotations

import argparse
import json
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
from torch.utils.data import DataLoader, Dataset
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


def train_one_epoch(model, loader, criterion, optimiser, device):
    model.train()
    total_loss = 0.0
    total = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimiser.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * len(labels)
        total += len(labels)
    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, frame):
    model.eval()
    total_loss = 0.0
    total = 0
    labels_all = []
    preds_all = []
    probs_all = []
    indices_all = []
    for images, labels, indices in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        total_loss += loss.item() * len(labels)
        total += len(labels)
        labels_all.extend(labels.cpu().tolist())
        preds_all.extend(probs.argmax(dim=1).cpu().tolist())
        probs_all.extend(probs.cpu().numpy())
        indices_all.extend(indices.tolist())
    labels_all = np.asarray(labels_all)
    preds_all = np.asarray(preds_all)
    probs_all = np.asarray(probs_all)
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
        rows.append(record)
    return metrics, pd.DataFrame(rows)


def load_split(splits_dir: Path, fold: int, repeat: int, split: str) -> pd.DataFrame:
    path = splits_dir / f"fold{fold}_rep{repeat}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def run_one(args, fold: int, repeat: int) -> dict:
    seed_everything(args.seed + 1000 * fold + repeat)
    train_frame = load_split(args.splits, fold, repeat, "train")
    val_frame = load_split(args.splits, fold, repeat, "val")
    test_frame = load_split(args.splits, fold, repeat, "test")
    if args.limit_train:
        train_frame = train_frame.groupby("class_label", group_keys=False).head(
            max(args.limit_train // 5, 1)
        )
    train_transform, eval_transform = make_transforms(args.image_size)
    train_loader = DataLoader(
        SplitDataset(train_frame, args.data_dir, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.device == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        SplitDataset(val_frame, args.data_dir, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device == "cuda",
    )
    test_loader = DataLoader(
        SplitDataset(test_frame, args.data_dir, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device == "cuda",
    )

    model = timm.create_model(
        args.model, pretrained=not args.no_pretrained, num_classes=len(CLASSES)
    ).to(args.device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(args.epochs, 1), eta_min=args.lr * 0.01
    )

    history = []
    best = {"epoch": 0, "macro_f1": -1.0, "state": None}
    stale = 0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimiser, args.device
        )
        val_metrics, _ = evaluate(
            model, val_loader, criterion, args.device, val_frame
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
            f"fold {fold} rep {repeat} epoch {epoch}: "
            f"train_loss={train_loss:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"({elapsed:.1f}s)"
        )
        if val_metrics["macro_f1"] > best["macro_f1"]:
            best = {
                "epoch": epoch,
                "macro_f1": val_metrics["macro_f1"],
                "state": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
            }
            stale = 0
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    test_metrics, test_rows = evaluate(
        model, test_loader, criterion, args.device, test_frame
    )
    print(
        f"fold {fold} rep {repeat} test macro_f1={test_metrics['macro_f1']:.4f} "
        f"balanced_acc={test_metrics['balanced_accuracy']:.4f}"
    )

    args.predictions.mkdir(parents=True, exist_ok=True)
    test_rows.insert(0, "split", "test")
    test_path = args.predictions / (
        f"finetuned_{args.run_name}_fold{fold}_rep{repeat}_test.csv"
    )
    test_rows.to_csv(test_path, index=False)

    result = {
        "config": {
            "model": args.model,
            "fold": fold,
            "repeat": repeat,
            "seed": args.seed + 1000 * fold + repeat,
            "epochs": args.epochs,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "n_train": int(len(train_frame)),
            "n_val": int(len(val_frame)),
            "n_test": int(len(test_frame)),
            "pretrained": not args.no_pretrained,
        },
        "best_epoch": best["epoch"],
        "best_val_macro_f1": best["macro_f1"],
        "history": history,
        "test": test_metrics,
        "test_predictions": str(test_path),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    result_path = args.out / (
        f"finetuned_{args.run_name}_fold{fold}_rep{repeat}.json"
    )
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model", default="resnet50.a1_in1k", help="timm model name"
    )
    ap.add_argument("--data-dir", type=Path, default=REPO / "data" / "dataset")
    ap.add_argument("--splits", type=Path, default=REPO / "splits" / "disjoint")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "finetuned")
    ap.add_argument(
        "--predictions",
        type=Path,
        default=REPO / "results" / "predictions",
    )
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--repeats", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()
    if args.run_name is None:
        args.run_name = args.model.replace(".", "_").replace("/", "_")
    args.device = torch.device(args.device)

    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    repeats = [int(value) for value in args.repeats.split(",") if value.strip()]
    for fold in folds:
        for repeat in repeats:
            run_one(args, fold, repeat)


if __name__ == "__main__":
    main()
