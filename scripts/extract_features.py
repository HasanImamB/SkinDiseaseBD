#!/usr/bin/env python3
"""Extract image metadata, ResNet-50 embeddings and dihedral correlations.

Outputs (under results/):
  file_index.csv        one row per image (path, version, class, size, dims)
  embeddings.npz        L2-normalised ResNet-50 pool5 features + paths
  dihedral.npz          for each augmented image: best matching raw image and
                        normalised cross-correlation under the 8 dihedral
                        transforms (flip + 90 deg rotations)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO / "data" / "dataset"
IMG_EXT = {".jpg", ".jpeg", ".png"}
DIH_SIZE = 32


def iter_images(data_dir: Path):
    raw_dir = data_dir / "ResearchImage"
    for p in sorted(raw_dir.glob("*")):
        if p.suffix.lower() in IMG_EXT:
            yield {"path": p, "version": "raw", "cls": "unlabeled", "raw_name": p.stem}
    for version in ("Images_v1", "Images_v2"):
        vdir = data_dir / version
        for cdir in sorted(p for p in vdir.iterdir() if p.is_dir()):
            for p in sorted(cdir.glob("*")):
                if p.suffix.lower() in IMG_EXT:
                    yield {"path": p, "version": version, "cls": cdir.name, "raw_name": ""}


def build_index(data_dir: Path) -> list[dict]:
    rows = []
    for r in iter_images(data_dir):
        with Image.open(r["path"]) as im:
            w, h = im.size
        rows.append(
            {
                "path": str(r["path"].relative_to(data_dir)),
                "version": r["version"],
                "class": r["cls"],
                "raw_name": r["raw_name"],
                "filename": r["path"].name,
                "width": w,
                "height": h,
                "bytes": r["path"].stat().st_size,
                "abs_path": str(r["path"]),
            }
        )
    return rows


def extract_embeddings(rows: list[dict], out_path: Path, batch_size: int = 32):
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    prep = weights.transforms()
    feats = np.empty((len(rows), 2048), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            batch = torch.stack([prep(Image.open(r["abs_path"]).convert("RGB")) for r in chunk])
            feats[start : start + len(chunk)] = model(batch).numpy()
            if start % (batch_size * 10) == 0:
                print(f"  embeddings {start + len(chunk)}/{len(rows)}", flush=True)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
    np.savez_compressed(out_path, features=feats, paths=np.array([r["abs_path"] for r in rows]))
    print(f"  saved {out_path} {feats.shape}")


def extract_dinov2_embeddings(rows: list[dict], out_path: Path, batch_size: int = 32):
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224)
    model.eval()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    prep = timm.data.create_transform(
        input_size=(3, 224, 224),
        is_training=False,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    dim = int(model.num_features)
    feats = np.empty((len(rows), dim), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            batch = torch.stack([prep(Image.open(r["abs_path"]).convert("RGB")) for r in chunk])
            feats[start : start + len(chunk)] = model(batch).numpy()
            if start % (batch_size * 10) == 0:
                print(f"  dinov2 {start + len(chunk)}/{len(rows)}", flush=True)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
    np.savez_compressed(out_path, features=feats, paths=np.array([r["abs_path"] for r in rows]))
    print(f"  saved {out_path} {feats.shape}")


def _thumb(path: str, size: int = DIH_SIZE) -> np.ndarray:
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((size, size), Image.Resampling.BILINEAR), np.float32)
    a -= a.mean()
    n = np.linalg.norm(a)
    return (a / (n + 1e-6)).ravel()


def _dihedral(x: np.ndarray, size: int = DIH_SIZE) -> np.ndarray:
    a = x.reshape(size, size)
    out = []
    for k in range(4):
        b = np.rot90(a, k)
        out.append(b.ravel())
        out.append(np.fliplr(b).ravel())
    return np.stack(out)


def dihedral_match(rows: list[dict], out_path: Path):
    raw = [r for r in rows if r["version"] == "raw"]
    aug = [r for r in rows if r["version"] != "raw"]
    gr = np.stack([_thumb(r["abs_path"]) for r in raw])
    dr = np.concatenate([_dihedral(x) for x in gr])
    raw_ids = np.repeat(np.arange(len(raw)), 8)

    scores = np.empty(len(aug), dtype=np.float32)
    best_raw = np.empty(len(aug), dtype=np.int32)
    for start in range(0, len(aug), 128):
        chunk = aug[start : start + 128]
        g = np.stack([_thumb(r["abs_path"]) for r in chunk])
        m = g @ dr.T
        j = m.argmax(1)
        scores[start : start + len(chunk)] = m[np.arange(len(chunk)), j]
        best_raw[start : start + len(chunk)] = raw_ids[j]
        if start % (128 * 10) == 0:
            print(f"  dihedral {start + len(chunk)}/{len(aug)}", flush=True)

    np.savez_compressed(
        out_path,
        aug_paths=np.array([r["abs_path"] for r in aug]),
        raw_paths=np.array([r["abs_path"] for r in raw]),
        best_raw=best_raw,
        corr=scores,
    )
    print(f"  saved {out_path}")
    print(
        "  corr quantiles:",
        {q: round(float(np.quantile(scores, q)), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95)},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--results", type=Path, default=REPO / "results")
    ap.add_argument("--skip-embeddings", action="store_true")
    ap.add_argument("--dinov2", action="store_true", help="also extract DINOv2 ViT-S/14 embeddings")
    args = ap.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)

    print("[1/4] indexing images")
    rows = build_index(args.data_dir)
    with open(args.results / "file_index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {len(rows)} images -> {args.results / 'file_index.csv'}")
    with open(args.results / "inventory.json", "w") as f:
        summary = {}
        for r in rows:
            key = f"{r['version']}/{r['class']}"
            summary.setdefault(key, 0)
            summary[key] += 1
        json.dump(summary, f, indent=2)

    print("[2/4] dihedral raw matching")
    dihedral_match(rows, args.results / "dihedral.npz")

    if not args.skip_embeddings:
        print("[3/4] ResNet-50 embeddings")
        extract_embeddings(rows, args.results / "embeddings.npz")
        if args.dinov2:
            print("[3b/4] DINOv2 ViT-S/14 embeddings")
            extract_dinov2_embeddings(rows, args.results / "embeddings_dinov2.npz")
    else:
        print("[3/4] skipped embeddings")


if __name__ == "__main__":
    main()
