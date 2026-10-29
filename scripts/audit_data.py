#!/usr/bin/env python3
"""Dataset inventory, checksums and duplicate audit.

Outputs (under results/):
  checksums_sha256.csv   sha256/md5/size for every image
  duplicates_exact.json  groups of byte-identical files
  duplicates_perceptual.json  near-duplicate groups (pHash / dHash)
  image_stats.csv        per-image quality statistics
  audit_summary.json     headline counts used in docs/data_audit.md
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageStat

REPO = Path(__file__).resolve().parents[1]


def hashes(path: Path) -> tuple[str, str]:
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), md5.hexdigest()


def image_stats(row: pd.Series) -> dict:
    with Image.open(row.abs_path) as im:
        rgb = im.convert("RGB")
        gray = np.asarray(rgb.convert("L"), np.float32)
        ph = str(imagehash.phash(rgb, hash_size=8))
        dh = str(imagehash.dhash(rgb, hash_size=8))
        wh = str(imagehash.whash(rgb, hash_size=8))
        # simple sharpness proxy: variance of Laplacian on a 256px thumbnail
        small = np.asarray(rgb.convert("L").resize((256, 256)), np.float32)
        lap = (
            -4 * small[1:-1, 1:-1]
            + small[:-2, 1:-1]
            + small[2:, 1:-1]
            + small[1:-1, :-2]
            + small[1:-1, 2:]
        )
        mean_rgb = ImageStat.Stat(rgb).mean
    return {
        "phash": ph,
        "dhash": dh,
        "whash": wh,
        "sharpness": float(lap.var()),
        "mean_r": mean_rgb[0],
        "mean_g": mean_rgb[1],
        "mean_b": mean_rgb[2],
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
    }


def group_by_key(records: list[dict], key: str) -> list[list[dict]]:
    buckets: dict[str, list[dict]] = collections.defaultdict(list)
    for r in records:
        buckets[r[key]].append(r)
    return [v for v in buckets.values() if len(v) > 1]


def near_duplicate_groups(stats: pd.DataFrame, threshold: int) -> list[list[str]]:
    """Union-find over perceptual hashes within each version."""
    out = []
    for version, sub in stats.groupby("version"):
        hashes_ = [imagehash.hex_to_hash(h) for h in sub.phash]
        paths = sub.path.tolist()
        parent = list(range(len(paths)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                if hashes_[i] - hashes_[j] <= threshold:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[rj] = ri
        buckets = collections.defaultdict(list)
        for i in range(len(paths)):
            buckets[find(i)].append(paths[i])
        out.extend(v for v in buckets.values() if len(v) > 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=REPO / "results" / "file_index.csv")
    ap.add_argument("--results", type=Path, default=REPO / "results")
    ap.add_argument("--phash-threshold", type=int, default=6)
    args = ap.parse_args()

    idx = pd.read_csv(args.index)
    print(f"[1/4] hashing {len(idx)} files")
    rows = []
    for n, row in enumerate(idx.itertuples(index=False), 1):
        sha, md5 = hashes(Path(row.abs_path))
        rows.append({"path": row.path, "sha256": sha, "md5": md5})
        if n % 500 == 0:
            print(f"  {n}/{len(idx)}", flush=True)
    checks = pd.DataFrame(rows)
    checks.to_csv(args.results / "checksums_sha256.csv", index=False)
    idx = idx.merge(checks, on="path", how="left")

    print("[2/4] exact duplicates")
    exact = group_by_key(checks.to_dict("records"), "sha256")
    (args.results / "duplicates_exact.json").write_text(json.dumps(exact, indent=2))
    print(f"  {len(exact)} exact duplicate groups")

    print("[3/4] perceptual hashes + image stats")
    st = pd.DataFrame([image_stats(r) for _, r in idx.iterrows()])
    stats = pd.concat([idx.reset_index(drop=True), st], axis=1)
    stats.to_csv(args.results / "image_stats.csv", index=False)

    print("[4/4] near-duplicate groups")
    near = near_duplicate_groups(stats, args.phash_threshold)
    (args.results / "duplicates_perceptual.json").write_text(json.dumps(near, indent=2))

    summary = {
        "total_files": int(len(idx)),
        "by_version": {str(k): int(v) for k, v in idx.version.value_counts().items()},
        "by_version_class": {
            f"{v}/{c}": int(n) for (v, c), n in idx.groupby(["version", "class"]).size().items()
        },
        "extensions": {
            str(k): int(v)
            for k, v in idx.filename.str.rsplit(".", n=1).str[-1].value_counts().items()
        },
        "dimensions": {
            f"{v}/{w}x{h}": int(n)
            for (v, w, h), n in idx.groupby(["version", "width", "height"]).size().items()
        },
        "exact_duplicate_groups": len(exact),
        "exact_duplicate_files": int(sum(len(g) for g in exact)),
        "perceptual_duplicate_groups": len(near),
        "perceptual_duplicate_files": int(sum(len(g) for g in near)),
        "phash_threshold": args.phash_threshold,
        "raw_index_range": [
            int(idx.loc[idx.version == "raw", "raw_name"].astype(float).min()),
            int(idx.loc[idx.version == "raw", "raw_name"].astype(float).max()),
        ],
    }
    (args.results / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
