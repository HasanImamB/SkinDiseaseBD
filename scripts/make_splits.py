#!/usr/bin/env python3
"""Build the evaluation splits defined in the Q1 roadmap.

Protocols
---------
P-leaked   Image-level stratified 70/15/15 split of ``Images_v1`` (mimics the
           original paper and intentionally allows derivatives of the same
           source image to cross splits).
P-disjoint Source-image-disjoint repeated StratifiedGroupKFold over labelled
           raw images. Test and validation contain raw images only; training
           uses raw + tier-A derivatives (``primary_usable``) from training
           groups only.
P-partial  Controlled leakage levels (0/25/50/100 % of test groups having
           derivatives in training) for the dose-response analysis. Built from
           fold 0 / repeat 0 of P-disjoint so that only the leakage level
           changes.

Inputs
------
``results/raw_groups.json`` produced by ``scripts/build_groups.py``.

Outputs
-------
``splits/<protocol>/fold{k}_rep{r}_{train,val,test}.csv`` (P-disjoint)
``splits/leaked/fold0_rep{r}_{train,val,test}.csv`` (P-leaked)
``splits/partial/leak{l}/fold0_rep0_{train,val,test}.csv`` (P-partial)
``results/split_stats.json`` and ``docs/split_stats.md``

CSV columns: filepath, raw_id, group_id, class_label, is_derivative,
transform_type. Paths are relative to ``data/dataset``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO / "results"
DEFAULT_SPLITS = REPO / "splits"
CLASSES = ("Dermatitis", "Eczema", "Scabies", "Tinea Ringworm", "Vitiligo")
COLUMNS = ["filepath", "raw_id", "group_id", "class_label", "is_derivative", "transform_type"]


def load_frame(results: Path) -> tuple[pd.DataFrame, dict]:
    payload = json.loads((results / "raw_groups.json").read_text())
    df = pd.DataFrame(payload["images"])
    for col in ("is_derivative", "primary_usable"):
        df[col] = df[col].fillna(False).astype(bool)
    # Raw images have no author-provided label; use the class inferred from the
    # augmented descendants. Derivatives keep their folder label.
    df["class_label"] = np.where(df["is_derivative"], df["class"], df["group_class"])
    df["transform_type"] = "unknown"
    df = df[df["class_label"].isin(CLASSES)].reset_index(drop=True)
    return df, payload["metadata"]


def write_split(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame[["path", *COLUMNS[1:]]].rename(columns={"path": "filepath"})
    out = out.sort_values("filepath").reset_index(drop=True)
    out.to_csv(path, index=False)


def summarize(frame: pd.DataFrame, role: str) -> dict:
    return {
        "role": role,
        "images": int(len(frame)),
        "raw": int((~frame["is_derivative"]).sum()),
        "derivatives": int(frame["is_derivative"].sum()),
        "groups": int(frame["group_id"].nunique()),
        "classes": {c: int((frame["class_label"] == c).sum()) for c in CLASSES},
    }


def write_triplet(out_dir: Path, tag: str, train, val, test, meta: dict, summaries: list) -> None:
    for role, frame in (("train", train), ("val", val), ("test", test)):
        write_split(out_dir / f"{tag}_{role}.csv", frame)
        summaries.append({"protocol": out_dir.name, "tag": tag, **summarize(frame, role)})
    meta_path = out_dir / f"{tag}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def disjoint_folds(df: pd.DataFrame, folds: int, repeats: int, seed: int):
    raw = df[~df["is_derivative"]]
    for rep in range(repeats):
        sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + rep)
        split_iter = sgkf.split(raw, raw["class_label"], raw["group_id"])
        for k, (train_idx, test_idx) in enumerate(split_iter):
            rest = raw.iloc[train_idx]
            test_raw = raw.iloc[test_idx]
            min_class = int(rest["class_label"].value_counts().min())
            inner_splits = max(2, min(folds, min_class))
            inner = StratifiedGroupKFold(
                n_splits=inner_splits, shuffle=True, random_state=seed + rep + 1000
            )
            tr_in, va_in = next(inner.split(rest, rest["class_label"], rest["group_id"]))
            train_raw = rest.iloc[tr_in]
            val_raw = rest.iloc[va_in]

            train_groups = set(train_raw["group_id"])
            train_files = df[df["primary_usable"] & df["group_id"].isin(train_groups)]
            meta = {
                "protocol": "P-disjoint",
                "fold": k,
                "repeat": rep,
                "seed": seed + rep,
                "train_groups": len(train_groups),
                "val_groups": int(val_raw["group_id"].nunique()),
                "test_groups": int(test_raw["group_id"].nunique()),
                "inner_splits": inner_splits,
            }
            yield k, rep, train_files, val_raw, test_raw, meta


def build_disjoint(df, out_dir, folds, repeats, seed, summaries):
    for k, rep, train, val, test, meta in disjoint_folds(df, folds, repeats, seed):
        write_triplet(out_dir, f"fold{k}_rep{rep}", train, val, test, meta, summaries)


def build_leaked(df, out_dir, repeats, seed, summaries):
    pool = df[df["version"] == "Images_v1"].reset_index(drop=True)
    idx = np.arange(len(pool))
    y = pool["class_label"].to_numpy()
    for rep in range(repeats):
        tv_idx, test_idx = train_test_split(
            idx, test_size=0.15, stratify=y, random_state=seed + rep
        )
        tr_idx, va_idx = train_test_split(
            tv_idx,
            test_size=15 / 85,
            stratify=y[tv_idx],
            random_state=seed + rep,
        )
        train, val, test = pool.iloc[tr_idx], pool.iloc[va_idx], pool.iloc[test_idx]
        meta = {
            "protocol": "P-leaked",
            "fold": 0,
            "repeat": rep,
            "seed": seed + rep,
            "note": "image-level split; derivatives of the same source may cross roles",
        }
        write_triplet(out_dir, f"fold0_rep{rep}", train, val, test, meta, summaries)


def build_partial(df, out_dir, folds, repeats, seed, levels, summaries):
    base = next(disjoint_folds(df, folds, repeats=repeats, seed=seed))
    k, rep, base_train, base_val, test_raw, _ = base
    test_groups = sorted(test_raw["group_id"].unique())
    rng = np.random.default_rng(seed)
    for level in levels:
        n_leak = 0 if level == 0 else max(1, int(round(len(test_groups) * level / 100)))
        chosen = set(rng.choice(test_groups, size=n_leak, replace=False)) if n_leak else set()
        leak = df[df["is_derivative"] & df["group_id"].isin(chosen)]
        train = pd.concat([base_train, leak]).drop_duplicates("path")
        leak_groups = set(test_groups) & set(chosen)
        achieved = len(leak_groups) / len(test_groups) if test_groups else 0.0
        meta = {
            "protocol": "P-partial",
            "fold": k,
            "repeat": rep,
            "seed": seed,
            "requested_leak_pct": level,
            "achieved_leak_pct": round(100 * achieved, 2),
            "leaked_test_groups": len(leak_groups),
            "test_groups": len(test_groups),
            "note": "leakage is intentional; do not use for headline metrics",
        }
        write_triplet(
            out_dir / f"leak{level}", f"fold{k}_rep{rep}",
            train, base_val, test_raw, meta, summaries,
        )


def write_report(results: Path, summaries: list, metadata: dict) -> None:
    (results / "split_stats.json").write_text(
        json.dumps({"dataset": metadata, "splits": summaries}, indent=2) + "\n"
    )
    lines = [
        "# Split statistics",
        "",
        "Generated by `scripts/make_splits.py`. Paths in the CSVs are relative to `data/dataset/`.",
        "",
        "## Caveats",
        "",
        f"- Raw images released: {metadata['n_raw']}.",
        f"- Raw images with an inferred class usable for evaluation: {metadata['labelled_raw']}.",
        f"- Raw images without a recoverable class (excluded): {metadata['unlabelled_raw']}.",
        "- Raw labels are inferred from their augmented descendants, not author-provided; "
        "report this explicitly.",
        "- Test and validation sets contain raw images only. Training uses raw + tier-A "
        "derivatives (`primary_usable`).",
        "- Minor classes have very few raw groups, so per-class test power is low "
        "(see the roadmap, Gate G2).",
        "",
        "## Raw groups per class",
        "",
        "| Class | Raw images | Groups |",
        "|---|---:|---:|",
    ]
    for cls, stats in metadata["raw_by_class"].items():
        lines.append(f"| {cls} | {stats['images']} | {stats['groups']} |")
    lines += [
        "",
        "## Split sizes",
        "",
        "| Protocol | Split | Images | Raw | Derivatives | Groups | "
        + " | ".join(CLASSES) + " |",
        "|---|---|---:|---:|---:|---:|" + "---:|" * len(CLASSES),
    ]
    for s in summaries:
        cls_counts = " | ".join(str(s["classes"][c]) for c in CLASSES)
        lines.append(
            f"| {s['protocol']} | {s['tag']} {s['role']} | {s['images']} | {s['raw']} | "
            f"{s['derivatives']} | {s['groups']} | {cls_counts} |"
        )
    (REPO / "docs" / "split_stats.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--protocol", choices=("leaked", "disjoint", "partial", "all"), default="all"
    )
    args = ap.parse_args()

    df, metadata = load_frame(args.results)
    raw = df[~df["is_derivative"]]
    raw_by_class = {}
    for cls in CLASSES:
        subset = raw[raw["class_label"] == cls]
        raw_by_class[cls] = {
            "images": int(len(subset)),
            "groups": int(subset["group_id"].nunique()),
        }
    metadata = {
        **metadata,
        "labelled_raw": int(len(raw)),
        "unlabelled_raw": int(metadata["n_raw"] - len(raw)),
        "raw_by_class": raw_by_class,
    }

    summaries: list[dict] = []
    if args.protocol in ("leaked", "all"):
        build_leaked(df, args.splits / "leaked", args.repeats, args.seed, summaries)
    if args.protocol in ("disjoint", "all"):
        build_disjoint(
            df, args.splits / "disjoint", args.folds, args.repeats, args.seed, summaries
        )
    if args.protocol in ("partial", "all"):
        build_partial(
            df, args.splits / "partial", args.folds, args.repeats, args.seed,
            levels=(0, 25, 50, 100), summaries=summaries,
        )

    write_report(args.results, summaries, metadata)
    print(f"wrote {len(summaries)} split files under {args.splits}")
    print(f"labelled raw images: {metadata['labelled_raw']} / {metadata['n_raw']}")


if __name__ == "__main__":
    main()
