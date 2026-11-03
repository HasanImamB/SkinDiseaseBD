#!/usr/bin/env python3
"""Aggregate fine-tuned per-fold test predictions.

Reads ``results/predictions/finetuned_{run}_fold*_rep*_test.csv``, writes a
pooled OOF file, and reports per-fold, pooled and group-bootstrap metrics plus
paired comparisons against the frozen-feature probes.

Example:
    python scripts/summarize_finetuned.py --run-name resnet50_128
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from leakage_probe import CLASSES, bootstrap, paired_delta, pooled_predictions, score

REPO = Path(__file__).resolve().parents[1]
METRICS = ("macro_f1", "balanced_accuracy", "accuracy")
PATTERN = re.compile(r"fold(\d+)_rep(\d+)_test\.csv$")


def load_frozen(path: Path, filepaths: pd.Series) -> pd.DataFrame:
    frame = pd.read_csv(path).set_index("filepath")
    return frame.loc[filepaths].reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", required=True)
    ap.add_argument(
        "--predictions",
        type=Path,
        default=REPO / "results" / "predictions",
    )
    ap.add_argument("--out", type=Path, default=REPO / "results" / "finetuned")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = sorted(
        args.predictions.glob(f"finetuned_{args.run_name}_fold*_rep*_test.csv")
    )
    if not files:
        raise FileNotFoundError(
            f"no predictions for run {args.run_name} in {args.predictions}"
        )
    frames = []
    per_fold = []
    for path in files:
        match = PATTERN.search(path.name)
        if match is None:
            continue
        fold, repeat = int(match.group(1)), int(match.group(2))
        frame = pd.read_csv(path)
        frames.append(frame)
        metrics = score(frame["y_true"], frame["y_pred"])
        metrics.update({"fold": fold, "repeat": repeat})
        per_fold.append(metrics)

    combined = pd.concat(frames, ignore_index=True)
    pooled = pooled_predictions(combined)
    pooled_metrics = score(pooled["y_true"], pooled["y_pred"])
    bootstrap_metrics = bootstrap(
        pooled["y_true"],
        pooled["y_pred"],
        pooled["group_id"],
        args.n_boot,
        args.seed,
    )
    per_fold = sorted(per_fold, key=lambda row: (row["repeat"], row["fold"]))
    fold_summary = {}
    for metric in METRICS:
        values = np.array([row[metric] for row in per_fold])
        fold_summary[metric] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }

    comparisons = {}
    filepaths = pooled["filepath"]
    for name in ("dinov2_linear", "dinov2_1nn", "resnet_linear", "resnet_1nn"):
        path = args.predictions / f"disjoint_{name}_oof.csv"
        if not path.exists():
            continue
        frozen = load_frozen(path, filepaths)
        frozen_metrics = score(frozen["y_true"], frozen["y_pred"])
        delta = paired_delta(
            pooled["y_true"].to_numpy(),
            frozen["y_pred"].to_numpy(),
            pooled["y_pred"].to_numpy(),
            pooled["group_id"].to_numpy(),
            args.n_boot,
            args.n_perm,
            args.seed,
        )
        comparisons[name] = {
            "frozen_macro_f1": frozen_metrics["macro_f1"],
            "frozen_balanced_accuracy": frozen_metrics["balanced_accuracy"],
            "delta_finetuned_minus_frozen": delta,
        }

    oof_path = args.predictions / f"finetuned_{args.run_name}_oof.csv"
    pooled.to_csv(oof_path, index=False)

    payload = {
        "config": {
            "run_name": args.run_name,
            "prediction_files": [path.name for path in files],
            "n_boot": args.n_boot,
            "n_perm": args.n_perm,
            "seed": args.seed,
            "classes": list(CLASSES),
        },
        "pooled": pooled_metrics,
        "bootstrap": bootstrap_metrics,
        "per_fold": per_fold,
        "fold_summary": fold_summary,
        "comparisons": comparisons,
        "pooled_predictions": str(oof_path),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / f"{args.run_name}_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"pooled macro-F1={pooled_metrics['macro_f1']:.4f} "
          f"[{bootstrap_metrics['macro_f1']['lo']:.4f}, "
          f"{bootstrap_metrics['macro_f1']['hi']:.4f}]")
    print(f"pooled balanced accuracy={pooled_metrics['balanced_accuracy']:.4f}")
    print(
        f"per-fold macro-F1 mean={fold_summary['macro_f1']['mean']:.4f} "
        f"sd={fold_summary['macro_f1']['sd']:.4f}"
    )
    print(f"wrote {summary_path} and {oof_path}")


if __name__ == "__main__":
    main()
