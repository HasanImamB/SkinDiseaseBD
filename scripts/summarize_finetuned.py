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
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from leakage_probe import CLASSES, bootstrap, paired_delta, pooled_predictions, score

REPO = Path(__file__).resolve().parents[1]
METRICS = (
    "macro_f1",
    "balanced_accuracy",
    "accuracy",
    "weighted_f1",
    "mcc",
    "macro_auroc",
)
BOOTSTRAP_EXTRA = ("weighted_f1", "mcc")
PATTERN = re.compile(r"fold(\d+)_rep(\d+)(?:_seed(\d+))?_test\.csv$")


def prob_columns(frame: pd.DataFrame) -> np.ndarray:
    return frame[[f"prob_{cls}" for cls in CLASSES]].to_numpy(dtype=float)


def clinical_metrics(y_true, y_pred, probs: np.ndarray | None = None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(CLASSES))
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - (tp + fn + fp)
    metrics["weighted_f1"] = float(
        f1_score(y_true, y_pred, average="weighted", labels=list(CLASSES), zero_division=0)
    )
    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    metrics["per_class_precision"] = {
        cls: float(tp[i] / (tp[i] + fp[i]) if tp[i] + fp[i] else 0.0)
        for i, cls in enumerate(CLASSES)
    }
    metrics["per_class_recall"] = {
        cls: float(tp[i] / (tp[i] + fn[i]) if tp[i] + fn[i] else 0.0)
        for i, cls in enumerate(CLASSES)
    }
    metrics["per_class_specificity"] = {
        cls: float(tn[i] / (tn[i] + fp[i]) if tn[i] + fp[i] else 0.0)
        for i, cls in enumerate(CLASSES)
    }
    metrics["per_class_f1"] = {
        cls: float(
            2 * tp[i] / (2 * tp[i] + fp[i] + fn[i])
            if 2 * tp[i] + fp[i] + fn[i]
            else 0.0
        )
        for i, cls in enumerate(CLASSES)
    }
    metrics["confusion_matrix"] = {
        "labels": list(CLASSES),
        "rows_true_cols_pred": cm.tolist(),
    }
    metrics["macro_auroc"] = None
    if probs is not None and probs.ndim == 2 and probs.shape[1] == len(CLASSES):
        if set(CLASSES).issubset(np.unique(y_true)):
            try:
                metrics["macro_auroc"] = float(
                    roc_auc_score(
                        y_true,
                        probs,
                        multi_class="ovr",
                        average="macro",
                        labels=list(CLASSES),
                    )
                )
            except ValueError:
                pass
    return metrics


def bootstrap_extra(
    y_true, y_pred, groups, n_boot: int, seed: int
) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    uniques = np.unique(groups)
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in BOOTSTRAP_EXTRA}
    for _ in range(n_boot):
        picked = rng.choice(uniques, size=len(uniques), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == group) for group in picked])
        metrics = clinical_metrics(y_true[idx], y_pred[idx])
        for key in BOOTSTRAP_EXTRA:
            samples[key].append(metrics[key])
    return {
        key: {
            "mean": float(np.mean(values)),
            "lo": float(np.percentile(values, 2.5)),
            "hi": float(np.percentile(values, 97.5)),
        }
        for key, values in samples.items()
    }


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
        seed = int(match.group(3)) if match.group(3) else None
        frame = pd.read_csv(path)
        frames.append(frame)
        metrics = clinical_metrics(
            frame["y_true"], frame["y_pred"], prob_columns(frame)
        )
        metrics.update({"fold": fold, "repeat": repeat, "seed": seed})
        per_fold.append(metrics)

    combined = pd.concat(frames, ignore_index=True)
    pooled = pooled_predictions(combined)
    pooled_metrics = clinical_metrics(
        pooled["y_true"], pooled["y_pred"], prob_columns(pooled)
    )
    bootstrap_metrics = bootstrap(
        pooled["y_true"],
        pooled["y_pred"],
        pooled["group_id"],
        args.n_boot,
        args.seed,
    )
    extra_bootstrap = bootstrap_extra(
        pooled["y_true"], pooled["y_pred"], pooled["group_id"], args.n_boot, args.seed
    )
    per_fold = sorted(
        per_fold, key=lambda row: (row["repeat"], row["fold"], row["seed"] or 0)
    )
    fold_summary = {}
    for metric in METRICS:
        present = [row[metric] for row in per_fold if row[metric] is not None]
        if not present:
            fold_summary[metric] = None
            continue
        values = np.asarray(present, dtype=float)
        fold_summary[metric] = {
            "n": int(len(values)),
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
    args.out.mkdir(parents=True, exist_ok=True)
    confusion = pd.DataFrame(
        pooled_metrics["confusion_matrix"]["rows_true_cols_pred"],
        index=list(CLASSES),
        columns=list(CLASSES),
    )
    confusion.index.name = "true\\pred"
    confusion_path = args.out / f"{args.run_name}_confusion_matrix.csv"
    confusion.to_csv(confusion_path)

    payload = {
        "config": {
            "run_name": args.run_name,
            "prediction_files": [path.name for path in files],
            "seeds": sorted({row["seed"] for row in per_fold if row["seed"] is not None}),
            "n_boot": args.n_boot,
            "n_perm": args.n_perm,
            "seed": args.seed,
            "classes": list(CLASSES),
        },
        "pooled": pooled_metrics,
        "bootstrap": bootstrap_metrics,
        "bootstrap_clinical": extra_bootstrap,
        "per_fold": per_fold,
        "fold_summary": fold_summary,
        "comparisons": comparisons,
        "pooled_predictions": str(oof_path),
        "confusion_matrix": str(confusion_path),
    }
    summary_path = args.out / f"{args.run_name}_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"pooled macro-F1={pooled_metrics['macro_f1']:.4f} "
          f"[{bootstrap_metrics['macro_f1']['lo']:.4f}, "
          f"{bootstrap_metrics['macro_f1']['hi']:.4f}]")
    print(f"pooled balanced accuracy={pooled_metrics['balanced_accuracy']:.4f}")
    print(
        f"pooled weighted F1={pooled_metrics['weighted_f1']:.4f} "
        f"MCC={pooled_metrics['mcc']:.4f} "
        f"macro AUROC={pooled_metrics['macro_auroc']}"
    )
    print(
        f"per-fold macro-F1 mean={fold_summary['macro_f1']['mean']:.4f} "
        f"sd={fold_summary['macro_f1']['sd']:.4f}"
    )
    print(f"wrote {summary_path} and {oof_path}")


if __name__ == "__main__":
    main()
