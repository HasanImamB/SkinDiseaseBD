#!/usr/bin/env python3
"""Frozen-feature leakage probe for the P-leaked, P-disjoint and P-partial splits.

Two probes are fitted on pre-computed embeddings:

* 1-nearest-neighbour with cosine distance,
* multinomial logistic regression on standardised features (C=1.0, balanced).

For each feature set the script reports per-repeat metrics, pooled out-of-fold
metrics, group-bootstrap confidence intervals, the paired P-partial 100% vs 0%
inflation, and a group-level permutation p-value. Predictions are written to
``results/predictions/`` and the summary to ``results/leakage_probe.json``.

Example:
    python scripts/leakage_probe.py --features dinov2 --probes 1nn linear
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "dataset"
CLASSES = ("Dermatitis", "Eczema", "Scabies", "Tinea Ringworm", "Vitiligo")
FEATURES = {
    "dinov2": REPO / "results" / "embeddings_dinov2.npz",
    "resnet": REPO / "results" / "embeddings.npz",
}
METRICS = ("macro_f1", "balanced_accuracy", "accuracy")


def relkey(path) -> str:
    path = Path(str(path))
    try:
        return path.resolve().relative_to(DATA.resolve()).as_posix()
    except ValueError:
        marker = "data/dataset/"
        text = path.as_posix()
        return text.split(marker, 1)[1] if marker in text else path.name


def load_features(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    archive = np.load(path, allow_pickle=True)
    features = archive["features"].astype(np.float32)
    index = {relkey(p): i for i, p in enumerate(archive["paths"])}
    return features, index


def matrix(frame: pd.DataFrame, features: np.ndarray, index: dict[str, int]) -> np.ndarray:
    missing = [p for p in frame["filepath"] if p not in index]
    if missing:
        raise KeyError(f"{len(missing)} split paths missing from embeddings, e.g. {missing[:3]}")
    return features[[index[p] for p in frame["filepath"]]]


def make_probe(name: str):
    if name == "1nn":
        return make_pipeline(
            Normalizer(norm="l2"),
            KNeighborsClassifier(n_neighbors=1, metric="cosine"),
        )
    if name == "linear":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, tol=1e-3),
        )
    raise ValueError(f"unknown probe: {name}")


def score(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=list(CLASSES), zero_division=0)
        ),
        "per_class_recall": {
            c: float(
                recall_score(y_true, y_pred, labels=[c], average="macro", zero_division=0)
            )
            for c in CLASSES
        },
    }


def fast_macro_f1(y_true, y_pred) -> float:
    return float(
        f1_score(y_true, y_pred, average="macro", labels=list(CLASSES), zero_division=0)
    )


def predict(probe, train: pd.DataFrame, test: pd.DataFrame, features, index) -> pd.DataFrame:
    probe.fit(matrix(train, features, index), train["class_label"].to_numpy())
    predicted = probe.predict(matrix(test, features, index))
    proba = probe.predict_proba(matrix(test, features, index))
    classes = list(getattr(probe, "classes_", probe[-1].classes_))
    out = pd.DataFrame(
        {
            "filepath": test["filepath"].to_numpy(),
            "group_id": test["group_id"].to_numpy(),
            "y_true": test["class_label"].to_numpy(),
            "y_pred": predicted,
        }
    )
    for cls in CLASSES:
        out[f"prob_{cls}"] = proba[:, classes.index(cls)] if cls in classes else 0.0
    return out


def bootstrap(y_true, y_pred, groups, n_boot: int, seed: int) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    uniques = np.unique(groups)
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in METRICS}
    for _ in range(n_boot):
        picked = rng.choice(uniques, size=len(uniques), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == group) for group in picked])
        sample = score(y_true[idx], y_pred[idx])
        for key in METRICS:
            samples[key].append(sample[key])
    return {
        key: {
            "mean": float(np.mean(values)),
            "lo": float(np.percentile(values, 2.5)),
            "hi": float(np.percentile(values, 97.5)),
        }
        for key, values in samples.items()
    }


def paired_delta(y_true, pred_low, pred_high, groups, n_boot: int, n_perm: int, seed: int) -> dict:
    y_true = np.asarray(y_true)
    pred_low = np.asarray(pred_low)
    pred_high = np.asarray(pred_high)
    groups = np.asarray(groups)
    observed = fast_macro_f1(y_true, pred_high) - fast_macro_f1(y_true, pred_low)
    uniques = np.unique(groups)
    rng = np.random.default_rng(seed)

    samples = []
    for _ in range(n_boot):
        picked = rng.choice(uniques, size=len(uniques), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == group) for group in picked])
        samples.append(
            fast_macro_f1(y_true[idx], pred_high[idx])
            - fast_macro_f1(y_true[idx], pred_low[idx])
        )

    permutations = []
    for _ in range(n_perm):
        swap = dict(zip(uniques, rng.random(len(uniques)) < 0.5))
        mask = np.array([swap[group] for group in groups])
        perm_low = np.where(mask, pred_high, pred_low)
        perm_high = np.where(mask, pred_low, pred_high)
        permutations.append(
            fast_macro_f1(y_true, perm_high) - fast_macro_f1(y_true, perm_low)
        )
    p_value = (np.sum(np.abs(permutations) >= abs(observed)) + 1) / (n_perm + 1)

    return {
        "delta": float(observed),
        "delta_lo": float(np.percentile(samples, 2.5)),
        "delta_hi": float(np.percentile(samples, 97.5)),
        "p_value": float(p_value),
    }


def split_tag(path: Path) -> str:
    return path.name[: -len("_test.csv")]


def run_leaked(splits: Path, features, index, probe_name, n_boot, seed) -> tuple[dict, pd.DataFrame]:
    rows = []
    for test_path in sorted((splits / "leaked").glob("fold*_rep*_test.csv")):
        tag = split_tag(test_path)
        train = pd.read_csv(test_path.parent / f"{tag}_train.csv")
        test = pd.read_csv(test_path)
        frame = predict(make_probe(probe_name), train, test, features, index)
        frame["repeat"] = int(tag.rsplit("rep", 1)[1])
        rows.append(frame)
    predictions = pd.concat(rows, ignore_index=True)
    per_repeat = []
    for repeat, frame in predictions.groupby("repeat"):
        per_repeat.append({"repeat": int(repeat), **score(frame["y_true"], frame["y_pred"])})
    return (
        {
            "per_repeat": per_repeat,
            "pooled": score(predictions["y_true"], predictions["y_pred"]),
            "bootstrap": bootstrap(
                predictions["y_true"], predictions["y_pred"], predictions["group_id"],
                n_boot, seed,
            ),
        },
        predictions,
    )


def run_disjoint(
    splits: Path, features, index, probe_name, n_boot, seed
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    rows = []
    for test_path in sorted((splits / "disjoint").glob("fold*_rep*_test.csv")):
        tag = split_tag(test_path)
        train = pd.read_csv(test_path.parent / f"{tag}_train.csv")
        test = pd.read_csv(test_path)
        frame = predict(make_probe(probe_name), train, test, features, index)
        frame["fold"] = int(tag.split("fold")[1].split("_")[0])
        frame["repeat"] = int(tag.rsplit("rep", 1)[1])
        rows.append(frame)
    predictions = pd.concat(rows, ignore_index=True)

    per_repeat = []
    for repeat, frame in predictions.groupby("repeat"):
        per_repeat.append({"repeat": int(repeat), **score(frame["y_true"], frame["y_pred"])})

    prob_cols = [f"prob_{c}" for c in CLASSES]
    pooled = predictions.groupby("filepath", as_index=False).agg(
        {**{col: "mean" for col in prob_cols}, "group_id": "first", "y_true": "first"}
    )
    pooled["y_pred"] = pooled[prob_cols].idxmax(axis=1).str.removeprefix("prob_")
    return (
        {
            "per_repeat": per_repeat,
            "pooled_oof": score(pooled["y_true"], pooled["y_pred"]),
            "bootstrap": bootstrap(
                pooled["y_true"], pooled["y_pred"], pooled["group_id"], n_boot, seed
            ),
        },
        predictions,
        pooled,
    )


def pooled_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    prob_cols = [f"prob_{c}" for c in CLASSES]
    pooled = predictions.groupby("filepath", as_index=False).agg(
        {**{col: "mean" for col in prob_cols}, "group_id": "first", "y_true": "first"}
    )
    pooled["y_pred"] = pooled[prob_cols].idxmax(axis=1).str.removeprefix("prob_")
    return pooled


def run_partial(
    splits: Path, features, index, probe_name, n_boot, n_perm, seed
) -> tuple[dict, pd.DataFrame]:
    results: dict[str, dict] = {}
    predictions = {}
    pooled = {}
    for level in (0, 25, 50, 100):
        directory = splits / "partial" / f"leak{level}"
        rows = []
        achieved = []
        for test_path in sorted(directory.glob("fold*_rep*_test.csv")):
            tag = split_tag(test_path)
            train = pd.read_csv(directory / f"{tag}_train.csv")
            test = pd.read_csv(test_path)
            frame = predict(make_probe(probe_name), train, test, features, index)
            frame["fold"] = int(tag.split("fold")[1].split("_")[0])
            frame["repeat"] = int(tag.rsplit("rep", 1)[1])
            rows.append(frame)
            achieved.append(json.loads((directory / f"{tag}_meta.json").read_text()))

        frame = pd.concat(rows, ignore_index=True)
        frame["leak_pct"] = level
        predictions[level] = frame
        pooled[level] = pooled_predictions(frame)

        per_fold = []
        for fold, fold_frame in frame.groupby("fold"):
            per_fold.append({"fold": int(fold), **score(fold_frame["y_true"], fold_frame["y_pred"])})
        per_repeat = []
        for repeat, repeat_frame in frame.groupby("repeat"):
            per_repeat.append(
                {"repeat": int(repeat), **score(repeat_frame["y_true"], repeat_frame["y_pred"])}
            )

        results[str(level)] = {
            "achieved_leak_pct": round(
                float(np.mean([m["achieved_leak_pct"] for m in achieved])), 2
            ),
            "achieved_leak_pct_by_fold": {
                f"fold{m['fold']}_rep{m['repeat']}": m["achieved_leak_pct"] for m in achieved
            },
            "per_fold": per_fold,
            "per_repeat": per_repeat,
            "pooled_oof": score(pooled[level]["y_true"], pooled[level]["y_pred"]),
            "macro_f1": score(pooled[level]["y_true"], pooled[level]["y_pred"])["macro_f1"],
            "bootstrap": bootstrap(
                pooled[level]["y_true"], pooled[level]["y_pred"],
                pooled[level]["group_id"], n_boot, seed,
            ),
        }

    low = pooled[0].set_index("filepath")
    high = pooled[100].set_index("filepath")
    common = low.index.intersection(high.index)
    low = low.loc[common]
    high = high.loc[common]
    delta = paired_delta(
        low["y_true"].to_numpy(),
        low["y_pred"].to_numpy(),
        high["y_pred"].to_numpy(),
        low["group_id"].to_numpy(),
        n_boot,
        n_perm,
        seed,
    )
    return {"levels": results, "delta_100_vs_0": delta}, pd.concat(
        predictions.values(), ignore_index=True
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=Path, default=REPO / "splits")
    ap.add_argument("--features", nargs="+", default=["dinov2", "resnet"], choices=FEATURES)
    ap.add_argument("--probes", nargs="+", default=["1nn", "linear"], choices=("1nn", "linear"))
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--permutations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "leakage_probe.json")
    ap.add_argument(
        "--predictions", type=Path, default=REPO / "results" / "predictions"
    )
    args = ap.parse_args()

    args.predictions.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict] = {}
    for feature_name in args.features:
        features, index = load_features(FEATURES[feature_name])
        feature_results = {}
        for probe_name in args.probes:
            leaked, leaked_pred = run_leaked(
                args.splits, features, index, probe_name, args.bootstrap, args.seed
            )
            disjoint, folds_pred, oof_pred = run_disjoint(
                args.splits, features, index, probe_name, args.bootstrap, args.seed
            )
            partial, partial_pred = run_partial(
                args.splits, features, index, probe_name, args.bootstrap,
                args.permutations, args.seed,
            )
            feature_results[probe_name] = {
                "leaked": leaked,
                "disjoint": disjoint,
                "partial": partial,
            }
            leaked_pred.to_csv(
                args.predictions / f"leaked_{feature_name}_{probe_name}.csv", index=False
            )
            folds_pred.to_csv(
                args.predictions / f"disjoint_{feature_name}_{probe_name}_folds.csv",
                index=False,
            )
            oof_pred.to_csv(
                args.predictions / f"disjoint_{feature_name}_{probe_name}_oof.csv",
                index=False,
            )
            partial_pred.to_csv(
                args.predictions / f"partial_{feature_name}_{probe_name}.csv", index=False
            )
            print(
                f"{feature_name}/{probe_name}: "
                f"disjoint macro-F1={disjoint['pooled_oof']['macro_f1']:.3f}, "
                f"leaked macro-F1={leaked['pooled']['macro_f1']:.3f}, "
                f"delta(100-0)={partial['delta_100_vs_0']['delta']:+.3f} "
                f"[{partial['delta_100_vs_0']['delta_lo']:+.3f}, "
                f"{partial['delta_100_vs_0']['delta_hi']:+.3f}]"
            )
        all_results[feature_name] = feature_results

    payload = {
        "config": {
            "features": args.features,
            "probes": args.probes,
            "bootstrap": args.bootstrap,
            "permutations": args.permutations,
            "seed": args.seed,
            "classes": list(CLASSES),
        },
        "results": all_results,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
