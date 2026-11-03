#!/usr/bin/env python3
"""Shortcut probes on the P-disjoint folds (analysis plan Amendment 1).

Three probe families are evaluated on the same 5 folds x 3 repeats as the main
P-disjoint protocol:

* metadata only (raw images): size, aspect ratio, file size, bytes/pixel,
  orientation, JPEG quantisation and available EXIF fields;
* border-only inputs: keep the outer 20% band and mask the centre;
* background-only proxy: mask the central 50% square and keep the remainder.

DINOv2 and ResNet-50 embeddings are recomputed for the masked variants and the
same 1-NN / linear probes are used. Stratified-dummy and majority-class
references are included for every table. Masked images live under
``results/shortcut/images/`` and embeddings under ``results/shortcut/``.

Example:
    python scripts/shortcut_probe.py --features dinov2 resnet --variants full border20 background50
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import ExifTags, Image
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torchvision.models import ResNet50_Weights, resnet50

from leakage_probe import (
    CLASSES,
    REPO,
    bootstrap,
    make_probe,
    paired_delta,
    pooled_predictions,
    predict,
    score,
)

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

DATA = REPO / "data" / "dataset"
DEFAULT_SPLITS = REPO / "splits"
DEFAULT_OUT = REPO / "results" / "shortcut"
IMG_EXT = {".jpg", ".jpeg", ".png"}
VARIANTS = ("full", "border20", "background50", "background75")
FEATURES = ("dinov2", "resnet")
NUMERIC_META = [
    "width",
    "height",
    "aspect_ratio",
    "bytes",
    "bytes_per_pixel",
    "orientation",
    "jpeg_quant_sum",
    "jpeg_quant_nonzero",
    "exif_datetime_epoch",
    "exif_hour",
]
CATEGORICAL_META = ["camera_make", "camera_model"]
METADATA_ABLATIONS = {
    "size_geometry": [
        "width",
        "height",
        "aspect_ratio",
        "bytes",
        "bytes_per_pixel",
        "orientation",
        "jpeg_quant_sum",
        "jpeg_quant_nonzero",
    ],
    "datetime": ["exif_datetime_epoch", "exif_hour"],
}


def selected_paths(splits: Path) -> list[str]:
    paths: set[str] = set()
    for csv_path in sorted((splits / "disjoint").glob("*_*.csv")):
        paths.update(pd.read_csv(csv_path)["filepath"])
    return sorted(paths)


def mask_variant(image: np.ndarray, variant: str) -> np.ndarray:
    if variant == "full":
        return image
    height, width = image.shape[:2]
    if variant == "border20":
        y0, y1 = int(round(0.2 * height)), int(round(0.8 * height))
        x0, x1 = int(round(0.2 * width)), int(round(0.8 * width))
    elif variant == "background50":
        y0, y1 = int(round(0.25 * height)), int(round(0.75 * height))
        x0, x1 = int(round(0.25 * width)), int(round(0.75 * width))
    elif variant == "background75":
        y0, y1 = int(round(0.125 * height)), int(round(0.875 * height))
        x0, x1 = int(round(0.125 * width)), int(round(0.875 * width))
    else:
        raise ValueError(f"unknown variant: {variant}")
    masked = image.copy()
    masked[y0:y1, x0:x1] = 0
    return masked


def build_images(data_dir: Path, paths: list[str], out_dir: Path, variants, force: bool) -> None:
    for index, rel in enumerate(paths, start=1):
        source = data_dir / rel
        with Image.open(source) as image:
            resized = np.asarray(
                image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            )
        for variant in variants:
            target = out_dir / variant / rel
            if target.exists() and not force:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask_variant(resized, variant)).save(target, quality=95)
        if index % 250 == 0:
            print(f"  masked images {index}/{len(paths)}", flush=True)


def embed_variant(
    images_dir: Path,
    variant: str,
    paths: list[str],
    feature: str,
    out_path: Path,
    batch_size: int,
    force: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    if out_path.exists() and not force:
        archive = np.load(out_path, allow_pickle=True)
        return archive["features"].astype(np.float32), {
            p: i for i, p in enumerate(archive["paths"])
        }

    if feature == "resnet":
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        prep = weights.transforms()
        dim = 2048
    elif feature == "dinov2":
        model = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            img_size=224,
        )
        prep = timm.data.create_transform(
            input_size=(3, 224, 224),
            is_training=False,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        dim = int(model.num_features)
    else:
        raise ValueError(f"unknown feature: {feature}")

    model.eval()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    feats = np.empty((len(paths), dim), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            batch = torch.stack(
                [prep(Image.open(images_dir / variant / rel).convert("RGB")) for rel in chunk]
            )
            feats[start : start + len(chunk)] = model(batch).numpy()
            if start % (batch_size * 20) == 0:
                print(f"  {feature}/{variant} {start + len(chunk)}/{len(paths)}", flush=True)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, features=feats, paths=np.array(paths))
    return feats, {p: i for i, p in enumerate(paths)}


def exif_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def metadata_frame(data_dir: Path, paths: list[str]) -> pd.DataFrame:
    rows = []
    for rel in paths:
        if not rel.startswith("ResearchImage/"):
            continue
        path = data_dir / rel
        with Image.open(path) as image:
            width, height = image.size
            exif = image.getexif()
            tags = {ExifTags.TAGS.get(key, key): value for key, value in exif.items()}
            quantisation = getattr(image, "quantization", None) or {}
        values = [value for table in quantisation.values() for value in table]
        quant_sum = float(np.sum(values)) if values else np.nan
        quant_nonzero = float(np.sum(np.asarray(values) > 1)) if values else np.nan

        stamp = tags.get("DateTime")
        try:
            parsed = datetime.strptime(str(stamp), "%Y:%m:%d %H:%M:%S")
            epoch = parsed.timestamp()
            hour = float(parsed.hour)
        except (TypeError, ValueError):
            epoch = np.nan
            hour = np.nan

        rows.append(
            {
                "filepath": rel,
                "width": float(width),
                "height": float(height),
                "aspect_ratio": width / height,
                "bytes": float(path.stat().st_size),
                "bytes_per_pixel": path.stat().st_size / (width * height),
                "orientation": exif_number(tags.get("Orientation")),
                "jpeg_quant_sum": quant_sum,
                "jpeg_quant_nonzero": quant_nonzero,
                "exif_datetime_epoch": epoch,
                "exif_hour": hour,
                "exif_iso": exif_number(tags.get("ISOSpeedRatings", tags.get("PhotographicSensitivity"))),
                "exif_exposure_time": exif_number(tags.get("ExposureTime")),
                "exif_f_number": exif_number(tags.get("FNumber")),
                "exif_focal_length": exif_number(tags.get("FocalLength")),
                "camera_make": tags.get("Make"),
                "camera_model": tags.get("Model"),
            }
        )
    return pd.DataFrame(rows)


def colour_frame(data_dir: Path, paths: list[str], bins: int = 8) -> pd.DataFrame:
    rows = []
    for rel in paths:
        if not rel.startswith("ResearchImage/"):
            continue
        with Image.open(data_dir / rel) as image:
            rgb = np.asarray(
                image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ) / 255.0
            hsv = np.asarray(
                image.convert("HSV").resize((224, 224), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ) / 255.0
        row = {"filepath": rel}
        for channel_index, channel in enumerate("rgb"):
            values = rgb[:, :, channel_index]
            row[f"rgb_mean_{channel}"] = float(values.mean())
            row[f"rgb_std_{channel}"] = float(values.std())
            histogram, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
            histogram = histogram / values.size
            for bucket, share in enumerate(histogram):
                row[f"hist_{channel}_{bucket}"] = float(share)
        for channel_index, channel in enumerate("hsv"):
            values = hsv[:, :, channel_index]
            row[f"hsv_mean_{channel}"] = float(values.mean())
            row[f"hsv_std_{channel}"] = float(values.std())
        rows.append(row)
    return pd.DataFrame(rows)


def metadata_pipeline(
    numeric_cols: list[str] = NUMERIC_META,
    categorical_cols: list[str] = CATEGORICAL_META,
) -> Pipeline:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    transformers = [("num", numeric, numeric_cols)]
    if categorical_cols:
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical, categorical_cols))
    preprocess = ColumnTransformer(transformers)
    return Pipeline(
        [
            ("preprocess", preprocess),
            (
                "clf",
                LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, tol=1e-3),
            ),
        ]
    )


def prediction_frame(test: pd.DataFrame, y_pred, proba, classes) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "filepath": test["filepath"].to_numpy(),
            "group_id": test["group_id"].to_numpy(),
            "y_true": test["class_label"].to_numpy(),
            "y_pred": np.asarray(y_pred),
        }
    )
    classes = list(classes)
    for cls in CLASSES:
        frame[f"prob_{cls}"] = proba[:, classes.index(cls)] if cls in classes else 0.0
    return frame


def run_metadata(
    splits: Path,
    metadata: pd.DataFrame,
    n_boot: int,
    seed: int,
    numeric_cols: list[str] = NUMERIC_META,
    categorical_cols: list[str] = CATEGORICAL_META,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    meta = metadata.set_index("filepath")
    outputs = {name: [] for name in ("metadata_linear", "stratified_dummy", "majority")}
    for test_path in sorted((splits / "disjoint").glob("fold*_rep*_test.csv")):
        tag = test_path.name[: -len("_test.csv")]
        train = pd.read_csv(test_path.parent / f"{tag}_train.csv")
        test = pd.read_csv(test_path)
        train_raw = train[~train["is_derivative"]]
        x_train = meta.loc[train_raw["filepath"]]
        x_test = meta.loc[test["filepath"]]
        y_train = train_raw["class_label"].to_numpy()

        model = metadata_pipeline(numeric_cols, categorical_cols).fit(x_train, y_train)
        frame = prediction_frame(
            test, model.predict(x_test), model.predict_proba(x_test), model.classes_
        )
        frame["repeat"] = int(tag.rsplit("rep", 1)[1])
        outputs["metadata_linear"].append(frame)

        for name, strategy in (
            ("stratified_dummy", "stratified"),
            ("majority", "most_frequent"),
        ):
            dummy = DummyClassifier(strategy=strategy, random_state=seed)
            dummy.fit(y_train, y_train)
            frame = prediction_frame(
                test, dummy.predict(x_test), dummy.predict_proba(x_test), dummy.classes_
            )
            frame["repeat"] = int(tag.rsplit("rep", 1)[1])
            outputs[name].append(frame)

    results = {}
    pooled_frames = {}
    for name, frames in outputs.items():
        predictions = pd.concat(frames, ignore_index=True)
        pooled = pooled_predictions(predictions)
        pooled_frames[name] = pooled
        per_repeat = [
            {"repeat": int(repeat), **score(frame["y_true"], frame["y_pred"])}
            for repeat, frame in predictions.groupby("repeat")
        ]
        results[name] = {
            "per_repeat": per_repeat,
            "pooled_oof": score(pooled["y_true"], pooled["y_pred"]),
            "bootstrap": bootstrap(
                pooled["y_true"], pooled["y_pred"], pooled["group_id"], n_boot, seed
            ),
        }
    return results, pooled_frames


def compare_pooled(
    pooled: dict[str, pd.DataFrame],
    base_name: str,
    references,
    n_boot: int,
    seed: int,
) -> dict:
    comparisons = {}
    base = pooled[base_name].set_index("filepath")
    for reference in references:
        other = pooled[reference].set_index("filepath")
        common = base.index.intersection(other.index)
        aligned_base = base.loc[common]
        aligned_other = other.loc[common]
        comparisons[f"{base_name}_vs_{reference}"] = paired_delta(
            aligned_base["y_true"].to_numpy(),
            aligned_other["y_pred"].to_numpy(),
            aligned_base["y_pred"].to_numpy(),
            aligned_base["group_id"].to_numpy(),
            n_boot,
            n_boot,
            seed,
        )
    return comparisons


def run_embeddings(
    splits: Path,
    features: np.ndarray,
    index: dict[str, int],
    probe_name: str,
    n_boot: int,
    seed: int,
) -> dict:
    rows = []
    for test_path in sorted((splits / "disjoint").glob("fold*_rep*_test.csv")):
        tag = test_path.name[: -len("_test.csv")]
        train = pd.read_csv(test_path.parent / f"{tag}_train.csv")
        test = pd.read_csv(test_path)
        frame = predict(make_probe(probe_name), train, test, features, index)
        frame["repeat"] = int(tag.rsplit("rep", 1)[1])
        rows.append(frame)
    predictions = pd.concat(rows, ignore_index=True)
    pooled = pooled_predictions(predictions)
    per_repeat = [
        {"repeat": int(repeat), **score(frame["y_true"], frame["y_pred"])}
        for repeat, frame in predictions.groupby("repeat")
    ]
    return {
        "per_repeat": per_repeat,
        "pooled_oof": score(pooled["y_true"], pooled["y_pred"]),
        "bootstrap": bootstrap(
            pooled["y_true"], pooled["y_pred"], pooled["group_id"], n_boot, seed
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--features", nargs="+", default=list(FEATURES), choices=FEATURES)
    ap.add_argument("--probes", nargs="+", default=["1nn", "linear"], choices=("1nn", "linear"))
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--force-images", action="store_true")
    ap.add_argument("--force-embeddings", action="store_true")
    ap.add_argument("--skip-images", action="store_true")
    ap.add_argument("--skip-embeddings", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    images_dir = args.out / "images"
    paths = selected_paths(args.splits)
    print(f"selected {len(paths)} split images")

    if not args.skip_images:
        print("[1/3] building masked variants")
        build_images(args.data_dir, paths, images_dir, args.variants, args.force_images)

    print("[2/3] metadata probing")
    raw_paths = [p for p in paths if p.startswith("ResearchImage/")]
    metadata = metadata_frame(args.data_dir, raw_paths)
    metadata.to_csv(args.out / "metadata_raw.csv", index=False)
    metadata_results, pooled_metadata = run_metadata(
        args.splits, metadata, args.bootstrap, args.seed
    )
    metadata_comparisons = compare_pooled(
        pooled_metadata,
        "metadata_linear",
        ("stratified_dummy", "majority"),
        args.bootstrap,
        args.seed,
    )
    metadata_ablation = {}
    for name, columns in METADATA_ABLATIONS.items():
        ablation_results, _ = run_metadata(
            args.splits,
            metadata,
            args.bootstrap,
            args.seed,
            numeric_cols=columns,
            categorical_cols=CATEGORICAL_META,
        )
        metadata_ablation[name] = ablation_results["metadata_linear"]

    print("[2b/3] colour probing")
    colour = colour_frame(args.data_dir, raw_paths)
    colour.to_csv(args.out / "colour_raw.csv", index=False)
    colour_columns = [column for column in colour.columns if column != "filepath"]
    colour_results, pooled_colour = run_metadata(
        args.splits,
        colour,
        args.bootstrap,
        args.seed,
        numeric_cols=colour_columns,
        categorical_cols=[],
    )
    colour_comparisons = compare_pooled(
        pooled_colour,
        "metadata_linear",
        ("stratified_dummy", "majority"),
        args.bootstrap,
        args.seed,
    )

    payload = {
        "config": {
            "variants": args.variants,
            "features": args.features,
            "probes": args.probes,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "split_images": len(paths),
            "raw_metadata_rows": len(metadata),
            "classes": list(CLASSES),
            "numeric_metadata": NUMERIC_META,
            "categorical_metadata": CATEGORICAL_META,
        },
        "metadata": metadata_results,
        "metadata_comparisons": metadata_comparisons,
        "metadata_ablation": metadata_ablation,
        "colour": colour_results,
        "colour_comparisons": colour_comparisons,
        "colour_features": colour_columns,
        "embeddings": {},
    }

    print("[3/3] masked-embedding probing")
    for variant in args.variants:
        payload["embeddings"][variant] = {}
        for feature in args.features:
            embeddings_path = args.out / f"embeddings_{feature}_{variant}.npz"
            features, index = embed_variant(
                images_dir,
                variant,
                paths,
                feature,
                embeddings_path,
                args.batch_size,
                args.force_embeddings,
            )
            payload["embeddings"][variant][feature] = {}
            for probe_name in args.probes:
                result = run_embeddings(
                    args.splits, features, index, probe_name, args.bootstrap, args.seed
                )
                payload["embeddings"][variant][feature][probe_name] = result
                print(
                    f"  {variant}/{feature}/{probe_name}: "
                    f"macro-F1={result['pooled_oof']['macro_f1']:.3f} "
                    f"[{result['bootstrap']['macro_f1']['lo']:.3f}, "
                    f"{result['bootstrap']['macro_f1']['hi']:.3f}]"
                )

    out_path = args.out / "shortcut_probe.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
