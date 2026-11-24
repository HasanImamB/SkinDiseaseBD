#!/usr/bin/env python3
"""Confident-learning label audit on pooled out-of-fold raw-image predictions.

Reads the P-disjoint OOF prediction files produced by ``scripts/leakage_probe.py``,
runs cleanlab's confident-learning label filter for every available feature/probe
combination, and aggregates the flags per raw image.

Outputs:
  results/label_audit.json   summary, per-probe issues and consensus flags
  results/label_audit.csv    one row per labelled raw image

Example:
    python scripts/label_audit.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores

REPO = Path(__file__).resolve().parents[1]
CLASSES = ("Dermatitis", "Eczema", "Scabies", "Tinea Ringworm", "Vitiligo")
CLASS_INDEX = {name: index for index, name in enumerate(CLASSES)}


def load_predictions(
    predictions_dir: Path, pattern: str
) -> tuple[pd.DataFrame, list[str]]:
    files = sorted(predictions_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no OOF predictions matching {pattern} in {predictions_dir}")
    frames = {}
    for path in files:
        name = path.name.removeprefix("disjoint_").removesuffix("_oof.csv")
        frame = pd.read_csv(path).set_index("filepath")
        probs = frame[[f"prob_{cls}" for cls in CLASSES]].to_numpy(dtype=float)
        probs = probs / probs.sum(axis=1, keepdims=True)
        frame["probs"] = list(probs)
        frames[name] = frame
    reference = next(iter(frames.values())).index
    for name, frame in frames.items():
        if set(frame.index) != set(reference):
            raise ValueError(f"{name} has a different raw-image set")
    combined = pd.DataFrame(index=reference)
    for name, frame in frames.items():
        frame = frame.loc[reference]
        combined[f"y_true_{name}"] = frame["y_true"]
        combined[f"y_pred_{name}"] = frame["y_pred"]
        combined[f"prob_true_{name}"] = [
            probs[CLASS_INDEX[label]] for probs, label in zip(frame["probs"], frame["y_true"])
        ]
        combined[f"probs_{name}"] = list(frame["probs"])
    combined["group_id"] = next(iter(frames.values())).loc[reference, "group_id"]
    labels = next(iter(frames.values())).loc[reference, "y_true"]
    combined["y_true"] = labels
    return combined, sorted(frames)


def write_contact_sheet(
    audit: pd.DataFrame,
    data_dir: Path,
    out: Path,
    count: int,
) -> None:
    rows = audit.head(count)
    thumb = 180
    caption = 54
    columns = 5
    n_rows = int(np.ceil(len(rows) / columns))
    canvas = Image.new("RGB", (columns * thumb, n_rows * (thumb + caption)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11
        )
    except OSError:
        font = ImageFont.load_default()
    for position, (filepath, row) in enumerate(rows.iterrows()):
        col = position % columns
        line = position // columns
        x = col * thumb
        y = line * (thumb + caption)
        try:
            image = Image.open(data_dir / filepath)
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
            offset = ((thumb - image.width) // 2, (thumb - image.height) // 2)
            canvas.paste(image, (x + offset[0], y + offset[1]))
        except FileNotFoundError:
            draw.rectangle((x, y, x + thumb, y + thumb), fill="#eeeeee")
        colour = "#b00000" if row["high_confidence_issue"] else "#b06000"
        text = (
            f"{Path(filepath).name}  {row['y_true']}\n"
            f"pred {row['consensus_prediction']}  votes {row['issue_votes']}/4\n"
            f"mean quality {row['mean_label_quality']:.3f}"
        )
        draw.multiline_text((x + 4, y + thumb + 4), text, fill=colour, font=font)
        draw.rectangle((x, y, x + thumb - 1, y + thumb + caption - 1), outline="#cccccc")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=90)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--predictions",
        type=Path,
        default=REPO / "results" / "predictions",
    )
    ap.add_argument("--out", type=Path, default=REPO / "results" / "label_audit.json")
    ap.add_argument("--csv", type=Path, default=REPO / "results" / "label_audit.csv")
    ap.add_argument("--data-dir", type=Path, default=REPO / "data" / "dataset")
    ap.add_argument(
        "--contact-sheet",
        type=Path,
        default=REPO / "results" / "label_audit" / "contact_sheet.jpg",
    )
    ap.add_argument("--sheet-count", type=int, default=35)
    ap.add_argument("--min-votes", type=int, default=3)
    ap.add_argument("--candidate-votes", type=int, default=2)
    ap.add_argument("--pattern", default="disjoint_*_oof.csv")
    args = ap.parse_args()

    combined, probe_names = load_predictions(args.predictions, args.pattern)
    labels = combined["y_true"].to_numpy()
    label_ids = np.array([CLASS_INDEX[label] for label in labels])

    per_probe = {}
    issue_votes = pd.Series(0, index=combined.index, dtype=int)
    quality = pd.DataFrame(index=combined.index)
    for name in probe_names:
        probs = np.stack(combined[f"probs_{name}"].to_numpy())
        issues = find_label_issues(
            labels=label_ids,
            pred_probs=probs,
            return_indices_ranked_by="self_confidence",
            n_jobs=1,
        )
        scores = get_label_quality_scores(label_ids, probs)
        quality[name] = scores
        issue_paths = combined.index[issues]
        issue_votes.loc[issue_paths] += 1
        ranked = [
            {
                "filepath": str(combined.index[index]),
                "group_id": str(combined.iloc[index]["group_id"]),
                "y_true": labels[index],
                "y_pred": str(combined[f"y_pred_{name}"].iloc[index]),
                "prob_true": float(combined[f"prob_true_{name}"].iloc[index]),
                "label_quality": float(scores[index]),
            }
            for index in issues
        ]
        per_probe[name] = {
            "n_issues": int(len(issues)),
            "issues": ranked,
            "mean_label_quality": float(np.mean(scores)),
            "median_label_quality": float(np.median(scores)),
        }

    combined["issue_votes"] = issue_votes
    combined["mean_label_quality"] = quality.mean(axis=1)
    combined["min_label_quality"] = quality.min(axis=1)
    combined["high_confidence_issue"] = combined["issue_votes"] >= args.min_votes
    combined["candidate_issue"] = combined["issue_votes"] >= args.candidate_votes
    combined["consensus_prediction"] = combined[
        [f"y_pred_{name}" for name in probe_names]
    ].mode(axis=1)[0]

    ranked_columns = [
        "group_id",
        "y_true",
        "consensus_prediction",
        "issue_votes",
        "mean_label_quality",
        "min_label_quality",
        "high_confidence_issue",
        "candidate_issue",
    ]
    audit = combined[ranked_columns].copy()
    audit.index.name = "filepath"
    audit = audit.sort_values(
        ["issue_votes", "mean_label_quality"], ascending=[False, True]
    )
    audit.to_csv(args.csv)

    high_confidence = audit[audit["high_confidence_issue"]]
    candidates = audit[audit["candidate_issue"]]
    class_counts = audit["y_true"].value_counts().to_dict()
    high_confidence_rate_by_class = {
        cls: round(len(high_confidence[high_confidence["y_true"] == cls]) / count, 3)
        for cls, count in class_counts.items()
    }
    payload = {
        "config": {
            "probes": probe_names,
            "n_raw_images": int(len(audit)),
            "min_votes": args.min_votes,
            "classes": list(CLASSES),
            "method": "cleanlab confident learning, self_confidence ranking",
        },
        "summary": {
            "probe_issue_counts": {
                name: per_probe[name]["n_issues"] for name in probe_names
            },
            "high_confidence_issues": int(len(high_confidence)),
            "candidate_issues": int(len(candidates)),
            "high_confidence_paths": high_confidence.index.tolist(),
            "candidate_paths": candidates.index.tolist(),
            "high_confidence_by_class": high_confidence["y_true"].value_counts().to_dict(),
            "candidate_by_class": candidates["y_true"].value_counts().to_dict(),
            "high_confidence_by_consensus": high_confidence[
                "consensus_prediction"
            ].value_counts().to_dict(),
            "class_counts": class_counts,
            "high_confidence_rate_by_class": high_confidence_rate_by_class,
        },
        "per_probe": per_probe,
        "ranked_images": audit.reset_index().to_dict(orient="records"),
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"probes: {', '.join(probe_names)}")
    print(
        "issue counts:",
        {name: per_probe[name]["n_issues"] for name in probe_names},
    )
    print(
        f"high-confidence issues (>={args.min_votes} votes): {len(high_confidence)}"
        f" / {len(audit)} raw images"
    )
    print(f"candidate issues (>=2 votes): {len(candidates)} / {len(audit)}")
    write_contact_sheet(audit, args.data_dir, args.contact_sheet, args.sheet_count)
    print(f"wrote {args.out}, {args.csv} and {args.contact_sheet}")


if __name__ == "__main__":
    main()
