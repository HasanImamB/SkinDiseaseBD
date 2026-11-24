#!/usr/bin/env python3
"""Build a master image metadata table from the audit and group reconstruction outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CLASSES = ("Dermatitis", "Eczema", "Scabies", "Tinea Ringworm", "Vitiligo")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, default=REPO / "results" / "file_index.csv")
    ap.add_argument("--groups", type=Path, default=REPO / "results" / "raw_groups.json")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "master_metadata.csv")
    args = ap.parse_args()

    index = pd.read_csv(args.index)
    groups = json.loads(args.groups.read_text())
    assignments = {row["path"]: row for row in groups["images"]}
    index_paths = set(index["path"])
    missing = [path for path in index["path"] if path not in assignments]
    extra = [path for path in assignments if path not in index_paths]
    if missing or extra:
        raise ValueError(
            f"index/group mismatch: {len(missing)} missing, {len(extra)} extra"
        )

    records = []
    for row in index.itertuples(index=False):
        group = assignments[row.path]
        inferred = group["class"]
        if row.version == "raw":
            label_source = (
                "unlabelled" if inferred == "unknown" else "inferred_from_descendants"
            )
        else:
            label_source = "folder_label"
        records.append(
            {
                "path": row.path,
                "version": row.version,
                "filename": row.filename,
                "class": inferred,
                "label_source": label_source,
                "labelled": inferred in CLASSES,
                "raw_name": row.raw_name,
                "raw_id": group["raw_id"],
                "group_id": group["group_id"],
                "raw_supergroup_id": group["raw_supergroup_id"],
                "is_derivative": group["is_derivative"],
                "tier": group["tier"],
                "primary_usable": group["primary_usable"],
                "group_class": group["group_class"],
                "width": row.width,
                "height": row.height,
                "bytes": row.bytes,
            }
        )

    frame = pd.DataFrame.from_records(records).sort_values("path").reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(
        f"wrote {args.out}: {len(frame)} images, "
        f"{frame['group_id'].nunique()} groups, "
        f"{int(frame['labelled'].sum())} labelled"
    )


if __name__ == "__main__":
    main()
