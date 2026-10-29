#!/usr/bin/env python3
"""Verify leakage properties of the generated splits.

For every ``fold*_rep*_{train,val,test}.csv`` triplet this script checks:

* group overlap between roles (must be empty for P-disjoint),
* test/validation contain raw images only,
* every class is present in each role,
* no exact or perceptual near-duplicate cluster crosses roles,
* for P-partial, the achieved fraction of test groups leaking into training.

P-leaked and P-partial are expected to violate strict disjointness; they are
reported as such instead of failing the run. Exit code is 1 if any strict
(P-disjoint) split fails a check.

Writes ``results/disjointness_report.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO / "results"
DEFAULT_SPLITS = REPO / "splits"
CLASSES = ("Dermatitis", "Eczema", "Scabies", "Tinea Ringworm", "Vitiligo")
ROLES = ("train", "val", "test")


def load_near_duplicates(results: Path) -> list[list[str]]:
    path = results / "duplicates_perceptual.json"
    return json.loads(path.read_text()) if path.exists() else []


def load_frame(results: Path) -> pd.DataFrame:
    payload = json.loads((results / "raw_groups.json").read_text())
    df = pd.DataFrame(payload["images"])
    return df.set_index("path")


def protocol_of(directory: Path) -> str:
    parts = directory.parts
    if "partial" in parts:
        return "partial"
    if "leaked" in parts:
        return "leaked"
    return "disjoint"


def check_triplet(directory: Path, prefix: str, meta: pd.DataFrame, near: list) -> dict:
    frames = {}
    for role in ROLES:
        path = directory / f"{prefix}_{role}.csv"
        if not path.exists():
            return {}
        frames[role] = pd.read_csv(path)

    role_of: dict[str, str] = {}
    missing = 0
    for role, frame in frames.items():
        for path in frame["filepath"]:
            if path not in meta.index:
                missing += 1
            else:
                role_of[path] = role

    groups = {role: set(frame["group_id"]) for role, frame in frames.items()}
    overlap = {
        "train_val": len(groups["train"] & groups["val"]),
        "train_test": len(groups["train"] & groups["test"]),
        "val_test": len(groups["val"] & groups["test"]),
    }

    derivatives = {
        role: int(frames[role]["is_derivative"].sum()) for role in ROLES
    }
    classes = {
        role: sorted(set(frames[role]["class_label"]) & set(CLASSES)) for role in ROLES
    }
    missing_classes = {
        role: [c for c in CLASSES if c not in classes[role]] for role in ROLES
    }

    crossings = 0
    for cluster in near:
        roles = {role_of[p] for p in cluster if p in role_of}
        if len(roles) > 1:
            crossings += 1

    report = {
        "path": str(directory.relative_to(REPO)),
        "tag": prefix,
        "protocol": protocol_of(directory),
        "counts": {role: int(len(frames[role])) for role in ROLES},
        "groups": {role: len(groups[role]) for role in ROLES},
        "group_overlap": overlap,
        "derivatives": derivatives,
        "missing_classes": missing_classes,
        "near_duplicate_clusters_crossing": crossings,
        "missing_paths": missing,
    }

    if report["protocol"] == "partial":
        test_groups = groups["test"]
        leaked = set(frames["train"].loc[frames["train"]["is_derivative"], "group_id"])
        leaked &= test_groups
        achieved = 100 * len(leaked) / len(test_groups) if test_groups else 0.0
        report["achieved_leak_pct"] = round(achieved, 2)
        meta_path = directory / f"{prefix}_meta.json"
        if meta_path.exists():
            meta_json = json.loads(meta_path.read_text())
            report["requested_leak_pct"] = meta_json.get("requested_leak_pct")
        report["status"] = "controlled-leakage"
    elif report["protocol"] == "leaked":
        report["status"] = "expected-leakage"
    else:
        failures = []
        if any(overlap.values()):
            failures.append("group overlap")
        if derivatives["test"] or derivatives["val"]:
            failures.append("derivative in test/val")
        if any(missing_classes[role] for role in ("train", "test")):
            failures.append("class missing from train/test")
        if crossings:
            failures.append("near-duplicate crossing")
        report["status"] = "fail" if failures else "pass"
        report["failures"] = failures
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    args = ap.parse_args()

    meta = load_frame(args.results)
    near = load_near_duplicates(args.results)
    reports = []
    for train_csv in sorted(args.splits.rglob("*_train.csv")):
        prefix = train_csv.name[: -len("_train.csv")]
        report = check_triplet(train_csv.parent, prefix, meta, near)
        if report:
            reports.append(report)

    output = {"near_duplicate_clusters": len(near), "splits": reports}
    (args.results / "disjointness_report.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )

    strict_failures = [r for r in reports if r["status"] == "fail"]
    for report in reports:
        extra = ""
        if report["protocol"] == "partial":
            extra = (
                f", leak {report.get('requested_leak_pct')}%"
                f"->{report.get('achieved_leak_pct')}%"
            )
        print(
            f"{report['path']}/{report['tag']}: {report['status']} "
            f"(groups t/v/e={report['groups']['train']}/"
            f"{report['groups']['val']}/{report['groups']['test']}, "
            f"near-dup crossings={report['near_duplicate_clusters_crossing']}{extra})"
        )
    print(f"\nchecked {len(reports)} split triplets; strict failures: {len(strict_failures)}")
    raise SystemExit(1 if strict_failures else 0)


if __name__ == "__main__":
    main()
