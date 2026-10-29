#!/usr/bin/env python3
"""Reconstruct source-image groups for SkinDiseaseBD.

Pipeline
  1. Merge near-identical raw images into super-groups (complete-linkage on
     DINOv2 embeddings, cosine distance <= 0.15).
  2. Match every augmented image to a raw image with two signals:
       - DINOv2 cosine similarity (shortlist)
       - SIFT + RANSAC inlier count (verification, rotation/zoom invariant)
     SIFT-confirmed matches are tier A; the rest fall back to the
     class-constrained embedding assignment (tier B).
  3. Map raw images to super-groups and emit one group id per image.

Outputs
  results/raw_groups.json     per-image group assignment + confidence
  results/group_summary.csv   per-group composition
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

REPO = Path(__file__).resolve().parents[1]
EMB = REPO / "results" / "embeddings_dinov2.npz"
DIH = REPO / "results" / "dihedral.npz"
INDEX = REPO / "results" / "file_index.csv"
RESULTS = REPO / "results"

RAW_CLUSTER_DIST = 0.15   # complete linkage, cosine distance (>= 0.85 similarity)
SIFT_SHORTLIST = 10
SIFT_MIN_INLIERS = 10
SIFT_DOMINANCE = 2.0


def make_sift():
    return cv2.SIFT_create(nfeatures=1500, contrastThreshold=0.008, edgeThreshold=15)


def sift_features(path: str, sift, max_dim: int = 640):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None, None
    h, w = im.shape
    scale = max_dim / max(h, w)
    if scale < 1:
        im = cv2.resize(im, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    kp, des = sift.detectAndCompute(im, None)
    if des is None:
        return None, None
    return np.float32([k.pt for k in kp]), des


def sift_inliers(f1, f2, bf, ratio: float = 0.8) -> int:
    p1, d1 = f1
    p2, d2 = f2
    if d1 is None or d2 is None or len(d1) < 8 or len(d2) < 8:
        return 0
    matches = bf.knnMatch(d1, d2, k=2)
    good = [a for a, b in matches if a.distance < ratio * b.distance]
    if len(good) < 6:
        return 0
    src = np.float32([p1[g.queryIdx] for g in good]).reshape(-1, 1, 2)
    dst = np.float32([p2[g.trainIdx] for g in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def main():
    idx = pd.read_csv(INDEX)
    emb = np.load(EMB, allow_pickle=True)
    dih = np.load(DIH, allow_pickle=True)
    path2i = {p: i for i, p in enumerate(emb["paths"])}
    feats = emb["features"]

    raw = idx[idx.version == "raw"].reset_index(drop=True)
    aug = idx[idx.version != "raw"].reset_index(drop=True)
    R = feats[[path2i[p] for p in raw.abs_path]]
    A = feats[[path2i[p] for p in aug.abs_path]]
    best_raw = dih["best_raw"]
    dih_corr = dih["corr"]
    aug_class = aug["class"].values

    print(f"[1/5] clustering {len(raw)} raw images")
    raw_sim = R @ R.T
    np.fill_diagonal(raw_sim, 1.0)
    dist = np.clip(1.0 - raw_sim, 0.0, 2.0)
    dist = (dist + dist.T) / 2.0
    link = linkage(squareform(dist, checks=False), method="complete")
    raw_group = fcluster(link, t=RAW_CLUSTER_DIST, criterion="distance") - 1
    n_groups = int(raw_group.max() + 1)
    print(f"  {len(raw)} raw -> {n_groups} super-groups (max {np.bincount(raw_group).max()} raws/group)")

    print("[2/5] group similarity scores")
    sim = A @ R.T
    group_score = np.full((len(aug), n_groups), -1.0, np.float32)
    for g in range(n_groups):
        group_score[:, g] = sim[:, raw_group == g].max(1)

    print(f"[3/5] SIFT verification against top-{SIFT_SHORTLIST} candidates")
    sift = make_sift()
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_feat = {}
    tier = np.array(["B"] * len(aug), dtype=object)
    assigned_raw = np.full(len(aug), -1, np.int32)
    sift_best = np.zeros(len(aug), np.int32)
    sift_second = np.zeros(len(aug), np.int32)
    for k in range(len(aug)):
        cand = list(np.argsort(sim[k])[::-1][:SIFT_SHORTLIST])
        if best_raw[k] not in cand:
            cand.append(int(best_raw[k]))
        fa = sift_features(aug.abs_path.iloc[k], sift)
        best_i, best_r, second_i = 0, -1, 0
        for r in cand:
            if r not in raw_feat:
                raw_feat[r] = sift_features(raw.abs_path.iloc[r], sift)
            inl = sift_inliers(fa, raw_feat[r], bf)
            if inl > best_i:
                second_i, best_i, best_r = best_i, inl, r
            elif inl > second_i:
                second_i = inl
        sift_best[k], sift_second[k] = best_i, second_i
        if best_i >= SIFT_MIN_INLIERS and best_i >= SIFT_DOMINANCE * max(second_i, 1):
            tier[k] = "A"
            assigned_raw[k] = best_r
        if (k + 1) % 250 == 0:
            print(f"  {k + 1}/{len(aug)} (tier A so far {(tier == 'A').sum()})", flush=True)
    print(f"  tier A (SIFT-confirmed): {(tier == 'A').sum()} / {len(aug)} ({(tier == 'A').mean():.3f})")

    print("[4/5] inferring group classes and assigning tier B")
    votes = collections.defaultdict(collections.Counter)
    for k in np.where(tier == "A")[0]:
        votes[int(raw_group[assigned_raw[k]])][aug_class[k]] += 1
    for k in np.where(dih_corr >= 0.9)[0]:
        votes[int(raw_group[best_raw[k]])][aug_class[k]] += 1
    group_class = {}
    for g, counter in votes.items():
        cls, n = counter.most_common(1)[0]
        if n >= 3:
            group_class[g] = cls

    for _ in range(5):
        score = group_score.copy()
        for g, cls in group_class.items():
            score[aug_class != cls, g] = -1.0
        pick = score.argmax(1)
        pick[tier == "A"] = raw_group[assigned_raw[tier == "A"]]
        # infer classes for groups that are still unknown
        assigned_votes = collections.defaultdict(collections.Counter)
        for g, cls in zip(pick, aug_class):
            assigned_votes[int(g)][cls] += 1
        added = 0
        for g, counter in assigned_votes.items():
            if g not in group_class:
                cls, n = counter.most_common(1)[0]
                if n >= 3:
                    group_class[g] = cls
                    added += 1
        if added == 0:
            break

    score = group_score.copy()
    for g, cls in group_class.items():
        score[aug_class != cls, g] = -1.0
    pick = score.argmax(1)
    pick[tier == "A"] = raw_group[assigned_raw[tier == "A"]]
    sorted_scores = np.sort(score, axis=1)
    group_margin = sorted_scores[:, -1] - sorted_scores[:, -2]

    # individual raw id for reporting
    for k in range(len(aug)):
        if tier[k] == "A":
            assigned_raw[k] = int(assigned_raw[k])
        else:
            members = np.where(raw_group == pick[k])[0]
            assigned_raw[k] = int(members[np.argmax(sim[k, members])])

    print("[5/5] writing group map")
    records = []
    for k, row in aug.reset_index(drop=True).iterrows():
        g = int(pick[k])
        records.append(
            {
                "path": row.path,
                "version": row.version,
                "class": row["class"],
                "group_id": f"G{g:03d}",
                "raw_id": f"{int(float(raw.raw_name.iloc[assigned_raw[k]])):03d}"
                if tier[k] == "A" or assigned_raw[k] >= 0
                else None,
                "raw_supergroup_id": f"G{g:03d}",
                "is_derivative": True,
                "tier": str(tier[k]),
                "sift_inliers": int(sift_best[k]),
                "sift_second": int(sift_second[k]),
                "embed_cos": float(sim[k, assigned_raw[k]]),
                "group_margin": float(group_margin[k]),
                "group_class": group_class.get(g, "unknown"),
                "primary_usable": bool(tier[k] == "A"),
            }
        )
    for i, row in raw.reset_index(drop=True).iterrows():
        g = int(raw_group[i])
        records.append(
            {
                "path": row.path,
                "version": "raw",
                "class": group_class.get(g, "unknown"),
                "group_id": f"G{g:03d}",
                "raw_id": f"{int(float(row.raw_name)):03d}",
                "raw_supergroup_id": f"G{g:03d}",
                "is_derivative": False,
                "tier": "raw",
                "sift_inliers": 0,
                "sift_second": 0,
                "embed_cos": 1.0,
                "group_margin": 0.0,
                "group_class": group_class.get(g, "unknown"),
                "primary_usable": True,
            }
        )

    meta = {
        "n_raw": int(len(raw)),
        "n_augmented": int(len(aug)),
        "n_supergroups": n_groups,
        "n_groups_used": int(len(set(pick))),
        "raw_cluster_distance": RAW_CLUSTER_DIST,
        "sift_shortlist": SIFT_SHORTLIST,
        "sift_min_inliers": SIFT_MIN_INLIERS,
        "tier_a": int((tier == "A").sum()),
        "tier_b": int((tier == "B").sum()),
        "group_class_counts": dict(collections.Counter(group_class.values())),
    }
    (RESULTS / "raw_groups.json").write_text(json.dumps({"metadata": meta, "images": records}, indent=2))

    rec = pd.DataFrame(records)
    summary = (
        rec.groupby(["group_id", "group_class", "version"]).size().unstack(fill_value=0).reset_index()
    )
    summary["total"] = summary.drop(columns=["group_id", "group_class"]).sum(axis=1)
    summary.to_csv(RESULTS / "group_summary.csv", index=False)

    # validation against independent dihedral anchors
    anchor = dih_corr >= 0.9
    anchor_correct = raw_group[assigned_raw] == raw_group[best_raw]
    for t in ("A", "B"):
        mask = anchor & (tier == t)
        if mask.any():
            print(f"  dihedral-anchor group agreement, tier {t}: {anchor_correct[mask].mean():.3f} (n={mask.sum()})")
    print("  group class counts:", meta["group_class_counts"])
    print("  groups used:", meta["n_groups_used"], "tier A:", meta["tier_a"])
    print(f"  wrote {RESULTS / 'raw_groups.json'} and {RESULTS / 'group_summary.csv'}")


if __name__ == "__main__":
    main()
