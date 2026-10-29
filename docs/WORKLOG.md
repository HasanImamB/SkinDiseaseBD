# Work Log

Chronological record of completed work on the SkinDiseaseBD leakage/benchmark project.
Each entry is appended after a task is finished so the summary is preserved for future reference.

---

## 2026-10-29 — Phase 1 audit pipeline

Phase 1 audit pipeline is built and run. Deliverables:

- `scripts/extract_features.py --dinov2` — index, dihedral matching, ResNet-50 + DINOv2 embeddings
- `scripts/audit_data.py` — SHA-256/MD5, exact + pHash duplicates, image stats
- `scripts/build_groups.py` — raw super-groups + SIFT/RANSAC + embedding group assignment
- `results/` — `file_index.csv`, `checksums_sha256.csv`, `raw_groups.json`, `group_summary.csv`, embeddings, duplicate maps
- `docs/data_audit.md` — full write-up

Key findings:

- 197 raw images (`1.jpg`–`198.jpg`, `169.jpg` missing); 250 is the paper's collection claim, not the release.
- 1,612 v1 + 1,612 v2 = 3,224 augmented images, not 1,612. v1/v2 are independent augmentation passes, so both must be grouped.
- 0 exact duplicates; 451 pHash near-duplicate groups (1,251 files); v1 has 90 filename collisions across classes.
- Grouping: 135 raw super-groups; tier A (SIFT-confirmed) 1,507 images at 97.5% anchor agreement, tier B 91%. Reconstruction is over-merged vs the claimed ~250 sources, so it is conservative for leakage but not ground-truth.

Use `primary_usable == true` (raw + tier-A, 1,704 images) for the valid primary protocol; treat tier B as secondary only. The leftover 137 MB `*.zip.part-*` file is an incomplete duplicate and can be deleted.

Next: Phase 2 split construction (`P-leaked`, `P-disjoint`, `P-partial`) using the raw-only test rule.

---

## 2026-10-29 — Cleanup

Removed redundant files to reclaim disk space:

- Deleted the incomplete `*.zip.part-*` download and all `__pycache__` folders.
- Deleted the three nested archives inside the extracted dataset (`Raw_Images.zip`, `Images_176x176_v1.zip`, `Images_512x512_v2.zip`, ~980 MB freed). Their contents were verified to match the extracted folders exactly before deletion.
- The top-level master archive is no longer present on disk; the extracted dataset remains complete: 197 raw + 1,612 v1 + 1,612 v2 images.

Data location: `data/dataset/` (single canonical copy; duplicate `Proposal/Data/` removed).

---

## 2026-10-29 — Data folder consolidation

Removed the duplicate data location. The dataset now lives in exactly one place:
`data/dataset/` (197 raw + 1,612 v1 + 1,612 v2 = 3,421 files). The old
`Proposal/Data/` folder and the `data/dataset` symlink were removed.

---

## 2026-10-29 — Review of today's work

Status: Phase 1 audit pipeline executed end-to-end. Deliverables exist and the headline data facts are verified; grouping remains provisional, so Gate G1 is only partially passed.

Completed:
1. Repo skeleton (`src/`, `data/`, `splits/`, `configs/`, `results/`, `notebooks/`, `assets/`, `scripts/`, `docs/`) and `.gitignore`.
2. Inventory, SHA-256/MD5 checksums, exact/perceptual duplicate audit, image statistics.
3. Feature extraction: ResNet-50 and DINOv2 ViT-S/14 embeddings, dihedral raw matching.
4. Source-group reconstruction: 135 raw super-groups, SIFT/RANSAC tier A + embedding tier B assignment.
5. Documentation: `docs/data_audit.md`, `docs/WORKLOG.md`.
6. Cleanup and consolidation: single data copy at `data/dataset/`; redundant nested archives removed.

Verified facts:
- 3,421 files: 197 raw, 1,612 v1 (176x176), 1,612 v2 (512x512).
- 168 of 197 raw images are 4000x3000; remaining raw images vary in size.
- 0 exact duplicates; 451 pHash near-duplicate groups covering 1,251 files.
- v1 and v2 are independent augmentation sets; there are 3,224 augmented images in the release, not 1,612.
- 197 raw images are present (`1`-`198`, `169` missing); the paper's 250 is a collection claim, not a released count.

Issues and risks:
- Group reconstruction is not ground truth: 100 augmented-populated groups vs the claimed ~250 sources, max group 175 images. Tier A covers 46.7% of augmented images at 97.5% anchor agreement; tier B is only ~91% and should not drive headline results.
- Tier B rows remain in `raw_groups.json`; use the `primary_usable` flag (1,704 images) to avoid accidentally treating them as reliable.
- The top-level master archive is no longer on disk. Only the extracted dataset remains; checksums exist but there is no local archive backup.
- No git repository, no pinned `requirements.txt`/environment file.
- Proposal text is stale: still says 1,612 images, 197-250 raw, and "subject-disjoint"; TODO files still contain malformed `[tag [ ]` checkboxes.
- No automated tests or `verify_disjointness.py` yet.

Assessment:
- Audit counts, checksums and duplicate findings are solid and reproducible.
- Grouping is the weak link; it is conservative for leakage (over-merging) but cannot support exact source-count claims.
- Correct next step is Phase 2 splits using raw-only test images and `primary_usable` training data, with the P-leaked protocol built from the 1,612 v1 images.

Recommended next actions:
1. Initialise git and pin the environment.
2. Update proposal/TODO numbers and terminology; fix malformed checkboxes.
3. Implement `scripts/make_splits.py` and `scripts/verify_disjointness.py` under the raw-only test rule.
4. Decide whether to re-download/archive the master zip and verify all checksums.

---

## 2026-10-29 — Phase 2 split construction

Implemented and ran the split protocols required by the roadmap.

Deliverables:
- `scripts/make_splits.py` — builds P-leaked, P-disjoint and P-partial splits.
- `scripts/verify_disjointness.py` — automated leakage assertions and report.
- `splits/` — 66 CSVs + 22 metadata files across 22 split triplets.
- `results/split_stats.json`, `results/disjointness_report.json`.
- `docs/split_stats.md` — split sizes, group counts and per-class counts.
- `requirements.txt` — pinned CPU reference environment.
- Git repository initialised (no commit yet); `Proposal/TODO_AI_Agent.md` updated.

Protocol results:
- P-leaked: image-level 70/15/15 on 1,612 v1 images, 3 repeats; intentionally leaks
  (133-141 near-duplicate clusters cross roles).
- P-disjoint: 5 folds x 3 repeats; all 15 triplets pass. No group overlap, no derivatives in
  validation/test, no near-duplicate crossings, all classes present in train and test.
- P-partial: achieved leakage 0.0 / 23.5 / 47.1 / 100.0 % for requested 0/25/50/100 %.

Critical caveat:
- Only 136 of 197 raw images have a recoverable class; 61 are unlabelled and excluded.
- Per-class raw groups are imbalanced (Eczema 42, Tinea 20, Vitiligo 10, Dermatitis 9,
  Scabies 9), so per-class test sets are small (2-14 raw images). Gate G2 power is limited.
- Raw labels are inferred from descendants, not author-provided.

Next: Phase 3 leakage quantification (reproduce the baseline, 1-NN and linear probe on
P-leaked vs P-disjoint, group-bootstrap inflation CIs, P-partial dose-response).

---

## 2026-10-29 — Repository published

- Added `README.md` (project overview, layout, reproduction commands, key findings, licence note).
- Committed the reproducible package as `first commit` and pushed `main` to
  https://github.com/HasanImamB/SkinDiseaseBD.
- Dataset images and large `.npz` caches remain excluded via `.gitignore`.

---
