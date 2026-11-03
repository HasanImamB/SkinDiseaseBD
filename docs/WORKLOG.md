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

## 2026-10-29 — Analysis plan and Phase 3 leakage probe

- Froze `docs/analysis_plan.md`: RQs, P-leaked/P-disjoint/P-partial, primary endpoint
  (pooled OOF raw macro-F1), group bootstrap, paired Delta and permutation test.
- Implemented `scripts/leakage_probe.py` (frozen DINOv2/ResNet-50 features; 1-NN and linear).
- Ran on all four feature/probe combinations; wrote `results/leakage_probe.json`,
  `results/predictions/` and `docs/leakage_analysis.md`.

Results (macro-F1):
- P-disjoint: 0.54-0.66; P-leaked: ~1.00.
- Paired P-partial Delta (100% - 0%): +0.35 to +0.51; CI excludes zero for 3/4 probes.
- Dose-response is monotone for 3/4 configurations; ResNet-linear is non-monotone at 50%.
- Scabies (0.30 recall) and Dermatitis (0.23-0.38) are the hardest classes.

Caveats: P-partial uses one fold (27 raw test images), so one Delta CI includes zero;
61/197 raw images remain unlabelled. Next: extend P-partial to all folds, then Phase 4.

---

## 2026-10-29 — Phase 3 probe finalised

- Validated `scripts/*.py`, `results/leakage_probe.json` and all 22 split triplets
  (`verify_disjointness.py` exit 0).
- Updated `README.md` and `Proposal/TODO_AI_Agent.md` to reflect the preliminary
  frozen-feature leakage result.
- Deferred the all-fold P-partial extension to the next work session.

---

## 2026-10-29 — Next-session shortcut-probe guidance recorded

Recorded Amendment 1 in `docs/analysis_plan.md`: reuse the 5x3 group folds and group bootstrap,
add stratified-dummy and majority-class references, reuse the DINOv2/ResNet-50 1-NN and linear
probes on border-only and background-only inputs, and test metadata-only predictability from
size/aspect/file size/EXIF. No shortcut probe was run today; the amendment is documentation only.

---

## 2026-11-03 — All-fold P-partial extension

Extended the controlled-leakage dose-response from one fold to all 5 folds x 3 repeats.

Deliverables:
- `scripts/make_splits.py` now builds P-partial for every P-disjoint fold/repeat, using a
  deterministic per-fold RNG (`seed + 1000*repeat + fold`).
- `scripts/leakage_probe.py` aggregates P-partial predictions across all folds/repeats and
  reports per-fold, per-repeat, pooled OOF, group-bootstrap CIs and the paired Delta.
- Regenerated `splits/partial/` (60 triplets, 180 CSVs), `results/leakage_probe.json`,
  `results/predictions/`, `results/split_stats.json`, `docs/split_stats.md`.
- `verify_disjointness.py`: 78 triplets checked, 0 strict failures.

Results (macro-F1, pooled OOF, P-partial 100% - 0%):
- DINOv2 / 1-NN: +0.250 [0.139, 0.373], p=0.001.
- DINOv2 / linear: +0.271 [0.153, 0.393], p=0.001.
- ResNet-50 / 1-NN: +0.362 [0.260, 0.480], p=0.001.
- ResNet-50 / linear: +0.311 [0.202, 0.433], p=0.001.
- Dose-response is monotone for all four configurations; achieved leakage is
  0.0 / 22.9 / 50.0 / 100.0% for requested 0 / 25 / 50 / 100%.

Interpretation change: the single-fold estimate (+0.35 to +0.51) was optimistic. The all-fold
paired inflation is +0.25 to +0.36 macro-F1, and every CI excludes zero. The descriptive
P-leaked vs P-disjoint gap is 33-46 points. `docs/leakage_analysis.md`, `README.md`,
`docs/analysis_plan.md` and `Proposal/TODO_AI_Agent.md` were updated accordingly.

Next: Amendment 1 shortcut probes, then the fine-tuned baseline / Phase 4 pilot.

## 2026-11-03 — Shortcut probes (Amendment 1)

Implemented and ran `scripts/shortcut_probe.py` on the same 5x3 P-disjoint folds.

Deliverables:
- `results/shortcut/shortcut_probe.json`, `results/shortcut/metadata_raw.csv`.
- Masked images under `results/shortcut/images/` (gitignored) and cached embeddings under
  `results/shortcut/embeddings_{feature}_{variant}.npz` (gitignored).
- `docs/shortcut_analysis.md` with tables, interpretation and caveats.

Results (pooled OOF macro-F1):
- Metadata-only linear probe: 0.343 [0.251, 0.432]; stratified dummy 0.230; majority 0.140.
  Paired metadata - stratified dummy = +0.114 [0.006, 0.220], p=0.095; metadata - majority =
  +0.203 [0.109, 0.287], p=0.003.
- Metadata ablation: size/geometry/JPEG 0.374; acquisition datetime 0.317. EXIF ISO/exposure
  fields are absent; camera model is nearly constant.
- Full-image reference: DINOv2 1-NN 0.669, linear 0.627; ResNet-50 1-NN 0.629, linear 0.648.
- Border20: 0.467-0.538 (9-20 points below full).
- Background50 (mild mask, keeps 75% of image): 0.537-0.621; ResNet-50 1-NN only -0.016
  versus full.

Interpretation: a weak-to-moderate metadata/acquisition shortcut exists, but metadata alone does
not explain the 0.54-0.66 disjoint floor. The masked variants stay above chance, yet the masks
are conservative, so no strong background shortcut can be claimed. Next: stricter background
mask, colour-histogram features, label-noise audit, then the fine-tuned baseline.

## 2026-11-03 — Shortcut sensitivity: stricter background and colour

Extended the shortcut probe with two sensitivity analyses requested by the roadmap.

- Added `background75` (central 75% x 75% masked, outer 12.5% band kept) to
  `scripts/shortcut_probe.py`.
- Added a colour-only feature family: mean/std RGB, mean/std HSV and 8-bin per-channel
  histograms on raw images.
- Regenerated `results/shortcut/shortcut_probe.json`, `results/shortcut/colour_raw.csv` and
  updated `docs/shortcut_analysis.md`.

Results (macro-F1):
- `background75`: 0.401-0.532, i.e. 10-27 points below full image. The earlier mild
  `background50` result (0.537-0.621, only 2-13 points below full) was partly an artefact of
  keeping 75% of the image.
- Colour-only: 0.246 [0.174, 0.305] vs stratified dummy 0.230; paired delta +0.016
  [-0.067, 0.112], p=0.789. No simple colour shortcut.
- Metadata remains the strongest shortcut family (0.343, size/geometry 0.374).

Interpretation: the full-image floor is not explained by metadata or global colour. A residual
border/crop cue or disease signal extending to the border remains; the crude masks cannot
disentangle them. Next: label-noise audit, then the fine-tuned baseline.

## 2026-11-03 — Automated label-quality triage

Added `scripts/label_audit.py`, which runs cleanlab confident learning on the pooled P-disjoint
OOF probabilities from the four frozen-feature probes and aggregates flags across probes.

Inputs: DINOv2 and ResNet-50, 1-NN and linear, 136 labelled raw images, 5x3 OOF predictions.
Outputs: `results/label_audit.json`, `results/label_audit.csv` and a gitignored review sheet at
`results/label_audit/contact_sheet.jpg`.

Results:
- Per-probe flags: DINOv2 1-NN 32, DINOv2 linear 29, ResNet-50 1-NN 50, ResNet-50 linear 28.
- 23/136 images flagged by at least 3 probes; 35/136 by at least 2.
- Flag rate by class: Scabies 70%, Dermatitis 54%, Tinea 24%, Vitiligo 9%, Eczema 1.4%.
- 18/23 high-confidence flags have Eczema as the consensus prediction.

Interpretation: the audit is inconclusive as evidence of label noise. With a 73/136 Eczema
majority and low minority-class recall, cleanlab is largely re-discovering model confusion. The
shortlist is useful for clinician review but no images were dropped or relabelled. Repeat the
audit after the fine-tuned/class-balanced benchmark; flags that survive a stronger model are
more credible.

Added `cleanlab==2.9.0` to `requirements.txt`, ignored the review-sheet directory, and documented
the method, tables and caveats in `docs/label_audit.md`.

## 2026-11-03 — Preliminary fine-tuned baseline (CPU)

Added `scripts/train.py`, a timm-based fine-tuning CLI for the P-disjoint splits (raw + tier-A
derivatives in train, raw-only val/test), and `scripts/summarize_finetuned.py` for pooled OOF
metrics, group bootstrap and paired comparisons against frozen probes.

Ran ResNet-50 (`resnet50.a1_in1k`) at 128 px, 5 folds x 1 repeat, max 10 epochs, patience 3,
AdamW lr 3e-4, label smoothing 0.1, batch 32, CPU. About 43 s per epoch.

Results:
- Per-fold macro-F1: 0.731, 0.572, 0.403, 0.630, 0.518; mean 0.571 +/- 0.123.
- Pooled OOF macro-F1 0.604 [0.477, 0.702], balanced accuracy 0.564.
- Per-class recall: Eczema 0.93, Tinea 0.62, Dermatitis 0.62, Vitiligo 0.45, Scabies 0.20.
- Paired deltas vs frozen probes: -0.019 vs ResNet-50 linear, -0.059 vs DINOv2 linear; all CIs
  include zero.

Interpretation: pipeline validated, but 128 px / 10 epochs / one repeat / 22-image validation is
too weak to challenge the frozen probes. Next: 160-224 px, class-balanced sampling, 3 seeds,
then the broader model zoo. Documented in `docs/finetuned_baseline.md`.
