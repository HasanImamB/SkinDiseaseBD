# Analysis plan (frozen)

Version 1, 2026-10-29. Frozen **before** running any test-set experiment. Any later deviation is
recorded in `docs/WORKLOG.md` with its rationale.

## 1. Research questions

- **RQ1 (leakage):** How much does image-level splitting of pre-augmented data inflate performance
  on this dataset, and is the inflation class-dependent?
- **RQ2 (benchmark):** Under source-image-disjoint evaluation, how do CNN, transformer and
  foundation-model features compare, and which training tricks help under variance-aware testing?
- **RQ3 (shortcuts):** Are models learning disease cues or acquisition shortcuts?
- **RQ4 (transfer/triage):** Do models generalise to SkinDisNet, and are their probabilities
  usable for triage (calibration, abstention)?
- **RQ5 (explainability):** Do saliency maps align with clinician-marked lesions?

Phase 3 addresses RQ1 with frozen-feature probes. RQ2-RQ5 are addressed in later phases.

## 2. Data snapshot

| Item | Value |
|---|---|
| Raw images released | 197 |
| Raw images with a recoverable class | 136 (61 excluded as unlabelled) |
| Augmented images | 3,224 (1,612 v1 + 1,612 v2) |
| Classes | Dermatitis, Eczema, Scabies, Tinea Ringworm, Vitiligo |
| Source groups | provisional, over-merged (`results/raw_groups.json`) |

Raw class labels are **inferred** from their augmented descendants (98.9% agreement), not provided
by the authors. Derivative labels are the folder labels. Every result must state this.

## 3. Evaluation unit and leakage definition

- **Primary evaluation unit:** raw image. Test and validation sets contain raw images only.
- **Training unit:** raw images plus tier-A derivatives (`primary_usable == true`) from training
  groups only.
- **Leakage definition:** a test group has a derivative in the training set. Leakage fraction is
  the proportion of test groups with at least one derivative in training.
- Derivative-level results are secondary and are reported only with group-level bootstrap.

## 4. Protocols

| Protocol | Construction | Role |
|---|---|---|
| **P-leaked** | Image-level stratified 70/15/15 over the 1,612 v1 images, 3 repeats. Derivatives of one source may cross roles. | Reproduction of the original protocol; unpaired descriptive comparison. |
| **P-disjoint** | Repeated `StratifiedGroupKFold`, 5 folds x 3 repeats, on the 136 labelled raw images, grouped by `group_id`. Test/val raw only; train = raw + tier-A derivatives of training groups. | **Primary protocol.** |
| **P-partial** | From every P-disjoint fold / repeat, add derivatives of 0/25/50/100% of test groups to training. Test raw images are unchanged. | Paired leakage dose-response. |

Files: `splits/leaked/`, `splits/disjoint/`, `splits/partial/`. Paths are relative to
`data/dataset/`.

## 5. Primary and secondary endpoints

- **Primary endpoint:** macro-F1 on pooled out-of-fold raw-image predictions under P-disjoint,
  averaged over the 3 repeats.
- **Secondary endpoints:** balanced accuracy, MCC, per-class sensitivity/specificity, AUROC
  (one-vs-rest) where probabilities exist, and calibration metrics in later phases.
- **Leakage effect (RQ1):** paired difference
  `Delta = macro-F1(P-partial 100%) - macro-F1(P-partial 0%)` on the **same** raw test images.
  P-leaked vs P-disjoint is reported separately and is **not** treated as a paired comparison.

## 6. Probe models for Phase 3

Frozen features (no fine-tuning) to isolate the effect of leakage:

- DINOv2 ViT-S/14 (`results/embeddings_dinov2.npz`, 384-d).
- ResNet-50 ImageNet (`results/embeddings.npz`, 2048-d).

Two probes:

1. **1-nearest-neighbour** with cosine distance.
2. **Multinomial logistic regression** on standardised features, `C=1.0`,
   `class_weight="balanced"`, `max_iter=2000`.

Hyperparameters are fixed in advance; no tuning on test data. Validation splits are not used for
tuning in this phase.

## 7. Statistics

- **Group bootstrap:** resample `group_id` clusters with replacement (1,000 resamples), recompute
  the metric on the resampled predictions, report the 2.5th-97.5th percentile interval.
- **Paired Delta:** use the same group resamples for both leakage levels; report the bootstrap CI.
- **Permutation test for Delta:** group-level random swap of the two models' predictions,
  1,000 permutations, two-sided p-value.
- **Multiple comparisons:** Holm correction within each family of model/probe comparisons.
- **No McNemar** between protocols with different test sets. McNemar is reserved for two models on
  the same test set, aggregated per group.
- Report per-fold mean +/- standard deviation alongside pooled OOF numbers.

## 8. Decision rules

- Gate G2 (power): if a class has fewer than 20 raw test images per fold, report the interval and
  do not claim class-specific supremacy.
- Gate G3 (leakage): the leakage claim requires a positive Delta whose 95% CI excludes zero and a
  monotone P-partial dose-response.
- If the grouping coverage or label provenance is challenged, downgrade claims to "released-data
  lower bound" rather than "true source-disjoint".

## 9. Limitations to report with every result

- 61/197 raw images are unlabelled and excluded; 136 labelled raw images remain.
- Per-class raw groups are imbalanced (Eczema 42, Tinea 20, Vitiligo 10, Dermatitis 9, Scabies 9).
- Groups are reconstructed empirically and over-merged; they are not author-provided IDs.
- Raw labels are inferred from descendants.
- P-leaked evaluates derivatives while P-disjoint evaluates raw images; the two are not paired.

## 10. Amendment 1 — shortcut probes (first run 2026-11-03)

Status: first run complete; results in `docs/shortcut_analysis.md` and
`results/shortcut/shortcut_probe.json`. The first run also added a stricter `background75`
sensitivity mask and a colour-histogram probe. Purpose: decide whether the 0.54-0.66 P-disjoint
floor reflects disease difficulty or class-correlated acquisition shortcuts.

Design:

- Reuse the same 5 x 3 P-disjoint group folds and the same group-bootstrap function
  (1,000 resamples) so results are directly comparable with the leakage table.
- Include a **stratified dummy** and a **majority-class** reference in every table.
- Run three probe families:
  1. **Metadata only** (labelled raw images): width, height, aspect ratio, file size,
     bytes/pixel, JPEG quality, and EXIF fields (camera model, ISO, exposure) when present.
     Augmented images are excluded from metadata analysis because v1/v2 dimensions are fixed
     and would leak version rather than disease.
  2. **Border-only**: keep an outer band (initial choice: outer 20% of width/height) and mask
     the centre; re-embed with DINOv2 and ResNet-50; reuse the 1-NN and linear probes.
  3. **Background-only proxy**: mask the central 50% region and keep the remainder. There are
     no lesion masks, so this is an explicit proxy, not a true background segmentation.
- Evaluate on raw images as the primary shortcut analysis; derivatives only as a secondary
  sensitivity check.
- Report macro-F1, balanced accuracy and group-bootstrap CIs against the full-image P-disjoint
  baseline.
- Interpretation rule: if metadata or border probes beat chance materially, report acquisition
  bias and soften disease-intrinsic claims. If they do not, state only that no simple acquisition
  shortcut was detected; do not claim the model has learned disease.

Implementation notes:

- Write masked/cropped inputs under `results/shortcut/`; never modify `data/dataset/`.
- Cache shortcut embeddings as `results/shortcut/embeddings_{feature}_{variant}.npz`.
- Check EXIF availability before committing to an EXIF-based table.

Caveats: no lesion segmentation; border/background separation is approximate; raw metadata
sample is small and class-imbalanced.

## 11. Amendment 2 — preliminary fine-tuned baseline (2026-11-03)

Purpose: validate a fine-tuning pipeline and establish a first compute-bounded reference, not to
finalise the Phase 4 benchmark.

Design:

- Model: `resnet50.a1_in1k` (timm, ImageNet pretrained), replaced 5-way classifier.
- Input 128 x 128; batch 32; AdamW lr 3e-4, weight decay 0.05; cross-entropy with label
  smoothing 0.1; max 10 epochs, early stopping patience 3 on validation macro-F1.
- Train on raw + tier-A derivatives from training groups; validate and test on raw images only.
- Splits: the existing P-disjoint 5 folds, **repeat 0 only**.
- Augmentation: random resized crop (0.65-1.0), horizontal flip, rotation 10 deg, colour jitter.
- Report per-fold metrics, pooled OOF group bootstrap (1,000 resamples) and paired comparisons
  against the frozen probes using the same 136 raw images.

Result: pooled macro-F1 0.604 [0.477, 0.702]; per-fold mean 0.571 +/- 0.123. Paired delta versus
DINOv2 linear is -0.059 [-0.210, 0.069]. The run is weaker than the frozen probes and is treated
as a pipeline check.

Caveats: one repeat, 128 px, 22-image validation set, no class balancing, CPU runtime. Details in
`docs/finetuned_baseline.md` and `docs/WORKLOG.md`.

## 12. Amendment 3 — stronger class-balanced fine-tuned reference (2026-11-24)

Purpose: separate under-training/low-resolution effects from model capacity and training imbalance
before committing compute to the full model zoo. This remains a Phase 4 pilot, not the final
benchmark.

Design:

- Model: `resnet50.a1_in1k` (timm, ImageNet pretrained), 5-way classifier.
- Input 160 x 160; batch 32; AdamW lr 3e-4, weight decay 0.05; cosine schedule.
- Cross-entropy with label smoothing 0.1; class-balanced WeightedRandomSampler so each class
  contributes equally per epoch. No focal loss or class-weighted loss in this run.
- Max 12 epochs, early stopping patience 3 on validation macro-F1.
- Full P-disjoint grid: 5 folds x 3 repeats; one seed (42) for the first pass.
- Train on raw + tier-A derivatives from training groups; validate and test on raw images only.
- Save the best checkpoint, per-fold test probabilities and logits, pooled OOF predictions,
  group-bootstrap intervals and paired comparisons against the frozen probes.
- AMP is enabled by the CLI but disabled on this CPU-only environment.

Analysis: report per-fold and per-repeat macro-F1, pooled OOF macro-F1 with group-bootstrap CI,
per-class recall, and paired deltas against the four frozen probes. Additional seeds (43, 44) are
added only if the first pass is computationally feasible. A final benchmark claim still requires
the model zoo, training-trick ablations and calibration analysis.

Result (2026-11-24): all 15 runs completed with seed 42. Per-run macro-F1 mean 0.603 +/- 0.137;
pooled OOF macro-F1 0.681 [0.546, 0.799]. Paired deltas versus the frozen probes are +0.018
(DINOv2 linear), +0.030 (DINOv2 1-NN), +0.058 (ResNet-50 linear) and +0.145 (ResNet-50 1-NN),
with only the last CI excluding zero. Seed 42 was computationally feasible but seeds 43/44 were
not added on this CPU-only host; seed variation remains unquantified. Details in
`docs/finetuned_baseline.md`, section 8.
