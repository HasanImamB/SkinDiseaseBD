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
| **P-partial** | From P-disjoint fold 0 / repeat 0, add derivatives of 0/25/50/100% of test groups to training. Test raw images are unchanged. | Paired leakage dose-response. |

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
