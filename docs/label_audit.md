# Label-quality audit (automated triage)

Status: automated confident-learning triage complete; clinician review pending.

## 1. Purpose

The raw class labels are **inferred** from augmented descendants, not supplied by the dataset
authors. This audit asks whether the pooled P-disjoint OOF predictions contain a subset of raw
images whose labels look inconsistent enough to prioritise for manual review. It is a triage
step, not a verdict on label correctness.

## 2. Inputs and method

- 136 labelled raw images, five classes.
- Pooled OOF probabilities from the four P-disjoint frozen-feature probes:
  `dinov2_1nn`, `dinov2_linear`, `resnet_1nn`, `resnet_linear`.
- Each prediction file has one out-of-sample probability vector per raw image, averaged over the
  5 folds x 3 repeats (derived from `results/predictions/disjoint_*_oof.csv`).
- For each probe, `cleanlab.filter.find_label_issues` (confident learning, `self_confidence`
  ranking) flags suspect labels. A raw image is a **high-confidence issue** when at least 3 of
  the 4 probes flag it, and a **candidate issue** when at least 2 probes flag it.

The raw flags are highly dependent on the underlying model. With macro-F1 around 0.63 and a
73/136 Eczema majority, many minority-class errors will be flagged as label issues even when the
label is correct.

## 3. Results

| Probe | Flagged raw images |
|---|---|
| DINOv2 1-NN | 32 |
| DINOv2 linear | 29 |
| ResNet-50 1-NN | 50 |
| ResNet-50 linear | 28 |

Consensus:

- **23 / 136** raw images (16.9%) are flagged by at least 3 probes.
- **35 / 136** raw images (25.7%) are flagged by at least 2 probes.

Flag rate by class:

| Class | Raw images | High-confidence flags | Flag rate |
|---|---:|---:|---:|
| Eczema | 73 | 1 | 1.4% |
| Tinea Ringworm | 29 | 7 | 24.1% |
| Dermatitis | 13 | 7 | 53.8% |
| Vitiligo | 11 | 1 | 9.1% |
| Scabies | 10 | 7 | 70.0% |

Among the 23 high-confidence flags, the consensus prediction is Eczema for 18 images, Dermatitis
for 2, Tinea Ringworm for 2 and Scabies for 1.

Top-ranked 4-vote flags (ranked by mean label quality):

| Image | Group | Label | Consensus prediction | Votes |
|---|---|---|---|---:|
| `ResearchImage/151.jpg` | G100 | Tinea Ringworm | Eczema | 4 |
| `ResearchImage/27.jpg` | G117 | Dermatitis | Eczema | 4 |
| `ResearchImage/135.jpg` | G075 | Tinea Ringworm | Eczema | 4 |
| `ResearchImage/80.jpg` | G032 | Scabies | Eczema | 4 |
| `ResearchImage/193.jpg` | G080 | Scabies | Eczema | 4 |
| `ResearchImage/26.jpg` | G117 | Dermatitis | Eczema | 4 |
| `ResearchImage/25.jpg` | G117 | Dermatitis | Eczema | 4 |
| `ResearchImage/81.jpg` | G032 | Scabies | Eczema | 4 |
| `ResearchImage/134.jpg` | G075 | Tinea Ringworm | Eczema | 4 |
| `ResearchImage/192.jpg` | G078 | Scabies | Dermatitis | 4 |

Several flags are repeated within groups (G117, G032, G075, G021, G027), so the 23 image-level
flags correspond to fewer independent source images.

## 4. Interpretation

The audit is **inconclusive as evidence of label noise**:

- Almost all high-confidence flags are minority-class images predicted as Eczema. This is the
  same pattern as the model's per-class errors: Scabies recall is only 0.1-0.4 and Dermatitis
  recall is low, while Eczema is the majority class. Cleanlab's confident joint is therefore
  dominated by the Eczema prior.
- A 17% high-confidence flag rate is far above what is plausible for a dataset whose descendant
  labels agreed 98.9% at group level. It is more consistent with model under-fitting on small,
  visually overlapping classes than with widespread mislabelling.
- The flags are useful as a **review shortlist**, not as a reason to drop images. No images were
  removed and no labels were changed.

A clinician review sheet is generated at
`results/label_audit/contact_sheet.jpg` (gitignored). It shows the 35 candidate images with their
current label, vote count and mean label quality. Review should start with the 4-vote flags.

## 5. Next steps

1. Clinician reviews the shortlist; record agree / disagree / ambiguous per image.
2. Repeat the audit after the fine-tuned benchmark, and after adding class-balanced training or
   sampling; flags that survive a stronger model are more credible label errors.
3. If labels change, rerun splits, leakage probes and the shortcut audit, because label edits
   affect group stratification.

## 6. Files

- `scripts/label_audit.py`
- `results/label_audit.json` - per-probe flags, consensus, per-class rates
- `results/label_audit.csv` - one row per raw image, sorted by votes then label quality
- `results/label_audit/contact_sheet.jpg` - review sheet (gitignored)
