# Leakage analysis (Phase 3, frozen-feature probes)

Preliminary results from `scripts/leakage_probe.py` on the frozen splits. No model was
fine-tuned; the goal is to isolate the effect of splitting on pre-augmented data.

## 1. Setup

- Features: DINOv2 ViT-S/14 (384-d) and ResNet-50 ImageNet (2048-d), both frozen.
- Probes: 1-NN with cosine distance; multinomial logistic regression (`C=1.0`, balanced).
- P-disjoint: 5 folds x 3 repeats, raw-only test, raw + tier-A training.
- P-leaked: image-level 70/15/15 on the 1,612 v1 images, 3 repeats (derivative-level test).
- P-partial: fold 0 / repeat 0, leakage levels 0/25/50/100%.
- Intervals: group bootstrap over `group_id`, 1,000 resamples. Delta p-value: group-level
  permutation, 1,000 permutations.

## 2. Main result

| Feature | Probe | P-disjoint macro-F1 [95% CI] | P-leaked macro-F1 | Delta (100%-0%) [95% CI] | p |
|---|---|---|---:|---|---:|
| DINOv2 | 1-NN | 0.651 [0.526, 0.775] | 1.000 | +0.345 [0.000, 0.458] | 0.226 |
| DINOv2 | linear | 0.663 [0.541, 0.778] | 0.997 | +0.509 [0.234, 0.625] | 0.015 |
| ResNet-50 | 1-NN | 0.536 [0.424, 0.633] | 0.999 | +0.471 [0.080, 0.591] | 0.018 |
| ResNet-50 | linear | 0.623 [0.499, 0.731] | 0.997 | +0.477 [0.223, 0.568] | 0.027 |

- Under P-disjoint, frozen features reach only **0.54-0.66 macro-F1**.
- Under P-leaked, the same probes reach **~1.00 macro-F1**, i.e. near-perfect memorisation of
  augmented derivatives.
- The paired P-partial contrast is **+0.35 to +0.51 macro-F1**, with the 95% CI excluding zero
  for three of four probe/feature combinations.
- The DINOv2 1-NN delta is large but not significant on the single 27-image test fold
  (CI includes zero, p=0.226); this is a power limitation, not evidence against the effect.

## 3. Dose-response (P-partial, macro-F1)

| Feature / probe | 0% | 25% | 50% | 100% |
|---|---:|---:|---:|---:|
| DINOv2 / 1-NN | 0.575 | 0.681 | 0.747 | 0.920 |
| DINOv2 / linear | 0.491 | 0.565 | 0.723 | 1.000 |
| ResNet-50 / 1-NN | 0.474 | 0.621 | 0.701 | 0.945 |
| ResNet-50 / linear | 0.523 | 0.781 | 0.691 | 1.000 |

The trend is monotone for three of four configurations; ResNet-50 / linear is non-monotone at
50% (0.691 < 0.781), consistent with fold-level noise on 27 test images.

## 4. Per-class difficulty under P-disjoint (pooled OOF recall)

| Class | DINOv2 1-NN | DINOv2 linear | ResNet 1-NN | ResNet linear |
|---|---:|---:|---:|---:|
| Dermatitis | 0.38 | 0.38 | 0.23 | 0.38 |
| Eczema | 0.92 | 0.85 | 0.68 | 0.82 |
| Scabies | 0.30 | 0.30 | 0.30 | 0.30 |
| Tinea Ringworm | 0.59 | 0.69 | 0.62 | 0.72 |
| Vitiligo | 0.82 | 0.91 | 0.82 | 0.73 |

Scabies and Dermatitis are the hardest classes; Eczema is the easiest. This matches the
class-skew of the released raw images (Eczema has 73 labelled raw images, Dermatitis 13,
Scabies 10).

## 5. Interpretation

- Image-level splitting of already-augmented data is catastrophic: a trivial 1-NN on frozen
  features reaches ~100% when derivatives of the test source are in the training set.
- A source-image-disjoint protocol removes this shortcut and exposes a much harder task.
- The inflation is not a small calibration artefact; it is a change in the apparent difficulty
  of the dataset by **35-51 macro-F1 points**.
- Because P-leaked tests derivatives and P-disjoint tests raw images, the two columns are **not
  paired**. The rigorous paired estimate is the P-partial Delta.

## 6. Caveats

- Only **136/197 raw images** are labelled; 61 are excluded.
- The P-partial experiment currently uses **one fold** (27 raw test images, 17 groups). The
  intervals are correspondingly wide and one comparison is non-significant.
- Raw labels are inferred from descendants, not author-provided.
- Source groups are empirically reconstructed and over-merged; they are not author IDs.
- 1-NN is deliberately simple; it is a leakage detector, not a benchmark model.

## 7. Next steps

1. Extend P-partial to all 5 folds x 3 repeats to increase power and stabilise the Delta.
2. Add image-level P-leaked macro-F1 as a descriptive reproduction of the original paper.
3. Move to Phase 4 (fine-tuned model zoo) once the paired Delta is stable.
