# Leakage analysis (Phase 3, frozen-feature probes)

Results from `scripts/leakage_probe.py` on the frozen splits. No model was fine-tuned;
the goal is to isolate the effect of splitting on pre-augmented data.

## 1. Setup

- Features: DINOv2 ViT-S/14 (384-d) and ResNet-50 ImageNet (2048-d), both frozen.
- Probes: 1-NN with cosine distance; multinomial logistic regression (`C=1.0`, balanced).
- P-disjoint: 5 folds x 3 repeats, raw-only test, raw + tier-A training.
- P-leaked: image-level 70/15/15 on the 1,612 v1 images, 3 repeats (derivative-level test).
- P-partial: all 5 folds x 3 repeats, leakage levels 0/25/50/100%; pooled out-of-fold
  predictions are averaged over folds and repeats, as for P-disjoint.
- Intervals: group bootstrap over `group_id`, 1,000 resamples. Delta p-value: group-level
  permutation, 1,000 permutations.

## 2. Main result

| Feature | Probe | P-disjoint macro-F1 [95% CI] | P-leaked macro-F1 | Delta (100%-0%) [95% CI] | p |
|---|---|---|---:|---|---:|
| DINOv2 | 1-NN | 0.651 [0.526, 0.775] | 1.000 | +0.250 [0.139, 0.373] | 0.001 |
| DINOv2 | linear | 0.663 [0.541, 0.778] | 0.997 | +0.271 [0.153, 0.393] | 0.001 |
| ResNet-50 | 1-NN | 0.536 [0.424, 0.633] | 0.999 | +0.362 [0.260, 0.480] | 0.001 |
| ResNet-50 | linear | 0.623 [0.499, 0.731] | 0.997 | +0.311 [0.202, 0.433] | 0.001 |

- Under P-disjoint, frozen features reach only **0.54-0.66 macro-F1**.
- Under P-leaked, the same probes reach **~1.00 macro-F1**, i.e. near-perfect memorisation of
  augmented derivatives; the descriptive gap to P-disjoint is **33-46 macro-F1 points**.
- The paired P-partial contrast (all 5 folds x 3 repeats) is **+0.25 to +0.36 macro-F1**, with
  every 95% CI excluding zero and every permutation p = 0.001.
- The all-fold extension reduced the point estimate relative to the earlier single-fold result
  (+0.35 to +0.51); the earlier estimate was optimistic because it rested on one 27-image fold.

## 3. Dose-response (P-partial, macro-F1)

| Feature / probe | 0% | 25% | 50% | 100% |
|---|---:|---:|---:|---:|
| DINOv2 / 1-NN | 0.651 | 0.655 | 0.829 | 0.901 |
| DINOv2 / linear | 0.663 | 0.670 | 0.854 | 0.934 |
| ResNet-50 / 1-NN | 0.536 | 0.569 | 0.780 | 0.898 |
| ResNet-50 / linear | 0.623 | 0.710 | 0.854 | 0.934 |

The trend is monotone for all four configurations. The jump is concentrated between 25% and
50% leakage, and 25% leakage adds little over the disjoint baseline. Achieved leakage averaged
0.0 / 22.9 / 50.0 / 100.0% for the requested 0 / 25 / 50 / 100%.

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
  of the dataset by **33-46 macro-F1 points** descriptively, and by **25-36 points** in the
  paired, all-fold dose-response estimate.
- Because P-leaked tests derivatives and P-disjoint tests raw images, the two columns are **not
  paired**. The rigorous paired estimate is the P-partial Delta.

## 6. Caveats

- Only **136/197 raw images** are labelled; 61 are excluded.
- P-partial now covers all 5 folds x 3 repeats, but each fold still has only 27-28 raw test
  images (17-19 groups), so per-fold metrics are noisy even though the pooled Delta is stable.
- Raw labels are inferred from descendants, not author-provided.
- Source groups are empirically reconstructed and over-merged; they are not author IDs.
- 1-NN is deliberately simple; it is a leakage detector, not a benchmark model.

## 7. Next steps

1. Shortcut probes are complete, including the stricter `background75` mask and the
   colour-histogram check (`docs/shortcut_analysis.md`).
2. A preliminary fine-tuned ResNet-50 baseline is complete at 128 px, 5 folds x 1 repeat
   (`docs/finetuned_baseline.md`); it validates the pipeline but is below the frozen probes.
3. Reproduce the original benchmark [1] under P-leaked and P-disjoint.
4. Run a stronger multi-seed fine-tuned baseline at 160-224 px with class balancing, then move to
   the Phase 4 model zoo and ablations.
