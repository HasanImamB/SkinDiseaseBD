# Fine-tuned ResNet-50 pilots (CPU)

Status: 128 px single-repeat pilot plus a completed 160 px class-balanced 5-fold x 3-repeat
reference. Complete enough to validate the training pipeline and set reference points, but
**not** the final Phase 4 benchmark.

## 1. Configuration

| Item | Value |
|---|---|
| Model | `resnet50.a1_in1k` (timm, ImageNet pretrained) |
| Input | 128 x 128 |
| Train set | raw + tier-A derivatives from training groups (1,102 images in fold 0) |
| Val/test | raw images only (22 val, 27 test per fold) |
| Folds/repeats | 5 folds x 1 repeat (`repeat 0`) |
| Epochs | max 10, early stopping patience 3 on val macro-F1 |
| Optimiser | AdamW, lr 3e-4, weight decay 0.05 |
| Loss | cross-entropy with label smoothing 0.1 |
| Batch size | 32 |
| Augmentation | random resized crop (0.65-1.0), horizontal flip, rotation 10 deg, colour jitter |
| Device | CPU (8 torch threads, 4 dataloader workers) |
| Runtime | about 43 s per epoch at 128 px |

The validation set has only 22 images, so early stopping and best-epoch selection are noisy.

## 2. Results

Per-fold test performance:

| Fold | macro-F1 | balanced accuracy | Scabies recall | Dermatitis recall |
|---|---:|---:|---:|---:|
| 0 | 0.7313 | 0.7333 | 0.00 | 0.67 |
| 1 | 0.5723 | 0.5533 | 0.00 | 0.67 |
| 2 | 0.4030 | 0.4333 | 0.50 | 0.00 |
| 3 | 0.6298 | 0.6000 | 0.00 | 1.00 |
| 4 | 0.5182 | 0.5714 | 0.50 | 1.00 |
| **Mean +/- SD** | **0.5709 +/- 0.1226** | - | - | - |

Pooled out-of-fold (136 raw images):

| Metric | Value |
|---|---:|
| macro-F1 [95% group-bootstrap CI] | **0.6044 [0.4766, 0.7021]** |
| balanced accuracy | 0.5644 |
| accuracy | 0.7426 |
| Dermatitis recall | 0.615 |
| Eczema recall | 0.932 |
| Scabies recall | 0.200 |
| Tinea Ringworm recall | 0.621 |
| Vitiligo recall | 0.455 |

The high accuracy relative to macro-F1 reflects the Eczema majority (73/136); Scabies and
Vitiligo remain weak.

## 3. Comparison with frozen probes

Paired group-bootstrap comparison on the same 136 raw OOF images:

| Frozen probe | Frozen macro-F1 | Fine-tuned - frozen | 95% CI | p |
|---|---:|---:|---|---:|
| DINOv2 linear | 0.663 | -0.059 | [-0.210, 0.069] | 0.425 |
| DINOv2 1-NN | 0.651 | -0.047 | [-0.200, 0.085] | 0.500 |
| ResNet-50 linear | 0.623 | -0.019 | [-0.159, 0.115] | 0.796 |
| ResNet-50 1-NN | 0.536 | +0.069 | [-0.071, 0.191] | 0.293 |

All intervals include zero. The preliminary fine-tuned model does **not** beat the frozen
features and is numerically below DINOv2. This is not evidence against fine-tuning: 128 px,
10 epochs, one repeat and a 22-image validation set are all unfavourable.

## 4. Interpretation and caveats

- This run validates the pipeline and gives a pooled reference, not a publishable benchmark
  result.
- The low resolution may remove diagnostic detail; the frozen probes use 224 px inputs.
- Early stopping on 22 validation images can stop before convergence and adds variance across
  folds.
- Only one repeat was run, so the interval reflects group sampling but not training-seed
  variation.
- No class-balanced sampling or focal loss was used; Scabies recall is near zero in three folds.
- Best checkpoints were kept in memory only; per-fold JSON files record the histories.

## 5. Next steps

1. Rerun the same 5 folds at 160 or 224 px with more epochs to separate resolution/under-training
   from model capacity.
2. Add class-balanced sampling or class-weighted loss, then check whether Scabies recall improves
   without collapsing Eczema.
3. Repeat with 3 seeds after fixing the resolution and schedule.
4. Re-run the label audit on the stronger fine-tuned OOF probabilities; flags that survive a
   better model are more credible label errors.
5. Only then expand to the model zoo (EfficientNet, ConvNeXt, ViT/Swin).

## 6. Files

- `scripts/train.py` - training CLI
- `scripts/summarize_finetuned.py` - pooling, bootstrap and frozen-probe comparisons
- `results/finetuned/resnet50_128_fold*_rep0.json` - per-fold histories and metrics
- `results/finetuned/resnet50_128_summary.json` - pooled summary
- `results/predictions/finetuned_resnet50_128_fold*_rep0_test.csv` - per-fold probabilities
- `results/predictions/finetuned_resnet50_128_oof.csv` - pooled OOF probabilities

## 7. Updated training CLI (2026-11-24)

`scripts/train.py` now supports `--seeds`, `--class-balanced`, `--class-weighted-loss`, `--amp`,
`--checkpoints` and `--resume`. Each fold/repeat/seed writes a distinct JSON, prediction CSV and
best checkpoint. Prediction CSVs include both probabilities and logits; `summarize_finetuned.py`
recognises the `_seed{n}` suffix and pools across all matching runs. The stronger 160 px
class-balanced reference is specified in `docs/analysis_plan.md`, Amendment 3.

## 8. Completed 160 px class-balanced reference (2026-11-24)

The full 5 folds x 3 repeats grid with seed 42 finished. Configuration follows Amendment 3:
160 px input, batch 32, AdamW lr 3e-4 with weight decay 0.05, cosine schedule, cross-entropy
with label smoothing 0.1, class-balanced `WeightedRandomSampler`, max 12 epochs with early
stopping patience 3 on validation macro-F1, CPU only. Best epoch averaged 4.2.

| Metric | Value |
|---|---:|
| Per-fold/repeat macro-F1 mean +/- SD | 0.603 +/- 0.137 (n=15) |
| Pooled OOF macro-F1 [95% group-bootstrap CI] | **0.681 [0.546, 0.799]** |
| Pooled weighted F1 [CI] | 0.772 [0.659, 0.859] |
| Pooled balanced accuracy [CI] | 0.621 [0.501, 0.760] |
| Pooled MCC [CI] | 0.670 [0.541, 0.785] |
| Pooled macro AUROC (one-vs-rest) | 0.862 |
| Pooled accuracy | 0.794 |
| Dermatitis / Eczema recall | 0.538 / 0.986 |
| Scabies / Tinea / Vitiligo recall | 0.200 / 0.655 / 0.727 |

Per-class precision / recall / specificity: Dermatitis 0.778 / 0.538 / 0.984, Eczema 0.766 /
0.986 / 0.651, Scabies 1.000 / 0.200 / 1.000, Tinea 0.826 / 0.655 / 0.963, Vitiligo 1.000 /
0.727 / 1.000. The pooled confusion matrix is in
`results/finetuned/resnet50_160_balanced_confusion_matrix.csv`; 5/13 Dermatitis and 10/29 Tinea
images are predicted as Eczema, and 5/10 Scabies images are also assigned to Eczema.

Per-repeat mean macro-F1 was 0.526, 0.622 and 0.661. The pooled OOF file contains one averaged
probability vector per raw image, so all 136 labelled raw images are represented once.

Paired comparisons against the four frozen probes (same 136 raw images):

| Frozen probe | Frozen macro-F1 | Fine-tuned - frozen | 95% CI | p |
|---|---:|---:|---|---:|
| DINOv2 linear | 0.663 | +0.018 | [-0.108, 0.118] | 0.738 |
| DINOv2 1-NN | 0.651 | +0.030 | [-0.091, 0.135] | 0.589 |
| ResNet-50 linear | 0.623 | +0.058 | [-0.067, 0.173] | 0.302 |
| ResNet-50 1-NN | 0.536 | +0.145 | [0.031, 0.261] | 0.009 |

Interpretation: higher resolution, class-balanced sampling and 15 training runs close the gap with
the frozen probes, but only the comparison against the weakest frozen probe is statistically
clear. Scabies recall remains 0.200, so class imbalance is not fully resolved. This is still a
single-architecture pilot, not the final Phase 4 benchmark.

The cleanlab triage was repeated on the stronger OOF probabilities
(`results/label_audit_finetuned.json`): 16/136 images flagged, 14 of which were also flagged by
the frozen-probe audit. Flag rate remains highest for Scabies (0.50) and Dermatitis (0.31), and
11/16 flags have Eczema as the predicted label, so the flags still look like minority-class
confusion rather than confirmed label errors.

Files: `results/finetuned/resnet50_160_balanced_summary.json`,
`results/finetuned/resnet50_160_balanced_confusion_matrix.csv`,
`results/finetuned/finetuned_resnet50_160_balanced_*_seed42.json`,
`results/predictions/finetuned_resnet50_160_balanced_oof.csv`,
`results/label_audit_finetuned.{json,csv}`.
