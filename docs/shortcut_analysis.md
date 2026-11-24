# Shortcut analysis (Phase 3 / RQ3, Amendment 1)

First shortcut probe run. The goal is to decide whether the 0.54-0.66 P-disjoint floor
reflects disease difficulty or class-correlated acquisition shortcuts.

## 1. Setup

- Same P-disjoint splits as the main analysis: 5 folds x 3 repeats, raw-image test/validation.
- Embedding probes: training uses raw + tier-A derivatives from training groups; test uses raw
  images only. Features are recomputed for each masked variant with the same 224x224 pipeline.
- Metadata probe: raw images only (augmented images excluded because v1/v2 have fixed dimensions).
- Probes: metadata linear model; DINOv2/ResNet-50 1-NN and linear probes.
- References: stratified dummy and majority class on the same folds.
- Intervals: group bootstrap over `group_id`, 1,000 resamples.
- Artifacts: `results/shortcut/shortcut_probe.json`, `results/shortcut/metadata_raw.csv`,
  masked images under `results/shortcut/images/`, embeddings under `results/shortcut/*.npz`.

Variants:

| Variant | Definition |
|---|---|
| `full` | 224x224 reference image with the shortcut pipeline. |
| `border20` | Keep the outer 20% band, mask the central 60% x 60% region. |
| `background50` | Keep the remainder, mask the central 50% x 50% region. |
| `background75` | Stricter sensitivity mask: central 75% x 75% removed, outer 12.5% band kept. |

`background50` is a mild mask (removes 25% of the image area and keeps 75%); `background75` is
the stricter check and keeps only 43.75% of the area.

## 2. Metadata-only result

| Method | macro-F1 [95% CI] | balanced accuracy [95% CI] |
|---|---|---|
| metadata linear | 0.343 [0.251, 0.432] | 0.354 [0.256, 0.465] |
| stratified dummy | 0.230 [0.163, 0.286] | 0.253 [0.183, 0.320] |
| majority class | 0.140 [0.117, 0.157] | 0.200 [0.200, 0.200] |

Paired comparisons on the same pooled out-of-fold raw images:

| Comparison | Delta macro-F1 [95% CI] | permutation p |
|---|---|---:|
| metadata - stratified dummy | +0.114 [0.006, 0.220] | 0.095 |
| metadata - majority | +0.203 [0.109, 0.287] | 0.003 |

Metadata feature ablation:

| Features | macro-F1 [95% CI] | balanced accuracy [95% CI] |
|---|---|---|
| size / geometry / JPEG | 0.374 [0.275, 0.465] | 0.392 [0.294, 0.501] |
| acquisition datetime | 0.317 [0.226, 0.402] | 0.382 [0.271, 0.488] |
| full metadata | 0.343 [0.251, 0.432] | 0.354 [0.256, 0.465] |

EXIF ISO, exposure, f-number and focal length were absent from all raw images. Camera model was
nearly constant (`M2101K7AI` for 168/170 raw images with EXIF), so the signal comes mainly from
size/geometry and acquisition time, not camera identity.

Colour-only probe (mean/std RGB, mean/std HSV, 8-bin per-channel histograms):

| Method | macro-F1 [95% CI] | balanced accuracy [95% CI] |
|---|---|---|
| colour linear | 0.246 [0.174, 0.305] | 0.252 [0.171, 0.337] |
| stratified dummy | 0.230 [0.163, 0.286] | 0.253 [0.183, 0.320] |
| majority class | 0.140 [0.117, 0.157] | 0.200 [0.200, 0.200] |

Paired colour - stratified dummy = +0.016 [-0.067, 0.112], p=0.789; colour - majority =
+0.106 [0.032, 0.167], p=0.055. Global colour statistics are essentially at chance, so there is
no simple colour shortcut.

## 3. Border and background probes

| Variant | Feature | Probe | macro-F1 [95% CI] | Delta vs full |
|---|---|---|---:|---:|
| full | DINOv2 | 1-NN | 0.669 [0.534, 0.789] | - |
| full | DINOv2 | linear | 0.627 [0.512, 0.733] | - |
| full | ResNet-50 | 1-NN | 0.629 [0.511, 0.739] | - |
| full | ResNet-50 | linear | 0.648 [0.530, 0.753] | - |
| border20 | DINOv2 | 1-NN | 0.467 [0.347, 0.573] | -0.202 |
| border20 | DINOv2 | linear | 0.533 [0.412, 0.646] | -0.095 |
| border20 | ResNet-50 | 1-NN | 0.538 [0.418, 0.641] | -0.092 |
| border20 | ResNet-50 | linear | 0.534 [0.408, 0.638] | -0.115 |
| background50 | DINOv2 | 1-NN | 0.537 [0.401, 0.645] | -0.132 |
| background50 | DINOv2 | linear | 0.560 [0.440, 0.669] | -0.067 |
| background50 | ResNet-50 | 1-NN | 0.614 [0.489, 0.720] | -0.016 |
| background50 | ResNet-50 | linear | 0.621 [0.492, 0.728] | -0.027 |
| background75 | DINOv2 | 1-NN | 0.401 [0.294, 0.501] | -0.269 |
| background75 | DINOv2 | linear | 0.435 [0.330, 0.527] | -0.193 |
| background75 | ResNet-50 | 1-NN | 0.532 [0.411, 0.628] | -0.098 |
| background75 | ResNet-50 | linear | 0.502 [0.382, 0.607] | -0.146 |
| reference | stratified dummy | - | 0.230 [0.163, 0.286] | -0.439 |
| reference | majority class | - | 0.140 [0.117, 0.157] | -0.529 |

All masked probes remain above the majority-class baseline (0.140 macro-F1); the stricter
`background75` probes are still at or above the stratified dummy (0.230), but the margin is
smaller. Border-only loses 9-20 macro-F1 points versus full image. The mild `background50` mask
loses only 2-13 points, and ResNet-50 1-NN is essentially unchanged (-0.016). The stricter
`background75` mask loses 10-27 points, so the near-full `background50` result was partly an
artefact of the mild mask.

## 4. Interpretation

- A metadata-only classifier reaches 0.34 macro-F1, above both references. The paired gain over
  the stratified dummy is +0.11 [0.006, 0.220], although the permutation p-value is 0.095; the
  gain over majority is clear (+0.20, p=0.003). This is a **weak-to-moderate acquisition
  shortcut**, not a dominant one.
- Size/geometry alone (0.374) is the strongest metadata family, consistent with Eczema being
  uniformly 4000x3000 while several other classes vary in resolution.
- Colour-only features are at chance (0.246 vs 0.230 dummy), so there is no simple global colour
  shortcut.
- Border and background probes stay above chance. The mild `background50` mask keeps 75% of the
  image and can hide lesion tissue; the stricter `background75` mask removes the central 75%
  square and drops performance by 10-27 points. The remaining 0.40-0.53 macro-F1 suggests either
  disease cues extending into the border or a residual acquisition/crop cue; it is not enough to
  claim a pure background classifier.
- The full-image floor is therefore not explained by metadata or colour alone. Disease-related
  signal is likely present, but acquisition bias contributes and must be reported as a
  limitation.

Per the pre-specified rule: no strong simple acquisition shortcut was detected; do not claim
that the model has learned disease, and do not dismiss acquisition bias.

## 5. Caveats

- Only 136 labelled raw images; per-fold test sets have 27-28 raw images (17-19 groups). Gate G2
  still applies: do not make class-specific supremacy claims.
- There are no lesion masks. Border/background separation is approximate, and the chosen masks
  are mild.
- Metadata EXIF coverage is limited; ISO, exposure and focal length are absent.
- The metadata model is logistic regression with fixed hyperparameters; no tuning was performed.
- Derivative-level shortcut sensitivity was not run because the P-disjoint protocol defines
  raw-only test sets. It remains a planned secondary check.

## 6. Next steps

1. Automated label-quality triage is complete (`docs/label_audit.md`); clinician review of the
   shortlist is still pending.
2. Repeat the label audit after the stronger fine-tuned/class-balanced benchmark; flags that
   survive a better model are more credible.
3. Consider clinician-marked lesion masks or learned segmentation to replace the crude central
   masks, and carry the acquisition-bias caveat into the Phase 4 benchmark and manuscript.
