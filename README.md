# SkinDiseaseBD

Leakage-aware evaluation and reproducible benchmarks for the **Bangladesh Common Skin Disease
Image Dataset** (Mendeley v3, DOI [10.17632/9ggd3shdr7.3](https://doi.org/10.17632/9ggd3shdr7.3)).

The project studies how image-level splits of pre-augmented dermatology datasets inflate
performance, and defines source-image-disjoint protocols for honest evaluation.

## Status

- **Phase 1 — data audit:** complete (`docs/data_audit.md`); shortcut probes run
  (`docs/shortcut_analysis.md`); automated label-noise triage done (`docs/label_audit.md`),
  clinician review pending.
- **Phase 2 — split design:** complete (`docs/split_stats.md`).
- **Phase 3 — leakage quantification:** frozen-feature result, 25–36 macro-F1 point paired
  inflation across all 5 folds x 3 repeats (`docs/leakage_analysis.md`); shortcut analysis
  complete; preliminary fine-tuned baseline run (`docs/finetuned_baseline.md`).
- **Phase 4 — benchmark:** training pipeline in place; 128 px single-repeat pilot gives pooled
  macro-F1 0.60, and the complete 160 px class-balanced 5-fold x 3-repeat reference reaches 0.68.
  Current frontier: additional seeds, then the CNN/ViT model zoo and ablations.

## Data

The dataset is **not** committed (licence: CC BY-NC 4.0). Download it from Mendeley and extract
so that the repository contains:

```
data/dataset/
  ResearchImage/      197 raw images
  Images_v1/          1,612 augmented images, 176x176
  Images_v2/          1,612 augmented images, 512x512
```

## Repository layout

```
scripts/    audit, feature extraction, group reconstruction, splits, probes, verification
results/    inventory, checksums, duplicates, embeddings metadata, group map, split stats
splits/     P-leaked, P-disjoint and P-partial split CSVs
docs/       data audit, split statistics, leakage and shortcut analyses, work log
Proposal/   research plan, roadmap and task lists
```

## Reproduce

```bash
python -m pip install -r requirements.txt

python scripts/extract_features.py --dinov2
python scripts/audit_data.py
python scripts/build_groups.py
python scripts/make_metadata.py
python scripts/make_splits.py
python scripts/verify_disjointness.py
python scripts/leakage_probe.py
python scripts/shortcut_probe.py
python scripts/label_audit.py

# preliminary CPU baseline: 5 folds x 1 repeat, 128 px
python scripts/train.py --folds 0,1,2,3,4 --repeats 0 --epochs 10 \
    --image-size 128 --run-name resnet50_128
python scripts/summarize_finetuned.py --run-name resnet50_128

# stronger class-balanced reference: 5 folds x 3 repeats, 160 px, seed 42
python scripts/train.py --folds 0,1,2,3,4 --repeats 0,1,2 --seeds 42 \
    --epochs 12 --image-size 160 --class-balanced --resume \
    --run-name resnet50_160_balanced
python scripts/summarize_finetuned.py --run-name resnet50_160_balanced
```

## Key findings

- **197** raw images were released (`1.jpg`–`198.jpg`, `169.jpg` missing); the paper's ~250 is a
  collection claim, not the public release.
- **3,224** augmented images (1,612 v1 + 1,612 v2); the versions are independent augmentation
  passes, not duplicates.
- **0** exact duplicates; 451 pHash near-duplicate clusters.
- Only **136/197** raw images have a recoverable class; 61 are unlabelled and excluded.
- Source groups are reconstructed conservatively (over-merged), so leakage control is strict but
  statistical power is limited.
- Image-level splitting inflates frozen-feature macro-F1 by **25–36 points** in the paired,
  all-fold dose-response estimate; metadata-only cues reach **0.34 macro-F1**, so a weak
  acquisition shortcut is present, while global colour features are at chance.
- Cleanlab flags **23/136** raw images from frozen probes and **16/136** from the stronger
  fine-tuned OOF (14 overlap), dominated by minority classes predicted as Eczema; this is model
  confusion, not proof of mislabelling. The shortlist awaits clinician review.
- The 128 px fine-tuned ResNet-50 pilot reaches pooled macro-F1 **0.60 [0.48, 0.70]**. The stronger
  160 px class-balanced, 5-fold x 3-repeat reference reaches **0.68 [0.55, 0.80]**, statistically
  clear only against the weakest frozen probe, with Scabies recall still 0.20. This is a
  single-architecture pilot, not the final benchmark.

## Running on Colab / Kaggle (free GPU)

Use `notebooks/cloud_224_benchmark.ipynb`. Push `main` first so the cloud clone contains the
latest code, then open the notebook in Colab or Kaggle, select a GPU runtime and run all cells.
The notebook installs `requirements_cloud.txt` (the pinned `requirements.txt` is CPU-only), finds
the dataset under Google Drive or `/kaggle/input`, links it to `data/dataset`, trains ResNet-50 at
224 px with `--amp --class-balanced --resume`, pools predictions and copies outputs to persistent
storage. The dataset parent folder must contain `ResearchImage/`, `Images_v1/` and `Images_v2/`;
images are never committed. Edit the `MODEL`, `RUN_NAME`, `SEEDS`, `EPOCHS` and `BATCH_SIZE`
variables in the config cell to extend the benchmark, and keep `--resume` so interrupted runs can
continue.

## Licence

Code is released for research use. Dataset images remain under the original Mendeley licence and
are **not** redistributed here.
