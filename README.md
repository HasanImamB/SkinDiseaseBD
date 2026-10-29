# SkinDiseaseBD

Leakage-aware evaluation and reproducible benchmarks for the **Bangladesh Common Skin Disease
Image Dataset** (Mendeley v3, DOI [10.17632/9ggd3shdr7.3](https://doi.org/10.17632/9ggd3shdr7.3)).

The project studies how image-level splits of pre-augmented dermatology datasets inflate
performance, and defines source-image-disjoint protocols for honest evaluation.

## Status

- **Phase 1 — data audit:** complete (`docs/data_audit.md`).
- **Phase 2 — split design:** complete (`docs/split_stats.md`).
- **Phase 3+ — leakage quantification and benchmarks:** in progress.

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
scripts/    audit, feature extraction, group reconstruction, splits, verification
results/    inventory, checksums, duplicates, embeddings metadata, group map, split stats
splits/     P-leaked, P-disjoint and P-partial split CSVs
docs/       data audit, split statistics, work log
Proposal/   research plan, roadmap and task lists
```

## Reproduce

```bash
python -m pip install -r requirements.txt

python scripts/extract_features.py --dinov2
python scripts/audit_data.py
python scripts/build_groups.py
python scripts/make_splits.py
python scripts/verify_disjointness.py
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

## Licence

Code is released for research use. Dataset images remain under the original Mendeley licence and
are **not** redistributed here.
