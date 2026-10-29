# TODO - AI Agent (Automatable Tasks)

This todo list covers everything I can do autonomously without requiring manual clinical input, human recruitment, or physical access. 

## Phase 0: Setup & Environment (Week 1-2)
- [x] **Create repo structure** - `src/`, `data/`, `splits/`, `configs/`, `results/`, `notebooks/`, `assets/`, `scripts/`, `docs/`
- [x] **Environment setup** - `requirements.txt` pins the CPU reference environment
- [x] **Git setup** - repo initialised, `.gitignore` in place (no commits yet)
- [x] **Research plan document** - `docs/analysis_plan.md` (RQs, primary endpoint, statistics)

## Phase 1: Data Audit (Week 2-4)
- [x] **Download dataset** - Download Bangladesh Common Skin Disease Image Dataset from Mendeley (v3, DOI: 10.17632/9ggd3shdr7.3)
- [x] **File inventory** - Count images per class, total count, verify 197 vs 250 raw discrepancy
- [x] **Structure analysis** - Inspect folder structure, naming patterns, file extensions
- [x] **Checksum generation** - `results/checksums_sha256.csv`
- [x] **Duplicate detection** - Exact duplicates (hash) + near-duplicates (perceptual hash pHash/dHash)
- [x] **Embedding-based clustering** - Use DINOv2/ResNet50 embeddings to find near-duplicate groups
- [ ] **Augmentation forensics** - partial: parent assignment + dihedral matching done; per-image transform labels still missing
- [x] **Build group maps** - Create `raw_groups.json` with `group_id` for each image
- [x] **Audit report** - Generate `docs/data_audit.md` with all findings

## Phase 2: Split Design (Week 3-6)
- [x] **Recreate leaked split (P-leaked)** - Image-level stratified 70/15/15 matching original paper
- [x] **Create source-image-disjoint splits (P-disjoint)** - StratifiedGroupKFold (5 folds × 3 repeats)
- [x] **Create partial leakage splits (P-partial)** - 0%, 25%, 50%, 100% leakage levels for dose-response
- [x] **Disjointness validation** - Automated checks: no group leakage, no duplicate leakage, class balance
- [x] **Export splits** - Save all splits as CSVs with metadata (filepath, group_id, class, is_derivative)
- [x] **Split statistics report** - Generate `docs/split_stats.md`

## Phase 3: Leakage Quantification (Week 5-8)
- [ ] **Reproduce original baseline** - Implement classical ML/DL from [1] as closely as possible
- [x] **1-NN baseline** - embedding-space 1-NN under P-leaked and P-disjoint
- [x] **Linear probe** - logistic regression on frozen ResNet50/DINOv2 features
- [x] **Compare leaked vs disjoint** - frozen-feature probes; fine-tuned baseline pending
- [x] **Dose-response curve** - single fold; all-fold extension pending
- [x] **Group bootstrap CI** - group bootstrap + permutation test
- [x] **Leakage analysis report** - `docs/leakage_analysis.md` (preliminary)

## Phase 4: Model Benchmarking (Week 7-12)
- [ ] **Setup model zoo** - Configure ResNet50, DenseNet121, EfficientNet-B0/B3, MobileNetV3, ViT-B/16, Swin-T, ConvNeXt-T via timm
- [ ] **Training pipeline** - Build modular training code (config-driven, checkpointing, resume, logging)
- [ ] **Data loaders** - Create group-aware dataloaders that respect splits
- [ ] **Training loop** - Implement with AMP, early stopping, cosine schedule, AdamW
- [ ] **Run benchmark** - 5-fold × 3-repeat × seeds for all models
- [ ] **Save OOF predictions** - Logits/probs for all test predictions
- [ ] **Efficiency metrics** - Params, FLOPs, inference time, model size
- [ ] **Benchmark results** - Generate `results/benchmarks.csv`, `docs/BENCHMARKS.md`

## Phase 5: Training Tricks Ablation (Week 11-14)
- [ ] **Ablation framework** - One-factor-at-a-time + cumulative setup
- [ ] **Implement tricks** - MixUp, CutMix, Focal Loss, Class-balanced sampling, Label smoothing
- [ ] **TTA implementation** - Test-time augmentation pipeline (inference only)
- [ ] **Run ablations** - On 3 representative models (EfficientNet-B3, Swin-T, ConvNeXt-T)
- [ ] **Statistical tests** - Wilcoxon signed-rank with Holm correction
- [ ] **Ablation report** - Generate `docs/ABLATION.md` with forest plots

## Phase 6: Analysis & Evaluation (Week 12-17)
- [ ] **Metrics computation** - All metrics (macro/weighted F1, balanced acc, MCC, per-class sens/spec)
- [ ] **Calibration** - ECE, Brier score, reliability diagrams, temperature scaling
- [ ] **Conformal prediction** - Implement APS/RAPS for selective prediction
- [ ] **Risk-coverage curves** - Generate triage analysis
- [ ] **Robustness testing** - Blur, JPEG, brightness/contrast, occlusion corruptions
- [ ] **Confusion analysis** - Detailed per-model confusion matrices
- [ ] **Quantitative explainability** - Grad-CAM, insertion/deletion AUC, sanity checks (model/label randomization)
- [ ] **Generate all figures** - `scripts/make_figures.py` to auto-generate
- [ ] **Analysis reports** - Create all docs in Phase 6 deliverables

## Phase 7: Writing & Reproducibility (Week 14-24)
- [ ] **Manuscript draft** - Write full paper in LaTeX/Markdown following structure
- [ ] **Figures & tables** - Auto-generate from results
- [ ] **Supplementary material** - Full tables, hyperparams, extra figures
- [ ] **Reproducibility package** - README, configs, requirements, dockerfile (optional), scripts
- [ ] **Model card & datasheet** - Generate `MODEL_CARD.md`, `DATASHEET.md`
- [ ] **Checklists** - CLAIM, TRIPOD+AI, STARD-AI templates filled
- [ ] **Citation verification** - Check all refs against sources where possible
- [ ] **Preprint prep** - Prepare arXiv submission files if needed

## Automation Tools to Build
- [ ] `scripts/download_data.py` - Data download/checksum
- [x] `scripts/audit_data.py` - Full audit pipeline
- [x] `scripts/build_groups.py` - Group reconstruction
- [x] `scripts/make_splits.py` - All split protocols
- [ ] `scripts/train.py` - Unified training CLI
- [ ] `scripts/eval.py` - Evaluation pipeline
- [ ] `scripts/make_figures.py` - All figures
- [x] `scripts/verify_disjointness.py` - Automated checks
