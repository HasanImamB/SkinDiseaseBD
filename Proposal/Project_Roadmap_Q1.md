# Updated Roadmap (Q1 Journal Target): Leakage-Aware Evaluation and Modern Deep Learning Benchmarks for the Bangladesh Common Skin Disease Image Dataset

> Version 2. Replaces the 8-week roadmap. Items marked **[NEW]** were added for Q1 readiness; **[FIX]** marks a correction to a methodological problem in v1.

**Current checkpoint (2026-11-24):** Phases 1-3 are largely complete. The data audit, source-group reconstruction, three split protocols, disjointness checks, all-fold leakage quantification, shortcut probes, label triage and fine-tuning pilots (128 px single-repeat and completed 160 px class-balanced 5x3) are in place. Phase 4 benchmarking, Phase 5 ablations, Phase 6 calibration/external validation/fairness, Phase 7 explainability, and all clinical/governance tasks are not yet complete.

---

## 1. Review of v1: what must change for a Q1 journal

### 1.1 Positioning problem (biggest risk)
The v1 contribution is "re-split an existing dataset, train 7 standard models, report metrics." Reviewers at Q1 venues will call this an incremental re-benchmark. The leakage finding is valuable but, alone, is a short communication, not a Q1 article. To clear the bar the paper needs at least **three** of the following, ideally all:

1. A rigorous, quantified leakage study (dose-response, not just one before/after number).
2. External validation on an independent Bangladeshi dataset (SkinDisNet [6]).
3. Shortcut-learning and acquisition-bias analysis (is the model learning disease or camera/background/session?).
4. Calibration, uncertainty and selective prediction (abstain/refer), which is what triage actually needs.
5. Quantitative explainability (not only pretty Grad-CAM figures).
6. Deployment relevance: model size, latency on CPU/mobile, robustness to smartphone-style corruptions.
7. A clinician co-author or clinician-validated annotation subset.

**Proposed paper claim (one sentence):** *"Image-level evaluation of augmented clinical skin datasets overstates performance by X points; under a source-image-disjoint, externally validated, calibrated protocol, modern models achieve Y, with these class-specific failure modes, and here is a reproducible standard."*

### 1.2 Methodological flaws in v1 and the proposal

| # | Issue | Why a reviewer will reject it | Fix |
|---|---|---|---|
| 1 **[FIX]** | "Subject-disjoint" is a misnomer. The dataset has no patient IDs; you can only guarantee **source-image-disjoint**. The same patient may contribute several raw photos. | Residual patient-level leakage remains; the term over-claims. | Rename to "source-image-disjoint" for this dataset. Add near-duplicate/same-lesion clustering (embedding + perceptual hash) to merge likely same-patient images. Use SkinDisNet, which has patient IDs, to demonstrate a true patient-disjoint protocol. |
| 2 **[FIX]** | Plan evaluates on all augmented derivatives of test raw images. | Derivatives of one photo are not independent; n is inflated, CIs and p-values are invalid (pseudo-replication). | Primary test metric on **raw (original) test images only**. Report derivative-level results as secondary, with **cluster (group) bootstrap**. |
| 3 **[FIX]** | McNemar's test between "leaked" and "disjoint" settings. | The two settings use **different test sets**, so predictions are not paired; McNemar is invalid there. | Quantify inflation as Δ = metric(leaked test) − metric(disjoint test) with a **group-bootstrap CI** or permutation test. Use McNemar only for two models on the **same** test set (and aggregate per group). |
| 4 **[FIX]** | Single 70/15/15 split of ~197-250 raw images. | Test set ≈ 30-37 raw images, ≈ 6-7 per class. One split is statistically meaningless; per-class sensitivity CIs span 40+ points. | **Repeated stratified group k-fold** (5 folds × 3 repeats) with nested validation; pool out-of-fold predictions; report CIs. Keep one frozen final holdout only if the raw count allows (see §4.2). |
| 5 **[FIX]** | Ablation described both as "in order (cumulative)" and "isolate each component." Also TTA listed as a training trick. | Contradictory; cumulative ordering confounds effects. TTA is inference-only. | One-factor-at-a-time against the baseline, then one cumulative "best recipe." TTA evaluated at inference on saved checkpoints (no retraining). |
| 6 **[FIX]** | 3 seeds with paired t-tests. | Almost no power; t-tests assume independence across seeds/folds. | 5 seeds on the primary models; test across fold×seed units with Wilcoxon signed-rank or Bayesian signed-rank (Benavoli), Holm correction, plus group-bootstrap CIs on the pooled predictions. |
| 7 **[FIX]** | Fallback inconsistent: the proposal holds raw images out as *validation*; the roadmap as *test*. | Reviewers notice. | Fix: confirmed raw images are used for **test**; validation is drawn from training groups. |
| 8 **[FIX]** | Hyperparameters selected without a stated protocol. | Test-set tuning risk. | Tune on inner validation groups only; freeze configs before touching the test folds; log every run. |
| 9 | Dataset provenance unresolved: 197 vs 250 raw images. | Cannot state N in the abstract. | Resolve in the audit, report both numbers and explain. |
| 10 | "Absence of metadata precludes fairness analysis." | A weak limitation for a Q1 clinical paper. | **[NEW]** Clinician-annotated Fitzpatrick skin type (FST) on raw images (≈200, feasible) or at minimum image-derived ITA as a proxy, clearly labeled as a proxy. Report subgroup performance. |
| 11 | Literature has 6 references. | Q1 papers typically cite 40-60 with proper related work on leakage, dermatology AI fairness, calibration. | Expand (see §9). Verify every existing citation (e.g., [3] Gessert's ISIC 2019 work was published as a journal article with metadata; check the exact title/venue; verify [1] arXiv ID and [5] DOI before submission). |
| 12 | Timeline 8-12 weeks. | Unrealistic for a Q1 paper including writing, external validation and internal review. | 22-26 weeks to submission, plus revision budget (§3). |

### 1.3 Things v1 does well (keep)
Clear problem statement; sensible architecture list; mixed-precision/early stopping; explicit risk table; fallback plan; reproducibility package; per-class triage framing.

---

## 2. Target venues and reporting standards

**Candidate Q1 journals (verify current quartile/scope on Scimago and the journal site before choosing; quartiles change yearly):**
- *Computers in Biology and Medicine*
- *Artificial Intelligence in Medicine*
- *Computer Methods and Programs in Biomedicine*
- *IEEE Journal of Biomedical and Health Informatics*
- *Scientific Reports* (broad, high volume; strong emphasis on rigor and data availability)
- Stretch: *Medical Image Analysis*, *npj Digital Medicine* (would need much stronger clinical validation)
- Dataset/benchmark angle: *Scientific Data* (only if you contribute new annotations such as FST, lesion masks, group maps)

**Reporting checklists to follow and attach:** CLAIM (Checklist for AI in Medical Imaging), TRIPOD+AI (prediction model reporting), STARD-AI (diagnostic accuracy). Add a model card and a dataset datasheet.

**Decision gate G0 (Week 2):** pick the primary journal and one backup; read 5 recent papers from it to calibrate length, tone and required validation.

---

## 3. Timeline (22-26 weeks to submission)

| Phase | Weeks | Focus | Gate |
|---|---|---|---|
| 0 | 1-2 | Scoping, protocol freeze, preregistration, ethics/licence check, clinician recruitment | G0 |
| 1 | 2-4 | Data audit, provenance, shortcut and label-quality audit | G1 |
| 2 | 3-6 | Group reconstruction and split design | G2 |
| 3 | 5-8 | Leakage quantification study | G3 |
| 4 | 7-12 | Main benchmark (CNN, ViT, foundation models) | G4 |
| 5 | 11-14 | Training-trick ablation | |
| 6 | 12-17 | Calibration, triage, robustness, external validation, fairness | G5 |
| 7 | 15-18 | Explainability (quantitative) | |
| 8 | 14-22 | Writing (starts early, overlaps) | |
| 9 | 21-24 | Release package, internal review, submission | G6 |
| 10 | After submission | Revision and rebuttal (budget 8-12 weeks) | |

Phases overlap; writing the Methods and Data sections should start in Phase 2.

---

## 4. Detailed tasks

### Phase 0: Scoping, protocol freeze and governance (Weeks 1-2)
- [ ] **Environment**: pinned conda/uv environment (PyTorch, torchvision, timm, scikit-learn, albumentations, pandas, numpy, matplotlib, seaborn, pytorch-grad-cam, tqdm, wandb or tensorboard, plus `open_clip`, `cleanlab`, `netcal` or `torchmetrics` calibration, `mapie` for conformal prediction). Record `pip freeze` and CUDA/driver versions.
- [ ] **Repo skeleton**: `src/`, `data/`, `splits/`, `configs/`, `results/`, `notebooks/`, `assets/`, `scripts/`, `docs/`. Git from day one; tag every experiment with commit hash.
- [ ] **[NEW] Research questions (RQ), written down and frozen:**
  - RQ1: How much does image-level splitting inflate performance on this dataset (and is the inflation class-dependent)?
  - RQ2: Under source-image-disjoint evaluation, how do CNNs, ViT/Swin and foundation-model features compare, and which training tricks actually help under variance-aware testing?
  - RQ3: Are models learning disease cues or acquisition shortcuts?
  - RQ4: Do models generalize to an independent Bangladeshi dataset (SkinDisNet), and are their probabilities usable for triage (calibration, abstention)?
  - RQ5: Do saliency maps align with clinician-marked lesions, quantitatively?
- [ ] **[NEW] Primary endpoint and analysis plan**: primary metric = macro-F1 on raw-image-level out-of-fold predictions; secondary = balanced accuracy, MCC, per-class sensitivity/specificity, ECE, selective-risk AUC. Pre-specify comparisons and multiplicity correction (Holm). Deposit the plan (OSF or a timestamped repo commit) **before** running test folds.
- [ ] **[NEW] Licence/ethics**: read the Mendeley licence (confirm redistribution rules; likely redistribute only IDs/splits, not images). Read the dataset paper's ethics and consent statement. Check for identifiable content (faces, tattoos, name tags). Write the ethics statement (secondary analysis of anonymized public data; confirm whether your institution requires an exemption letter).
- [ ] **[NEW] Clinician partner**: recruit a dermatologist (co-author or acknowledged annotator). Scope of their work: validate labels on a sample, assign FST, mark lesion regions on ~100-150 images, help define referral tiers. Q1 clinical-AI reviewers consistently ask for this.
- [ ] **[NEW] Compute plan**: estimate runs (see §6), decide platform (local GPU, Kaggle/Colab-class), checkpoint and resume strategy.
- [ ] **Literature**: build a reference library (Zotero) with the expanded list in §9; write a one-page related-work map.

### Phase 1: Data audit, provenance and bias audit (Weeks 2-4)
- [ ] **File inventory**: counts per class and overall; resolve 197 vs 250 raw; resolve 1,612 total and class range (302-381). Record dataset version, download date, checksums (SHA-256 per file).
- [ ] **Naming/structure inspection**: folder structure, prefixes, indices, suffixes (`_aug`, `_rot`, `_flip`, ...). Build `raw_to_aug_map.json` where possible.
- [ ] **Exact and near-duplicate detection (independent of filenames)**: MD5/SHA, perceptual hash (pHash/dHash), then embedding similarity (DINOv2 and ResNet-50) with thresholds chosen from a labeled sample of known parent-child pairs.
- [ ] **Augmentation forensics [NEW]**: infer which transformations were applied (flip, rotate, crop, brightness) by registration against the parent; record per-image transform type. Needed for the leakage dose-response analysis and for deciding whether derivatives should be used in training at all.
- [ ] **[NEW] Shortcut probes (RQ3)** run on the full dataset with the *group-aware* CV from Phase 2 once available (run provisional version now):
  - Class predictability from EXIF/camera model, image size, aspect ratio, JPEG quality, file size.
  - Class predictability from color-histogram/mean-RGB features (simple classifier).
  - Class predictability from **border-only** or **background-only** crops (center masked out).
  - Class predictability from **lesion-only** vs **background-masked** inputs on the clinician-marked subset.
  - If acquisition-only features predict class well above chance, state it prominently; this is itself a major finding.
- [ ] **[NEW] Label-quality audit**: confident-learning (cleanlab) on out-of-fold probabilities to flag suspect labels; clinician reviews a stratified sample (e.g., 60 images, plus all flagged ones if feasible). Report agreement (Cohen's kappa) and label-noise estimate.
- [ ] **[NEW] Image quality/diversity summary**: resolution distribution, blur score, skin-tone proxy distribution (ITA with caveats), body-site if inferable.
- [ ] **Audit report**: `docs/data_audit.md` with counts, grouping method and confidence, unresolved cases, shortcut findings, label-noise estimate, licence notes.

**Gate G1:** if grouping confidence < 90% by filename, switch to embedding-reconstructed groups; if label noise > ~10% or strong acquisition shortcuts appear, redesign claims (the paper becomes more about dataset validity, which is still publishable but changes the framing).

### Phase 2: Group reconstruction and split design (Weeks 3-6)
- [ ] **Define groups**: final `raw_id` for each image; merge near-identical raw images (likely same lesion/patient) into **super-groups** (`group_id`). Report counts: raw images, groups, images per group, groups per class.
- [ ] **Group-reconstruction validation**: spot-check 20 groups per class visually (more if clinician time allows); compute precision/recall of reconstruction on filename-confirmed pairs; report as "reconstruction fidelity."
- [ ] **Split protocols (all exported):**
  - **P-leaked**: image-level stratified 70/15/15 random split (mimics [1]).
  - **P-disjoint (primary)**: repeated **StratifiedGroupKFold** (5 folds × 3 repeats, fixed seeds), groups = `group_id`. Inner validation = group-level split from training folds.
  - **P-partial [NEW]**: controlled leakage levels (0%, 25%, 50%, 100% of test groups having derivatives in train) for the dose-response curve.
  - **P-holdout (optional)**: if raw count supports it, a frozen ~20% group-level holdout never touched until final reporting.
- [ ] **Evaluation unit rule [FIX]**: test and validation sets contain **raw images only** (primary). Derivatives of training groups may be used for training. Derivative-level test results are secondary.
- [ ] **Disjointness tests (automated, run in CI)**: assert no `group_id` crosses splits; assert no exact/near duplicate pair (similarity ≥ threshold) crosses splits; assert class presence in every fold.
- [ ] **Exports**: `splits/<protocol>/fold{k}_rep{r}_{train,val,test}.csv` with `filepath, raw_id, group_id, class_label, is_derivative, transform_type`; plus `raw_groups.json`.
- [ ] **Fallback**: if groups cannot cover all images, evaluate only on confirmed raw images and state limitations explicitly.

**Gate G2:** minimum viable test power. With ~40 groups per class, per-class sensitivity CIs are wide (e.g., 36/40 ≈ 90% has a Wilson 95% CI of about 77-96%). Say so in the paper and rely on pooled out-of-fold predictions and external validation for stronger claims.

### Phase 3: Leakage quantification (Weeks 5-8) (RQ1)
- [ ] **Reproduce [1] faithfully**: implement the exact classical ML/DL pipeline, features, hyperparameters; if details are missing, document assumptions and test sensitivity.
- [ ] **Run baseline under P-leaked and P-disjoint** with the same code, same hyperparameters, same seeds.
- [ ] **Trivial memorization baselines [NEW]**: 1-NN on pixel/embedding space and a linear probe on frozen features under both protocols. If 1-NN scores near the leaked "state-of-the-art" accuracy, that is a compelling, intuitive demonstration of leakage.
- [ ] **Quantify inflation [FIX]**: overall and per-class Δ with group-bootstrap CIs; permutation test for Δ; confusion-matrix differences. Use McNemar only for paired model comparisons on one shared test set.
- [ ] **Dose-response [NEW]**: accuracy vs leakage fraction (P-partial); a monotone curve is a strong figure.
- [ ] **Do the numbers transfer?** Evaluate the leaked-trained model on the disjoint test set (paired, McNemar valid) to show what a user would actually experience in deployment.
- [ ] **Document**: `docs/leakage_analysis.md` with tables and figures.

### Phase 4: Main benchmark (Weeks 7-12) (RQ2)
- [ ] **Model zoo (timm, ImageNet-pretrained unless noted):**
  - CNNs: ResNet-50, DenseNet-121, EfficientNet-B0, EfficientNet-B3, MobileNetV3-Large, **ConvNeXt-T [NEW]** (and optionally EfficientNetV2-S).
  - Transformers: ViT-B/16, Swin-T (add DeiT-S or Swin-S only if compute allows).
  - **Foundation/self-supervised [NEW]:** DINOv2 (ViT-S/14 or B/14) with linear probe and fine-tune; a medical/dermatology foundation model if weights are accessible under licence (check availability and terms before committing; do not plan around a model you cannot download).
  - **Strong simple baselines [NEW]:** logistic regression/SVM on frozen embeddings; the classical ML pipeline from [1].
- [ ] **Training protocol**: phased fine-tuning (head, then partial, then full unfreezing), layer-wise LR decay for ViT/Swin, AdamW, cosine schedule with warmup, weight decay, AMP, early stopping (patience 10-15) on validation macro-F1, gradient accumulation if needed. Fix hyperparameter search budget per model (e.g., 8 configs via inner validation on one fold) and log it.
- [ ] **Preprocessing**: resize to 224×224; per-model normalization; training-time augmentation (flip, rotation, mild color jitter, RandomResizedCrop with conservative scale). Think about whether training on pre-augmented derivatives plus online augmentation double-augments; ablate "raw-only training" vs "raw + provided derivatives" [NEW].
- [ ] **Resolution sensitivity**: 224 vs 384/512 on the top 3 models only.
- [ ] **Seeds/folds**: 5 folds × 3 repeats × (3 seeds for primary models, 5 seeds on the final shortlist). Save all out-of-fold predictions (logits) to disk; all later analysis reads these files.
- [ ] **Efficiency table [NEW]**: params, FLOPs, peak memory, CPU and (if possible) mobile latency after ONNX/TFLite export; GPU-hours.
- [ ] **Output**: `results/benchmarks.csv` (one row per model × fold × repeat × seed) and aggregated tables.

**Gate G4:** if every model reaches near-ceiling performance under disjoint evaluation, the leakage story is weaker and the dataset may be "easy" or shortcut-driven; shift emphasis to shortcut analysis and external validation.

### Phase 5: Training-trick ablation (Weeks 11-14)
- [ ] Run on **3 representative models** (e.g., EfficientNet-B3, Swin-T or ViT-B/16, ConvNeXt-T), not all 8.
- [ ] **One-factor-at-a-time vs baseline:** MixUp/CutMix; focal loss (γ=2.0, also try 1.0); class-balanced sampling; label smoothing (ε=0.1); online augmentation strength; then one cumulative best-recipe run.
- [ ] **TTA** evaluated at inference on saved checkpoints (flip/rotate/multi-crop; report the TTA set).
- [ ] **Statistics**: paired comparison on matched fold×seed units; Wilcoxon or Bayesian signed-rank with Holm correction; effect sizes with group-bootstrap CIs. Report negative results honestly.
- [ ] **Output**: ablation table with mean ± std **and** CIs; one forest plot.

### Phase 6: Calibration, triage, robustness, external validation, fairness (Weeks 12-17) (RQ3, RQ4)
- [ ] **Core metrics (per class and pooled):** accuracy, balanced accuracy, macro/weighted F1, MCC, precision, recall (sensitivity), specificity, AUROC (one-vs-rest), with group-bootstrap 95% CIs.
- [ ] **Confusion analysis**: eczema vs dermatitis, tinea vs other scaly dermatoses, scabies vs inflammatory dermatoses; show which confusions persist across models and folds.
- [ ] **Calibration [NEW]**: ECE, Brier score, reliability diagrams; temperature scaling fit on validation groups only; before/after tables.
- [ ] **Selective prediction and triage [NEW]**: risk-coverage curves; abstention thresholds; **conformal prediction sets** (APS/RAPS) with empirical coverage per class. Interpret as "refer when the set is large/uncertain." Define referral tiers with the clinician rather than inventing clinical weights.
- [ ] **Robustness [NEW]**: simulate smartphone conditions (blur, JPEG compression, brightness/contrast shifts, color temperature, resolution downsampling, partial occlusion, crop). Report relative drop per model.
- [ ] **External validation [NEW] on SkinDisNet [6]:**
  - Map the overlapping four classes; document mapping and label-definition differences; exclude vitiligo.
  - Zero-shot (train on this dataset, test on SkinDisNet) and optionally fine-tuned transfer.
  - Because SkinDisNet has patient IDs, also run a true **patient-disjoint** split there; this is where "subject-disjoint" can be claimed honestly.
  - Report performance drop and calibration drift under domain shift.
- [ ] **Fairness/subgroup analysis [NEW]:** by FST (clinician-annotated) or ITA proxy, and by image-quality tertile; report sensitivity/specificity gaps with CIs; state clearly that sample sizes limit conclusions.
- [ ] **Shortcut re-test [NEW]:** repeat Phase 1 probes using trained-model attribution (e.g., how much accuracy remains with background-only or lesion-masked inputs).

### Phase 7: Explainability, quantitative (Weeks 15-18) (RQ5)
- [ ] **Methods:** Grad-CAM/Grad-CAM++ for CNNs; attention rollout and a transformer attribution method (e.g., Chefer et al.) for ViT/Swin.
- [ ] **Sanity checks [NEW]:** model-randomization and label-randomization tests for saliency (Adebayo et al.) so you are not presenting edge detectors as explanations.
- [ ] **Quantitative metrics [NEW]:** insertion/deletion AUC; pointing-game and IoU/energy-in-mask against clinician-marked lesion masks on ~100-150 images; compare correct vs misclassified cases.
- [ ] **Qualitative panel:** per class, 3 correct + 3 misclassified examples, with clinician commentary on at least a subset.
- [ ] Clinical validity of saliency remains a stated limitation unless the clinician review covers it.

### Phase 8: Writing (Weeks 14-22, overlapping)
- [ ] **Skeleton paper by Week 14** with empty figure/table placeholders.
- [ ] **Planned structure:** Abstract; Introduction (clinical need, leakage problem, contributions as 4-5 bullets); Related Work; Dataset and audit; Evaluation protocols; Models and training; Experiments (leakage, benchmark, ablation, calibration/triage, external validation, robustness, fairness, explainability); Discussion (clinical interpretation, why leakage matters for the field, recommendations); Limitations; Conclusion; Data/code availability; Ethics; Author contributions; Competing interests.
- [ ] **Key figures (target 8-10):** pipeline/protocol diagram; group-size distribution; leaked-vs-disjoint bars with CIs; leakage dose-response; benchmark forest plot; per-class sensitivity/specificity; confusion matrix (pooled); reliability and risk-coverage plots; external-validation drop; saliency panel with quantitative overlay.
- [ ] **Key tables:** dataset audit; split statistics; main benchmark (mean ± std and CI); ablation; efficiency; external validation; subgroup performance; comparison with prior work.
- [ ] **Language/claims discipline:** say "source-image-disjoint" for this dataset; avoid "clinically validated"; avoid claiming deployment readiness; frame triage as decision-support research.
- [ ] **Tools:** keep all numbers generated by scripts into `results/`; a single `make_figures.py` rebuilds every figure so nothing is pasted by hand.

### Phase 9: Release, internal review and submission (Weeks 21-24)
- [ ] **Reproducibility package:** `splits/`, `raw_groups.json`, `configs/*.yaml`, `requirements.txt` or lockfile, `Dockerfile` (optional), `README.md` with exact commands, trained-model checkpoints (if licence allows), a Zenodo DOI for the release.
- [ ] **Do not redistribute images** unless the licence explicitly allows; publish IDs, hashes, split files and download scripts.
- [ ] **Pre-submission checks:** run the full pipeline from scratch on a clean machine for at least one fold; verify disjointness assertions; verify every number in the paper traces to a result file; CLAIM and TRIPOD+AI checklists completed; plagiarism/similarity check; confirm all references.
- [ ] **Internal review:** one co-author or advisor plays hostile reviewer (list likely reviewer comments and answer them in the Discussion/limitations pre-emptively); clinician reads the clinical sections.
- [ ] **Submission materials:** cover letter stating the three main contributions and why they matter beyond this dataset; highlights; suggested reviewers; graphical abstract if required; supplementary material (full tables, hyperparameters, extra figures). Preprint (arXiv) after checking journal policy.

### Phase 10: Revision readiness (post-submission)
- [ ] Reserve compute and time for requested extra experiments (typical: additional baselines, extra external dataset, ablations, statistical clarifications).
- [ ] Keep an experiment log so additional runs use identical code and seeds.
- [ ] Prepare a point-by-point response template; never argue without a new result or a precise citation.

---

## 5. Deliverables (updated)

| Deliverable | Description | Status |
|---|---|---|
| `docs/data_audit.md` | Counts, naming, dedup, augmentation forensics, shortcut and label audit | Complete; clinician label/group review pending |
| `raw_groups.json`, `splits/*` | Group maps and all protocols (leaked, disjoint, partial) | Complete; groups provisional/over-merged |
| `docs/leakage_analysis.md` | Frozen-probe inflation with CIs, dose-response, 1-NN/linear baselines | Complete; reproduction of [1] still pending |
| `results/benchmarks.csv` + OOF predictions | All model × fold × repeat × seed results | Not started; 128 px and 160 px ResNet-50 pilot OOF only |
| `docs/BENCHMARKS.md`, `docs/ABLATION.md` | Benchmark and variance-aware ablation | Not started |
| `docs/CALIBRATION_TRIAGE.md` | Calibration, conformal sets, risk-coverage | Not started |
| `docs/EXTERNAL_VALIDATION.md` | SkinDisNet zero-shot and patient-disjoint results | Not started |
| `docs/FAIRNESS_ROBUSTNESS.md` | Subgroup and corruption analyses | Not started |
| `docs/EXPLAINABILITY.md` | Saliency sanity checks and quantitative metrics | Not started |
| Reproducibility package + DOI | Code, configs, splits, README, model card, datasheet | Partial: README, requirements, splits and scripts exist; configs, model card, datasheet and DOI pending |
| Manuscript + supplement + cover letter + checklists | CLAIM, TRIPOD+AI, ethics, data statement | In progress: `Proposal/ALL contain` |
| Preprint and presentation slides | Derived from the final manuscript | Not started |

---

## 6. Compute and experiment budget (rough planning estimate; recompute after pilot)

| Block | Runs |
|---|---|
| Main benchmark: 9 models × 5 folds × 3 repeats × 3 seeds | ≈ 405 (reduce by using 1 seed on the first pass, then adding seeds only for the shortlist) |
| Ablation: 3 models × 6 settings × 5 folds × 3 seeds | ≈ 270 |
| Resolution: 3 models × 5 folds × 3 seeds | ≈ 45 |
| Leakage study: baselines × 4 protocols × seeds | ≈ 50-100 |

With ~1,100-1,300 training images per fold at 224×224, a run is typically minutes on a single modern GPU, so the budget is tractable, but 500+ runs make checkpoint management, logging and automated resume essential. Do a pilot (one model, one fold) in Week 6 to measure real time per run, then trim the grid to what fits. Prioritize the shortlist over breadth: fewer models with more rigorous statistics beat many models with single runs.

---

## 7. Decision gates and contingencies

| Gate | Question | If no |
|---|---|---|
| G0 | Journal, clinician partner, licence OK? | Delay; do not start experiments before the protocol is frozen. |
| G1 | Grouping confidence ≥ 90%? Label noise acceptable? Shortcuts? | Use embedding groups; clinician label review; reframe toward dataset-validity paper. |
| G2 | Enough groups for stable folds? | Reduce to 3-fold group CV; emphasize pooled CIs and external validation. |
| G3 | Is inflation substantial and statistically clear? | Report honestly; shift weight to benchmark + external validation. |
| G4 | Benchmark differentiates models? | If all saturate, emphasize shortcuts, calibration, robustness, efficiency. |
| G5 | External validation workable (class mapping, domain gap)? | Use a second public smartphone/clinical dataset or a cross-site subset; state limits. |
| G6 | All checklist items passing? | Do not submit until reproduction from scratch succeeds. |

---

## 8. Updated risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Incremental-novelty rejection | High | Contribution bundle (leakage dose-response, external validation, shortcuts, calibration, quantitative XAI); clinician co-author; strong framing. |
| Cannot confirm patient-level separation | High | Rename protocol; super-group merging; patient-disjoint demo on SkinDisNet; state the limitation. |
| Tiny raw set, unstable estimates | High | Repeated group k-fold, pooled OOF, cluster bootstrap CIs, honest power statement. |
| Pseudo-replication from derivatives | High | Raw-image-only test; cluster-aware statistics. |
| Acquisition shortcuts/class-correlated backgrounds | High | Shortcut probes; background-masked experiments; external validation. |
| Label noise or inconsistent diagnoses | High | Confident learning + clinician review; report noise estimates. |
| Transformer instability on small data | Medium-High | Phased fine-tuning, layer-wise LR decay, warmup, smaller LR, label smoothing; DINOv2 features as stable alternative. |
| Multiple-comparison inflation | Medium | Preregistered comparisons, Holm correction. |
| Licence prevents redistribution | Medium | Publish IDs/hashes/splits + scripts, not images. |
| SkinDisNet class mapping mismatch | Medium | Document definitions with clinician; evaluate only on mappable classes. |
| Compute/time overrun | Medium | Pilot-based grid trimming; shortlist-first strategy; automated resume. |
| Reviewer asks for more datasets or clinicians | Medium | Reserve revision time; keep code modular so a new dataset is a config change. |
| Citation errors | Low-Medium | Verify every reference against the source before submission. |

---

## 9. Reading list to add (verify details when citing)

- **Leakage and reproducibility:** Kapoor & Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science" (Patterns, 2023); Roberts et al., common pitfalls in ML for COVID-19 imaging (Nature Machine Intelligence, 2021); Bussola et al., "AI slipping on tiles" (leakage in digital pathology); Bissoto et al., "Debiasing skin lesion datasets and models? Not so fast" (CVPR workshops, 2020).
- **Dermatology fairness/diverse data:** Daneshjou et al., Diverse Dermatology Images (DDI) (Science Advances, 2022); Groh et al., Fitzpatrick17k; Google SCIN dataset; PAD-UFES-20 (smartphone clinical images).
- **Architectures/training:** Tan & Le (EfficientNet); Dosovitskiy et al. (ViT); Liu et al. (Swin); Liu et al. (ConvNeXt); Oquab et al. (DINOv2); Zhang et al. (mixup); Yun et al. (CutMix); Lin et al. (focal loss); Szegedy et al. / Müller et al. (label smoothing).
- **Calibration/uncertainty:** Guo et al., "On calibration of modern neural networks"; Geifman & El-Yaniv (selective classification); Angelopoulos & Bates (conformal prediction tutorial); Romano et al. (APS), Angelopoulos et al. (RAPS).
- **Statistics:** Dietterich (comparing classifiers); Demšar (statistical comparisons across datasets); Benavoli et al. (time for a change: Bayesian signed-rank); Efron & Tibshirani (bootstrap); cluster bootstrap references.
- **Explainability:** Selvaraju et al. (Grad-CAM); Chefer et al. (transformer interpretability); Adebayo et al. (sanity checks for saliency); Petsiuk et al. (RISE, insertion/deletion).
- **Reporting:** CLAIM 2020 update; TRIPOD+AI (2024); STARD-AI; Mitchell et al. (model cards); Gebru et al. (datasheets).
- **Your existing six references:** keep [1]-[6] but verify [1] (arXiv ID), [3] (exact title and venue), [5] (DOI) and [6] against the original pages.

---

## 10. Completion checklist (Q1 version)

- [ ] Journal and backup journal chosen; scope fit confirmed (G0)
- [ ] Clinician partner engaged; ethics/licence statements written
- [ ] Analysis plan frozen before test evaluation
- [ ] Data audit, dedup, shortcut and label-noise audit complete
- [ ] Source-image-disjoint groups validated; all protocols exported; disjointness tests automated
- [ ] Leakage quantified with CIs, dose-response and 1-NN baseline
- [ ] Main benchmark completed with OOF predictions saved
- [ ] Ablation done one-factor-at-a-time with proper statistics
- [ ] Calibration, conformal triage, robustness and subgroup analyses complete
- [ ] External validation on SkinDisNet (including patient-disjoint split there)
- [ ] Quantitative explainability with saliency sanity checks
- [ ] Reproducibility package with DOI; from-scratch run verified
- [ ] CLAIM and TRIPOD+AI checklists completed; references verified
- [ ] Internal hostile-review pass done; cover letter and supplement ready
- [ ] Submitted; revision plan and compute reserved
