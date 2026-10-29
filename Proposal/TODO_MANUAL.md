# TODO - Manual Tasks (Require Human Intervention)

These tasks cannot be done by an AI agent and require you to do them personally or with help from others.

## Phase 0: Governance & Ethics (Week 1-2) - CRITICAL
- [ ] **Read Mendeley licence** - Check exact redistribution terms for dataset v3 (DOI: 10.17632/9ggd3shdr7.3). Confirm if images can be shared or only splits/IDs.
- [ ] **Ethics review** - Check if your institution requires ethics exemption/approval for secondary analysis of publicly available anonymized data.
- [ ] **Journal selection** - Pick primary + backup Q1 journal (Computers in Biology and Medicine, AI in Medicine, CMPB, JBHI, Scientific Reports). Read 5 recent papers from chosen journal.
- [ ] **Clinician recruitment** - Find and recruit a dermatologist co-author/consultant. Discuss scope (label validation, FST annotation, lesion marking, referral tiers).
- [ ] **OSF pre-registration** - Register analysis plan on OSF with timestamp before running experiments.

## Phase 1: Data & Clinical Annotations (Week 2-6)
- [ ] **Verify dataset download** - Manually verify files downloaded correctly from Mendeley, compare checksums if possible.
- [ ] **Manual inspection** - Visually spot-check 20-30 groups to verify group reconstruction accuracy.
- [ ] **Label validation** - Have clinician review stratified sample (~60 images) + suspect labels flagged by confident learning.
- [ ] **Fitzpatrick Skin Type (FST)** - Clinician annotates FST I-VI on all/most raw images (~200). This is labor intensive.
- [ ] **Lesion region marking** - Clinician marks lesion masks on ~100-150 images for quantitative XAI validation (pointing game/IoU).
- [ ] **Clinical interpretation** - Discuss confusion patterns with clinician (eczema vs dermatitis etc.) to define meaningful referral tiers.

## Phase 2: External Validation Setup (Week 12-17)
- [ ] **Access SkinDisNet** - Download/access SkinDisNet dataset (DOI: 10.17632/yj3md44hxg.2) from Mendeley. Check licence terms.
- [ ] **Class mapping discussion** - Collaborate with clinician to resolve class mapping between datasets and label differences.
- [ ] **Verify patient IDs** - Confirm SkinDisNet has patient IDs for true patient-disjoint validation.

## Phase 3: Writing, Review & Submission (Week 14-24)
- [ ] **Clinical sections review** - Clinician reviews clinical relevance, triage framing, limitations.
- [ ] **Internal review** - Get advisor/co-author to do hostile reviewer pass.
- [ ] **Journal submission** - Submit manuscript, cover letter, supplementary files via journal portal.
- [ ] **Preprint decision** - Decide on arXiv preprint based on journal policy.
- [ ] **Revision response** - Respond to reviewers with point-by point, conduct extra experiments as requested.
