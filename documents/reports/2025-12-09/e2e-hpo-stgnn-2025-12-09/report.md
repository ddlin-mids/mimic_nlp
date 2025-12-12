# Daily Report: E2E Gated Fusion HPO & STGNN Infrastructure

**Date:** 2025-12-09 (Pacific Time)  
**Agent:** Claude Code  
**Project:** MIMIC-IV 30-Day Readmission Prediction  
**Subtask:** Intensive HPO for E2E Gated Fusion + STGNN Monitoring

---

## 1) Starting Plan of the Day

- Monitor STGNN job (47018768) submitted previously
- Prepare intensive HPO for best-performing models
- "Pick a best model and HPO it to the max" per user direction
- Evaluate STGNN results when complete

**Context Sources Used:**
- Previous session conversation (E2E Gated Fusion results: AUC 0.6246, F1 0.41)
- MLflow experiment logs
- SLURM job outputs

---

## 2) What Was Done

### Data
- **No new data pulled** - using existing preprocessed artifacts
- Documented data lineage for training jobs:
  - EHR sequences: `data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_cat_embedding.pkl` (7.3 GB)
  - Cohort: `data/interim/readmit_analysis/long_los_cohort.csv` (15,659 admissions)
  - Text embeddings: `data/interim/embeddings/notes/` (~300 MB total)

### Modeling / Experiments

**1. E2E Gated Fusion HPO Script Enhancement**
- Rewrote `src/models/pytorch/train_e2e_gated_fusion_hpo.py` with comprehensive search space:
  - `hidden_dim`: [64, 128, 256]
  - `dropout`: [0.1, 0.5]
  - `lr`: [5e-5, 5e-3] (log scale)
  - `encoder_lr`: [1e-6, 1e-4] (log scale)
  - `num_layers`: [1, 3]
  - `num_heads`: [2, 4, 8]
  - `pca_components`: [32, 64, 128]
  - `warmup_ratio`: [0.05, 0.2]
  - `batch_size`: [16, 32, 64]
  - `weight_decay`: [1e-4, 1e-1] (log scale)
- Added TPE sampler + Median pruner for efficient search
- Implemented final model training with best params + test evaluation
- Full MLflow tracking integration

**2. Jobs Submitted**
| Job ID | Name | Status | Time Elapsed | Partition |
|--------|------|--------|--------------|-----------|
| 47019078 | E2E HPO | Running | ~3 min | free-gpu |
| 47018768 | STGNN | Running | ~50 min (stuck) | free-gpu |

**3. STGNN Data Loading Issue Identified**
- STGNN job stuck for 50+ minutes loading 7.3 GB pickle file
- Root cause: Python pickle unpickling is single-threaded and slow on HPC shared filesystem (CRSP)
- HPO job succeeded in loading same data faster (likely better I/O or cached)

### Analysis / Interpretation

**E2E Gated Fusion Baseline Results (from previous session):**
- Test AUC: 0.6246 (below XGBoost 0.641)
- Test F1: **0.410** (best among all neural models)
- Test Recall: 0.753 (75% of readmissions flagged)
- Mean Gate Value: 0.48 (balanced EHR/text weighting)

**Key Insight:** E2E architecture trades ~0.016 AUC for +0.12 F1 improvement over GRU Gated Fusion. This makes it attractive for clinical alerting where high recall matters more than perfect ranking.

### Artifacts Produced

| Artifact | Purpose |
|----------|---------|
| `src/models/pytorch/train_e2e_gated_fusion_hpo.py` | Enhanced HPO script with 10-dimensional search space |
| `scripts/slurm/train_e2e_gated_fusion_hpo.sbatch` | SLURM job script for 12-hour HPO run |
| Updated `documents/reports/final_project_report/final_project_report_v2.md` Section 5.5 | Added E2E results |

---

## 3) Results Snapshot

### Current Model Leaderboard (Pre-HPO)

| Rank | Model | Test AUC | Test AUPRC | Test F1 | Notes |
|------|-------|----------|------------|---------|-------|
| 1 | XGBoost (GRU EHR) | **0.641** | 0.356 | 0.00 | Best AUC, no recall |
| 2 | Gated Fusion (GRU) | 0.638 | 0.350 | 0.29 | Baseline multimodal |
| 3 | Early Fusion MLP (HPO) | 0.638 | 0.354 | 0.35 | Strong balanced |
| 4 | E2E Gated Fusion | 0.625 | 0.353 | **0.41** | Best F1 (neural) |
| 5 | GNN (GraphSAGE) | 0.596 | 0.319 | 0.41 | High recall (72%) |

*No new experimental results today - HPO job in progress*

---

## 4) Impact Assessment

### Accuracy / Utility
- HPO expected to improve E2E Gated Fusion from 0.625 → potentially 0.64+ AUC
- If successful, would combine best AUC with best F1

### Reliability / Robustness
- 50 Optuna trials with TPE sampler should find stable configuration
- Median pruning prevents wasted compute on poor configurations

### Decision-readiness
- Pending HPO results (~12 hours)
- If HPO succeeds: ready for final model selection
- If STGNN unsticks: additional architecture comparison available

### Risk & Ethics
- No PHI exposure (working with preprocessed embeddings)
- HPC I/O bottleneck may delay STGNN evaluation

---

## 5) Deviations from Plan

| Planned | Actual | Rationale |
|---------|--------|-----------|
| Evaluate STGNN results | Monitoring stuck job | Pickle loading I/O bottleneck on HPC |
| Quick HPO setup | Comprehensive HPO rewrite | Original script too basic; needed proper search space |

---

## 6) Open Questions & Unknowns

1. **Will STGNN job complete?**
   - Evidence needed: Wait for pickle load or cancel/resubmit to different node
   
2. **Can HPO push E2E AUC above 0.64?**
   - Evidence needed: HPO trial results (expected in ~12 hours)

3. **Is pickle format the right choice for 7GB+ data?**
   - Evidence needed: Benchmark zarr/HDF5 vs pickle loading times

---

## 7) Next Steps (Ranked)

### Immediate (Tonight/Tomorrow Morning)
1. **Monitor HPO job 47019078** - Check Optuna trial progress
   - Owner: Agent
   - Success: 50 trials complete, best_val_auc logged

2. **Decide on STGNN job** - If still stuck after 2 hours, cancel and investigate
   - Owner: Agent
   - Success: Either results or root cause identified

### Short-term (This Week)
3. **Evaluate HPO results** - Train final model with best params
   - Owner: Agent
   - Success: Test AUC >= 0.63, F1 >= 0.40

4. **Update final report** - Incorporate HPO results into Section 5.5
   - Owner: Agent
   - Success: Report reflects best model configuration

### Nice-to-have
5. **Convert pickle to faster format** - If I/O remains bottleneck
   - Owner: Agent
   - Success: Data load time < 5 minutes

---

## 8) Reproducibility Notes

### Entry Points
1. `scripts/slurm/train_e2e_gated_fusion_hpo.sbatch` - Submit HPO job
2. `src/models/pytorch/train_e2e_gated_fusion_hpo.py` - Main HPO script

### Minimal Config
```bash
# Submit HPO job
sbatch scripts/slurm/train_e2e_gated_fusion_hpo.sbatch

# Or run directly
uv run python src/models/pytorch/train_e2e_gated_fusion_hpo.py \
    --ehr_seq_path data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_cat_embedding.pkl \
    --text_dir data/interim/embeddings/notes \
    --cohort_path data/interim/readmit_analysis/long_los_cohort.csv \
    --n_trials 50 \
    --epochs_per_trial 15 \
    --final_epochs 50
```

### Randomness
- Seed: 42 (set in script)
- TPE sampler seeded for reproducibility
- PyTorch manual seed set

### Data Lineage
```
MIMIC-IV (physionet.org) 
  → ehr/get_mimic_cohort.py (cohort filtering)
  → ehr/preprocess_ehr.py (sequence extraction)
  → ehr/embed_notes_bioclinical_mbert.py (text encoding)
  → data/interim/ehr_long_los/*.pkl + data/interim/embeddings/notes/*.npz
```

---

## 9) Appendices

### A. HPO Search Space Details

| Parameter | Type | Range | Scale |
|-----------|------|-------|-------|
| hidden_dim | categorical | [64, 128, 256] | - |
| dropout | float | [0.1, 0.5] | linear |
| lr | float | [5e-5, 5e-3] | log |
| encoder_lr | float | [1e-6, 1e-4] | log |
| num_layers | int | [1, 3] | linear |
| num_heads | categorical | [2, 4, 8] | - |
| pca_components | categorical | [32, 64, 128] | - |
| warmup_ratio | float | [0.05, 0.2] | linear |
| batch_size | categorical | [16, 32, 64] | - |
| weight_decay | float | [1e-4, 1e-1] | log |

### B. Job Status Commands

```bash
# Check job status
squeue -u $USER --format="%.10i %.12P %.30j %.8T %.10M %.6D %R"

# Check HPO logs
tail -f logs/slurm/e2e_hpo_47019078.log

# Check STGNN logs
tail -f logs/slurm/stgnn_47018768.log
```
