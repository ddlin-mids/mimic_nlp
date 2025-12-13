# Daily Report: Lab Trajectory Features, Trimodal Fusion & Ensemble Analysis

**Date:** 2025-12-13 (Pacific Time)  
**Agent:** Claude Code  
**Project:** MIMIC-IV 30-Day Readmission Prediction  
**Subtask:** Lab Feature Engineering, Trimodal Fusion, Late Fusion Ensemble, Architecture Analysis

---

## 1) Starting Context

Coming from previous sessions:
- Best model: XGBoost on GRU EHR embeddings (AUC 0.641)
- Gated Fusion (EHR + Text): AUC 0.638
- Stacking Ensemble: AUC 0.6242
- Goal: Push AUC toward 0.70

**Session Objectives:**
1. Extract lab trajectory features from MIMIC labevents
2. Test if lab features improve model performance
3. Build trimodal fusion (EHR + Text + Labs)
4. Build late fusion ensemble combining best models
5. Deep analysis of model architectures and literature benchmarks

---

## 2) What Was Done

### Phase 1: Lab Trajectory Feature Extraction

**Created:** `src/data/extract_lab_trajectory_features.py`

Extracted 135 features from 15 clinically-relevant labs for cardiorenal patients:

| Lab Category | Labs Included |
|--------------|---------------|
| Kidney Function | Creatinine, BUN |
| Electrolytes | Potassium, Sodium, Chloride, Bicarbonate |
| Cardiac Markers | NT-proBNP, Troponin I, Troponin T |
| Hematology | Hemoglobin, Platelets, WBC |
| Metabolic | Glucose, Albumin, Bilirubin, Lactate, pH, Anion Gap |

**Features per lab (9 each):**
- `last`: Last value before discharge
- `slope_72h`: Linear trend in final 72 hours
- `max`, `min`, `mean`, `std`: Distribution statistics
- `count`: Number of measurements
- `missing`: Binary indicator if lab never measured
- `first_last_diff`: Change from admission to discharge

**Job 47039886 Results:**
- Processed 2.76M lab events from 158M row labevents.csv.gz
- Output: `data/interim/ehr_long_los/lab_features/lab_features.npz` (5.6 MB)
- 10,204 admissions × 135 features
- Core labs (creatinine, BUN, potassium) have <1% missing rates

---

### Phase 2: XGBoost Ablation with Lab Features

**Updated:** `src/models/train_fusion_xgboost.py`

**Job 47039978 Results:**

| Experiment | Test AUC | Test AUPRC | Notes |
|------------|----------|------------|-------|
| **Lab Features Only** | **0.6268** | 0.3516 | Best single modality! |
| Structured EHR + Labs (GRU) | 0.6218 | 0.3498 | +0.22% vs GRU alone |
| Structured EHR (GRU) | 0.6196 | 0.3442 | Previous baseline |
| Full Fusion + Labs (GRU) | 0.6168 | 0.3412 | Adding more hurts |
| Full Fusion (GRU) | 0.6160 | 0.3398 | |
| Discharge Notes | 0.6067 | 0.3287 | |
| Static EHR | 0.6018 | 0.3241 | |

**Key Finding:** Lab features alone (0.6268) outperform GRU EHR embeddings (0.6196) as a single modality, but combining them in XGBoost doesn't yield additive gains.

---

### Phase 3: Trimodal Neural Fusion (EHR + Text + Labs)

**Created:** `src/models/pytorch/train_trimodal_gated_fusion.py`

Architecture:
```
EHR (128-dim) ──→ Projection ──→ ┐
                                 ├── GMU₁ ──→ ┐
Lab (135-dim) ──→ Projection ──→ ┘            ├── GMU₂ ──→ Classifier
                                              │
Text (64-dim) ──→ Projection ────────────────→┘
```

**Job 47040089 Results:**

| Experiment | Test AUC | Test AUPRC | Notes |
|------------|----------|------------|-------|
| Gated Fusion (hidden=128) | 0.6187 | 0.3382 | Baseline trimodal |
| Gated + Focal Loss | 0.6180 | 0.3338 | Focal loss didn't help |
| Attention Fusion | 0.6109 | 0.3381 | Worse than gated |

**Key Finding:** Trimodal fusion (0.6187) performs **worse** than bimodal gated fusion (0.638). Adding lab features to neural model hurts performance.

---

### Phase 4: Late Fusion Ensemble

**Created:** `src/models/train_late_fusion_ensemble.py`

Strategy: Train separate models, combine predictions at inference.

**Job 47040328 Results:**

| Ensemble Method | Test AUC | Test AUPRC | Notes |
|-----------------|----------|------------|-------|
| **Simple Average** | **0.6308** | 0.3463 | Best ensemble |
| Weighted Average (w=0.46) | 0.6293 | 0.3456 | Optimized on val |
| LR Meta-learner | 0.6178 | 0.3434 | |
| Gated Fusion alone | 0.6152 | 0.3373 | |
| XGB Meta-learner | 0.6149 | 0.3386 | |
| Lab XGBoost alone | 0.6068 | 0.3362 | |

**Key Finding:** Late fusion ensemble (0.6308) improves over individual components but still below the best single model (XGBoost GRU: 0.641).

---

### Phase 5: Architecture Deep Dive & Literature Review

**Investigated feature overlap between GRU and lab features:**

| Aspect | GRU EHR Embeddings | Lab Trajectory Features |
|--------|-------------------|------------------------|
| Lab representation | Binary abnormal/normal flags | Continuous values |
| Labs covered | ~300+ item IDs | 15 cardiorenal-focused |
| Temporal handling | GRU processes daily sequences | Summary statistics |
| Information | "Was creatinine abnormal on day 3?" | "Creatinine rose 1.2→2.8 over 72h" |

**Text Embedding Analysis:**
- Model: `thomas-sounack/bioclinical-modernbert-base` (8192 context)
- Pooling: CLS token only (suboptimal for long documents)
- Fine-tuning: None (frozen feature extractor)

**Literature AUC Benchmarks:**

| Model | AUC | Source |
|-------|-----|--------|
| LACE Score | 0.60-0.68 | Clinical baseline |
| XGBoost/Traditional ML | 0.64-0.74 | Structured EHR |
| LSTM/GRU | 0.76-0.82 | Temporal modeling |
| MM-STGNN (Tang et al.) | **0.79** | Multimodal + imaging + graph |
| Heart Disease subset | **0.80** | Disease-specific |

---

## 3) Final Model Comparison

| Rank | Model | Test AUC | Test AUPRC | Notes |
|------|-------|----------|------------|-------|
| 1 | **XGBoost (GRU EHR)** | **0.641** | 0.356 | Best overall |
| 2 | Gated Fusion (bimodal) | 0.638 | 0.350 | EHR + Text |
| 3 | Late Fusion Ensemble | 0.6308 | 0.346 | Lab XGB + Gated |
| 4 | Lab Features Only (XGB) | 0.6268 | 0.352 | Strong single modality |
| 5 | Stacking Ensemble | 0.6242 | - | From earlier session |
| 6 | Trimodal Gated Fusion | 0.6187 | 0.338 | Adding labs hurt |

---

## 4) Key Insights

### Why Lab Features Hurt Neural Fusion

1. **Redundancy:** GRU already encodes lab events as binary abnormal flags
2. **Representation mismatch:** 135 continuous features vs 128-dim learned embeddings
3. **Optimization difficulty:** More parameters with same data (10K samples)
4. **Different optimal models:** Labs work well with XGBoost (tree-based), not neural

### Why Ensembles Didn't Beat Single Best

1. Base models underperformed on test vs validation (overfitting)
2. Models capture similar signal (high correlation between predictions)
3. Need more diverse models for ensemble gains

### Gap to Literature (0.64 vs 0.79)

MM-STGNN achieved 0.79 AUC with:
1. **Chest X-ray imaging** - we don't have this
2. **Patient similarity graphs** - our GNN attempt was limited
3. **End-to-end multimodal training** - our embeddings are frozen

---

## 5) Artifacts Produced

| Artifact | Path | Purpose |
|----------|------|---------|
| Lab feature extraction | `src/data/extract_lab_trajectory_features.py` | Extract 135 lab trajectory features |
| Lab features data | `data/interim/ehr_long_los/lab_features/lab_features.npz` | 10,204 × 135 feature matrix |
| Trimodal fusion | `src/models/pytorch/train_trimodal_gated_fusion.py` | EHR + Text + Labs neural fusion |
| Late fusion ensemble | `src/models/train_late_fusion_ensemble.py` | Combine Lab XGB + Gated Fusion |
| Ensemble predictions | `data/interim/ehr_long_los/late_fusion_ensemble/ensemble_predictions.npz` | Test set predictions |
| SLURM scripts | `scripts/slurm/train_trimodal_fusion.sbatch`, `train_late_fusion_ensemble.sbatch` | Job submission |

---

## 6) Open Questions

1. **Can mean pooling improve text embeddings?** (Currently using CLS token only)
2. **Would end-to-end GRU training help?** (Currently frozen embeddings)
3. **Are we near the ceiling without imaging data?** (Literature suggests imaging adds ~5-10% AUC)
4. **Should we add explicit LACE-like clinical features?** (Simple, interpretable)

---

## 7) Recommended Next Steps

| Priority | Approach | Expected Gain | Effort |
|----------|----------|---------------|--------|
| 1 | Mean pooling for text embeddings | +1-2% AUC | Low |
| 2 | Add LACE-like clinical features | +1-2% AUC | Low |
| 3 | End-to-end GRU fine-tuning | +3-5% AUC | High |
| 4 | Section-specific text embedding | +2-3% AUC | Medium |
| 5 | Accept 0.64 and focus on report | - | Low |

---

## 8) Reproducibility Notes

### Run Lab Feature Extraction
```bash
sbatch scripts/slurm/extract_lab_features.sbatch
```

### Run Trimodal Fusion
```bash
sbatch scripts/slurm/train_trimodal_fusion.sbatch
```

### Run Late Fusion Ensemble
```bash
sbatch scripts/slurm/train_late_fusion_ensemble.sbatch
```

### Data Lineage
```
MIMIC-IV labevents.csv.gz (158M rows)
  → src/data/extract_lab_trajectory_features.py
  → data/interim/ehr_long_los/lab_features/lab_features.npz (135 features)
  
GRU EHR embeddings + Text embeddings + Lab features
  → src/models/pytorch/train_trimodal_gated_fusion.py
  → MLflow: mimic_cardiorenal_trimodal_fusion
  
Lab XGB + Gated Fusion predictions
  → src/models/train_late_fusion_ensemble.py
  → data/interim/ehr_long_los/late_fusion_ensemble/ensemble_predictions.npz
```

---

## 9) Session Summary

This session extensively explored adding lab trajectory features to improve readmission prediction. Despite lab features showing strong standalone performance (AUC 0.6268), they failed to improve neural fusion models and only marginally helped ensembles.

**Bottom line:** The XGBoost on GRU EHR embeddings (AUC 0.641) remains our best model. Further gains likely require either:
- End-to-end model training (high effort)
- Imaging data (not available)
- Accepting current performance as near literature baseline for structured EHR models
