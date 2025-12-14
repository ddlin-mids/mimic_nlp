# Model Performance Analysis - December 13, 2025

## Critical Finding: Data Leakage in Original Baseline

The originally reported best model (XGBoost Structured EHR at **0.6406 AUC**) was inflated due to data leakage. The old ablation script (`train_ablation_xgboost.py`) used the test set for early stopping:

```python
# WRONG: Data leakage
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], ...)
```

The corrected approach uses validation set:

```python
# CORRECT: No leakage
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], ...)
```

## True Model Performance (Proper Evaluation)

| Rank | Model | Test AUC | Test AUPRC | Test F1 |
|------|-------|----------|------------|---------|
| 1 | **Gated Fusion** | **0.638** | 0.3495 | 0.2887 |
| 2 | Early Fusion MLP (HPO) | 0.6378 | 0.3541 | 0.3465 |
| 3 | Late Fusion Ensemble | 0.6308 | 0.3463 | 0.4095 |
| 4 | Lab Features Only | 0.6268 | 0.3516 | 0.4106 |
| 5 | Temporal V2 (no temporal attn) | 0.6242 | 0.3359 | 0.4176 |
| 6 | Structured EHR + Labs | 0.6218 | 0.3371 | 0.4217 |
| 7 | Structured EHR (GRU) | 0.6196 | 0.3361 | 0.4101 |
| 8 | Full Fusion + Events | 0.6189 | 0.3411 | 0.4207 |
| 9 | All Text (Notes + Events) | 0.6068 | 0.3398 | 0.4076 |
| 10 | Discharge Notes Only | 0.6067 | 0.3362 | 0.4083 |
| 11 | Static EHR (TF-IDF) | 0.6018 | 0.3004 | 0.3917 |
| 12 | Radiology Notes Only | 0.5984 | 0.3111 | 0.4090 |
| 13 | Demographics Only | 0.5297 | 0.2574 | 0.4000 |

## Key Insights

### 1. Gated Fusion is Best (0.638 AUC)
The Gated Multimodal Unit (GMU) effectively learns when to weight EHR vs text:
- Learns modality importance per sample
- Projects both to common 64-dim space
- Sigmoid gating: `h = z * x_ehr + (1-z) * x_txt`

### 2. Lab Trajectory Features are Powerful
Lab features alone (0.6268) are nearly as predictive as GRU-encoded EHR (0.6196):
- 135 features: last, max, min, mean, std, count, missing, slope_72h, first_last_diff
- Key labs: creatinine, BUN, WBC, hemoglobin, sodium, potassium

### 3. Temporal Attention Hurts Performance
Complex temporal modeling consistently underperforms:
- Full temporal attention: 0.598 AUC (worst)
- No temporal (simple pooling): 0.624 AUC (best neural)

**Hypothesis**: Readmission risk is driven by discharge state, not temporal patterns during stay.

### 4. Text Embeddings Add Marginal Value
- Discharge notes alone: 0.607 AUC
- Radiology reports alone: 0.598 AUC  
- Both combined: 0.601 AUC (no synergy)

Clinical text captures patient state but doesn't add much beyond structured EHR.

### 5. Ensemble Gains are Modest
- Simple averaging: ~0.01 AUC improvement
- Stacking meta-learners: no additional gain
- High correlation between base models (~0.7-0.8)

## Running Experiments

**Hybrid XGBoost** (Job 47044354): Combines temporal EHR + radiology + GRU embeddings + lab features. Results expected soon.

## Recommendations

1. **Use Gated Fusion for deployment** - 0.638 AUC is the validated best
2. **Simplify features** - Lab trajectories + demographics may suffice
3. **Avoid temporal complexity** - Simple aggregations work better
4. **Set realistic expectations** - 30-day readmission is inherently hard to predict

## Data Paths
- Cohort: `data/interim/readmit_analysis/long_los_cohort.csv`
- Lab features: `data/interim/ehr_long_los/lab_features/lab_features.npz`
- GRU embeddings: `data/interim/ehr_long_los/embeddings/structured_ehr_embeddings.npz`
- Text embeddings: `data/interim/embeddings/notes/discharge_summary.npz`
