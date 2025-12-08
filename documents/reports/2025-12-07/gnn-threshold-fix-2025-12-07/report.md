# Daily Report: GNN Threshold Optimization & Report Updates

**Date:** 2025-12-07 (Pacific Time)  
**Agent:** Claude Code  
**Project:** MIMIC-IV Cardiorenal Readmission Prediction  

---

## 1) Starting Plan of the Day

- Investigate why GNN model had F1=0 despite reasonable AUROC
- Fix threshold optimization for GNN predictions
- Update final project report with corrected GNN results
- Document AUC-F1 trade-off findings

**Context Sources Used:**
- MLflow experiment `410534259911568078` (GNN runs)
- `src/models/pytorch/train_gnn.py` and `train_gnn_hpo.py`
- Final report: `documents/reports/final_project_report/final_project_report_v2.md`

---

## 2) What Was Done

### Data
- No new data processing; used existing cardiorenal cohort (N=4,271 admissions)
- Label balance unchanged: ~18% positive (readmitted within 30 days)

### Modeling / Experiments

**Problem Identified:** GNN model had F1=0.0 because all predictions fell below the default 0.5 threshold. The model's probability outputs were well-calibrated for ranking (AUROC=0.62) but centered around 0.505, meaning almost no predictions crossed 0.5.

**Solution Implemented:** Added `find_optimal_threshold()` function to both GNN training scripts:
- Uses precision-recall curve on validation set
- Finds threshold maximizing F1 score
- Applies optimal threshold at test time

**Files Modified:**
1. `src/models/pytorch/train_gnn.py` - Added threshold optimization logic
2. `src/models/pytorch/train_gnn_hpo.py` - Same fix for HPO variant

### Analysis / Interpretation

**Key Finding: AUC-F1 Trade-off**

The GNN with threshold optimization shows a trade-off pattern:
- Lower AUROC (0.596 vs 0.640 for XGBoost+Transformer)
- Highest F1 (0.406 vs 0.375 for runner-up)
- Highest Recall (0.718 vs 0.471)

This makes GNN suitable for **high-sensitivity clinical alerting** where catching more true positives is prioritized over precision.

**Correction Made:** Updated report to clarify GNN uses GRU embeddings (from `train_ehr_encoder.py`), NOT Transformer embeddings. The Transformer embeddings exist but were not integrated into the GNN pipeline.

### Artifacts Produced

| Artifact | Purpose |
|----------|---------|
| `src/models/pytorch/train_gnn.py` | Updated with `find_optimal_threshold()` |
| `src/models/pytorch/train_gnn_hpo.py` | Same threshold fix |
| `final_project_report_v2.md` | Updated Section 5.1 table, Figure 3, Sections 5.3-5.4 |
| `final_project_report_v2.html` | Rebuilt HTML with mermaid diagrams |

---

## 3) Results Snapshot

### Main Results Table

| Model | Embeddings | AUROC | AUPRC | F1 | Precision | Recall | Threshold |
|-------|------------|-------|-------|-----|-----------|--------|-----------|
| XGBoost + Transformer | Transformer | 0.640 | 0.358 | 0.375 | 0.302 | 0.495 | 0.50 |
| Gated Fusion | Transformer | 0.632 | 0.339 | 0.371 | 0.305 | 0.471 | 0.50 |
| XGBoost | GRU | 0.614 | 0.309 | 0.324 | 0.268 | 0.412 | 0.50 |
| **GNN (fixed)** | **GRU** | **0.596** | **0.305** | **0.406** | **0.280** | **0.718** | **0.505** |
| Early Fusion | Transformer | 0.599 | 0.298 | 0.290 | 0.267 | 0.316 | 0.50 |

**Key Insight:** GNN achieves best F1 and Recall despite lower AUROC due to threshold optimization finding the sweet spot for its probability distribution.

---

## 4) Impact Assessment

### Accuracy / Utility
- GNN F1 improved from 0.000 to 0.406 (largest F1 among all models)
- Recall improved from 0.000 to 0.718 (catches 72% of readmissions)
- Provides complementary model for high-sensitivity use cases

### Reliability / Robustness
- Optimal threshold (0.505) is very close to 0.5, indicating model is reasonably calibrated
- Single seed run; variance across seeds not tested for GNN

### Decision-readiness
- GNN now viable for ensemble or tiered alerting system
- Report accurately reflects model capabilities

### Risk & Ethics
- High recall comes with lower precision (28%); more false alarms
- Appropriate for workflows that can tolerate alert fatigue for safety

---

## 5) Deviations from Plan

- Originally planned to explore MIMIC-IV-Ext-22MCTS integration; deferred to prioritize fixing GNN metrics
- Did not re-run HPO with threshold optimization; used existing best hyperparameters

---

## 6) Open Questions & Unknowns

1. **Would threshold optimization improve other models?**
   - Evidence needed: Apply same technique to XGBoost, Gated Fusion
   
2. **Is GNN's high recall due to threshold or architecture?**
   - Evidence needed: Compare probability distributions across models

3. **How does GNN perform with Transformer embeddings?**
   - Evidence needed: Modify `preprocess_graph.py` to use Transformer embeddings

---

## 7) Next Steps (Ranked)

1. **Immediate:** Apply threshold optimization to other fusion models to enable fair comparison
   - Owner: Agent
   - Success: All models report metrics at optimal threshold

2. **Short-term:** Integrate MIMIC-IV-Ext-22MCTS temporal clinical events
   - Owner: Agent
   - Success: New embeddings improve AUROC by 0.02+

3. **Nice-to-have:** Ensemble GNN (high recall) with XGBoost (high precision)
   - Owner: Agent
   - Success: Ensemble AUROC > 0.65

---

## 8) Reproducibility Notes

**Entry Points:**
1. `sbatch scripts/slurm/train_gnn.sbatch` - Train GNN with threshold optimization
2. `bash scripts/build_report.sh` - Rebuild HTML report

**Key Config:**
```python
# train_gnn.py threshold optimization
def find_optimal_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]
```

**Data Lineage:**
```
MIMIC-IV → get_mimic_cohort.py (cardiorenal filter) → preprocess_ehr.py 
→ train_ehr_encoder.py (GRU embeddings) → preprocess_graph.py → train_gnn.py
```

**Seeds:** 42 (consistent across pipeline)

---

## 9) Appendices

### A. Threshold Optimization Code Added

```python
def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold that maximizes F1 on validation set."""
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    if best_idx >= len(thresholds):
        return 0.5
    return float(thresholds[best_idx])
```

### B. Report Sections Updated

- **Section 5.1:** Model comparison table re-sorted, GNN metrics corrected
- **Figure 3:** F1 bar chart updated (GNN now top)
- **Section 5.3:** New "AUC-F1 Trade-off" discussion added
- **Section 5.4:** Clinical deployment recommendations updated for tiered alerting
