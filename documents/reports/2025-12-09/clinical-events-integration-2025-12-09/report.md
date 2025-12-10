# Daily Report: MIMIC-IV-Ext-22MCTS Clinical Events Integration

**Date:** 2025-12-09 (Pacific Time)  
**Agent:** Claude Code  
**Project:** NLP-Driven 30-Day Readmission Prediction on MIMIC-IV  
**Subtask:** Evaluate MIMIC-IV-Ext-22MCTS temporal clinical events as additional modality

## Starting Plan of the Day

1. Identify low-hanging fruit from recent reports (2025-12-06, 2025-12-07) and final project report
2. Investigate integration of MIMIC-IV-Ext-22MCTS dataset (22.6M timestamped clinical events)
3. Determine approach: qLoRA fine-tuning vs. static BioClinicalBERT embeddings
4. Create/update HPC pipeline scripts for clinical events processing

## Context Sources Used

- `documents/reports/final_project_report/final_project_report_v2.md`
- `documents/reports/2025-12-06/` and `2025-12-07/` session reports
- `physionet.org/files/mimic-iv-ext-22mcts/1.0.0/` dataset
- PhysioNet documentation and arXiv paper (arXiv:2505.00827)

---

## What Was Done

### Data

- **Reviewed MIMIC-IV-Ext-22MCTS structure:**
  - 22,588,586 total clinical events across 267,284 admissions
  - 3,265,484 unique event strings (high cardinality due to LLM extraction noise)
  - Events are timestamped relative to discharge (hours); Time_bin provides 10-bin discretization
  - Top events: "hypertension" (457k), "hypotension" (290k), "abdominal pain" (215k), "fever" (181k)

- **Discovered existing artifacts (already generated Dec 8):**
  - `data/interim/clinical_events/clinical_events_filtered.csv` (43.7 MB) - cohort-filtered events
  - `data/interim/clinical_events/clinical_events_vocab.csv` - top 2,000 events with IDs
  - `data/interim/embeddings/notes/clinical_events.npz` (17.3 MB) - admission-level embeddings

- **Embedding generation approach (confirmed):**
  - Top 10,000 most frequent events embedded via BioClinicalBERT (768-dim)
  - Mean-pooling aggregation per admission
  - Final shape: (12,177 admissions, 768 dims) filtered to cardiorenal cohort

### Modeling / Experiments

**Ablation study already completed (experiment ID: 837588739691436548)**

| Experiment | Data Slice | Model | Key Change | Test AUC | Test F1 |
|------------|------------|-------|------------|----------|---------|
| Structured EHR (Transformer) | Cardiorenal | XGBoost | Baseline | **0.623** | 0.403 |
| Full Fusion (Transformer) | Cardiorenal | XGBoost | +Notes | 0.609 | 0.396 |
| Full Fusion + Events | Cardiorenal | XGBoost | +22MCTS | 0.608 | 0.394 |
| All Text (Notes + Events) | Cardiorenal | XGBoost | Notes+Events only | 0.607 | 0.408 |
| Discharge Notes | Cardiorenal | XGBoost | Notes only | 0.607 | 0.408 |
| **Clinical Events** | Cardiorenal | XGBoost | 22MCTS only | **0.553** | 0.406 |
| Baseline (Demo) | Cardiorenal | XGBoost | Demographics | 0.530 | 0.400 |

### Analysis / Interpretation

**Key Finding: Clinical Events underperform expectations**

1. **Clinical Events alone (AUC 0.553)** perform worse than discharge notes (0.607) and barely above demographics baseline (0.530)

2. **Adding events to Full Fusion provides no lift** (0.608 vs 0.609 without events)

3. **Likely causes:**
   - LLM extraction noise: LLaMA-3.1-8B generated events with hallucinations (per dataset README)
   - Extreme vocabulary sparsity: 3.2M unique strings from 22M rows = 6.9x compression only
   - Mean-pooling loses temporal signal: aggregating all events removes time-awareness
   - Redundancy: event content overlaps with structured ICD codes and note embeddings

4. **Structured EHR Transformer remains best** (0.623 AUC) - daily labs/meds/ICD provide cleaner signal

### Artifacts Produced

| Artifact | Purpose |
|----------|---------|
| `ehr/extract_unique_clinical_events.py` | Enhanced script with cohort filtering & daily sequence generation |
| `scripts/slurm/extract_unique_events.sbatch` | Updated SLURM job with cohort integration |
| `data/interim/clinical_events/` | Filtered events, vocabulary, daily sequences |
| `data/interim/embeddings/notes/clinical_events.npz` | Admission-level 768-dim embeddings |

---

## Results Snapshot

### Main Results Table

| Experiment | Test AUC | Test AUPRC | Test F1 | Delta vs Baseline |
|------------|----------|------------|---------|-------------------|
| Structured EHR (Transformer) | 0.623 | - | 0.403 | +0.093 |
| Full Fusion + Events | 0.608 | - | 0.394 | +0.078 |
| Clinical Events Only | 0.553 | 0.274 | 0.406 | +0.023 |
| Baseline (Demographics) | 0.530 | - | 0.400 | - |

**Conclusion:** MIMIC-IV-Ext-22MCTS clinical events do not improve model performance in current form.

---

## Impact Assessment

### Accuracy / Utility
- Clinical events modality adds **no incremental value** over existing structured EHR + notes
- The +0.023 AUC over demographics baseline is negligible compared to structured EHR (+0.093)

### Reliability / Robustness
- High variance in event extraction (LLM-generated) introduces noise
- Mean-pooling aggregation may mask useful temporal patterns

### Decision-readiness
- **Not recommended** for production integration in current form
- Alternative approaches (temporal attention, sequence modeling) may unlock value

### Risk & Ethics
- LLM extraction hallucinations noted in dataset documentation
- No additional PHI exposure (events derived from existing notes)

---

## Deviations from Plan

1. **Discovered work already completed:** Clinical events embeddings and ablation were run on Dec 8; today's session was primarily verification and documentation
2. **Cancelled redundant job:** Extraction job 47018753 was submitted before discovering existing artifacts; cancelled after 10 min
3. **Shifted focus to analysis:** Instead of new experiments, documented findings and assessed integration value

---

## Open Questions & Unknowns

1. **Would temporal attention over event sequences improve results?**
   - Evidence needed: Implement attention mechanism over daily event counts (similar to Transformer EHR encoder)
   - Hypothesis: Time-aware aggregation may recover signal lost by mean-pooling

2. **Is vocabulary filtering too aggressive?**
   - Current: Top 10k events from 3.2M unique strings
   - Evidence needed: Ablate at 5k, 20k, 50k vocabulary sizes

3. **Does the LLM extraction quality vary by event type?**
   - Evidence needed: Manual review of top-50 events for accuracy
   - Some events (e.g., medications) may be more reliable than symptoms

---

## Next Steps (Ranked, Time-boxed)

### Immediate (Tomorrow)
1. **Apply threshold optimization to all models** (Owner: David)
   - Expected: +0.10-0.20 F1 lift (based on GNN threshold fix)
   - Success: F1 > 0.30 for XGBoost and Gated Fusion

### Short-term (This Week)
2. **Ensemble GNN + XGBoost** (Owner: David)
   - Expected: Break 0.64 AUC ceiling
   - Success: AUC > 0.65, Recall > 0.60

### Nice-to-have
3. **Temporal attention over clinical events** (Owner: TBD)
   - Feed daily event counts into Transformer encoder (similar to labs/meds)
   - Success: Clinical Events AUC > 0.58 (surpass discharge notes)

---

## Reproducibility Notes

### Entry Points (in order)
1. `ehr/extract_unique_clinical_events.py` - Extract and filter events
2. `ehr/embed_clinical_events.py` - Generate BioClinicalBERT embeddings
3. `src/models/train_fusion_xgboost.py --embedding_type Transformer` - Run ablation

### Minimal Config
```bash
# Extraction
uv run python ehr/extract_unique_clinical_events.py \
  --cohort-path data/interim/readmit_analysis/long_los_cohort.csv \
  --output_dir data/interim/clinical_events \
  --min-freq 10 --max-vocab-size 2000

# Embedding
uv run python ehr/embed_clinical_events.py \
  --cohort-path data/interim/readmit_analysis/long_los_cohort.csv \
  --top-k 10000
```

### Data Lineage
```
MIMIC-IV-Ext-22MCTS (22.6M events)
  → Filter to cardiorenal cohort (long_los_cohort.csv)
  → Temporal leakage prevention (Time <= 0, before discharge)
  → Top 10k events by frequency
  → BioClinicalBERT embedding (768-dim)
  → Mean-pool per admission
  → clinical_events.npz (12,177 x 768)
```

---

## Appendices

### A. MIMIC-IV-Ext-22MCTS Dataset Summary

| Metric | Value |
|--------|-------|
| Total events | 22,588,586 |
| Unique events | 3,265,484 |
| Unique admissions | 267,284 |
| Cohort-filtered events | ~1.5M (est.) |
| Cohort admissions with events | 12,177 |
| Top event | "hypertension" (456,862 occurrences) |

### B. Event Categories (Top 10)

1. hypertension (457k)
2. hypotension (290k)
3. abdominal pain (215k)
4. fever (181k)
5. chest pain (173k)
6. nausea (est. ~150k)
7. vomiting (est. ~140k)
8. shortness of breath (est. ~130k)
9. diabetes (est. ~120k)
10. anemia (est. ~100k)

---

```json
{
  "date": "2025-12-09",
  "agents": ["claude-code"],
  "project": "mimic-nlp-readmission",
  "starting_plan": [
    "Identify low-hanging fruit from recent reports",
    "Investigate MIMIC-IV-Ext-22MCTS integration",
    "Determine qLoRA vs embedding approach",
    "Create HPC pipeline scripts"
  ],
  "data": {
    "sources": ["MIMIC-IV-Ext-22MCTS", "long_los_cohort.csv"],
    "rows_after_filters": 12177,
    "transforms": ["cohort_filter", "temporal_leakage_prevention", "top_10k_vocab", "mean_pool"],
    "quality": {
      "missing_pct": null,
      "label_balance": {"pos": 0.245, "neg": 0.755}
    }
  },
  "experiments": [
    {
      "id": "clinical_events_only",
      "slice": "cardiorenal",
      "model": "xgboost",
      "key_change": "22MCTS embeddings only",
      "metrics": {"auroc": 0.553, "auprc": 0.274, "f1": 0.406},
      "notes": "Underperforms discharge notes"
    },
    {
      "id": "full_fusion_events",
      "slice": "cardiorenal",
      "model": "xgboost",
      "key_change": "Add 22MCTS to full fusion",
      "metrics": {"auroc": 0.608, "f1": 0.394},
      "notes": "No improvement over fusion without events"
    }
  ],
  "analysis": {
    "top_features": ["structured_ehr", "discharge_notes"],
    "sanity_checks": ["temporal_leakage_prevented", "cohort_filter_applied"]
  },
  "artifacts": [
    {"name": "clinical_events.npz", "purpose": "Admission-level embeddings (12177x768)"},
    {"name": "clinical_events_vocab.csv", "purpose": "Top 2000 events with IDs"},
    {"name": "extract_unique_clinical_events.py", "purpose": "Enhanced extraction script"}
  ],
  "impact": {
    "utility": "No incremental value over existing modalities",
    "robustness": "High noise from LLM extraction",
    "decision_readiness": "Not recommended for production",
    "risks": ["LLM hallucinations in event extraction"]
  },
  "deviations": [
    "Discovered existing artifacts from Dec 8",
    "Cancelled redundant extraction job",
    "Shifted to analysis and documentation"
  ],
  "open_questions": [
    {
      "question": "Would temporal attention over events improve AUC?",
      "evidence_needed": "Implement attention mechanism, compare to mean-pool"
    },
    {
      "question": "Is top-10k vocabulary optimal?",
      "evidence_needed": "Ablate at 5k, 20k, 50k sizes"
    }
  ],
  "next_steps": [
    {
      "owner": "david",
      "action": "Threshold optimization for all models",
      "success": "F1 > 0.30 for XGBoost"
    },
    {
      "owner": "david",
      "action": "Ensemble GNN + XGBoost",
      "success": "AUC > 0.65"
    }
  ],
  "reproducibility": {
    "entry_points": [
      "ehr/extract_unique_clinical_events.py",
      "ehr/embed_clinical_events.py",
      "src/models/train_fusion_xgboost.py"
    ],
    "config": {
      "seed": 42,
      "data_alias": "long_los_cohort",
      "top_k_events": 10000,
      "embedding_dim": 768
    },
    "lineage": "22MCTS → cohort_filter → temporal_filter → top_10k → BioClinicalBERT → mean_pool"
  }
}
```
