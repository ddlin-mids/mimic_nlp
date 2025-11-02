# End of Day Report — 2025-11-01

## 1) Report Header

- **Date & Agent:** 2025-11-01 (PT), codex-agent
- **Project / Subtask:** MIMIC-IV 30-Day Readmission / Risk Stratification Analytics
- **Starting Plan of the Day:**
  - Quantify 30-day readmission rates across LOS segments and cardio-renal/sepsis clusters.
  - Document findings in the daily report following `documents/report_guide/report_guide.md`.
  - Explore discharge disposition, follow-up documentation, time-to-readmit windows, lab instability, medication exposure, prior utilization, and social-text indicators.
- **Context Sources Used:** `data/interim/cohort.csv`, PhysioNet MIMIC-IV hosp tables (`diagnoses_icd.csv.gz`, `labevents.csv.gz`, `prescriptions.csv.gz`), saved outputs in `data/interim/readmit_analysis/`, repository guide `documents/report_guide/report_guide.md`.

## 2) What Was Done

### Data
- Enriched cohort with ICD-derived condition flags (AKI, HF, hyponatremia, post-haemorrhagic anaemia, sepsis) to define cardio-renal/sepsis long-stay cluster (`data/interim/readmit_analysis/2025-11-01_readmit_condition_flags.csv`).
- Recomputed LOS-segment readmission rates and cardio-renal/sepsis prevalence (`data/interim/readmit_analysis/2025-11-01_readmit_los_readmit_rates.csv`).
- Generated discharge disposition and follow-up-language tables tying documentation gaps to risk (`data/interim/readmit_analysis/2025-11-01_discharge_*`).
- Captured readmission lag distributions by LOS and condition cluster (`data/interim/readmit_analysis/2025-11-01_readmit_gap_distribution_*.csv`).
- Extracted end-of-stay creatinine/sodium labs within 48h of discharge for LOS≥15d (`data/interim/readmit_analysis/2025-11-01_lab_last48h_summary.csv`).
- Tallied medication exposures (loop diuretics, ACE/ARB, K-sparing/thiazide) near discharge (`data/interim/readmit_analysis/2025-11-01_medications_near_discharge_summary.csv`).
- Profiled prior utilization buckets and age bands vs readmit risk; derived simple social-text indicators from discharge notes (`data/interim/readmit_analysis/2025-11-01_*utilization*.csv`, `2025-11-01_social_text_indicators_*.csv`).

### Analysis / Interpretation
- Long LOS cardio-renal/sepsis cluster shows 25.7% readmit rate vs 23.6% in other long stays, confirming added risk beyond length-of-stay (`data/interim/readmit_analysis/2025-11-01_readmit_cluster_readmit_rates.csv`).
- Discharge to psychiatric facilities yields 45.9% readmission, highlighting intensive follow-up needs, while follow-up language in notes associates with ~1.8pp lower readmit probability (`data/interim/readmit_analysis/2025-11-01_discharge_*`).
- 33–37% of readmissions occur within 7 days across LOS bins; cardio-renal cluster front-loads early returns (34.3% ≤7d) reinforcing transitional-care urgency (`data/interim/readmit_analysis/2025-11-01_readmit_gap_distribution_*.csv`).
- Cardio-renal patients exit with elevated creatinine (median 1.1 mg/dL vs 0.7) and 37% loop-diuretic exposure, suggesting residual renal stress (`data/interim/readmit_analysis/2025-11-01_lab_last48h_summary.csv`, `2025-11-01_medications_near_discharge_summary.csv`).
- Prior utilization is a strong predictor (26.3% readmit for ≥3 prior admissions vs 12.1% for first-time stays) and compounds with long LOS and older age (`data/interim/readmit_analysis/2025-11-01_prior_utilization_readmit_rates.csv`).
- Simple NLP heuristics flag transportation barriers (20% readmit rate) and housing insecurity (16.4%) dominated by short stays, offering candidates for social risk stratification (`data/interim/readmit_analysis/2025-11-01_social_text_indicators_summary.csv`).

### Artifacts Produced
- `ehr/cohort_eda.py`: reproducible script for generating all exploratory tables.
- CSV tables under `data/interim/readmit_analysis/` (prefixed `2025-11-01_…`).
- Daily report package in `documents/reports/readmission-risk-stratification-2025-11-01/` (report + JSON + results table).

## 3) Results Snapshot

| Experiment ID | Data Slice | Model | Key Change | Metric 1 | Metric 2 | Notes |
| ------------- | ---------- | ----- | ---------- | -------- | -------- | ----- |
| E-LOS | LOS segments | aggregate | Stratified admissions by LOS bins | >=22d rate 26.8% | <=4d rate 13.4% | Risk rises monotonically with LOS; longest stays carry ~2x short-stay risk. |
| E-Cluster | LOS≥15d cardio-renal/sepsis dx | aggregate | Added ICD-driven cardio-renal/sepsis flag | Cardio-renal readmit 25.7% | Other long LOS 23.6% | High-acuity cluster adds +2.1pp absolute risk despite similar LOS. |
| E-Disposition | Discharge location (≥100 admissions) | aggregate | Grouped dispositions & computed rates | Psych facility 45.9% | Home health 18.7% | Psych transfers dominate; skilled nursing/home health remain elevated. |
| E-Followup | Discharge note follow-up language | aggregate | Keyword detection for follow-up documentation | No mention 18.7% | Mention present 16.9% | Documentation gap aligns with +1.8pp higher readmits across LOS bins. |
| E-Labs | LOS≥15d labs (48h before discharge) | aggregate | Pulled terminal creatinine/sodium | Cardio-renal Cr median 1.1 | Non-cardio long Cr median 0.7 | Renal burden persists near discharge among high-risk cluster. |
| E-PriorUtil | Prior admissions buckets | aggregate | Counted prior stays per subject | 3+ prior readmit 26.3% | 0 prior 12.1% | Utilization history doubles risk; median age ~66 in high bucket. |

*(See `documents/reports/readmission-risk-stratification-2025-11-01/results.csv` for CSV version.)*

## 4) Impact Assessment
- **Accuracy / Utility:** Quantified which clinical and social clusters carry the highest 30-day risk, enabling focused model targets (cardio-renal, psych discharge, high prior utilization).
- **Reliability / Robustness:** All stats derive from cohort-wide counts (no sampling); consistent uplift across LOS slices and condition flags reduces chance findings.
- **Decision-readiness:** Tables can feed cohort filters and feature engineering immediately; `ehr/cohort_eda.py` encodes reproducible extraction.
- **Risk & Ethics:** Social-text heuristics are coarse and may miss nuance; psychiatric cohorts may require additional privacy review before targeted interventions.

## 5) Deviations from Plan
- Attempted to render a combined LOS vs cardio-renal figure but `matplotlib` is absent in the runtime; proceeded without figures while preserving tabular outputs.

## 6) Open Questions & Unknowns
1. Do medication overlaps within 48h accurately capture administered doses, or should MAR data be incorporated? Need administration (emar/inputevents) cross-check.
2. Are transportation/housing keyword hits precise enough, or do we need clinician validation to avoid false positives? Sample chart review recommended.
3. Should prior utilization be normalized by observation window length (e.g., per-year admissions) to distinguish chronic frequent flyers? Requires anchor-year alignment.

## 7) Next Steps
1. **Immediate (tomorrow) — Owner: codex-agent**: Integrate loop-diuretic, creatinine slope, and follow-up language flags into exploratory modeling dataset; **Success:** engineered features land in staged Parquet with coverage ≥90% for LOS≥15d admissions.
2. **Short term (this week) — Owner: analytics-team**: Validate psych-facility discharge workflows with care management to design targeted triage rules; **Success:** documented intervention plan with responsible service lines.
3. **Nice-to-have — Owner: modeling-team**: Prototype dual-branch model separating short-stay transitional vs long-stay cardio-renal cohorts; **Success:** achieve ≥0.02 AUROC lift over single-model baseline on validation split.

## 8) Reproducibility Notes
- **Entry points:** `ehr/cohort_eda.py` (this script generates all tables). Optional: reuse prior notebook outputs under `notebooks/notebooks_dc/` for legacy comparison.
- **Minimal config:** Requires access to `data/interim/cohort.csv`, PhysioNet hosp tables (`diagnoses_icd.csv.gz`, `labevents.csv.gz`, `prescriptions.csv.gz`); run `uv run python ehr/cohort_eda.py` (pass `--skip-labs`/`--skip-meds` to shorten runtime).
- **Randomness:** None; all computations deterministic group-bys.
- **Data lineage:** MIMIC-IV v3.1 hosp tables → cohort filters (`cohort.csv`) → ICD flags & LOS segmentation → derived analysis tables in `data/interim/readmit_analysis/`.

## 9) Appendices
- None.

```json
{
  "date": "2025-11-01",
  "agents": ["codex-agent"],
  "project": "MIMIC-IV 30-Day Readmission / Risk Stratification Analytics",
  "starting_plan": [
    "Quantify readmission rates across LOS and cardio-renal/sepsis clusters",
    "Document findings per daily report guide",
    "Explore discharge disposition, follow-up documentation, time-to-readmit, labs, medications, prior utilization, social text indicators"
  ],
  "data": {
    "sources": [
      "data/interim/cohort.csv",
      "physionet.org/files/mimiciv/3.1/hosp/diagnoses_icd.csv.gz",
      "physionet.org/files/mimiciv/3.1/hosp/labevents.csv.gz",
      "physionet.org/files/mimiciv/3.1/hosp/prescriptions.csv.gz"
    ],
    "rows_after_filters": 184154,
    "transforms": [
      "ICD-derived cardio-renal/sepsis flags",
      "LOS segmentation",
      "Discharge disposition cleaning",
      "Follow-up keyword detection",
      "Readmission gap bucketing",
      "Lab pull (48h pre-discharge)",
      "Medication overlap within 48h",
      "Prior utilization counts",
      "Social-text heuristics"
    ],
    "quality": {
      "missing_pct": null,
      "label_balance": {"pos": 0.17086243035720103, "neg": 0.829137569642799}
    }
  },
  "experiments": [
    {
      "id": "E-LOS",
      "slice": "LOS segments",
      "model": "aggregate",
      "key_change": "Stratified admissions by LOS bins",
      "metrics": {"readmit_high": 0.268, "readmit_low": 0.134},
      "notes": "Risk rises with LOS"
    },
    {
      "id": "E-Cluster",
      "slice": "LOS>=15 cardio-renal/sepsis",
      "model": "aggregate",
      "key_change": "ICD-driven cardio-renal flag",
      "metrics": {"cardiorenal": 0.257, "non_cardio_long": 0.236},
      "notes": "+2.1pp uplift"
    },
    {
      "id": "E-Disposition",
      "slice": "Discharge location",
      "model": "aggregate",
      "key_change": "Disposition grouping",
      "metrics": {"psych_facility": 0.459, "home": 0.156},
      "notes": "Psych transfers highest risk"
    },
    {
      "id": "E-Followup",
      "slice": "Follow-up language",
      "model": "aggregate",
      "key_change": "Keyword heuristic",
      "metrics": {"no_followup": 0.187, "with_followup": 0.169},
      "notes": "Documentation correlates with lower risk"
    }
  ],
  "analysis": {
    "top_features": ["LOS segment", "cardio-renal/sepsis flags", "discharge disposition", "prior admissions"],
    "sanity_checks": [
      "Compared long LOS clusters vs peers",
      "Validated readmission gap totals vs 31,465 positives",
      "Confirmed lab coverage counts for LOS>=15d"
    ]
  },
  "artifacts": [
    {"name": "ehr/cohort_eda.py", "purpose": "Reproduce exploratory tables"},
    {"name": "data/interim/readmit_analysis/2025-11-01_readmit_los_readmit_rates.csv", "purpose": "LOS stratified rates"},
    {"name": "documents/reports/readmission-risk-stratification-2025-11-01/results.csv", "purpose": "Daily results snapshot"}
  ],
  "impact": {
    "utility": "Identified priority risk cohorts (cardio-renal, psych discharge, high prior utilization)",
    "robustness": "Full-cohort aggregations across multiple source tables",
    "decision_readiness": "Tables ready for feature engineering and intervention design",
    "risks": ["Social-text heuristics may misclassify", "Medication overlap approximates orders, not administrations"]
  },
  "deviations": ["Skipped plotting due to missing matplotlib"] ,
  "open_questions": [
    {"question": "Should MAR data refine medication exposure?", "evidence_needed": "Join emar/inputevents prior to discharge"},
    {"question": "Are transportation/housing keywords precise?", "evidence_needed": "Sample chart review"},
    {"question": "Normalize prior utilization by observation window?", "evidence_needed": "Anchor-year aware calculation"}
  ],
  "next_steps": [
    {"owner": "codex-agent", "action": "Integrate new features into modeling dataset", "success": ">=90% coverage in engineered feature file"},
    {"owner": "analytics-team", "action": "Validate psych discharge intervention plan", "success": "Documented triage workflow"},
    {"owner": "modeling-team", "action": "Prototype dual-branch readmission model", "success": "AUROC lift >=0.02"}
  ],
  "reproducibility": {
    "entry_points": ["ehr/cohort_eda.py"],
    "config": {"date_prefix": "2025-11-01", "skip_labs": false, "skip_meds": false},
    "lineage": "cohort.csv -> ICD/labs/meds joins -> readmit_analysis tables"
  }
}
```
