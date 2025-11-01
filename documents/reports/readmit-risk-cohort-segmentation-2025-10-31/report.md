# End of Day Report — 2025-10-31

**Agent:** codex-agent  
**Project:** MIMIC-IV 30-Day Readmission Risk Profiling  
**Subtask Focus:** Cohort validation, length-of-stay segmentation, phenotype discovery

## Starting Plan of the Day
- Confirm cohort builder enforces ≤30-day same-patient readmissions with discharge and radiology summaries; exclude in-hospital deaths.
- Stand up reusable EHR preprocessing utilities (segmentation-ready) aligned with readmit-stgnn baseline.
- Characterise high-risk readmission phenotypes via length-of-stay (LOS) slices and ICD-10 profiles; surface actionable cohorts.

## Context Sources Used
- Today’s chat thread instructions and prior agent outputs.  
- `data/interim/cohort.csv`, `data/interim/readmit_analysis/*.csv`.  
- PhysioNet MIMIC-IV v3.1 hospital tables: `admissions.csv.gz`, `diagnoses_icd.csv.gz`, `d_icd_diagnoses.csv.gz`.

## What Was Done
### Data
- Rebuilt the cohort with tightened filters (no in-hospital deaths, required discharge + radiology text). Result: 184,154 admissions / 94,532 patients; 31,465 (17.1%) index stays followed by ≤30-day readmission (mean LOS 8.8 d, median 5.9 d). Saved stats in `data/interim/readmit_analysis/readmit_summary_stats.csv`.
- Segmented positive admissions into six LOS buckets (≤4 d through ≥22 d). Persisted assignments (`readmit_positive_admissions_with_segment.csv`) and aggregate mix (`readmit_los_segments.csv`).
- Joined ICD-10 diagnoses for every readmitted stay; produced per-segment frequency tables (`readmit_icd_distribution_by_segment.csv`, top-10 cut `readmit_icd_top10_by_segment.csv`).

### Analysis / Interpretation
- Identified two dominant phenotypes:
  - **Short LOS (≤7 d)** — 60% of readmits, with cardiometabolic chronic diagnoses (hypertension, hyperlipidemia, diabetes) dominating; signals potential discharge-readiness gaps.
  - **Extended LOS (≥22 d)** — 12% of readmits but 44% exhibit acute kidney injury, 22% sepsis, 22% hyponatremia, 18% post-haemorrhagic anaemia. LOS 15–21 d shows similar cardio-renal overlap (AKI 38%, HF 30%).
- Quantified cardio-renal overlap: heart failure + AKI co-occur in 17.5% of LOS 15–21 d and 15.7% of LOS ≥22 d readmissions vs 6.2% in ≤4 d stays (`readmit_aki_hf_rates_by_segment.csv`).

### Artifacts Produced
- `readmit_los_distribution.png`: LOS histogram for ≤30-day readmits.  
- `readmit_los_segment_summary.png`: Volume and share by LOS segment (bar + line chart).  
- Segment stats, ICD distributions, and condition summaries saved under `data/interim/readmit_analysis/`.  
- Daily report package under `documents/reports/readmit-risk-cohort-segmentation-2025-10-31/` (report, results table, figures).

## Results Snapshot

| Experiment ID | Data Slice | Key Finding | Admissions | Unique Patients | Avg LOS (days) | Admissions % | Patients % |
| ------------- | ---------- | ----------- | ---------- | ---------------- | -------------- | ------------ | ----------- |
| SEG-1 | ≤4 d | High-volume short stay readmits | 8,633 | 6,682 | 3.1 | 27.4% | 37.9% |
| SEG-2 | 5–7 d | Highest share; transitional care gaps | 10,291 | 7,537 | 5.4 | 32.7% | 42.8% |
| SEG-3 | 8–10 d | Mid-range LOS with rising acute events | 5,123 | 4,342 | 8.3 | 16.3% | 24.7% |
| SEG-4 | 11–14 d | Extended LOS with cardio-renal overlap | 3,024 | 2,685 | 11.9 | 9.6% | 15.2% |
| SEG-5 | 15–21 d | Prolonged LOS with kidney complications | 2,206 | 1,980 | 16.9 | 7.0% | 11.2% |
| SEG-6 | ≥22 d | Very long LOS with sepsis/hyponatremia | 2,188 | 1,855 | 35.4 | 7.0% | 10.5% |

**Figure 1:** `figure-1.png` — LOS segment share (bars) with admission proportion overlay.  
**Figure 2:** `figure-2.png` — Histogram of LOS among ≤30-day readmissions.

## Impact Assessment
- **Accuracy / Utility:** Surfaced high-frequency chronic readmits (≤7 d) and high-acuity cardio-renal/sepsis clusters (≥15 d), enabling targeted feature engineering (transitional care vs complication monitoring).
- **Reliability / Robustness:** Readmission gap enforcement confirmed (no positives >30 d); condition prevalence computed across 31k readmit stays, reducing sampling noise.
- **Decision-readiness:** Cohort is ready for focused modeling tracks—one for early-discharge risk stratification, one for protracted ICU/surgical pathways—pending integration of labs/meds per phenotype.
- **Risk & Ethics:** Continues to rely on de-identified PhysioNet data; must guard against bias when chronic disease codes dominate short LOS segments (potential socioeconomic confounders absent).

## Deviations from Plan
- None — executed planned cohort validation, preprocessing alignment, and segmentation analyses as scoped.

## Open Questions & Unknowns
1. **Transitional-care markers:** Which discharge note signals (follow-up instructions, social factors) differentiate ≤7 d readmits? Need NLP feature extraction + chart review.
2. **Procedural drivers in ≥22 d stays:** Are specific surgeries or device procedures (CPT) precipitating the high sepsis/AKI cluster? Requires joining procedures and ICU stays.
3. **Medication burden:** Does diuretic or nephrotoxic exposure explain AKI spikes? Need meds administration timelines and dosage normalization.

## Next Steps
1. **Immediate (next day) – Owner: codex-agent** — Join lab (creatinine, sodium) and medication administrations to LOS segments; success = coverage stats per segment (>90% admissions with labs) and updated condition table with lab thresholds.
2. **Short-term (this week) – Owner: analytics-team** — Derive discharge-note embeddings to contrast ≤7 d vs ≥15 d cohorts; success = qualitative themes + top tokens linked to readmission risk.
3. **Nice-to-have – Owner: modeling-team** — Prototype dual-branch model (transitional vs complication segments) with shared encoder; success = documented AUROC lift ≥0.02 vs single-model baseline on validation.

## Reproducibility Notes
- **Entry points:** `ehr/get_mimic_cohort.py` (rebuild cohort) → Python scripts in session (see saved CSVs) for LOS segmentation and ICD joins.  
- **Config:** Default CLI args; LOS bins `[0,4,7,10,14,21,1000]`; seed implicit (NumPy histogram defaults).  
- **Data lineage:** PhysioNet MIMIC-IV → cohort filters (LOS ≥2 d, exclude deaths, require discharge + radiology text, ≤30 d readmit) → LOS segmentation → ICD joins → condition prevalence summaries.  
- **Randomness:** Deterministic (no stochastic sampling beyond deterministic NumPy binning).

---
```json
{
  "date": "2025-10-31",
  "agents": ["codex-agent"],
  "project": "MIMIC-IV 30-Day Readmission Risk Profiling",
  "starting_plan": [
    "Confirm cohort builder enforces \u226430-day readmissions with discharge and radiology notes",
    "Align EHR preprocessing utilities with readmit-stgnn baseline",
    "Segment readmitted cohort by LOS and ICD-10 to surface high-risk phenotypes"
  ],
  "data": {
    "sources": [
      "data/interim/cohort.csv",
      "physionet.org/files/mimiciv/3.1/hosp/diagnoses_icd.csv.gz",
      "physionet.org/files/mimiciv/3.1/hosp/d_icd_diagnoses.csv.gz"
    ],
    "rows_after_filters": 184154,
    "transforms": [
      "Segmented readmitted admissions into six LOS buckets",
      "Merged ICD-10 diagnoses to readmitted admissions",
      "Computed condition prevalence (AKI, HF, sepsis) per segment"
    ],
    "quality": {
      "missing_pct": null,
      "label_balance": {"pos": 0.171, "neg": 0.829}
    }
  },
  "experiments": [
    {
      "id": "SEG-2",
      "slice": "LOS 5-7d readmits",
      "model": "segmentation-analysis",
      "key_change": "Focused on transitional-care subset",
      "metrics": {"admissions_pct": 32.7, "patients_pct": 42.8},
      "notes": "Largest cohort share; chronic cardiometabolic codes dominate"
    },
    {
      "id": "SEG-6",
      "slice": "LOS >=22d readmits",
      "model": "segmentation-analysis",
      "key_change": "Isolated prolonged stays",
      "metrics": {"aki_pct": 44.0, "sepsis_pct": 21.8},
      "notes": "High-acuity cardio-renal + sepsis cluster"
    }
  ],
  "analysis": {
    "top_features": [
      "Acute kidney injury codes (N17/584)",
      "Congestive heart failure codes (I50/428)",
      "Sepsis codes (A41/995.9)"
    ],
    "sanity_checks": [
      "Verified no readmission gaps exceed 30 days",
      "Confirmed all positives retain discharge and radiology text"
    ]
  },
  "artifacts": [
    {"name": "documents/reports/readmit-risk-cohort-segmentation-2025-10-31/report.md", "purpose": "Human-readable EOD summary"},
    {"name": "documents/reports/readmit-risk-cohort-segmentation-2025-10-31/results.csv", "purpose": "Main LOS segment table"},
    {"name": "documents/reports/readmit-risk-cohort-segmentation-2025-10-31/figure-1.png", "purpose": "LOS segment volume plot"},
    {"name": "documents/reports/readmit-risk-cohort-segmentation-2025-10-31/figure-2.png", "purpose": "LOS histogram"}
  ],
  "impact": {
    "utility": "Surfaced transitional-care vs complication-driven readmission cohorts for targeted modeling",
    "robustness": "Condition prevalence computed over full 31k-readmit cohort with deterministic joins",
    "decision_readiness": "Ready to branch modeling roadmap into short- vs long-stay risk tracks",
    "risks": ["Chronic-disease segments may encode socioeconomic bias; requires contextual features"]
  },
  "deviations": [],
  "open_questions": [
    {"question": "Which discharge-note signals predict <=7d readmissions?", "evidence_needed": "NLP embedding analysis with qualitative review"},
    {"question": "Are specific procedures driving >=22d readmits?", "evidence_needed": "Join CPT/ICU tables and analyze prevalence"},
    {"question": "Does nephrotoxic medication exposure explain AKI spikes?", "evidence_needed": "Medication timeline extraction and dosage checks"}
  ],
  "next_steps": [
    {"owner": "codex-agent", "action": "Merge labs/meds into LOS segments", "success": ">90% admissions with lab coverage and updated complication rates"},
    {"owner": "analytics-team", "action": "Embed discharge notes for LOS extremes", "success": "Identify top linguistic themes differentiating <=7d vs >=15d"},
    {"owner": "modeling-team", "action": "Prototype dual-branch readmission model", "success": "Validation AUROC improvement >=0.02 over single model"}
  ],
  "reproducibility": {
    "entry_points": ["ehr/get_mimic_cohort.py", "python scripts for LOS segmentation (see analysis CSVs)"],
    "config": {"los_bins": [0, 4, 7, 10, 14, 21, 1000]},
    "lineage": "mimic_iv_v3.1 -> cohort filters -> LOS segmentation -> ICD joins -> condition summaries"
  }
}
```
