# Data & Cohort Blueprint

## Data Sources
- **MIMIC-IV v3.1 (hosp, icu modules)** — structured admissions, labs, medications, procedures; accessed under PhysioNet credentialed DUA (read-only mount: `physionet.org/files/mimiciv/3.1/`).
- **MIMIC-IV-Note v2.2** — discharge summaries, radiology reports, progress notes; primary text corpus for embeddings.
- **Ancillary assets** — MIMIC-IV-ED v2.2 for emergency context, MIMIC-IV-Ext-22MCTS for event timing if needed, MIMIC-CXR reports (text only).
- **Local intermediates** — `data/interim/cohort.csv` with cohort filters; derived artifacts stored in `data/interim/readmit_analysis/` (dates embedded in filenames).

## Cohort Definition (Current)
1. Adult inpatient admissions with valid discharge and radiology documentation.
2. Exclude in-hospital deaths, hospice discharges, newborn admissions per `cohort.yaml` spec.
3. Label positive if next admission for same subject occurs ≤30 days after discharge; enforce patient-wise splitting to prevent leakage.
4. Derived statistics tracked in `cohort_stats.json` (LOS mean 8.76 d, median 5.9 d; 17.1% readmit rate, 31 465 positives / 184 154 admissions).

## Dual Focus Groups
| Group | Definition | Signal | Rationale |
|-------|------------|--------|-----------|
| **Transitional short stays** | LOS ≤7 d (124 715 admissions, 18 924 readmits, 15.2% rate) | Chronic cardiometabolic diagnoses dominate; follow-up documentation absence pushes risk to 18.7% | Largest patient volume; improvement hinges on discharge planning and social determinants. |
| **Cardio-renal/sepsis long stays** | LOS ≥15 d with AKI, HF, hyponatremia, sepsis, or post-haemorrhagic anaemia (10 204 admissions, 2 618 readmits, 25.7% rate) | Creatinine median 1.1 mg/dL in final 48 h, 37% loop-diuretic exposure near discharge | Small but high-risk cohort requiring renal optimization and infection management. |

Additional slices to monitor:
- **Psychiatric facility discharges:** 45.9% readmission, indicates coordination priority.
- **Home health vs routine home:** 18.7% vs 15.6%, reflecting transitional-care needs.
- **Transportation / housing barriers (text heuristics):** 724 admissions with transport issues (20% readmit), 1 780 with housing insecurity keywords (16.4%).

## Data Management Practices
- Store derived CSVs with run date prefixes (e.g., `2025-11-01_*`) for auditability.
- Rebuild tables via `uv run python ehr/cohort_eda.py` (supports `--skip-labs`, `--skip-meds` to manage runtime).
- Keep raw PhysioNet paths read-only; never persist PHI or large exports in git.
- Document new pulls or schema adjustments in `documents/reports/{date}/…/report.md` and update checksums in `documents/` when mirroring data.
