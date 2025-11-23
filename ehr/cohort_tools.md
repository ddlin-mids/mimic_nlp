# Cohort Tools Overview (`ehr/`)

This document describes the three main cohort-related scripts:

- `ehr/get_mimic_cohort.py`
- `ehr/cohort_eda.py`
- `ehr/compare_long_los_cohort.py`

and how they work together to define and inspect the readmission cohort.

---

## 1. `ehr/get_mimic_cohort.py` — Build the canonical cohort

**Purpose**

Construct the primary MIMIC-IV 30-day readmission cohort from raw PhysioNet tables and save it under `data/interim/`. This is the single source of truth that all downstream analysis and modeling should use.

**Key behavior**

- Delegates the heavy lifting to `src.data.build_cohort`:
  - Applies inclusion/exclusion criteria (adult stays, index admissions, 30-day window, etc.).
  - Computes `readmitted_within_window` and `readmission_gap_in_days`.
  - Attaches demographics, admission type, and discharge location.
  - Joins discharge and chest-radiology notes where present.
- Writes:
  - `data/interim/cohort.csv` — full cohort with identifiers, LOS, labels, and text.
  - `data/interim/cohort_splits.csv` — patient-level train/val/test splits.
  - `data/interim/cohort_stats.json` — summary counts and sanity checks.

**CLI usage**

From the repo root:

```bash
uv run python ehr/get_mimic_cohort.py \
  --mimic-root physionet.org/files/mimiciv/3.1 \
  --mimic-note-root physionet.org/files/mimic-iv-note/2.2 \
  --mimic-cxr-root physionet.org/files/mimic-cxr/2.1.0 \
  --output-dir data/interim \
  --min-los-days 1 \
  --readmit-window-days 30 \
  --val-size 0.1 \
  --test-size 0.2 \
  --seed 17
```

In most cases the defaults are sufficient; you only override paths if your PhysioNet mount points differ.

---

## 2. `ehr/cohort_eda.py` — Enrich cohort and build EDA tables

**Purpose**

Augment `data/interim/cohort.csv` with ICD-derived condition flags and generate a set of exploratory tables under `data/interim/readmit_analysis/` used in reports and feature engineering. It also materializes the long-stay cohort used by Daniel’s notebooks.

**Key steps**

1. **Input checks**
   - Verifies the presence of:
     - `data/interim/cohort.csv`
     - `physionet.org/files/mimiciv/3.1/hosp/diagnoses_icd.csv.gz`
     - `physionet.org/files/mimiciv/3.1/hosp/labevents.csv.gz`
     - `physionet.org/files/mimiciv/3.1/hosp/prescriptions.csv.gz`
   - Creates `data/interim/readmit_analysis/` if missing.

2. **Load cohort**
   - Reads `cohort.csv` and adds `los_segment` bins based on `length_of_stay_days`.

3. **Apply condition flags**
   - `_apply_condition_flags` scans `diagnoses_icd.csv.gz` in chunks to set:
     - `has_acute_kidney_injury`
     - `has_heart_failure`
     - `has_hyponatremia`
     - `has_posthemorrhagic_anemia`
     - `has_sepsis`
   - Derives summary flags:
     - `has_any_cardiorenal_sepsis`
     - `has_aki_and_hf`
     - `cardiorenal_sepsis_long` = (LOS ≥ 15) & `has_any_cardiorenal_sepsis`
     - `long_stay_no_cardiorenal` = (LOS ≥ 15) & ~`has_any_cardiorenal_sepsis`

4. **Long-stay cohort for downstream modeling**
   - `save_long_los_cohort`:
     - Canonical long-stay definition: filters to `length_of_stay_days >= 15`.
     - Uses `_apply_condition_flags` output to attach ICD-derived condition flags.
     - Defines `is_cardiorenal_long = cardiorenal_sepsis_long` (LOS ≥15 with any cardio-renal/sepsis diagnosis).
     - Selects 20 columns (IDs, LOS, labels, demographics, condition flags).
     - Writes `data/interim/readmit_analysis/long_los_cohort.csv` as the canonical long-LOS cohort for modeling.
   - **Legacy LOS≥14 cohort:**
     - On the first run after this refactor, if an older `long_los_cohort.csv` already exists with LOS≥14 semantics, it is copied to `data/interim/readmit_analysis/long_los_cohort_los14.csv` before being overwritten.
     - This legacy file preserves the partner-provided LOS≥14 long-stay cohort that Daniel used originally in `notebooks/notebooks_dc/02B_link_notes_to_admissions.ipynb`, for reference and comparison.

5. **EDA tables**

The script then generates multiple analysis tables, all prefixed with the current date, for example:

- LOS and cluster-level readmission rates:
  - `YYYY-MM-DD_readmit_los_readmit_rates.csv`
  - `YYYY-MM-DD_readmit_cluster_readmit_rates.csv`
  - `YYYY-MM-DD_readmit_condition_flags.csv`
- Discharge disposition and follow-up language:
  - `YYYY-MM-DD_discharge_disposition_readmit_rates.csv`
  - `YYYY-MM-DD_discharge_followup_language_readmit_rates.csv`
  - `YYYY-MM-DD_discharge_followup_language_by_los.csv`
- Readmission gap distributions:
  - `YYYY-MM-DD_readmit_positive_gaps_raw.csv`
  - `YYYY-MM-DD_readmit_gap_distribution_by_los.csv`
  - `YYYY-MM-DD_readmit_gap_distribution_cardiorenal.csv`
  - `YYYY-MM-DD_readmit_gap_distribution_aki_hf.csv`
- Labs near discharge (LOS ≥ 15):
  - `YYYY-MM-DD_lab_last48h_records.csv`
  - `YYYY-MM-DD_lab_last48h_summary.csv`
  - `YYYY-MM-DD_lab_last48h_aki.csv`
- Medications near discharge (LOS ≥ 15):
  - `YYYY-MM-DD_medications_near_discharge_summary.csv`
- Prior utilization and age bands:
  - `YYYY-MM-DD_prior_utilization_readmit_rates.csv`
  - `YYYY-MM-DD_prior_utilization_by_los.csv`
  - `YYYY-MM-DD_ageband_los_readmit_rates.csv`
- Social-text indicators:
  - `YYYY-MM-DD_social_text_indicators_summary.csv`
  - `YYYY-MM-DD_social_text_indicators_by_los.csv`

**CLI usage**

From the repo root:

```bash
# Full EDA (labs + meds; may be heavy)
uv run python ehr/cohort_eda.py

# Faster run, skip lab and medication pulls
uv run python ehr/cohort_eda.py --skip-labs --skip-meds
```

Either invocation will (re)generate:

- `data/interim/readmit_analysis/long_los_cohort.csv`
- The family of `YYYY-MM-DD_*.csv` analysis tables.

---

## 3. `ehr/compare_long_los_cohort.py` — Compare legacy vs regenerated long-LOS cohorts

**Purpose**

Provide a sanity check when refactoring or regenerating `long_los_cohort.csv`. It compares the legacy file against a newly generated one and reports:

- Shape and column differences.
- Admissions present only in one file.
- A sample of admissions where field values differ.

**Default behavior**

- Compares:
  - Original: `data/interim/readmit_analysis/long_los_cohort_los14.csv` (legacy LOS≥14 cohort, if present).
  - New: `data/interim/readmit_analysis/long_los_cohort.csv` (canonical LOS≥15 cohort).
- Uses `(subject_id, hadm_id)` as the admission key.
- Ignores the join keys themselves when checking value differences, focusing on the other columns.

**CLI usage**

From the repo root:

```bash
uv run python ehr/compare_long_los_cohort.py \
  --original-path data/interim/readmit_analysis/long_los_cohort_los14.csv \
  --new-path data/interim/readmit_analysis/long_los_cohort.csv \
  --max-diff-rows 20
```

You can override `--original-path` / `--new-path` to compare arbitrary cohorts with the same schema.

**Typical workflow**

1. Run `ehr/cohort_eda.py` to regenerate both:
   - `long_los_cohort.csv` (LOS≥15 canonical cohort).
   - `long_los_cohort_los14.csv` (legacy snapshot, if an older file existed).
2. Run `ehr/compare_long_los_cohort.py` to:
   - Check admissions alignment.
   - Inspect any value differences.
3. Once satisfied, update `save_long_los_cohort` to write directly to `long_los_cohort.csv` (as it does now).

---

## 4. Recommended usage order

For a clean rebuild of everything related to the readmission cohort:

1. **Build the cohort**

```bash
uv run python ehr/get_mimic_cohort.py --output-dir data/interim
```

2. **Generate EDA tables and the long-LOS cohort**

```bash
uv run python ehr/cohort_eda.py --skip-labs --skip-meds
```

3. *(Optional during refactors)* **Compare old vs new long-LOS cohorts**

```bash
uv run python ehr/compare_long_los_cohort.py
```

This sequence ensures that:

- `data/interim/cohort.csv` is up to date and reproducible.
- `data/interim/readmit_analysis/long_los_cohort.csv` is regenerated from the canonical cohort using consistent ICD semantics.
- Downstream notebooks and embedding scripts can rely on a stable, documented long-LOS cohort. 
