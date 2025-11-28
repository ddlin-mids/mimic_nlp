# EHR Pipeline Fix and Features Report

**Date:** 2025-11-28  
**Project:** MIMIC-IV Readmission / EHR Embedding  
**Agents:** Claude Code, Gemini CLI  

## Executive Summary
This report documents the refinement of the EHR preprocessing pipeline for the long-LOS cohort. Key updates include robust patient-level splitting to prevent data leakage and the inclusion of critical cohort-specific features (cardiorenal status, admission type) in the downstream embedding artifacts. The cohort has been successfully stratified and validated.

## Data & Cohort
- **Source:** MIMIC-IV (hosp/admissions, diagnoses, labs, meds) + MIMIC-IV-Note
- **Final Cohort Size:** 15,659 admissions
- **Splitting Strategy:** 80/10/10 Patient-Level Stratified Split
- **Stratification Keys:** Readmission Status, Cardiorenal Status, Gender, Age (Median Split)

### Stratification Statistics
The following table confirms the balance of key covariates across the generated splits:

| Split | N | Readmit Rate | Cardiorenal % | Gender (F) % | Median Age |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 12,522 | 24.9% | 65.2% | 45.0% | 63.0 |
| **Val** | 1,568 | 25.6% | 64.2% | 45.3% | 64.0 |
| **Test** | 1,569 | 24.6% | 65.8% | 44.8% | 64.0 |

## Key Changes
1.  **Robust Splitting:** Implemented strict patient-level splitting in `ehr/cohort_eda.py` to ensure zero patient overlap between train/val/test sets.
2.  **Feature Preservation:** Updated `ehr/preprocess_ehr.py` to explicitly preserve and encode cohort-specific features like `is_cardiorenal_long`, `admission_type`, and `discharge_location` which were previously being dropped.
3.  **Composite Stratification:** Verified that stratification logic correctly balances multiple competing demographic and clinical factors.

## Experiments & Artifacts
- **Job ID:** Job-46743661 (Ready to Submit)
- **Goal:** Generate `cat_embedding` (for GRU/RNNs) and `one_hot` (for Baselines) representations.
- **Key Artifacts:**
    - `data/interim/readmit_analysis/long_los_cohort.csv`: The canonical cohort file with valid splits.
    - `ehr/cohort_eda.py`: The source of truth for cohort generation and splitting.
    - `ehr/preprocess_ehr.py`: The preprocessing logic for feature extraction.

## Next Steps
1.  Submit the preprocessing job (`scripts/slurm/preprocess_ehr_long_los.sbatch`).
2.  Train the GRU-based EHR encoder using the generated artifacts.
3.  Generate baseline embeddings (SVD/PCA) for comparison.
