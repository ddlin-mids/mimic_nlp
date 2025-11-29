# EHR Pipeline Fix and Features Report

**Date:** 2025-11-28  
**Project:** MIMIC-IV Readmission / EHR Embedding  
**Agents:** Claude Code, Gemini CLI  

## Executive Summary
This report documents the refinement of the EHR preprocessing pipeline for the long-LOS cohort. Key updates include robust patient-level splitting to prevent data leakage and the inclusion of critical cohort-specific features (cardiorenal status, admission type) in the downstream embedding artifacts. The cohort has been successfully stratified and validated. The preprocessing pipeline has successfully completed.

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

## Technical Challenges & Solutions (OOM Fix)
**Issue:** Job 46745045 failed with an OOM error after successfully generating the `cat_embedding` artifacts but before completing the `one_hot` encoding. The sequential execution of both memory-intensive tasks in a single process caused fragmentation exceeding 128GB.

**Resolution:**
1.  **Idempotency:** Updated `ehr/preprocess_ehr.py` to check if output files exist before starting processing.
2.  **Job Separation:** Submitted Recovery Job 46745623 with a new flag `--skip-cat-embedding`. This forces the script to bypass the already completed steps and dedicate all resources to the `one_hot` generation.
3.  **Status:** **Success.** The pipeline is now complete.

## Strict Splitting Enforcement
**Issue:** Both the neural encoder (`train_ehr_encoder.py`) and the static baseline (`encode_ehr_static_boe.py`) were previously performing operations (random splitting or global fitting) that ignored the rigorous pre-defined cohort splits, risking data leakage.

**Resolution:**
1.  **Temporal Encoder (`ehr/train_ehr_encoder.py`)**: Modified to strictly read the `split` column. Test set is now completely excluded from the training loop.
2.  **Static Baseline (`ehr/encode_ehr_static_boe.py`)**: Modified to `fit()` the TF-IDF/SVD pipeline **only** on the training set, then `transform()` the remaining data.

## Artifact Definitions
Two types of artifacts are being generated for different modeling paradigms:

1.  **`ehr_preprocessed_all_cat_embedding.pkl` (Flat/Tabular)**
    *   **Structure:** Pandas DataFrame (Rows = Days).
    *   **Use Case:** Baseline models (XGBoost, Logistic Regression) and EDA. Treat days as independent samples.

2.  **`ehr_preprocessed_seq_by_day_cat_embedding.pkl` (Sequential/Nested)**
    *   **Structure:** Dictionary of Numpy Arrays (Key = Admission ID).
    *   **Use Case:** Deep Learning Sequence Models (GRU, LSTM, Transformers). Preserves strict temporal order per patient.

## Pipeline Architecture
The EHR processing pipeline consists of three distinct stages designed to produce complementary embedding types:

1.  **Preprocessing (`ehr/preprocess_ehr.py`)**
    *   **Role:** Foundation. Cleans raw MIMIC-IV data (labs, meds, ICDs).
    *   **Outputs:**
        *   *Flat Data:* For static baselines (one-hot).
        *   *Sequential Data:* For temporal deep learning (cat_embedding).

2.  **Static Baseline (`ehr/encode_ehr_static_boe.py`)**
    *   **Method:** Bag-of-Words (BoW) → TF-IDF → Truncated SVD.
    *   **Logic:** Aggregates all events in an admission into a single vector, ignoring time.
    *   **Output:** `static_embeddings.npz`. Good for non-sequential models (XGBoost).

3.  **Temporal Encoder (`ehr/train_ehr_encoder.py`)**
    *   **Method:** GRU/LSTM Neural Network.
    *   **Logic:** Processes daily feature vectors sequentially to capture patient trajectory and state evolution. Trained via supervised learning on the readmission target.
    *   **Output:** `structured_ehr_embeddings.npz`. Captures temporal dynamics.

## Experiments & Artifacts
- **Job ID:** Job-46745045 (Partial), Job-46745623 (Success)
- **Goal:** Generate `cat_embedding` (for GRU/RNNs) and `one_hot` (for Baselines) representations.
- **Key Artifacts:**
    - `data/interim/readmit_analysis/long_los_cohort.csv`: The canonical cohort file with valid splits.
    - `ehr/cohort_eda.py`: The source of truth for cohort generation and splitting.
    - `ehr/preprocess_ehr.py`: The preprocessing logic for feature extraction.

## Next Steps
1.  Submit the preprocessing job (`scripts/slurm/preprocess_ehr_long_los.sbatch`) - **COMPLETED**.
2.  Submit `scripts/slurm/train_ehr_encoder.sbatch` (Temporal Embeddings).
3.  Submit `scripts/slurm/encode_ehr_static.sbatch` (Static Baseline Embeddings).
