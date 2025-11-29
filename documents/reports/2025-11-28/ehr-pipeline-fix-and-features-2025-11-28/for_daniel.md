# EHR Embeddings Handover (2025-11-28)

**To:** Daniel  
**From:** David
**Subject:** Updated EHR Embeddings (Strict Patient-Level Splitting)

Here are the definitive EHR embedding artifacts generated today. These use the new **strict patient-level splitting** (80/10/10) to ensure zero data leakage between train/val/test sets.

## 1. Static Embeddings (Baseline)
*   **File:** `data/interim/ehr_long_los/embeddings/static_ehr_embeddings.npz`
*   **Method:** Bag-of-Words → TF-IDF → Truncated SVD.
*   **Usage:**
    *   Load with `numpy.load()`.
    *   **Keys:** `embeddings`, `hadm_ids`, `node_names` (format: `subject_hadm`), `splits`.
    *   **Note:** The metadata (`hadm_ids`) is embedded directly in this `.npz` file.

## 2. Structured Embeddings (Temporal/Neural)
*   **Embeddings:** `data/interim/ehr_long_los/embeddings/structured_ehr_embeddings.npz`
*   **Mapping:** `data/interim/ehr_long_los/embeddings/structured_ehr_mapping.csv`
*   **Method:** GRU-based sequential encoding of daily clinical events.
*   **Usage:**
    *   The `.npz` contains the raw embedding vectors.
    *   **CRITICAL:** You must use the accompanying `structured_ehr_mapping.csv` to map the row indices in the `.npz` back to `hadm_id` / `node_name`.

## 3. Reference Cohort with EHR info per date
*   **File:** `data/interim/ehr_long_los/ehr_combined.csv`
*   **Purpose:** Contains the ground truth `split` columns and labels (`readmitted_within_30days`). Use this to ensure we are all evaluating on the exact same test set.

