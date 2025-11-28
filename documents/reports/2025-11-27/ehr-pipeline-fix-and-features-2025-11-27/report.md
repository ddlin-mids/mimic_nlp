# End of Day Report — 2025-11-27

## 1) Report Header
- **Date & Agent:** 2025-11-27 (PT), Claude Code
- **Project/Subtask:** MIMIC-IV 30-Day Readmission / Structured EHR Preprocessing & Embedding
- **Starting Plan:**
    *   Investigate and fix the "garbage output/OOM" error in the EHR preprocessing job.
    *   Implement a memory-safe "chunking" mechanism for processing 140k+ admissions.
    *   Integrate "Prior Admission History" features (count, frequency, gaps).
    *   Prepare the Deep Learning Encoder (GRU/LSTM) script for Phase 2 embedding generation.
- **Context Sources Used:**
    *   Failed log: `logs/slurm/preprocess_ehr_long_los_46629533.log` (garbage output).
    *   Reference code: `refs/readmit-stgnn/ehr/preprocess_ehr.py` (iterative vs. vectorized comparison).
    *   Feature request: "Prior 30-day readmission" calculation.

## 2) What Was Done

### Data Engineering & Pipeline Optimization
- **Root Cause Analysis:** Identified that the "garbage" log output was an Out-Of-Memory (OOM) corruption caused by trying to "explode" the entire 140k-patient temporal matrix at once in pandas.
- **Implemented Chunking:** Refactored `ehr/preprocess_ehr.py` to process admissions in batches (chunks of 2,000).
    - **Impact:** Peak memory usage dropped significantly (estimated <32GB), preventing crashes on 128GB nodes.
    - **Resumability:** Added logic to save intermediate chunks (`temp_chunks/`). If the job fails, it resumes from the last successful chunk.
- **Feature Engineering (New):** Integrated prior utilization features directly into the pipeline.
    - `num_prior_admissions`: Total count of previous hospitalizations.
    - `num_prior_30d_readmissions`: Frequency of "frequent flyer" events.
    - `days_since_last_discharge`: Recency of last contact (proxy for frailty).
- **Global Consistency:** Added logic to pre-scan all raw files (ICD, Labs, Meds) to establish a consistent feature vocabulary (Global Columns) across all chunks, ensuring valid concatenation.

### Modeling Prep (Phase 2)
- **Encoder Script:** Created `ehr/train_ehr_encoder.py`.
    - **Architecture:** Customizable GRU/LSTM sequence encoder.
    - **Input:** Takes the processed sequence pickle.
    - **Leakage Fix:** Implemented *subject-wise* splitting (vs. random admission split) to prevent data leakage from train to val.
    - **Speed:** Replaced row-iteration with vectorized loading.
    - **Output:** Produces 128-dim (default) static embeddings per admission.
- **Baseline Script:** Created `ehr/encode_ehr_static_boe.py`.
    - **Method:** Bag-of-Embeddings (summed counts) -> TF-IDF -> SVD (PCA).
    - **Purpose:** Fast, interpretable baseline that ignores temporal order.

### Cohort Validation
- **Text Availability Check:** Verified that 100% of the Long-LOS cohort (15,659 admissions) has both Discharge Summaries and Radiology Reports available.
- **Sample Radiology Report:**
  > "INDICATION: Fever, tachycardia and history of bronchiectasis, evaluate for pneumonia... FINDINGS: Frontal and lateral views of the chest were performed. The lung volumes are low... there appear to be bibasilar, right greater than left, nodular opacities..."
- **Sample Discharge Summary:**
  > "History of Present Illness: Mrs. ___ is a ___ F with history of recurrent diverticulitis, originally diagnosed ___..."

### Code Quality & Optimization
- **Refactoring:** Cleaned up `preprocess_ehr.py` (type hints, remove hardcoded paths).
- **Cleanup:** Deleted obsolete documentation (`ehr/cohort_tools.md`, `ehr/preprocess_ehr.md`) and redundant scripts (`ehr/add_history_features.py`).
- **Encoding Optimization (Final Polish):** Refined the final encoding step (`preproc_ehr`) to:
    - **Avoid Loop:** Replaced iterating 6000 columns (17hr eta) with vectorized ops (seconds).
    - **Reduce Dim:** Switched Labs to Binary (`abnormal`=1, else 0) and Gender to Binary (`drop_first=True`) to halve sparsity overhead.
    - **Correctness:** Changed `fillna(mean)` to `fillna(0)` for count-based features (ICD/Meds) to preserve sparse semantics.

## 3) Methodology Comparison: Classification Strategies

We evaluated three architectures for handling the structured EHR data. We selected Strategy 3 (Sequence RNN) as the primary method.

| Feature | 1. Patient Graph (Mental Model) | 2. Concept Graph (Reference STGNN) | 3. Sequence RNN/GRU (Our Choice) |
| :--- | :--- | :--- | :--- |
| **Nodes** | **Patients**. (Edges = Similarity) | **Medical Codes**. (Edges = Co-occurrence) | N/A (Implicit latent space). |
| **Mechanism** | "Crowd Wisdom" (Node Classification). Labels propagate from neighbor patients. | "Concept Enrichment". Patient is a signal flowing through the concept graph. | "Temporal Trajectory". Learning sequential patterns (worsening vs improving). |
| **Pros** | Powerful for look-alike modeling. | Handles rare diseases well by using "neighbor" codes. | Fast, standard, inductive (handles new patients easily). |
| **Cons** | **Leakage Risk**: Hard to separate Train/Test. **Scale**: $N^2$ edges expensive. | **Complex**: Requires massive adjacency matrix in memory. | Data hungry (needs more samples to generalize). |
| **Fusion** | Difficult to fuse with Text embeddings. | End-to-end training only. | **Modular**: Output is a simple dense vector, perfect for concatenating with BERT. |

**Decision:** We proceed with **Strategy 3 (GRU)** because it provides the best balance of performance and compatibility with our multimodal fusion goal. We may apply a KNN-smoothing layer (Strategy 1 concept) *after* fusion if performance boost is needed.

## 4) Results Snapshot

| Experiment ID | Task | Status | Key Metric | Notes |
| ------------- | ---- | ------ | ---------- | ----- |
| `Job-46629533` | Vectorized Preproc (Old) | **Failed** | N/A | OOM / Garbage Log |
| `Job-46740657` | Chunked Preproc (v1) | **Cancelled** | N/A | Stopped to apply encoding optimization |
| `Job-46740996` | Chunked Preproc (v2) | **Success** | **100%** | Generated sequence pickle (20GB) and CSV (5.7GB) |
| `Job-46741264` | Repair One-Hot | **Success** | **100%** | Generated missing one-hot artifacts |
| `EHR-Encoder` | Embedding Training | **Running** | Val AUC ~0.63 | Initial epochs show good learning |

*(Detailed results will be available once the currently running preprocessing job completes.)*

## 5) Impact Assessment

### Accuracy / Utility
- **Prior History:** Adding "prior readmissions" is historically one of the strongest predictors. This is expected to significantly boost the model's ability to identify high-risk "frequent flyers."
- **Sequence Modeling:** Moving to a GRU (vs. just static counts) allows the model to capture *trajectories* (e.g., worsening labs over time), which static baselines miss.

### Reliability / Robustness
- **Memory Safety:** The chunking mechanism ensures the pipeline is robust to dataset growth. We can now process the full MIMIC-IV dataset without requesting exotic high-mem nodes.
- **Leakage Prevention:** Fixing the validation split ensures our reported metrics will be realistic and not inflated by patient identity leakage.

## 6) Deviations from Plan

- **Integrated History:** Initially planned to just "fix the crash," but expanded scope to "add history features" immediately to ensure the dataset is "clean" for the baseline run. This required cancelling and restarting the job but saves a retraining cycle later.

## 7) Open Questions & Unknowns

- **Training Time:** How long will the GRU take to converge on 140k sequences? We allotted 4 hours (`scripts/slurm/train_ehr_encoder.sbatch`), which should be sufficient, but TBD.
- **Embedding Dimension:** We are defaulting to 128 dimensions. Is this optimal? We might need to tune this if the representation is too sparse or too compressed.

## 8) Next Steps (Ranked)

1.  **Immediate:** Monitor ongoing training jobs (`46741258`, `46741259`) for completion.
2.  **Short-term:** Validate the learned embeddings. (Do patients with similar conditions cluster together? Is the supervised training AUROC reasonable?)
3.  **Downstream:** Hand off `structured_ehr_embeddings.npz` to Daniel for multimodal fusion.

## 9) Reproducibility Notes

- **Entry Point:** `scripts/slurm/preprocess_ehr_long_los.sbatch`
- **Config:**
    - `chunk_size`: 2000
    - `dataset`: Long LOS Cohort (LOS >= 15 days)
    - `history`: Enabled (`augment_cohort_with_history`)
- **Lineage:** Raw MIMIC tables -> `preprocess_ehr.py` (Chunked) -> `ehr_combined.csv` & `ehr_preprocessed_seq_by_day_cat_embedding.pkl`