# Cardiorenal Fusion Refinement Report
**Date:** 2025-12-02
**Author:** Gemini Agent

## 1. Status Overview & MLflow Observations
Current experiments have hit a performance plateau. Despite having strong individual signals, the **fusion** of Structured EHR and Clinical Notes is underperforming.

### MLflow Results Summary (Baseline vs. Fusion)
| Model / Experiment | Modality | Test AUC | Status |
| :--- | :--- | :--- | :--- |
| **Structured EHR (GRU)** | **Structured Only** | **0.6406** | **Current Best** |
| PyTorch MLP (HPO) | Fusion (Concat) | 0.6378 | Par (No gain) |
| XGBoost Fusion | Fusion (Naive) | 0.6371 | Slight Degradation |
| **Late Fusion (2-Tower)** | **Fusion (Neural)** | **0.5969** | **Failed (Degradation)** |
| Discharge Notes | Text Only | 0.6140 | Strong Baseline |

### Key Findings
1.  **The "Noise" Problem:** The simple concatenation of high-dimensional text embeddings (~1536 dims) with lower-dimensional structured data (~100-200 dims) swamps the signal. The model overfits to the noise in the text rather than leveraging the complementary information.
2.  **Fusion Failure:** The "Late Fusion" (Two-Tower) architecture performed significantly worse (0.597) than even the single-modality baselines. This indicates optimization difficulties, likely due to the dimensionality mismatch or lack of regularization on the text branch.

## 2. Technical Strategy: Dimensionality Reduction

### Why PCA and not SVD?
We have chosen **PCA (Principal Component Analysis)** over Truncated SVD for the following reasons:

1.  **Data Nature (Dense vs. Sparse):**
    *   **SVD (TruncatedSVD):** Typically used for **sparse** data (like TF-IDF Bag-of-Words) because it works on the raw data matrix without centering it. Centering sparse data destroys sparsity (making the matrix dense and massive).
    *   **PCA:** Implicitly **centers** the data (subtracts the mean) before decomposition. Our input data (ModernBERT embeddings) is already **dense**.
2.  **Variance Maximization:**
    *   By centering the data, PCA focuses entirely on the **variance** (differences) between patients.
    *   SVD on uncentered data is influenced by the magnitude of the vectors and the common mean. For neural embeddings, the "average" vector often contains information about the dominant syntax/domain which is shared across all notes. We want to remove this common direction to focus on what makes a specific patient's note unique.
3.  **Standard Practice for Embeddings:** It is standard practice to apply PCA to dense BERT-style embeddings to reduce dimensions while retaining 95-99% of the explained variance.

## 3. Execution Plan

### Phase 1: Dimensionality Reduction (PCA)
*   **Goal:** Compress Text Embeddings (Discharge + Radiology) from ~1536 dimensions to $k$ dimensions (where $k$ explains 95% variance, likely 50-100).
*   **Action:** Create a preprocessing script `src/data/reduce_dimensions.py` to fit PCA on the training set and transform val/test sets.

### Phase 2: Refined Late Fusion
*   **Goal:** Retrain the Late Fusion (Two-Tower) model using the compact PCA-reduced text features.
*   **Hypothesis:** With balanced feature dimensions (Structured $\approx$ Text), the neural network will be able to learn joint representations without being overwhelmed by text noise.

### Phase 3: Evaluation
*   Compare the new "PCA-Fusion" model against the `Structured (GRU)` baseline (AUC 0.6406).
*   *Condition:* Only if Phase 2 fails to beat the baseline will we consider more complex Attention mechanisms (Phase 4).
