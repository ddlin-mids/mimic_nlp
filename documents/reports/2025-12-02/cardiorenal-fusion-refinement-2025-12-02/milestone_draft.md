# Project Milestone / Progress Report
**Date:** 2025-12-02
**Author:** Gemini Agent (on behalf of team)

## 1. Introduction
We address the critical challenge of predicting 30-day all-cause hospital readmission for patients with cardiorenal syndrome. This complex comorbidity profile (heart failure + kidney disease) is associated with high readmission rates and significant healthcare costs. Our goal is to leverage the rich, multimodal data available in MIMIC-IV—specifically combining structured temporal events (labs, vitals) with unstructured clinical notes (discharge summaries, radiology reports)—to improve prediction accuracy beyond standard baselines.

## 2. Methodology

### 2.1 Data Processing & Cohort
*   **Source:** MIMIC-IV (v3.1) and MIMIC-IV-Note (v2.2).
*   **Cohort Definition:** Patients with concurrent Heart Failure and Acute Kidney Injury (ICD-based) and a "long" length of stay (>= 15 days), representing the most complex cases.
*   **Sample Size:** ~11,500 admissions (Train: 80%, Val: 10%, Test: 10%).

### 2.2 Modality Encoders
We employ distinct encoding strategies for the two primary data types:
1.  **Structured Data:**
    *   **Static Features:** Demographics (Age, Gender, Race) encoded via One-Hot/StandardScaler.
    *   **Temporal Features:** Daily sequences of lab values and vital signs processed via a **GRU (Gated Recurrent Unit)** to capture trajectory (e.g., worsening creatinine levels).
2.  **Unstructured Text:**
    *   **Encoder:** **BioClinical ModernBERT**, a state-of-the-art transformer model pre-trained on biomedical text.
    *   **Dimensionality:** We extract 768-dimensional `[CLS]` token embeddings from both Discharge Summaries and Radiology Reports.
    *   **Refinement:** To address the "curse of dimensionality" and noise, we experiment with **PCA (Principal Component Analysis)** to compress embeddings while retaining 95% of variance.

### 2.3 Fusion Architectures
We evaluate three fusion strategies to combine these modalities:
1.  **Early Fusion (XGBoost):**
    *   Concatenation of all feature vectors (Static + Structured GRU State + Text Embeddings).
    *   Optimized via **Optuna** for hyperparameters (Depth, Learning Rate, Regularization).
    *   *Hypothesis:* Gradient boosting handles tabular/dense mixtures robustly.
2.  **Early Fusion (MLP):**
    *   A standard Feed-Forward Neural Network trained on the concatenated input.
    *   Serves as a baseline for neural optimization.
3.  **Late Fusion (Two-Tower Network):**
    *   **Architecture:** Two separate neural branches ("Towers")—one for Structured Data, one for Text—that process modalities independently before merging at a final classification layer.
    *   *Hypothesis:* Allows the model to learn modality-specific non-linearities without text noise overwhelming the structured signal.

## 3. Preliminary Results

### 3.1 Quantitative Performance (Test AUC)
| Model | Modality | Test AUC | Key Observation |
| :--- | :--- | :--- | :--- |
| **XGBoost (Optimized)** | **Fusion (All)** | **0.645** | **Current SOTA.** Robust to noise. |
| Structured GRU | Structured Only | 0.641 | Strong baseline; text adds minimal gain. |
| Discharge Notes | Text Only | 0.614 | Predictive, but weaker than structured data. |
| Late Fusion (Naive) | Fusion | 0.597 | Failed. Likely due to high-dim text noise. |
| Demographic Baseline | Static Only | ~0.530 | Poor performance. |

### 3.2 Analysis
*   **The "Noise" Barrier:** Adding high-dimensional text embeddings (1536 dims) to the structured model (128 dims) initially degraded neural performance. XGBoost was able to ignore the noise (via feature selection), maintaining performance, whereas the naive MLP overfitted.
*   **Fusion Gain is Marginal:** The lift from 0.641 (Structured) to 0.645 (Fusion) is small. This suggests that the text may be redundant with the structured data (e.g., the note says "Creatinine rising" and the lab values show it), or that we need better extraction of *unique* text signals (e.g., social determinants).

## 4. Next Steps
1.  **PCA Refinement:** We are currently running experiments (Jobs 46810528/46810530) to test if PCA-reduced text embeddings (64 dims) allow the Neural Networks to converge better.
2.  **Error Analysis:** We will investigate false negatives to see if they correspond to patients with missing notes or specific clinical subtypes.
3.  **Final Report:** Synthesize these findings into the final paper, focusing on the trade-off between model complexity (Fusion) and robustness (Single Modality).
