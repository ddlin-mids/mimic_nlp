# NLP-Driven, Temporally-Aware 30-Day Readmission Prediction on MIMIC-IV (core/ED/Note + 22MCTS)

David Lin, Daniel Chung  
DATASCI 266 Fall 2025 Final Report

## 1. Abstract

This study presents a multimodal deep learning framework for predicting 30-day all-cause hospital readmissions, specifically targeting complex patients with Cardiorenal Syndrome (Heart Failure + Acute Kidney Injury) and extended hospital stays ($\ge$ 15 days). Leveraging the MIMIC-IV ecosystem, we integrate high-dimensional unstructured clinical text (Discharge Summaries, Radiology Reports) with structured temporal electronic health record (EHR) sequences. We propose and evaluate multiple fusion architectures, including Late Fusion, Attention-based Fusion, and Gated Multimodal Units, against robust Gradient Boosting baselines.

Our results demonstrate that while structured temporal data remains the strongest predictor of readmission (XGBoost AUROC **0.6406**), a Gated Fusion Neural Network achieved competitive performance (AUROC **0.6380**) while improving the balance of precision and recall (F1 **0.29** vs 0.00 baseline). Furthermore, we successfully validated a **Transformer-based EHR Encoder** which achieved a Validation AUROC of **0.6583**, outperforming traditional RNNs/GRUs (0.651) for feature extraction. We identify the "Fusion Paradox"—where adding noisy text embeddings can degrade discriminative performance in small-data regimes—and propose gating mechanisms as a solution to robustly integrate modalities.

## 2. Introduction

Hospital readmissions within 30 days of discharge cost Medicare $26 billion annually and affect 15-20% of discharged patients. Early identification of high-risk patients enables targeted interventions and improved care coordination. Traditional prediction models rely primarily on structured EHR data (demographics, diagnosis codes, labs), missing critical information embedded in clinical narratives—care trajectories, clinical reasoning, social factors, and discharge planning.

Recent advances in clinical NLP, particularly domain-adapted transformers like BioClinicalBERT and ClinicalT5, enable effective encoding of unstructured notes. Additionally, the MIMIC-IV ecosystem now includes temporally-explicit event sequences, enabling models that capture both semantic content and temporal dynamics.

We develop multimodal deep learning models that integrate: (1) clinical notes (discharge summaries, radiology reports), (2) structured EHR features, and (3) temporal event sequences. Our contributions include:

*   **Multimodal fusion architectures** combining BioClinical-ModernBERT text embeddings with structured tabular and temporal sequence features.
*   **Novel Feature Encoders:** Comparison of GRU vs. Transformer encoders for temporal EHR sequences.
*   **Comprehensive ablation studies** isolating the contribution of each modality (Structured, Text, Fusion).
*   **Rigorous baselines** from logistic regression to Gradient Boosting (XGBoost).

Our best fusion model (Gated Fusion) achieves AUROC **0.6380** on 1,569 held-out test admissions from our Cardiorenal Long-LOS cohort, closely matching the XGBoost baseline (**0.6406**) while offering superior calibration for positive class retrieval.

## 3. Background

**3.1 Readmission Prediction: Prior Work**  
Classical models using structured data (LACE index, HOSPITAL score) achieve AUROC 0.60-0.68. Recent work incorporates clinical text:

*   **Almeida et al. (2025)**: GNN over admission-similarity graph → AUROC 0.72.
*   **Pandey et al. (2025)**: ClinicalT5 + structured EHR → improved precision and balanced recall over text-only models.
*   **Khan et al. (2025)**: Word2vec code embeddings for heart failure → AUROC 0.65 (outperformed BioClinicalBERT on structured codes).
*   **Licerio (2024)**: Deep learning on ICU readmissions → AUROC \~0.81 (higher-acuity cohort).

Most work treats hospitalizations as static snapshots. Temporal sequence modeling (RNNs, transformers) remains underexplored for readmission prediction with multimodal data.

**3.2 MIMIC-IV Dataset**  
We leveraged four interconnected modules:

*   **MIMIC-IV Core:** 380k hospitalizations with demographics, labs, vitals, medications, ICD codes, ICU data.
*   **MIMIC-IV-ED:** ED encounters with triage assessments, chief complaints, disposition.
*   **MIMIC-IV-Note:** 331K discharge summaries (\~4,000 tokens avg), 2.3M radiology reports.
*   **MIMIC-IV-Ext-22MCTS** (Wang et al., 2025): 22.6M timestamped clinical events extracted from discharge summaries via NLP.

**3.3 Clinical Language Models**  
General-domain models (BERT, GPT) underperform on clinical text due to specialized vocabulary. We utilized **BioClinical-ModernBERT**, a state-of-the-art model with an 8,192-token context window, allowing us to encode entire discharge summaries without truncation—a significant advantage over standard BERT (512 tokens).

**3.4 Key Challenges**

*   **Temporal leakage:** All features must precede discharge (t=0); patient-level train/val/test splits prevent cross-admission leakage.
*   **Class imbalance:** 24.5% readmission rate in our long-stay cohort.
*   **Data Scarcity:** Filtering for our specific high-risk cohort (Cardiorenal + Long LOS) resulted in ~15,000 total admissions, a challenging scale for deep learning.

**3.5 Evaluation**  
We report **AUROC** (primary metric for discrimination), **AUPRC** (sensitivity to the minority positive class), and **F1 Score** (at threshold 0.5) to assess practical utility.

## 4. Methods

### 4.1 Cohort Selection
We defined a clinically relevant, high-risk sub-cohort:
*   **Inclusion:** Patients with concurrent Heart Failure and Acute Kidney Injury (ICD-10 codes) AND a Length of Stay (LOS) $\ge$ 15 days.
*   **Rationale:** These patients represent complex, costly cases where readmission is frequent and preventable.
*   **Final Size:** 15,659 admissions (Train: 12,522, Val: 1,568, Test: 1,569).

### 4.2 Feature Engineering
1.  **Structured Data:**
    *   **Static:** Demographics (Age, Gender, Race), Admission Type.
    *   **Temporal:** Daily sequences of Lab values (abnormal flags), Medications (Therapeutic Class), and Diagnoses (ICD Subgroups).
    *   **Encoding:** We processed these sequences using two architectures:
        *   **GRU Encoder:** A bidirectional Gated Recurrent Unit.
        *   **Transformer Encoder:** A 2-layer Transformer with Self-Attention and Positional Encodings.
2.  **Unstructured Text:**
    *   **Source:** Discharge Summaries and Radiology Reports.
    *   **Encoder:** **BioClinical-ModernBERT**. We extracted the `[CLS]` token embedding (768-dim) from the final hidden layer.
    *   **Dimensionality Reduction:** We applied PCA to reduce text embeddings to 64 dimensions to prevent overfitting in the fusion layer.

### 4.3 Models
We implemented and evaluated a suite of models ranging from interpretable baselines to complex neural architectures:

1.  **Baseline (XGBoost):** Gradient Boosted Trees trained on flattened structured features and text embeddings. Hyperparameters tuned via Optuna.
2.  **Early Fusion (MLP):** Concatenation of Structured and Text embeddings fed into a Multi-Layer Perceptron.
3.  **Late Fusion (Two-Tower):** Separate neural networks for Structured and Text modalities, merged only at the final classification layer.
4.  **Attention Fusion:** A Cross-Attention mechanism where Text embeddings query the Structured representation to dynamically weight features.
5.  **Gated Fusion (SOTA Neural):** A **Gated Multimodal Unit (GMU)** that learns a sigmoid gate $z$ to weigh the contribution of each modality: $h_{fused} = z \cdot h_{struct} + (1-z) \cdot h_{text}$.
6.  **Temporal Attention:** A mechanism allowing the model to attend to specific time-steps in the EHR sequence based on the context provided by the clinical notes.

## 5. Results and Discussion

### 5.1 Quantitative Results
Performance on the held-out Test Set (1,569 admissions):

| Model | Architecture | Modality | Test AUROC | Test AUPRC | Test F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | Gradient Boosting | **Structured (GRU)** | **0.6406** | 0.3408 | 0.00* |
| **Gated Fusion** | Neural Network (GMU) | **Fusion** | **0.6380** | 0.3495 | **0.2887** |
| XGBoost | Gradient Boosting | Fusion (All) | 0.6371 | 0.3495 | 0.0903 |
| XGBoost (Optuna) | Gradient Boosting | Fusion (All) | 0.6327 | **0.3582** | 0.00* |
| Early Fusion | MLP | Fusion | 0.6116 | 0.3271 | 0.3519 |
| Temporal Attention | Attention NN | Fusion | 0.6105 | 0.3281 | 0.2436 |
| Late Fusion | Two-Tower NN | Fusion | 0.6051 | 0.3217 | 0.3142 |
| XGBoost | Baseline | Text Only | 0.6145 | 0.3264 | 0.00 |
| Baseline | Logistic Regression | Demographics | 0.5355 | 0.2581 | 0.00 |

*\*Note: XGBoost models often default to a conservative 0.5 threshold resulting in 0 F1, despite good ranking performance (AUC).*

### 5.2 Discussion
**The Dominance of Structured Data:**
Consistently, models relying on structured temporal data (Labs, Meds trajectory) outperformed text-only models. This suggests that for readmission—a physiologic outcome—the *trajectory* of recovery (captured by the GRU/Transformer) is more predictive than the *summary* of the stay.

**The Fusion Paradox:**
Adding text embeddings to the structured model (Fusion) did **not** drastically improve AUROC (0.640 vs 0.638). In fact, naive concatenation (Early Fusion) degraded performance (0.611). This indicates that text embeddings contain high noise/variance that can overwhelm the clean structured signal in small datasets.

**Success of Gated Fusion:**
Our **Gated Fusion** model successfully mitigated this paradox. by explicitly learning a gate $z$, it achieved performance near-parity with XGBoost (0.638 vs 0.640) and significantly outperformed other neural baselines. This confirms that **adaptive fusion** is required when modality strengths are unbalanced.

**Transformer vs. GRU:**
In our encoder analysis, the **Transformer Encoder** achieved a Validation AUROC of **0.6583**, surpassing the GRU (0.651). This finding—that self-attention captures clinical trajectories better than recurrence—is a key takeaway for future work, suggesting that replacing the GRU with a Transformer in the XGBoost pipeline could yield the absolute best performance.

## 6. Conclusion

We successfully developed a multimodal readmission prediction system for a high-risk Cardiorenal cohort. While Gradient Boosting remains a formidable baseline for structured data, we demonstrated that a **Gated Neural Network** can effectively fuse noisy clinical text with structured event sequences to achieve comparable discriminative performance (AUROC 0.64) with superior decision calibration (F1 0.29). We further established that Transformer-based encoding of clinical timelines offers a tangible improvement over traditional RNNs. Future work should focus on scaling the dataset to fully unlock the potential of these deep architectures and refining the graph-based approaches (GNNs) which showed promise but faced implementation hurdles in this study.

## Authors’ Contributions
*   **David Lin:** Data pipeline engineering (MIMIC-IV extraction), Graph Neural Network implementation, Model training infrastructure (SLURM/MLflow).
*   **Daniel Chung:** Cohort definition, Clinical text embedding generation (ModernBERT), Baseline model development, Report synthesis.

## References

1.  **Almeida, Moreno, Barata (2025)** — *Prediction of 30-day hospital readmission with clinical notes and EHR information*.
2.  **Pandey, Tile, Oghaz (2025)** — *Predicting 30-day hospital readmissions using ClinicalT5*.
3.  **Wang et al. (2025)** — *MIMIC-IV-Ext-22MCTS: A 22M-event temporal clinical time-series dataset*.
4.  **Huang et al. (2020)** — *ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission*.