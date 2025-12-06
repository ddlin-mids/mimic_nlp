# NLP-Driven, Temporally-Aware 30-Day Readmission Prediction on MIMIC-IV (core/ED/Note \+ 22MCTS)

David Lin, Daniel Chung  
DATASCI 266 Fall 2025 Final Report

## 1\. Abstract

## 2\. Introduction

Hospital readmissions within 30 days of discharge cost Medicare $26 billion annually and affect 15-20% of discharged patients. Early identification of high-risk patients enables targeted interventions and improved care coordination. Traditional prediction models rely primarily on structured EHR data (demographics, diagnosis codes, labs), missing critical information embedded in clinical narratives–care trajectories, clinical reasoning, social factors, and discharge planning.

Recent advances in clinical NLP, particularly domain-adapted transformers like BioClinicalBERT and ClinicalT5, enable effective encoding of unstructured notes. Additionally, the MIMIC-IV ecosystem now includes temporally-explicit event sequences, enabling models that capture both semantic content and temporal dynamics.

We develop multimodal deep learning models that integrate: (1) clinical notes (discharge summaries, radiology reports), (2) structure EHR features, and (3) temporal event sequences. Our contributions include:

* **Multimodal fusion architectures** combining BioClinical-ModernBERT text embeddings with structured tabular and temporal sequence features  
* **Comprehensive ablation studies** isolating the contribution of each modality  
* **Rigorous baselines** from logistic regression to neural architectures

Our best model achieves AUROC 0.6327 on 1,569 long-stay admissions (LOS ≥ 15 days), with structured temporal features achieving 0.6295–comparable to text-based models and suggesting complementary information.

## 3\. Background

**3.1 Readmission Prediction: Prior Work**  
Classical models using structured data (LACE index, HOSPITAL score) achieve AUROC 0.60-0.68. Recent work incorporates clinical text:

* **Almeida et al. (2025)**: GNN over admission-similarity graph → AUROC 0.72  
* **Pandey et al. (2025)**: ClinicalT5 \+ structured EHR → improved precision and balanced recall over text-only models  
* **Khan et al. (2025)**: Word2vec code embeddings for heart failure → AUROC 0.65 (outperformed BioClinicalBERT on structured codes)  
* **Licerio (2024)**: Deep learning on ICU readmissions → AUROC \~0.81 (higher-acuity cohort)

Most work treats hospitalizations as static snapshots. Temporal sequence modeling (RNNs, transformers) remains underexplored for readmission prediction with multimodal data.

**3.2 MIMIC-IV Dataset**  
We leveraged four interconnected modules:

* **MIMIC-IV** Core: 380k hospitalizations with demographics, labs, vitals, medications, ICD codes, ICU data  
* **MIMIC-IV-ED**: ED encounters with triage assessments, chief complaints, disposition  
* **MIMIC-IV-Note**: 331K discharge summaries (\~4,000 tokens avg), 2.3M radiology reports  
* **MIMIC-IV-Ext-22MCTS** (Wang et al., 2025): 22.6M timestamped clinical events extracted from discharge summaries via NLP, normalized to RxNorm/SNOMED-CT

**3.3** **Clinical Language Models**  
General-domain model (BERT, GPT) underperform on clinical text due to specialized vocabulary and semantics. Domain-adapted models include:

* **ClinicalBERT**: BERT fine-tuned on MIMIC-III notes  
* **BioClinicalBERT**: Further pre-trained on PubMed \+ MIMIC  
* **BioClinical-ModernBERT**: Recent variant with 8,192-token context, accommodating 96.9% of our combined discharge \+ radiology text without truncation or chunking

**3.4** **Key Challenges**

* **Temporal leakage**: All features must precede discharge (t=0); patient-level train/val/test splits prevent cross-admission leakage  
* **Class imbalance**: 24.5% readmission rate in our long-stay cohort  
* **Long documents**: Average discharge summary \~4,300 tokens; 3.1% exceeds 8K context and will be removed  
* **Data alignment**: Matching notes, structured events, and temporal sequences across heterogeneous sources

**3.5 Evaluation**  
We report AUROC (primary metric), AUPRC (minority class sensitivity), and Expected Calibration Error (ECE) for reliable risk estimates

## 4\. Methods

**Baseline Model**

## 5\. Results and discussion

## 6\. Conclusion

## References

1. **Almeida, Moreno, Barata (2025)** — *Prediction of 30-day hospital readmission with clinical notes and EHR information* (GNN over admissions; AUROC≈0.72).  
2. **Pandey, Tile, Oghaz (2025)** — *Predicting 30-day hospital readmissions using ClinicalT5 with structured & unstructured EHR* (hybrid LLM \+ tabular; PLOS ONE).  
3. **Licerio (2024)** — *Predicting 30-Day Unplanned ICU Readmissions Using Deep Learning and NLP: A MIMIC-IV Analysis* (high-signal ICU cohort; AUROC≈0.81, thesis)  
4. **Khan et al. (2025)** — *Predicting 30-Day Readmission for Heart Failure Using Code Embeddings on MIMIC-IV* (word2vec/BioClinicalBERT vs one-hot).  
5. **Wang et al. (2025)** — *MIMIC-IV-Ext-22MCTS: A 22M-event temporal clinical time-series dataset from MIMIC-IV-Note* (dataset paper).

Dataset context: MIMIC-IV (core/ED/Note) documentation and overview. [MIMIC-IV](https://physionet.org/content/mimiciv/?utm_source=chatgpt.com)

## Authors’ Contributions
