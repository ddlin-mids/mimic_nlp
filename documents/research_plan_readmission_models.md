# Research Plan: State-of-the-Art 30-Day Readmission Models

**Objective:** To systematically identify, evaluate, and synthesize the latest high-quality research on predicting 30-day hospital readmissions using EHR data. This plan prioritizes models with publicly available code to ensure reproducibility and direct application to our project, which is based on the MIMIC-IV dataset.

This document outlines a repeatable methodology for staying current with the state-of-the-art, building upon the existing literature review found in `documents/readmission_on_mimic.md`.

## 1. Scoping & Keyword Definition

- **Timeframe:** Focus on recent publications (2024-2025) and significant benchmark papers from 2023.
- **Core Task:** 30-day all-cause readmission prediction.
- **Dataset Focus:** Primarily MIMIC-IV, but also consider other large EHR datasets (e.g., eICU).
- **Technical Focus:** Advanced machine learning models, including but not limited to:
    - Graph Neural Networks (GNNs)
    - Transformer-based architectures (e.g., HiBEHRT, Long-context models)
    - Large Language Models (LLMs) for clinical text or code sequences (e.g., ClinicalT5, CPLLM)
    - Multimodal fusion techniques.
- **Keywords:** A combination of the following will be used for searching: `("30-day readmission" OR "hospital readmission")`, `(EHR OR "electronic health record")`, `MIMIC-IV`, `("deep learning" OR GNN OR transformer OR LLM)`, `(NLP OR "clinical notes")`, `(github OR "code available" OR repository)`.

## 2. Search & Discovery

A multi-platform search strategy will be employed to ensure comprehensive coverage:

- **Academic Search Engines:**
    - **Google Scholar:** Broad search for initial discovery.
    - **arXiv:** Access to the latest pre-prints in machine learning.
    - **PubMed/PMC:** Focus on peer-reviewed biomedical and clinical publications.
- **Social & Code Platforms:**
    - **Papers with Code:** Directly search for the readmission task to find papers linked to code.
    - **GitHub:** Search for repositories related to MIMIC-IV and readmission to potentially discover papers or projects not yet formally published.

## 3. Filtering & Quality Assessment

Papers and repositories discovered will be filtered based on the following criteria:

1.  **Relevance:** Must directly address the 30-day readmission task using EHR data.
2.  **Code Availability:** A public repository with usable code is a primary requirement. The repository will be briefly inspected for completeness (e.g., has a README, data processing scripts, model definitions).
3.  **Methodological Rigor:** The paper should clearly describe its cohort selection, feature engineering, model architecture, and evaluation metrics.
4.  **Performance:** The reported performance (e.g., AUROC, AUPRC) will be compared against established benchmarks and our project's targets.
5.  **Venue:** Peer-reviewed publications are preferred, but high-quality, well-documented pre-prints will be included.

## 4. Synthesis & Reporting

The findings from the selected papers will be synthesized into a structured report. This report will update or supplement `documents/readmission_on_mimic.md`.

The synthesis for each paper will include:

- **Citation:** Full citation details.
- **Summary:** A concise overview of the paper's main contribution.
- **Model Architecture:** A clear description of the model used.
- **Dataset & Cohort:** Details on the dataset (e.g., MIMIC-IV version) and the final cohort size and characteristics.
- **Key Results:** Primary performance metrics (AUROC, AUPRC).
- **Code Repository:** A direct link to the public code.

A summary table will be maintained for quick comparison across different approaches.

## 5. Integration with Project Goals

The final step is to analyze how the findings from the new research can be integrated into our existing project plan (`documents/updated_project_plan.md`). This involves:

- Identifying novel techniques that could improve our HiBEHRT + BioClinical Modern BERT architecture.
- Comparing our expected performance targets with the newly discovered state-of-the-art.
- Considering alternative fusion strategies or feature engineering methods.

This structured research process will ensure our project remains aligned with the cutting edge of the field and leverages the best available methods and tools.
