**Datasets:**  
Timestamps: [https://physionet.org/content/mimic-iv-ext-22mcts/1.0.0/](https://physionet.org/content/mimic-iv-ext-22mcts/1.0.0/)  
Core: [https://physionet.org/content/mimiciv/3.1/hosp/\#files-panel](https://physionet.org/content/mimiciv/3.1/hosp/#files-panel)  
Notes: [https://physionet.org/content/mimic-iv-note/2.2/note/\#files-panel](https://physionet.org/content/mimic-iv-note/2.2/note/#files-panel)  
ED (optional): [https://physionet.org/content/mimic-iv-ed/2.2/ed/\#files-panel](https://physionet.org/content/mimic-iv-ed/2.2/ed/#files-panel)

**Project Update: MIMIC-IV 30-Day Readmission Prediction**  
**Completed Notebooks 1-7 (Prototype Phase)**

Built:  
A multimodal deep learning system that predicts 30-day hospital readmissions  
by combining clinical notes, patient demographics, and temporal event sequences.

Notebook 1: Label Construction

- Created binary labels (readmitted=1, not=0) for 518K hospital admissions  
- Applied exclusion criteria (removed deaths, transfers)  
- Output: 518K labeled admissions, 19.6% readmission rate (good balance)

Notebook 2: Link Clinical Notes

- Merged discharge summaries from MIMIC-IV-Note with labeled admissions  
- Coverage: 311K admissions have both labels AND discharge notes (60%)  
- Average note length: \~10,000 characters

Notebook 3: Link Temporal Events

- Integrated 22M timestamped clinical events (symptoms, diagnoses, procedures)  
- Critical: Filtered events to BEFORE discharge (prevents data leakage)  
- Average \~85 events per admission

Notebook 4: Encode Notes with ClinicalBERT

- Used ClinicalBERT (pre-trained on MIMIC clinical text) to encode notes  
- Chunked long notes into 512-token segments with overlap  
- Mean pooling across chunks → 768-dim embedding per admission  
- Prototype: Encoded 10K sample in \~8 minutes on A100 GPU

Notebook 5: Tabular Feature Engineering

- Extracted 26 structured features from EHR data:  
  - Demographics (race, insurance, marital status)  
  - Admission context (admission type, discharge location)  
  - Clinical complexity (diagnosis count, procedure count, length of stay)  
- Normalized features using StandardScaler

Notebook 6: Encode Temporal Event Sequences

- Vocabulary: \~1,200 unique clinical events (filtered rare events)  
- Encoding: Bag of Events (count-based, ignores order)  
- Dimensionality reduction: TruncatedSVD to 128-dim embeddings  
- Note: Plan to upgrade to LSTM for full dataset (captures temporal patterns)

Notebook 7: Multimodal Fusion & Training

- Combined all three modalities: Text (768) \+ Tabular (26) \+ Events (128) \= 922-dim  
- Architecture: Modality projections → Concatenation → MLP classifier  
- Trained on 10K sample (7K train, 1.5K val, 1.5K test)  
- Results: Test AUROC \~0.60-0.62 (expected for small dataset)

KEY FINDINGS:

- Complete pipeline validated end-to-end  
- All data sources successfully integrated  
- Model trains but overfits on small 10K sample (expected)  
- Ready to scale to full 311K dataset

PROTOTYPE LIMITATIONS:

- Small dataset (10K) causes overfitting \- train AUROC 0.74, val AUROC 0.62  
- Limited features (26) \- missing labs, medications, age, comorbidity scores  
- Bag of Events ignores temporal order \- LSTM will improve this  
- Performance is proof-of-concept, not final results

NEXT PHASE: SCALING TO FULL DATASET (311K)

1. Add comprehensive features: age, lab values, medications, comorbidity indices  
   1. Increase from 26 to \~100 features  
2. Encode all 311K notes with ClinicalBERT (\~2-3 hours on A100)  
3. Upgrade event encoding to LSTM (capture temporal disease progression)  
4. Train on full dataset (220K training samples \- 30x more data)  
5. Run ablation studies: text-only, tabular-only, events-only vs full multimodal  
6. Compare to baselines: Logistic Regression, XGBoost

EXPECTED FINAL PERFORMANCE:

- Target AUROC: 0.70-0.75+ (literature benchmark for MIMIC readmission)  
- With 30x more data, overfitting will reduce significantly  
- Richer features \+ temporal modeling will improve signal

## **Dataset & Results Summary**

**Dataset Statistics:**

* Started with 523,740 total admissions  
* After exclusions: 518,349 eligible admissions  
* With discharge notes: 311,459 admissions (60% coverage)  
* With all three modalities (text \+ tabular \+ events): \~9,500 admissions in 10K sample

**Label Distribution:**

* Overall readmission rate: 19.6-20.6% (good balance, not too imbalanced)  
* Class ratio: 4:1 (not readmitted : readmitted)

**Data Characteristics:**

* Average note length: \~10,000 characters (\~2,500 tokens)  
* Average events per admission: \~85 events  
* Event vocabulary: \~1,200 unique clinical events after filtering  
* Most common events: hypertension, fever, abdominal pain, hypotension

**Feature Coverage:**

* Diagnosis data: 99% of admissions have diagnoses (avg 12 diagnoses per admission)  
* Procedure data: Lower coverage (avg 1.6 procedures per admission)  
* Event sequences: \~85% of admissions have timestamped events

**Prototype Model Performance (10K sample):**

* Test AUROC: \~0.60-0.62  
* Shows clear overfitting (train AUROC 0.74, val AUROC 0.62)  
* Expected due to small dataset (7K training samples vs 200K parameters)

