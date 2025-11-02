# Master Overview

## Mission & Scope
- **Task:** Predict 30-day readmissions at discharge using multimodal MIMIC-IV v3.1 data (structured EHR + clinical narratives + radiology reports).
- **Care focus:** Dual-track targeting (1) short-stay transitional-care gaps (LOS ≤7 d) and (2) long-stay cardio-renal/sepsis trajectories (LOS ≥15 d with acute complications).
- **Deliverable:** Calibrated classifier (MLP or GNN head) suitable for HPC3 deployment with explainable subgroup monitoring and documented care pathways.

## Accomplished to Date
- **Architecture definition:** Meeting notes (2025-10-27) locked modular repo layout, BioClinical Modern BERT for notes, HiBEHRT-inspired structured encoder, fusion MLP.
- **Pipeline prototyping:** Colab arm built four-notebook chain mirroring planned modules (cohort, structured events, text embeddings, utilities) while preserving data lineage.
- **EDA segmentation:** Daily analysis (2025-10-31 to 2025-11-01) quantified LOS bins, cardio-renal/sepsis phenotype, discharge disposition risk, lab/med instability, social-text cues.
- **Literature scan:** Research plan catalogued contemporary readmission models with reproducible code (GraphSAGE, ClinicalT5 hybrids) to benchmark final system.

## Strategic Threads
1. **Cohort stewardship:** Maintain reproducible filters (`ehr/cohort_eda.py`, `ehr/get_mimic_cohort.py`) and keep stats in `data/interim/readmit_analysis/`.
2. **Modality pipelines:** Finalize embeddings for notes/radiology (BioClinical Modern BERT, long-context variants as available) and select structured encoder (HiBEHRT or alternative).
3. **Model family exploration:** Start with fusion MLP; evaluate Graph Neural approaches if neighbor context adds lift.
4. **Clinical translation:** Align dual focus groups with discharge planning stakeholders (psych transfers, cardio-renal management) and codify intervention hooks.

## Upcoming Decision Gates
- **Structured encoder:** Determine whether to train HiBEHRT from scratch, adapt published checkpoints, or substitute with transformer-on-event sequences (e.g., RETAIN/BERT-style).
- **Fusion head:** Compare MLP baseline vs GraphSAGE variant using shared embeddings.
- **Infrastructure:** Transition notebooks to src/scripts with Hydra configs, add pytest coverage, and prep Slurm scripts.
