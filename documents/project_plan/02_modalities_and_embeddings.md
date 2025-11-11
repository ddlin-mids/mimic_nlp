# Modality Pipelines & Embeddings

## Clinical Text (Discharge & Radiology Notes)
- **Backbone:** BioClinical Modern BERT (long-context variant preferred; fall back to sliding-window 512-token stride 128).
- **Preprocessing:** Deduplicate note IDs, enforce pre-discharge timestamps, redact residual PHI tokens, segment by admission.
- **Embedding Strategy:** Generate admission-level vectors via attention pooling; store in Parquet with metadata (subject_id, hadm_id, note types).
- **Enhancements Under Evaluation:**
  - Longformer-style models (4096 token length) to reduce sliding-window information loss.
  - Domain-adapted adapters for radiology vs discharge contexts to preserve nuanced vocabulary.
  - Prompt-based extractions for social determinants (transport, housing, caregiver support) to complement keyword heuristics.

### ModernBERT Encoding Runbook
1. **Dry run on CPU node**
   ```bash
   UV_CACHE_DIR=.uv-cache uv run python ehr/encode_notes_modernbert.py \
     --input-path notebooks/notebooks_dc/_data/processed/02_link_notes_to_admissions/labeled_admissions_with_discharge_and_radiology.csv \
     --output-dir notebooks/notebooks_dc/_data/processed/04B_encode_notes_bioclinical_modernbert/dry_run \
     --limit 256 --batch-size 2 --device cpu
   ```
   Confirms schema + manifest generation before booking GPUs.
2. **Submit full job on HPC**
   ```bash
   sbatch scripts/slurm/encode_notes_modernbert.sbatch
   ```
   - Jobs must target the **free-gpu** partition with A30s (request handled inside the script).  
   - Override defaults via env vars: `MODERNBERT_INPUT_PATH`, `MODERNBERT_OUTPUT_DIR`, `MODERNBERT_BATCH_SIZE`, `MODERNBERT_DTYPE`.
3. **Outputs**
   - Embeddings and hadm_id mapping saved under `notebooks/notebooks_dc/_data/processed/04B_encode_notes_bioclinical_modernbert/<run>` as compressed `.npz`.
   - `manifest.json` captures CLI arguments, device info, and per-column stats for reproducibility.
4. **Reporting**
   - Log runtime/resource notes in the day’s `documents/reports/YYYY-MM-DD/modernbert-encoding-*` package.

## Structured EHR Events
- **Target Encoder:** HiBEHRT (patient → admission → event hierarchy) with temporal embeddings and max 2 048 events per admission.
- **Current Status:** Published weights not available; alternatives being investigated:
  - Train HiBEHRT from scratch on MIMIC-IV using event tokenization derived from notebooks.
  - Use transformer encoders on flattened event sequences (e.g., BEHRT-style) if hierarchical training proves unstable.
  - Evaluate graph-temporal models (e.g., GNN on admission-event graph) if embeddings lag.
- **Event Engineering:** Harmonize labs, vitals, procedures, medications, diagnoses into shared vocabulary; encode numerical values via discretized bins + z-score embeddings; enforce pre-discharge window alignment.
- **Data Contracts:** Align with notebook outputs (`02_structured_events_hibehrt.ipynb`) and migrate to `src/data/features_structured.py` once finalized.

## Fusion Interface
- **Intermediate Storage:** Persist text and structured embeddings in `data/interim/_data/processed/` alongside cohort splits to enable reproducible modeling.
- **Metadata Synchronization:** Maintain Hydra configs describing embedding versions, hyperparameters, and preprocessing flags for reproducibility.
- **Open Questions:**
  - Optimal vocabulary cutoff for structured events (frequency threshold vs coverage trade-off).
  - Need for modality-specific normalization (e.g., per-service-line) before fusion.
  - Impact of delayed radiology reports spanning admission boundaries; policy to cap inclusion at discharge + X hours.
