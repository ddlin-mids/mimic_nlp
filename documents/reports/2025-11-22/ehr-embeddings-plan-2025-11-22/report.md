# End of Day Report — 2025-11-22

## 1) Report Header

- **Date & Agent:** 2025-11-22 (PT), codex-agent
- **Project / Subtask:** MIMIC-IV 30-Day Readmission / Preliminary EHR Embeddings for Multimodal Fusion
- **Starting Plan of the Day:**
  - Capture a concrete plan for generating structured EHR embeddings compatible with Daniel's ModernBERT note embeddings and fusion notebook.
  - Align HPC scripts (`ehr/*`) and Colab notebooks (`notebooks/notebook_dl`, `notebooks/notebooks_dc`) around shared artifacts and paths.
  - Prioritize a fast, reliable Stage 1 EHR baseline so Daniel can start multimodal training.
- **Context Sources Used:** chat thread; `documents/report_guide/report_guide.md`; `documents/project_plan/02_modalities_and_embeddings.md`; `documents/research_plan_readmission_models.md`; `notebooks/notebooks_dc/02B_link_notes_to_admissions.ipynb`; `notebooks/notebooks_dc/04B_encode_notes_bioclinical_modernbert.ipynb`; `notebooks/notebooks_dc/06_encode_temporal_events.ipynb`; `notebooks/notebooks_dc/07_multimodal_fusion.ipynb`; `ehr/preprocess_ehr.py`; `data/data_utils/*`; `refs/readmit-stgnn/*`.

## 2) What Was Done (Planning & Design)

- **Clarified environment split:**
  - `notebook_dl` + `ehr` + `scripts/slurm` are the HPC-facing, `uv`-managed path for full-cohort runs (ModernBERT encoding, future structured encoders).
  - `notebooks/notebooks_dc` is Daniel's Colab space (A100) copied into the repo for reference; we treat its outputs as upstream artifacts to be mirrored under `_data/processed/` for HPC scripts.
- **Reviewed note pipelines:**
  - `02B_link_notes_to_admissions.ipynb` links long-LOS admissions to single discharge summaries and final pre-discharge chest X-ray reports, adding length/token stats per note.
  - `04B_encode_notes_bioclinical_modernbert.ipynb` (and `ehr/encode_notes_modernbert.py`) encode discharge and radiology notes separately with BioClinical-ModernBERT (CLS token, `max_length=8192`, no chunking), saving `.npz` embeddings plus a `hadm_id` mapping.
- **Reviewed existing structured EHR tooling:**
  - On Colab, Notebook 6 builds a prototype Bag-of-Events (BoE) embedding from `events_before_discharge.csv` for a 10K sample via counts, L2 normalization, and TruncatedSVD to 128 dimensions per admission.
  - On HPC, `ehr/preprocess_ehr.py` and `data/data_utils/*` already implement a STGNN-style EHR pipeline: per-day ICD, lab, and medication features plus demographics, combined into `ehr_combined.csv` and preprocessed into both cat-embedding and one-hot representations, then rearranged into day-level sequences per admission (`ehr_preprocessed_seq_by_day_cat_embedding.pkl` / `_one_hot.pkl`).
- **Analyzed STGNN reference (`refs/readmit-stgnn`):**
  - Reference `ehr/preprocess_ehr.py` matches our local `ehr/preprocess_ehr.py` in spirit: daily bag-of-words counts for ICD/meds, daily abnormal flags for labs, demographic columns duplicated per day, and day sequences assembled per admission node.
  - `model/embedder.EmbeddingGenerator` learns embeddings for categorical columns and concatenates them with continuous features, producing richer per-day vectors without precomputing EHR embeddings.
  - `model/model.GraphRNN` then feeds these sequences through graph-aware GRU layers over an admission graph, pooling over time to produce a node-level representation used directly for readmission prediction (no separate EHR embedding file).
- **Framed near-term EHR embedding strategy:**
  - **Stage 1:** fast, static admission-level EHR embeddings using existing day-level features and SVD, designed to be easy to fuse with ModernBERT note embeddings and Daniel's Notebook 7 pipeline.
  - **Stage 2:** sequence-aware EHR embeddings that respect day-level trajectories via a lightweight GRU or temporal encoder (without requiring DGL), building on the same STGNN-style feature engineering.

## 3) Results Snapshot (Plan Artifacts)

| Experiment ID     | Data Slice                         | Model                        | Key Change                                                                                         | Metric 1 | Metric 2 | Notes                                                                                                              |
| ----------------- | ---------------------------------- | ---------------------------- | -------------------------------------------------------------------------------------------------- | -------- | -------- | ------------------------------------------------------------------------------------------------------------------ |
| PLAN-EHR-Stage1   | All admissions with labs/ICD/meds | N/A (design only)           | Define BoE+SVD static EHR embedding aligned with ModernBERT note embeddings                       | N/A      | N/A      | Uses `ehr/preprocess_ehr.py` outputs to aggregate per-admission vectors and compress to 128–256 dimensions.       |
| PLAN-EHR-Stage2   | All admissions with day sequences | GRU/temporal encoder (plan) | Design sequence encoder over `ehr_preprocessed_seq_by_day_cat_embedding.pkl` to produce 128–256-dim embeddings | N/A      | N/A      | Encodes daily EHR trajectories without graph layers to keep dependencies light.                                   |

*(No models were trained yet; this table captures plan milestones, not empirical metrics.)*

## 4) Impact Assessment

- **Accuracy / Utility:** A clear Stage 1 EHR embedding plan lets us quickly supply Daniel with structured embeddings that incorporate ICD, medications, and labs beyond what the current 10K event BoE captures, supporting more informative multimodal baselines.
- **Reliability / Robustness:** Reusing the STGNN-aligned `ehr/preprocess_ehr.py` ensures that structured features are consistent across HPC and any future graph or sequence models; SVD-based dimensionality reduction is deterministic and easy to reproduce.
- **Decision-readiness:** The documented stages give us an executable path: first deliver static embeddings for fusion, then incrementally upgrade to sequence-aware encoders once Daniel has a working multimodal training loop.
- **Risk & Ethics:** All work stays within existing de-identified MIMIC-IV artifacts; no new PHI fields are introduced. Main risk is overfitting or miscalibration when we later train models, which will need explicit evaluation.

## 5) Deviations from Plan

- No prior daily plan existed for today in this repo; this report establishes the initial EHR-embedding roadmap rather than summarizing completed experiments.

## 6) Open Questions & Unknowns

1. **Aggregation design for Stage 1**
   - Open: Exact rules for collapsing daily ICD/med/lab sequences into a single admission-level vector (e.g., sum/LOS vs ever-abnormal vs last-day snapshot).
   - Evidence needed: Small-scale experiments comparing a few aggregation strategies on a subset (e.g., AUROC deltas when used with a simple classifier).
2. **Target dimensionality and normalization**
   - Open: Whether 128-dimensional EHR embeddings (to match event BoE) are sufficient, or if 256 dimensions are needed to preserve variance without hurting downstream training.
   - Evidence needed: Explained variance from SVD and downstream performance in Daniel's fusion notebook.
3. **Sequence encoder architecture for Stage 2**
   - Open: Whether a single-layer GRU over days with 128 hidden units is enough, or if we should adopt a small transformer-style encoder for day-level features.
   - Evidence needed: Benchmarking 1–2 candidate architectures on a held-out validation set once the data loader is wired up.

## 7) Next Steps (Ranked)

1. **Immediate (today/next session) — Owner: codex-agent / ddlin**
   - Action: Run `ehr/preprocess_ehr.py` on the chosen readmission cohort (same splits as ModernBERT notes) and persist `ehr_combined.csv` plus `ehr_preprocessed_seq_by_day_{cat_embedding,one_hot}.pkl` under `data/interim/ehr/`.
   - Success: Files present with expected row counts and documented in the next report, ready for aggregation.
2. **Short-term (this week) — Owner: codex-agent / ddlin**
   - Action: Implement Stage 1 static EHR embedding generator (BoE+SVD) as a CLI, writing `ehr_embeddings_boe_full.npz` and a manifest, and coordinate with Daniel to consume it in his multimodal fusion notebook alongside ModernBERT discharge/radiology embeddings.
   - Success: Daniel can load all embeddings (notes plus structured) for a cohort-level training run with aligned `hadm_id` indices.
3. **Short-term (this week/next) — Owner: codex-agent / ddlin**
   - Action: Design and prototype `ehr/encode_structured_events.py` that reads `ehr_preprocessed_seq_by_day_cat_embedding.pkl` and trains or applies a small GRU-based sequence encoder to export 128–256-dimensional EHR embeddings.
   - Success: A working CLI with a CPU-bounded smoke test (small subset) and clear I/O schema, ready for later Slurm scaling.
4. **Nice-to-have — Owner: modeling team (Daniel + ddlin)**
   - Action: Evaluate more expressive temporal encoders (HiBEHRT-style transformer over day-level EHR tokens) or graph-based encoders as stretch goals once baseline multimodal performance is established.
   - Success: At least one alternative EHR encoder compared against Stage 1/2 embeddings on AUROC/AUPRC with documented trade-offs.

## 8) Reproducibility Notes

- **Entry points (planned):**
  - `uv run python ehr/get_mimic_cohort.py ...` for cohort builds (already available).
  - `uv run python ehr/preprocess_ehr.py --demo_file ... --icd_file ... --lab_file ... --med_file ... --save_dir data/interim/ehr` to generate STGNN-style day-level EHR features.
  - Future: `uv run python ehr/encode_ehr_static_boe.py ...` (Stage 1) and `uv run python ehr/encode_structured_events.py ...` (Stage 2) once implemented.
- **Minimal config:**
  - Uses existing `pyproject.toml` with `torch`, `pandas`, `numpy`, `sklearn`, and `tqdm` via `uv sync`.
  - Data assumed under `physionet.org/files/...` (read-only) and `data/interim/` / `notebooks/notebooks_dc/_data/processed/`.
- **Randomness:**
  - SVD seed and any future model seeds will be set explicitly (e.g., `random_state=42` for `TruncatedSVD`, fixed PyTorch seeds for sequence encoders).
- **Data lineage:**
  - MIMIC-IV raw tables (`physionet.org/files/...`) → scripted cohort (`ehr/get_mimic_cohort.py`) → structured EHR preprocessing (`ehr/preprocess_ehr.py`) → Stage 1 static EHR embeddings (planned) and Stage 2 sequence-aware embeddings (planned) → Daniel's fusion notebook (`notebooks/notebooks_dc/07_multimodal_fusion.ipynb`) or its HPC mirror in `notebook_dl`.

## 9) Appendices

- None for now; once embeddings are generated, future reports should link to specific `.npz` and manifest files plus any diagnostic plots.

```json
