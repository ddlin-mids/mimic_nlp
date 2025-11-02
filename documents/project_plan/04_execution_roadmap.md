# Execution Roadmap

## Phase 1 — Infrastructure Hardening (Week of 2025-11-03)
- Port Colab notebooks into `src/` modules (`src/data/build_cohort.py`, `src/data/features_structured.py`, `src/text/encoder.py`).
- Wire Hydra configs (`configs/data.yaml`, `configs/model/*.yaml`) and add `uv` CLI entrypoints for cohort build, embedding generation, training.
- Implement pytest smoke tests for cohort filters, embedding shape consistency, and fusion head forward pass.

## Phase 2 — Embedding Production (Weeks 2–3)
- Generate discharge/radiology embeddings (document version in configs and persist to `_data/processed/`).
- Finalize structured encoder choice (HiBEHRT training or alternative) and produce aligned embeddings.
- Validate coverage (≥90% admissions for each modality) and reconcile missing data via fallback strategies (mask tokens, imputation metadata).

## Phase 3 — Modeling Experiments (Weeks 3–5)
- Train fusion MLP baseline; log metrics per focus group and disposition slice.
- Run GraphNN variant using nearest-neighbor graphs; compare AUROC/AUPRC and calibration.
- Conduct ablations: text-only, structured-only, dual focus subsets, and prior-utilization feature inclusion.
- Package best-performing model with reproducibility bundle (config, commit hash, evaluation notebook).

## Phase 4 — Clinical Integration (Weeks 5–6)
- Present findings for dual focus groups to clinical stakeholders (cardio-renal management, psych discharge coordinators).
- Define action thresholds and alert routing; align with social work resources (transport, housing).
- Draft documentation for deployment (API contract, monitoring dashboards, alert review workflow).

## Continuous Activities
- Maintain `documents/reports/{date}/…` cadence for daily progress.
- Update literature tracker (`research_plan_readmission_models.md`) with new papers/weights.
- Monitor data drift by re-running `ehr/cohort_eda.py` after major data updates.
