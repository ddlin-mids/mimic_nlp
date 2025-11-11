# Research Plan — Dual Execution Strategy

This plan codifies how we will progress the 30-day readmission project while supporting both HPC-first scripting and partner-led Colab notebooks. Every experiment, artifact, and report should map to one (or both) execution arms outlined below.

---

## 1. Objectives
1. **Structured + Text Fusion**: Deliver reproducible ModernBERT embeddings for discharge and radiology notes plus structured EHR features ready for modeling.
2. **Dual-Environment Coverage**: Keep Daniel’s Colab workflow productive (A100, 100-credit budget) while ensuring the same logic can run headless via Slurm on the UCI free-gpu A30 partition.
3. **Governance & Reporting**: Maintain dated daily reports, manifests, and checksums for every major compute run so the team can trace results regardless of platform.

---

## 2. Execution Tracks

### Track A — HPC Scripts (uv-managed)
- **Scope**: Anything that needs full-cohort throughput, reproducibility, or direct Slurm scheduling (e.g., ModernBERT encoding, HiBEHRT training, batch inference).
- **Inputs**: CSV/Parquet artifacts under `notebooks/notebooks_dc/_data/processed/` or `data/interim/`.
- **Entry Points**:
  - `uv run python ehr/get_mimic_cohort.py …` for cohort rebuilds.
  - `uv run python ehr/encode_notes_modernbert.py …` (new CLI) for text embeddings.
  - `scripts/slurm/*.sbatch` templates targeting **free-gpu** A30 nodes (no `--pk_account` usage).
- **Operational Notes**:
  - Cache heavy deps inside repo-local `.uv-cache` / `.cache/huggingface` to avoid HOME quota overruns (50 GB cap).
  - Dry-run scripts on CPU with `--limit` before submitting `sbatch`.
  - Log outputs + manifests in `_data/processed/<notebook>/run_id` folders, then capture results in `documents/reports/YYYY-MM-DD/...`.

### Track B — Colab/Notebook_DL (shared with Daniel)
- **Scope**: Interactive EDA, exploratory modeling, and demonstrations that benefit from Colab Pro’s A100 access.
- **Location**: `notebooks/notebook_dl/` (dual-arm parity notebooks) plus Daniel’s legacy `notebooks/notebooks_dc/`.
- **Practices**:
  - Keep notebooks numbered (`01_…`, `02_…`) and stash derived data in sibling `_data/processed/` directories so HPC scripts can pick them up.
  - When Colab produces a new artifact (e.g., `labeled_admissions_with_discharge_and_radiology.csv`), sync the file back into the repo under the matching `_data/processed/` path.
  - Note GPU/credit usage inside the per-day report for budgeting.

---

## 3. Coordination Rules
1. **Single Source of Truth**: Wherever possible, promote notebook logic into `ehr/` scripts once stabilized; notebooks remain references or for interactive slicing.
2. **Versioning**: Tag every embedding/feature drop with a semantic suffix (e.g., `full_run`, `dry_run`, `v2025_11_04`) and list hyperparameters in the generated `manifest.json`.
3. **Testing**: Before large Slurm jobs, run `--limit` smoke tests locally; for notebooks, add parity assertions (counts, LOS stats) to guard against drift from scripts.
4. **Reporting**: Each day’s primary changes must be summarized under `documents/reports/YYYY-MM-DD/<task>/` with CSV snapshots + JSON metadata.

---

## 4. Near-Term Milestones
| Date Target | Deliverable | Owner |
|-------------|-------------|-------|
| ASAP | Run ModernBERT dry-run (CPU) + full Slurm job; save embeddings + manifest | ddlin |
| This Week | Mirror ModernBERT output paths in Daniel’s Colab notebooks so modeling can consume HPC embeddings | Daniel |
| Next Week | Stand up `ehr/encode_structured_events.py` (HiBEHRT-ready) with analogous Slurm script | ddlin |
| Rolling | Keep Colab notebooks synced (outputs copied into repo) and document GPU credit consumption | Daniel |

---

## 5. Risk & Mitigation
- **Home quota** (50 GB) may block installs → keep caches under repo dir; if quota trips, coordinate before cleaning shared dirs.
- **Notebook drift** vs scripts → parity assertions + regular promotion of stable code into `ehr/`.
- **GPU availability** on free-gpu partition → maintain ability to fall back to Colab for urgent runs; schedule Slurm jobs during off-peak hours when possible.

---

## 6. Reporting Template Reminder
Every agent day should capture:
1. Context sources used (scripts, notebooks, data paths).
2. Concrete changes (code, docs, data artifacts) with file references.
3. Results snapshot table (metrics, counts, or statuses).
4. Next steps + owners.
This keeps both partners and instructors aligned despite the dual-environment setup.
