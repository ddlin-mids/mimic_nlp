# MIMIC‑IV 30‑Day Readmission — Project Plan & Agent Brief

**Owners:** David Lin (ddlin), Daniel C.
**Cluster:** UCI HPC3
**Primary repos:**

* Class repo: `refs/2025-fall-main/`
* Research baseline: `refs/readmit-stgnn/`
* This project: `mimic_nlp/` (under `/share/c/l/p/d/mids/datasci-266/`)

**Primary data roots (read‑only):**

* `/share/c/l/p/d/mids/datasci-266/mimic_nlp/physionet.org/files/`

  * `mimiciv/3.1` (core tables)
  * `mimic-iv-note/2.2` (clinical & discharge notes)
  * `mimic-iv-ed/2.2` (ED)
  * `mimic-iv-ext-22mcts/1.0.0` (timestamp ext)
  * `mimic-cxr/2.1.0` → **use reports only**: `mimic-cxr-reports.zip`

---

## 0) Objectives (agreed 2025‑10‑27)

* **Task:** Admission‑level binary classification — predict **30‑day readmission** at time of discharge.
* **Exclusions:**

  * Exclude admissions where **in‑hospital death** occurred.
  * Only count readmissions **within 30 days** of index **discharge**.
  * Exclude specific **discharge locations** (e.g., death, hospice, etc.) and **admission types** (e.g., newborn) per config.
* **Modalities:**

  * **Structured EHR (labs, meds, procedures, diagnoses, vitals)** → encode with **HiBEHRT‑style hierarchical Transformer** (patient → admission → events).
  * **Clinical text (discharge notes, clinical notes, radiology reports)** → encode with **BioClinical *Modern BERT*** (long‑context variant or sliding‑window pooling). **No imaging**.
* **Analysis:** Global metrics (AUROC/AUPRC), calibration, and **vulnerable subgroup discovery** (age bands, sex, insurance, Elixhauser, LOS, service line, discharge disposition, etc.).

---

## 1) Repository Layout (single‑repo, modular)

```
mimic_nlp/
├── pyproject.toml            # Python 3.10+, uv‑managed env
├── uv.lock
├── README.md
├── Makefile
├── .env.example              # runtime env vars
├── configs/                  # Hydra configs
│   ├── config.yaml
│   ├── data.yaml
│   ├── cohort.yaml
│   ├── model/
│   │   ├── hibehrt.yaml
│   │   └── bioclinical_modern_bert.yaml
│   └── train.yaml
├── notebooks/
│   ├── eda.ipynb
│   └── notebooks_dc/        # Daniel’s prior work (read‑only)
├── refs/
│   ├── readmit-stgnn/       # reference code & papers (read‑only)
│   └── 2025-fall-main/      # class repo (read‑only)
├── scripts/
│   ├── submit_train.sbatch
│   ├── submit_eval.sbatch
│   └── sync_env.sh
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── io.py
│   │   ├── time.py
│   │   ├── text.py
│   │   └── split.py
│   ├── data/
│   │   ├── build_cohort.py
│   │   ├── admissions.py
│   │   ├── labels.py
│   │   └── features_structured.py
│   ├── text/
│   │   ├── cxr_reports.py
│   │   ├── discharge_notes.py
│   │   └── encoder_biocl_bert.py
│   ├── models/
│   │   ├── hibehrt.py
│   │   ├── fusion.py
│   │   └── heads.py
│   ├── train.py
│   └── eval.py
├── tests/
│   ├── test_cohort.py
│   ├── test_labels.py
│   ├── test_text_encoder.py
│   └── test_model_shapes.py
├── artifacts/                # git‑ignored (checkpoints, logs, metrics)
└── data/                     # git‑ignored (intermediate parquet)
```

---

## 2) Environment & Tooling

* **Package manager:** `uv` (locked via `uv.lock`).
* **Core libs:** `torch`, `transformers`, `datasets`, `pytorch-lightning` or `lightning`, `scikit-learn`, `pandas`/`polars`, `pyarrow`, `hydra-core`, `omegaconf`, `rich`, `tqdm`.
* **Optional:** `wandb` (off by default), `matplotlib`, `seaborn` (for EDA only).

**Install**

```bash
uv sync
```

**Run tests**

```bash
uv run python -m pytest -q
```

**Common dev loop**

```bash
uv run python -m src.data.build_cohort + uv run python -m src.train
```

---

## 3) Data Contracts

**Inputs (read‑only):** PhysioNet folders under `physionet.org/files/` as listed above.

**Intermediate tables (parquet under `data/`):**

* `data/cohort.parquet` — one row per admission (hadm_id), filtered by cohort rules.
* `data/labels.parquet` — `hadm_id`, `readmit_30d` (0/1), `readmit_hadm_id`, `days_to_readmit`.
* `data/structured_events.parquet` — exploded time‑stamped events with normalized codes (LOINC/ICD/ATC if available), columns: `subject_id`, `hadm_id`, `admittime`, `charttime`, `code`, `value`, `value_num`, `unit`.
* `data/text_discharge.parquet` — discharge notes with `hadm_id`, `note_text`, `len_tokens`.
* `data/text_cxr.parquet` — radiology reports (study‑level mapped to `hadm_id` when possible).
* `data/splits.parquet` — `hadm_id`, `split` (train/val/test), patient‑wise split (no leakage across `subject_id`).

---

## 4) Cohort & Label Logic

**Index admission definition:** an admission in `hosp.admissions` meeting:

* Not died in hospital (`hospital_expire_flag = 0`).
* Discharge disposition not in excluded set (config `cohort.discharge_exclude`).
* Admission type not in excluded set (config `cohort.admission_type_exclude`).

**Outcome (label):**

* `readmit_30d = 1` if there exists a later admission for the **same patient** starting **>0 and ≤30 days** after index **dischtime**.
* Else `0`.
* If patient **dies in‑hospital** on index admission → **drop** that index admission.
* If patient is discharged to **death/hospice** and policy says exclude → drop by config.

**Splits:**

* `subject_id` stratified across **train/val/test** (e.g., 70/15/15). Maintain outcome prevalence across splits.

---

## 5) Feature Engineering

### 5.1 Structured EHR → HiBEHRT encoder

* **Hierarchy:** patient → admission (time‑ordered) → event sequences (codes/time/value bins).
* **Event types:** labs, vitals, procedures, diagnoses, meds (as available). Map to shared token space with type embeddings.
* **Temporal encoding:** relative time since admission start; optional bucketing.
* **Numerical values:** discretize (quantile bins) **and** pass normalized z‑score as value embedding.
* **Config:** `configs/model/hibehrt.yaml` controls vocab sizes, embedding dims, Transformer layers, max events/admission.

### 5.2 Text (discharge notes, clinical notes, CXR reports)

* **Encoder:** BioClinical **Modern BERT** (long context) or fallback to BioClinicalBERT with **sliding window** + attention pooling.
* **Token budget:** `max_seq_len` (e.g., 4096 if long‑context model available; else 512 with stride 128 aggregation).
* **Outputs:** admission‑level pooled embedding per note type; fuse via attention or gated sum.

### 5.3 Modality Fusion

* Concatenate **HiBEHRT admission embedding** with **text embedding(s)**.
* Apply **fusion MLP + dropout** → classification head.
* Calibrated probabilities via **temperature scaling** on validation set.

---

## 6) Training & Evaluation

**Loss:** BCEWithLogitsLoss (class weighting optional).
**Optimizer:** AdamW; **Scheduler:** cosine/plateau.
**Metrics:** AUROC, AUPRC (primary), F1@threshold, Brier, ECE.
**Calibration:** temperature scaling; **PR/ROC** plots.
**Subgroup/Vulnerability:** evaluate by age bands, sex, insurance, Elixhauser count, LOS, discharge service, admission type.

**Reproducibility:** set `seed`, deterministic flags; log versions & configs.

---

## 7) Configs (Hydra)

`configs/config.yaml`

```yaml
defaults:
  - data: data
  - cohort: cohort
  - model: hibehrt
  - train: train
  - _self_

project_root: ${oc.env:PROJECT_ROOT, .}
artifacts_dir: ${project_root}/artifacts
```

`configs/data.yaml`

```yaml
physionet_root: /share/c/l/p/d/mids/datasci-266/mimic_nlp/physionet.org/files
mimiciv_version: 3.1
note_version: 2.2
ed_version: 2.2
ext_version: 1.0.0
out_dir: ${project_root}/data
```

`configs/cohort.yaml`

```yaml
admission_type_exclude: ["NEWBORN"]
discharge_exclude: ["DIED", "HOSPICE", "LEFT AGAINST MEDICAL ADVICE"]
readmit_window_days: 30
min_gap_hours: 0
split:
  seed: 42
  ratios: {train: 0.7, val: 0.15, test: 0.15}
```

`configs/model/hibehrt.yaml`

```yaml
name: hibehrt
struct:
  d_model: 256
  n_heads: 4
  n_layers: 4
  dropout: 0.1
  max_events_per_adm: 2048
text:
  model_name: ${oc.env:TEXT_MODEL, "emilyalsentzer/Bio_ClinicalBERT"}
  max_len: 512
fusion:
  d_hidden: 256
```

`configs/train.yaml`

```yaml
batch_size: 8
max_epochs: 10
lr: 2e-4
weight_decay: 0.01
scheduler: cosine
precision: bf16
num_workers: 8
use_wandb: false
```

---

## 8) CLI & Makefile

**Makefile**

```Makefile
.PHONY: env test cohort train eval clean

env:
	uv sync

test:
	uv run python -m pytest -q

cohort:
	uv run python -m src.data.build_cohort

train:
	uv run python -m src.train

eval:
	uv run python -m src.eval

clean:
	rm -rf artifacts/* data/*.parquet
```

**Common runs**

```bash
make env
make cohort
make train
make eval
```

---

## 9) HPC3 Slurm Templates

`scripts/submit_train.sbatch`

```bash
#!/bin/bash
#SBATCH -A <account>
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -t 24:00:00
#SBATCH -J readmit-train
#SBATCH -o artifacts/slurm/%x-%j.out

module load cuda/12
source ~/.bashrc
cd $SLURM_SUBMIT_DIR
uv run python -m src.train
```

`scripts/submit_eval.sbatch`

```bash
#!/bin/bash
#SBATCH -A <account>
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -t 04:00:00
#SBATCH -J readmit-eval
#SBATCH -o artifacts/slurm/%x-%j.out

module load cuda/12
source ~/.bashrc
cd $SLURM_SUBMIT_DIR
uv run python -m src.eval
```

---

## 10) Module Stubs (high‑level)

**`src/data/build_cohort.py`**

* Load admissions; apply `cohort.yaml` filters.
* Build labels via patient‑level next‑admission within 30 days.
* Save `cohort.parquet`, `labels.parquet`, `splits.parquet`.

**`src/data/features_structured.py`**

* Extract time‑stamped events (labs/vitals/meds/diagnoses/procedures).
* Normalize codes → token ids; bucket timestamps; quantize values.

**`src/text/encoder_biocl_bert.py`**

* Tokenize notes with long‑context model or sliding‑window aggregator; output per‑admission embeddings.

**`src/models/hibehrt.py`**

* Hierarchical encoders (admission‑level Transformer pooled to embedding).

**`src/models/fusion.py`**

* Fuse structured + text embeddings; apply classification head.

**`src/train.py` / `src/eval.py`**

* Lightning training loop; checkpointing; metrics; subgroup reports under `artifacts/metrics/*.json` and plots under `artifacts/plots/`.

---

## 11) Data Mapping Notes

* **Radiology reports:** unpack `mimic-cxr-reports.zip`; map `study_id`/`subject_id` to nearest `hadm_id` by time when explicit mapping exists.
* **Discharge vs general notes:** prefer **discharge summary** as the index‑time text; optionally concatenate last N notes within X hours pre‑discharge.
* **Time windows:** use only **pre‑discharge** data for index admission to avoid label leakage.

---

## 12) Vulnerable‑Type Analysis (Who is most at risk?)

* Train global model, then:

  * Compute subgroup metrics (AUROC/AUPRC) across demographics & clinical strata.
  * SHAP/Integrated Gradients on fusion layer to interpret drivers.
  * Report **top‑5 highest‑risk strata** (by calibrated risk lift vs population baseline).

Artifacts:

* `artifacts/metrics/subgroups.csv`
* `artifacts/plots/calibration_{split}.png`
* `artifacts/plots/shap_summary.png`

---

## 13) Guardrails & QA

* **Leakage checks:**

  * Ensure **test subjects** do not appear in train/val.
  * Ensure only **pre‑discharge** events/notes are used for the index admission.
* **Completeness:** assert row counts & nulls per intermediate table.
* **Determinism:** seed, hash of cohort SQL, config fingerprints.

---

## 14) Agent (Codex) Task List

> The following atomic tasks are designed for a code agent. Each step is self‑verifiable with tests.

1. **Scaffold repo** from the layout above; generate `pyproject.toml` with required deps; add `.gitignore` for `data/` and `artifacts/`.
2. **Implement cohort builder** (`src/data/build_cohort.py`) reading MIMIC tables, applying config filters, writing `cohort.parquet`, `labels.parquet`, `splits.parquet`.

   * *Acceptance:* `tests/test_cohort.py` & `tests/test_labels.py` pass with small mocked tables.
3. **Implement structured featurizer** (`src/data/features_structured.py`) to build tokenized event sequences with time/value buckets.

   * *Acceptance:* `tests/test_model_shapes.py` validates max sequence sizes and vocab bounds.
4. **Implement text loader + encoder** (`src/text/encoder_biocl_bert.py`) with sliding window pooling; configurable `TEXT_MODEL`.

   * *Acceptance:* `tests/test_text_encoder.py` validates pooling shapes for synthetic long notes.
5. **Implement HiBEHRT** (`src/models/hibehrt.py`) and **fusion head** (`src/models/fusion.py`).
6. **Training loop** (`src/train.py`) with Lightning, checkpointing best AUROC on `val`.
7. **Evaluation** (`src/eval.py`) producing overall and subgroup metrics; save calibration curves.
8. **Slurm scripts** under `scripts/`; verify `sbatch scripts/submit_train.sbatch` launches.
9. **Docs** (`README.md`) with quickstart, data contracts, and command examples.

---

## 15) Quickstart

```bash
# 0) Environment
uv sync

# 1) Build cohort & labels (reads PhysioNet dirs; writes to data/)
make cohort

# 2) Train baseline fusion model
make train

# 3) Evaluate + subgroup analysis
make eval
```

**Env vars** (`.env.example`):

```
PROJECT_ROOT=/share/c/l/p/d/mids/datasci-266/mimic_nlp
TEXT_MODEL=emilyalsentzer/Bio_ClinicalBERT
WANDB_DISABLED=true
```

---

## 16) Stretch Goals

* Swap note encoder to **longformer/modern‑bert clinical** once available on cluster.
* Add **ED visits** features from `mimic-iv-ed` preceding index admission.
* Contrastive pretraining between structured and text embeddings.
* Risk decomposition report for care teams (top features + actionable factors).

---

## 17) Open Questions (track in Issues)

* Preferred long‑context model checkpoint for “BioClinical Modern BERT” on HPC3? (temporary sliding window in place)
* Final exclude lists for discharge disposition/admission type (align with lecture rubric).
* Radiology report → hadm mapping policy when multiple studies surround admission.

---

*End of plan (v0.1).*


https://pmc.ncbi.nlm.nih.gov/articles/PMC11073780/

I organized everything into a clean, ready-to-implement plan with repo structure, configs, cohort rules, model choices (HiBEHRT + BioClinical “modern” BERT), Slurm templates, Makefile targets, and a step-by-step “Codex agent” task list.

Open the canvas on the right: **“MIMIC-IV Readmission Project — Dev Plan & Codex Agent Brief (v0.1)”**.

If you want, I can also drop in:

* a `pyproject.toml` with pinned deps,
* stub Python files for each module,
* and a tiny mock dataset + pytest scaffolding so the agent can TDD its way through Tasks 1–4.

Say the word and I’ll add those directly.
