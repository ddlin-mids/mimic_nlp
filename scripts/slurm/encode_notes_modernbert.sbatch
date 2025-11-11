#!/bin/bash
#SBATCH -J modernbert-encode
#SBATCH -p free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/modernbert_encode_%j.log

# Harden execution
set -euo pipefail

REPO_ROOT="/share/crsp/lab/pkaiser/ddlin/mids/datasci-266/mimic_nlp"
cd "$REPO_ROOT"

# Allow callers to override these via environment.
INPUT_PATH="${MODERNBERT_INPUT_PATH:-notebooks/notebooks_dc/_data/processed/02_link_notes_to_admissions/labeled_admissions_with_discharge_and_radiology.csv}"
OUTPUT_DIR="${MODERNBERT_OUTPUT_DIR:-notebooks/notebooks_dc/_data/processed/04B_encode_notes_bioclinical_modernbert/full_run}"
BATCH_SIZE="${MODERNBERT_BATCH_SIZE:-16}"
MAX_LENGTH="${MODERNBERT_MAX_LENGTH:-8192}"
DTYPE="${MODERNBERT_DTYPE:-float32}"

mkdir -p "$(dirname "$OUTPUT_DIR")"

# Cache locations stay in the workspace so we avoid permission issues on shared /home.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.uv-cache}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"

echo "[`date`] syncing virtual environment..."
uv sync --frozen

echo "[`date`] launching ModernBERT encoder..."
uv run python ehr/encode_notes_modernbert.py \
  --input-path "$INPUT_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --max-length "$MAX_LENGTH" \
  --dtype "$DTYPE"

echo "[`date`] encoding completed."
