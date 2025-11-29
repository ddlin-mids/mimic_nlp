#!/bin/bash
#
# Embed Clinical Notes (Discharge & Radiology) using BioClinical ModernBERT
#
# Resources: GPU (A30), 8 CPUs, 64GB Memory.
#
# Usage:
#   sbatch scripts/slurm/embed_notes_bioclinical_mbert.sbatch
#

#SBATCH --job-name=embed-mbert
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/embed_notes_bioclinical_mbert_%j.log

set -euo pipefail

# Environment Setup
export PYTHONUNBUFFERED=1
export HF_HOME="/share/crsp/lab/pkaiser/ddlin/mids/datasci-266/mimic_nlp/.cache/huggingface"

# Define Paths
# Cohort file for filtering
COHORT_PATH="data/interim/readmit_analysis/long_los_cohort.csv"

# Output directory for embeddings
OUTPUT_DIR="data/interim/embeddings/notes"

# Input MIMIC-IV Note files (gzipped CSVs)
DISCHARGE_PATH="physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"
RADIOLOGY_PATH="physionet.org/files/mimic-iv-note/2.2/note/radiology.csv.gz"

# Model Config
MODEL="thomas-sounack/bioclinical-modernbert-base"
BATCH_SIZE=32
MAX_LEN=8192
DTYPE="bfloat16"

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Cohort: $COHORT_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Model: $MODEL ($DTYPE)"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

# 1. Embed Discharge Summaries
echo ""
echo ">>> STEP 1: Embedding Discharge Summaries"
uv run python ehr/embed_notes_bioclinical_mbert.py \
  --input-path "$DISCHARGE_PATH" \
  --cohort-path "$COHORT_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --text-column "text" \
  --output-prefix "discharge_summary" \
  --model-name "$MODEL" \
  --batch-size $BATCH_SIZE \
  --max-length $MAX_LEN \
  --dtype "$DTYPE"

# 2. Embed Radiology Reports
echo ""
echo ">>> STEP 2: Embedding Radiology Reports"
uv run python ehr/embed_notes_bioclinical_mbert.py \
  --input-path "$RADIOLOGY_PATH" \
  --cohort-path "$COHORT_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --text-column "text" \
  --output-prefix "radiology_report" \
  --model-name "$MODEL" \
  --batch-size $BATCH_SIZE \
  --max-length $MAX_LEN \
  --dtype "$DTYPE"

echo ""
echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
