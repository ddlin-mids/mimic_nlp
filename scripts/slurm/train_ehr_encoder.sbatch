#!/bin/bash
#
# Train EHR Encoder (GRU) to generate embeddings for the long-LOS cohort.
#
# Resources: GPU (A30), 8 cores, 64GB memory.
#
# Usage:
#   sbatch scripts/slurm/train_ehr_encoder.sbatch
#

#SBATCH --job-name=train-ehr-enc
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/train_ehr_encoder_%j.log

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# Define paths
INPUT_PKL="data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_cat_embedding.pkl"
COHORT_CSV="data/interim/readmit_analysis/long_los_cohort.csv"
SAVE_DIR="data/interim/ehr_long_los/embeddings"

echo "Starting EHR Encoder training..."
echo "Working dir: $(pwd)"
echo "Input pickle: $INPUT_PKL"
echo "Cohort CSV: $COHORT_CSV"
echo "Timestamp: $(date)"

# Create output directory
mkdir -p "$SAVE_DIR"

# Run training
uv run python ehr/train_ehr_encoder.py \
  --input_path "$INPUT_PKL" \
  --cohort_path "$COHORT_CSV" \
  --save_dir "$SAVE_DIR" \
  --hidden_dim 128 \
  --batch_size 64 \
  --epochs 10 \
  --max_len 100

echo "Completed EHR Encoder training."
