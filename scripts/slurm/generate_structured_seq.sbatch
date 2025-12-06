#!/bin/bash
#
# Generate Structured EHR Embeddings (Full Sequence)
#
# Resources: GPU (A30), 8 CPUs, 64GB Memory.
#
# Usage:
#   sbatch scripts/slurm/generate_structured_seq.sbatch
#

#SBATCH --job-name=gen-ehr-seq
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/gen_ehr_seq_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Script: ehr/train_ehr_encoder.py"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

# Paths match original preprocessing
SEQ_DATA="data/interim/ehr_long_los/sequences.pkl"
COHORT="data/interim/readmit_analysis/long_los_cohort.csv"
SAVE_DIR="data/interim/ehr_long_los/embeddings"

echo ">>> Generating Sequential Embeddings..."
uv run python ehr/train_ehr_encoder.py \
    --input_path "$SEQ_DATA" \
    --cohort_path "$COHORT" \
    --save_dir "$SAVE_DIR" \
    --hidden_dim 128 \
    --batch_size 64 \
    --epochs 5 \
    --max_len 100 \
    --model_type gru

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
