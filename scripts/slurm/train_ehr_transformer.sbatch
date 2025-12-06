#!/bin/bash
#
# Train EHR Transformer
#
# Resources: GPU (A30), 8 CPUs, 64GB Memory.
#
# Usage:
#   sbatch scripts/slurm/train_ehr_transformer.sbatch
#

#SBATCH --job-name=ehr-trans
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/ehr_trans_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Script: ehr/train_ehr_transformer.py"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

SEQ_DATA="data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_cat_embedding.pkl"
COHORT="data/interim/readmit_analysis/long_los_cohort.csv"
SAVE_DIR="data/interim/ehr_long_los/embeddings_transformer"

echo ">>> Training EHR Transformer..."
uv run python ehr/train_ehr_transformer.py \
    --input_path "$SEQ_DATA" \
    --cohort_path "$COHORT" \
    --save_dir "$SAVE_DIR" \
    --hidden_dim 128 \
    --num_layers 2 \
    --num_heads 4 \
    --epochs 10 \
    --max_len 100

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
