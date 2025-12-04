#!/bin/bash
#
# Train Attention Fusion Models (Raw vs PCA)
#
# Resources: GPU (A30), 8 CPUs, 64GB Memory.
#
# Usage:
#   sbatch scripts/slurm/train_attention_fusion.sbatch <pca_components>
#

#SBATCH --job-name=attn-fusion
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/attn_fusion_%j.log

set -euo pipefail

PCA=${1:-0} # Default to 0 (Raw)

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${SLURM_SUBMIT_DIR}/mlruns"

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Script: src/models/pytorch/train_attention_fusion.py"
echo "PCA Components: $PCA"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

echo ">>> Starting Training..."
uv run python src/models/pytorch/train_attention_fusion.py \
    --lr 2e-4 \
    --dropout 0.3 \
    --embed_dim 64 \
    --num_heads 4 \
    --epochs 30 \
    --pca_components "$PCA"

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="

