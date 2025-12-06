#!/bin/bash
#
# Train Temporal Attention Fusion (Plan 1)
#
# Resources: GPU (A30), 8 CPUs, 64GB Memory.
#
# Usage:
#   sbatch scripts/slurm/train_temporal_attention_fusion.sbatch
#

#SBATCH --job-name=temp-attn
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/temp_attn_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${SLURM_SUBMIT_DIR}/mlruns"

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Script: src/models/pytorch/train_temporal_attention_fusion.py"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

echo ">>> Starting Training..."
uv run python src/models/pytorch/train_temporal_attention_fusion.py \
    --lr 2e-4 \
    --dropout 0.3 \
    --embed_dim 64 \
    --num_heads 4 \
    --epochs 30 \
    --pca_components 64

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
