#!/bin/bash
#
# Train PyTorch MLP for Fusion Readmission Prediction
#
# Resources: GPU (A30), 8 CPUs, 64GB Memory.
#
# Usage:
#   sbatch scripts/slurm/train_fusion_pytorch.sbatch
#

#SBATCH --job-name=train-fusion-pt
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/train_fusion_pytorch_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${SLURM_SUBMIT_DIR}/mlruns"

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Script: src/models/pytorch/train_fusion_pytorch.py"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

echo ">>> Starting Training..."
# Initial run with default params to verify stability and performance
uv run python src/models/pytorch/train_fusion_pytorch.py \
    --lr 1e-3 \
    --weight_decay 1e-2 \
    --dropout 0.4 \
    --warmup_steps 200 \
    --epochs 50

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
