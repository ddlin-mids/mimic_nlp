#!/bin/bash
#
# Train Late Fusion PyTorch Model
#
# Resources: GPU (A30), 4 CPUs, 64GB Memory.
#
# Usage:
#   sbatch scripts/slurm/train_late_fusion.sbatch
#

#SBATCH --job-name=late-fusion
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/late_fusion_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${SLURM_SUBMIT_DIR}/mlruns"

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Script: src/models/pytorch/train_late_fusion.py"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

echo ">>> Starting Training..."
uv run python src/models/pytorch/train_late_fusion.py \
    --lr 5e-4 \
    --dropout 0.5 \
    --weight_decay 0.05 \
    --hidden_dim 128 \
    --epochs 50

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
