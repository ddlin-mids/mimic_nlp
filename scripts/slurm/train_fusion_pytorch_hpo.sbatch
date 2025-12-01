#!/bin/bash
#
# PyTorch HPO for Fusion Readmission Prediction
#
# Resources: GPU (A30), 8 CPUs, 64GB Memory.
#
# Usage:
#   sbatch scripts/slurm/train_fusion_pytorch_hpo.sbatch
#

#SBATCH --job-name=hpo-pt
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/hpo_pytorch_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${SLURM_SUBMIT_DIR}/mlruns"

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Script: src/models/pytorch/train_fusion_pytorch_hpo.py"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

echo ">>> Starting HPO..."
uv run python src/models/pytorch/train_fusion_pytorch_hpo.py --trials 30

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
