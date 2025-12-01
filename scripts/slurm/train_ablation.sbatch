#!/bin/bash
#
# Run Feature Ablation Experiments (XGBoost)
#
# Resources: 16 CPUs, 64GB Memory.
#
# Usage:
#   sbatch scripts/slurm/train_ablation.sbatch
#

#SBATCH --job-name=ablation-xgb
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/ablation_xgb_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${SLURM_SUBMIT_DIR}/mlruns"

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Script: src/models/train_ablation_xgboost.py"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

echo ">>> Starting Ablation Experiments..."
uv run python src/models/train_ablation_xgboost.py

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
