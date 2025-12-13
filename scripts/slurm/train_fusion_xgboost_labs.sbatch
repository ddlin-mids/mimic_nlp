#!/bin/bash
#
# Train Fusion XGBoost with Lab Trajectory Features
#
# Tests the impact of adding lab trajectory features to the fusion model.
#
# Resources: 8 CPUs, 64GB Memory.
#
# Usage:
#   sbatch scripts/slurm/train_fusion_xgboost_labs.sbatch
#

#SBATCH --job-name=xgb-labs
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/xgb_labs_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${SLURM_SUBMIT_DIR}/mlruns"

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Script: src/models/train_fusion_xgboost.py (with lab features)"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

# Ensure logs directory exists
mkdir -p logs/slurm

echo ">>> Starting XGBoost Training with Lab Features..."
uv run python src/models/train_fusion_xgboost.py \
    --embedding_dir "data/interim/ehr_long_los/embeddings" \
    --embedding_type GRU

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
