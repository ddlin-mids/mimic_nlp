#!/bin/bash
#
# Train Fusion XGBoost (Transformer Embeddings)
#
# Resources: 8 CPUs, 32GB Memory.
#
# Usage:
#   sbatch scripts/slurm/train_fusion_xgboost.sbatch
#

#SBATCH --job-name=xgb-fus
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/xgb_fus_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${SLURM_SUBMIT_DIR}/mlruns"

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Script: src/models/train_fusion_xgboost.py"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

echo ">>> Starting XGBoost Training..."
uv run python src/models/train_fusion_xgboost.py \
    --embedding_dir "data/interim/ehr_long_los/embeddings_transformer" \
    --embedding_type Transformer

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
