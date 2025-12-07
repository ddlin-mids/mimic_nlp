#!/bin/bash
#
# Evaluate XGBoost Fusion with Transformer Embeddings
#
# Resources: GPU (A30), 8 CPUs, 64GB Memory.
#
# Usage:
#   sbatch scripts/slurm/eval_xgboost_trans.sbatch
#

#SBATCH --job-name=xgb-trans
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/xgb_trans_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${SLURM_SUBMIT_DIR}/mlruns"
export EMBEDDING_DIR="${SLURM_SUBMIT_DIR}/data/interim/ehr_long_los/embeddings_transformer"

cd "$SLURM_SUBMIT_DIR"

echo "===================================================================="
echo "Job Started: $(date)"
echo "Script: src/models/train_fusion_xgboost.py"
echo "Embedding Dir: ${EMBEDDING_DIR}"
echo "===================================================================="

echo ">>> Running XGBoost Fusion with Transformer Embeddings..."
uv run python src/models/train_fusion_xgboost.py \
    --embedding_dir "${EMBEDDING_DIR}"

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="

