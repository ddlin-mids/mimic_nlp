#!/bin/bash
#
# Hyperparameter Optimization for Multimodal Readmission (Transformer EHR)
#
# Resources: 16 CPUs, 64GB Memory.
#
# Usage:
#   sbatch scripts/slurm/train_fusion_hpo_trans.sbatch
#

#SBATCH --job-name=hpo-fusion-trans
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/hpo_fusion_trans_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${SLURM_SUBMIT_DIR}/mlruns"

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Model: xgboost (Transformer EHR)"
echo "MLflow URI: $MLFLOW_TRACKING_URI"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

echo ">>> Starting HPO for xgboost with Transformer embeddings..."
uv run python src/models/train_fusion_hpo.py \
    --trials 20 \
    --embedding_dir "data/interim/ehr_long_los/embeddings_transformer"

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="

