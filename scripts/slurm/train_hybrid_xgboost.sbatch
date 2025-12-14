#!/bin/bash
#SBATCH --job-name=hybrid_xgb
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/hybrid_xgb_%j.log

# Hybrid XGBoost with Neural Embeddings
# Combines temporal/fusion embeddings with XGBoost

set -euo pipefail

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${SLURM_SUBMIT_DIR}/mlruns"

EHR_PCA_DIM=${1:-64}

echo "===================================================================="
echo "Hybrid XGBoost Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "EHR PCA Dim: ${EHR_PCA_DIM}"
echo "Start time: $(date)"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/slurm results

uv run python src/models/train_hybrid_xgboost.py --ehr_pca_dim ${EHR_PCA_DIM}

echo ""
echo "===================================================================="
echo "Training complete!"
echo "End time: $(date)"
echo "===================================================================="
