#!/bin/bash
#
# Evaluate Models with Transformer Embeddings
#
# Resources: GPU (A30), 8 CPUs, 64GB Memory.
#
# Usage:
#   sbatch scripts/slurm/evaluate_transformer_fusion.sbatch
#

#SBATCH --job-name=eval-trans
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/eval_trans_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${SLURM_SUBMIT_DIR}/mlruns"
export EMBEDDING_DIR="${SLURM_SUBMIT_DIR}/data/interim/ehr_long_los/embeddings_transformer"

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Embedding Dir: ${EMBEDDING_DIR}"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

# 1. XGBoost Fusion (Runs multiple ablations)
echo ">>> [1/3] Running XGBoost Fusion..."
uv run python src/models/train_fusion_xgboost.py \
    --embedding_dir "${EMBEDDING_DIR}"

# 2. Early Fusion (PyTorch)
echo ">>> [2/3] Running Early Fusion (PyTorch)..."
uv run python src/models/pytorch/train_early_fusion.py \
    --embedding_dir "${EMBEDDING_DIR}" \
    --epochs 50 \
    --lr 1e-4 \
    --pca_components 64

# 3. Gated Fusion (PyTorch)
echo ">>> [3/3] Running Gated Fusion (PyTorch)..."
uv run python src/models/pytorch/train_gated_fusion.py \
    --embedding_dir "${EMBEDDING_DIR}" \
    --epochs 30 \
    --lr 2e-4 \
    --hidden_dim 64 \
    --dropout 0.3 \
    --pca_components 64

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
