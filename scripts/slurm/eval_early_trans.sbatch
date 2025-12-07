#!/bin/bash
#
# Evaluate Early Fusion (PyTorch) with Transformer Embeddings
#
# Resources: GPU (A30), 8 CPUs, 64GB Memory.
#
# Usage:
#   sbatch scripts/slurm/eval_early_trans.sbatch
#

#SBATCH --job-name=early-trans
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/early_trans_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${SLURM_SUBMIT_DIR}/mlruns"
export EMBEDDING_DIR="${SLURM_SUBMIT_DIR}/data/interim/ehr_long_los/embeddings_transformer"

cd "$SLURM_SUBMIT_DIR"

echo "===================================================================="
echo "Job Started: $(date)"
echo "Script: src/models/pytorch/train_early_fusion.py"
echo "Embedding Dir: ${EMBEDDING_DIR}"
echo "===================================================================="

echo ">>> Running Early Fusion (PyTorch) with Transformer Embeddings..."
uv run python src/models/pytorch/train_early_fusion.py \
    --embedding_dir "${EMBEDDING_DIR}" \
    --epochs 50 \
    --lr 1e-4 \
    --pca_components 64

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
