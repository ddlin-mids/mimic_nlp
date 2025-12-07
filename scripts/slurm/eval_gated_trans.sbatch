#!/bin/bash
#
# Evaluate Gated Fusion (PyTorch) with Transformer Embeddings
#
# Resources: GPU (A30), 8 CPUs, 64GB Memory.
#
# Usage:
#   sbatch scripts/slurm/eval_gated_trans.sbatch
#

#SBATCH --job-name=gate-trans
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/gate_trans_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${SLURM_SUBMIT_DIR}/mlruns"
export EMBEDDING_DIR="${SLURM_SUBMIT_DIR}/data/interim/ehr_long_los/embeddings_transformer"

cd "$SLURM_SUBMIT_DIR"

echo "===================================================================="
echo "Job Started: $(date)"
echo "Script: src/models/pytorch/train_gated_fusion.py"
echo "Embedding Dir: ${EMBEDDING_DIR}"
echo "===================================================================="

echo ">>> Running Gated Fusion (PyTorch) with Transformer Embeddings..."
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
