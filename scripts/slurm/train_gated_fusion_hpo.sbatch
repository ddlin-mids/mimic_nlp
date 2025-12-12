#!/bin/bash
#
# Tune Gated Fusion (HPO)
#
# Resources: GPU (A30), 8 CPUs, 32GB Memory.
#
# Usage:
#   sbatch scripts/slurm/train_gated_fusion_hpo.sbatch
#

#SBATCH --job-name=gate-hpo
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/gate_hpo_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${SLURM_SUBMIT_DIR}/mlruns"

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Script: src/models/pytorch/train_gated_fusion_hpo.py"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

echo ">>> Starting HPO Sweep with Transformer embeddings (50 trials)..."
uv run python src/models/pytorch/train_gated_fusion_hpo.py \
    --trials 50 \
    --pca 128 \
    --embedding_dir "data/interim/ehr_long_los/embeddings_transformer"

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
