#!/bin/bash
#
# Train Stacking Ensemble
#
# Combines XGBoost (Struct), XGBoost (Full), and Gated Fusion
# with out-of-fold predictions and a meta-learner.
#
# Resources: GPU (A30), 8 CPUs, 64GB Memory.
#
# Usage:
#   sbatch scripts/slurm/train_stacking_ensemble.sbatch
#

#SBATCH --job-name=stack-ens
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/stack_ens_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${SLURM_SUBMIT_DIR}/mlruns"

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Script: src/models/train_stacking_ensemble.py"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

# Ensure logs directory exists
mkdir -p logs/slurm

echo ">>> Starting Stacking Ensemble Training..."
uv run python src/models/train_stacking_ensemble.py \
    --embedding_dir "data/interim/ehr_long_los/embeddings" \
    --n_folds 5 \
    --output_dir "data/interim/ehr_long_los/ensemble" \
    --run_name "stacking_ensemble_v1"

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
