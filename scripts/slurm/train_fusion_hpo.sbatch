#!/bin/bash
#
# Hyperparameter Optimization for Multimodal Readmission
#
# Resources: 16 CPUs, 64GB Memory.
#
# Usage:
#   sbatch scripts/slurm/train_fusion_hpo.sbatch <xgboost|mlp>
#

#SBATCH --job-name=hpo-fusion
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/hpo_fusion_%j.log

set -euo pipefail

MODEL_TYPE=${1:-xgboost} # Default to xgboost if not provided

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${SLURM_SUBMIT_DIR}/mlruns"

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Model: $MODEL_TYPE"
echo "MLflow URI: $MLFLOW_TRACKING_URI"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

echo ">>> Starting HPO for $MODEL_TYPE..."
uv run python src/models/train_fusion_hpo.py --model "$MODEL_TYPE" --trials 20

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
