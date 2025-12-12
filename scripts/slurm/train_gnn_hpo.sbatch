#!/bin/bash
#
# Train Multimodal GNN HPO (Optuna)
#
# Resources: GPU (A30), 8 CPUs, 32GB Memory.
#
# Usage:
#   sbatch scripts/slurm/train_gnn_hpo.sbatch
#

#SBATCH --job-name=gnn-hpo
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/gnn_hpo_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${SLURM_SUBMIT_DIR}/mlruns"

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Script: src/models/pytorch/train_gnn_hpo.py"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

# Use the main graph (will be re-indexed to valid nodes automatically)
GRAPH_PATH="data/interim/ehr_long_los/graph/graph_pyg.pt"
COHORT="data/interim/readmit_analysis/long_los_cohort.csv"
SAVE_DIR="data/interim/ehr_long_los/models_gnn_hpo"

# Verify graph exists
if [ ! -f "$GRAPH_PATH" ]; then
    echo "Error: Graph file not found at $GRAPH_PATH"
    exit 1
fi

mkdir -p "$SAVE_DIR"

echo ">>> Starting GNN HPO Sweep (50 trials)..."
uv run python src/models/pytorch/train_gnn_hpo.py \
    --graph_path "$GRAPH_PATH" \
    --cohort_path "$COHORT" \
    --save_dir "$SAVE_DIR" \
    --epochs 100 \
    --n_trials 50

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
