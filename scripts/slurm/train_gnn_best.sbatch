#!/bin/bash
#
# Train Best Multimodal GNN (from HPO)
#
# Best Params: hidden_dim=64, num_layers=3, dropout=0.29, lr=0.00019, wd=8e-5
#
# Usage:
#   sbatch scripts/slurm/train_gnn_best.sbatch
#

#SBATCH --job-name=gnn-best
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/gnn_best_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${SLURM_SUBMIT_DIR}/mlruns"

cd "$SLURM_SUBMIT_DIR"

GRAPH_PATH="data/interim/ehr_long_los/graph_multimodal/graph_pyg.pt"
COHORT="data/interim/readmit_analysis/long_los_cohort.csv"
SAVE_DIR="data/interim/ehr_long_los/models_gnn_best"

echo ">>> Training Best GNN Config..."
uv run python src/models/pytorch/train_gnn.py \
    --graph_path "$GRAPH_PATH" \
    --cohort_path "$COHORT" \
    --save_dir "$SAVE_DIR" \
    --hidden_dim 64 \
    --num_layers 3 \
    --use_residual \
    --lr 0.00018845 \
    --dropout 0.28695 \
    --weight_decay 0.00007945 \
    --epochs 300

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
