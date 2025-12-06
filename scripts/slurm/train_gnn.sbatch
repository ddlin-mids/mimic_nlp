#!/bin/bash
#
# Train GNN (GraphSAGE) with Transformer Embeddings
#
# Resources: GPU (A30), 8 CPUs, 32GB Memory.
#
# Usage:
#   sbatch scripts/slurm/train_gnn.sbatch
#

#SBATCH --job-name=gnn-sage
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/gnn_sage_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${SLURM_SUBMIT_DIR}/mlruns"

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Script: src/models/pytorch/train_gnn.py"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

# Using the NEW Transformer embeddings
FEAT_PATH="data/interim/ehr_long_los/embeddings/structured_ehr_embeddings.npz"
# Using the NEW PyG graph
GRAPH_PATH="data/interim/ehr_long_los/graph/graph_pyg.pt"
COHORT="data/interim/readmit_analysis/long_los_cohort.csv"
SAVE_DIR="data/interim/ehr_long_los/models"

echo ">>> Training GraphSAGE..."
uv run python src/models/pytorch/train_gnn.py \
    --feature_path "$FEAT_PATH" \
    --graph_path "$GRAPH_PATH" \
    --cohort_path "$COHORT" \
    --save_dir "$SAVE_DIR" \
    --hidden_dim 128 \
    --lr 0.005 \
    --dropout 0.4 \
    --epochs 200

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="

