#!/bin/bash
#
# Train Refined GNN (GraphSAGE) with Transformer Embeddings & Structure
#
# Usage:
#   sbatch scripts/slurm/train_gnn_refined.sbatch
#

#SBATCH --job-name=gnn-refined
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/gnn_refined_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${SLURM_SUBMIT_DIR}/mlruns"

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

EMB_PATH="data/interim/ehr_long_los/embeddings/structured_ehr_embeddings.npz"
COHORT="data/interim/readmit_analysis/long_los_cohort.csv"
# We don't strictly need mapping_path for the new mode, but good to have
MAPPING="data/interim/ehr_long_los/graph/graph_node_map.csv"
GRAPH_DIR="data/interim/ehr_long_los/graph_refined"
MODEL_DIR="data/interim/ehr_long_los/models_gnn_refined"

# 1. Build Graph
echo ">>> Building Refined Graph (KNN on Embeddings)..."
uv run python ehr/preprocess_graph.py \
    --use_embeddings "$EMB_PATH" \
    --cohort_path "$COHORT" \
    --mapping_path "$MAPPING" \
    --save_dir "$GRAPH_DIR" \
    --k 15

# 2. Train GNN
echo ">>> Training GNN with Refined Graph..."
uv run python src/models/pytorch/train_gnn.py \
    --feature_path "$EMB_PATH" \
    --graph_path "$GRAPH_DIR/graph_pyg.pt" \
    --cohort_path "$COHORT" \
    --save_dir "$MODEL_DIR" \
    --hidden_dim 128 \
    --lr 0.0005 \
    --dropout 0.5 \
    --epochs 300

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
