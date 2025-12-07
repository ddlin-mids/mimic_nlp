#!/bin/bash
#
# Train Multimodal GNN (GraphSAGE) with Transformer + Text Embeddings
#
# Resources: GPU (A30), 8 CPUs, 32GB Memory.
#
# Usage:
#   sbatch scripts/slurm/train_gnn_multimodal.sbatch
#

#SBATCH --job-name=gnn-multi
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/gnn_multi_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${SLURM_SUBMIT_DIR}/mlruns"

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Script: src/models/pytorch/train_gnn.py"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

# Input paths (from preprocess_graph_multimodal.sbatch)
GRAPH_PATH="data/interim/ehr_long_los/graph_multimodal/graph_pyg.pt"
COHORT="data/interim/readmit_analysis/long_los_cohort.csv"
SAVE_DIR="data/interim/ehr_long_los/models_gnn_multimodal"

echo "Graph Path: $GRAPH_PATH"
echo "Cohort Path: $COHORT"
echo "Save Dir: $SAVE_DIR"

# Verify graph exists
if [ ! -f "$GRAPH_PATH" ]; then
    echo "Error: Graph file not found at $GRAPH_PATH"
    echo "Please run scripts/slurm/preprocess_graph_multimodal.sbatch first."
    exit 1
fi

echo ">>> Training Multimodal GraphSAGE..."
uv run python src/models/pytorch/train_gnn.py \
    --graph_path "$GRAPH_PATH" \
    --cohort_path "$COHORT" \
    --save_dir "$SAVE_DIR" \
    --hidden_dim 128 \
    --num_layers 3 \
    --use_residual \
    --lr 0.0005 \
    --dropout 0.4 \
    --weight_decay 0.0005 \
    --epochs 300

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
