#!/bin/bash
#SBATCH --job-name=stgnn_inductive
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/stgnn_inductive_%j.out
#SBATCH --error=logs/stgnn_inductive_%j.err

# STGNN with Inductive Learning - Fixes data leakage from transductive setup
# The previous STGNN achieved 0.999 AUC due to test nodes having edges to train nodes
# This version isolates test nodes properly

set -e

# Create logs directory
mkdir -p logs

echo "=================================================="
echo "STGNN Inductive Learning Pipeline"
echo "Fixing data leakage from transductive setup"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
date

# Step 1: Build inductive graph (train-only graph + inference edges)
echo ""
echo "Step 1: Building inductive patient graph..."
echo "  - Train nodes only have edges to other train nodes"
echo "  - Val/Test nodes connect only to train nodes at inference"
echo ""

uv run python src/models/pytorch/build_patient_graph_inductive.py \
    --features_path data/interim/ehr_long_los/integrated_features.csv \
    --output_path data/interim/ehr_long_los/patient_graph_inductive.pkl \
    --k_neighbors 10

# Step 2: Train STGNN with inductive learning
echo ""
echo "Step 2: Training STGNN with inductive learning..."
echo "  - Training uses only train subgraph"
echo "  - No val/test information leaks during training"
echo ""

uv run python src/models/pytorch/train_stgnn_inductive.py \
    --graph_path data/interim/ehr_long_los/patient_graph_inductive.pkl \
    --output_dir data/interim/ehr_long_los/stgnn_inductive \
    --hidden_dim 128 \
    --n_layers 2 \
    --dropout 0.3 \
    --lr 0.001 \
    --weight_decay 0.01 \
    --epochs 200 \
    --patience 30

echo ""
echo "=================================================="
echo "Pipeline completed!"
echo "Results saved to: data/interim/ehr_long_los/stgnn_inductive/"
echo "=================================================="
date
