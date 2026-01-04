#!/bin/bash
#SBATCH --job-name=stgnn_inductive_cpu
#SBATCH --partition=free
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/stgnn_inductive_cpu_%j.out
#SBATCH --error=logs/stgnn_inductive_cpu_%j.err

# STGNN with Inductive Learning - CPU version (GPU has PyG compatibility issues)

set -e
mkdir -p logs

echo "=================================================="
echo "STGNN Inductive Learning (CPU)"
echo "=================================================="
date

# Skip graph building if already exists
if [ ! -f data/interim/ehr_long_los/patient_graph_inductive.pkl ]; then
    echo "Building inductive graph..."
    uv run python src/models/pytorch/build_patient_graph_inductive.py \
        --features_path data/interim/ehr_long_los/integrated_features.csv \
        --output_path data/interim/ehr_long_los/patient_graph_inductive.pkl \
        --k_neighbors 10
fi

# Train with full epochs
echo ""
echo "Training STGNN with inductive learning..."
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
echo "=================================================="
date
