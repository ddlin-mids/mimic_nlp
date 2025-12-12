#!/bin/bash
#SBATCH --job-name=stgnn
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/stgnn_%j.log

source ~/.bashrc
cd /share/crsp/lab/pkaiser/ddlin/mids/datasci-266/mimic_nlp

echo "======================================================================"
echo "STGNN Training - Spatiotemporal Graph Neural Network"
echo "Using PCA-reduced EHR features (256 dims) + AMP + Checkpointing"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "======================================================================"

mkdir -p logs/slurm

# Check if PCA data exists
PCA_DATA="data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_pca256.pkl"
if [ ! -f "$PCA_DATA" ]; then
    echo "ERROR: PCA-reduced data not found at $PCA_DATA"
    echo "Please run: sbatch scripts/slurm/reduce_ehr_pca.sbatch first"
    exit 1
fi

# Experiment configurations with PCA-reduced data
# 1. STGNN EHR-only (no text, no graph fusion)
echo ""
echo ">>> Experiment 1: STGNN EHR-only (GConvGRU with GraphSAGE)"
uv run python src/models/pytorch/train_stgnn.py \
    --ehr_seq_path "$PCA_DATA" \
    --cohort_path data/interim/readmit_analysis/long_los_cohort.csv \
    --graph_path data/interim/ehr_long_los/graph/graph_pyg.pt \
    --hidden_dim 128 \
    --num_gru_layers 2 \
    --dropout 0.3 \
    --lr 1e-3 \
    --epochs 100 \
    --patience 15 \
    --max_seq_len 30 \
    --conv_type sage \
    --fusion_type none \
    --use_amp \
    --use_checkpointing \
    --run_name "stgnn_pca256_ehr_only"

# 2. STGNN + Text (Gated Fusion)
echo ""
echo ">>> Experiment 2: STGNN + Text (Gated Fusion)"
uv run python src/models/pytorch/train_stgnn.py \
    --ehr_seq_path "$PCA_DATA" \
    --cohort_path data/interim/readmit_analysis/long_los_cohort.csv \
    --graph_path data/interim/ehr_long_los/graph/graph_pyg.pt \
    --text_dir data/interim/embeddings/notes \
    --hidden_dim 128 \
    --num_gru_layers 2 \
    --dropout 0.3 \
    --lr 1e-3 \
    --epochs 100 \
    --patience 15 \
    --max_seq_len 30 \
    --conv_type sage \
    --fusion_type gated \
    --use_amp \
    --use_checkpointing \
    --run_name "stgnn_pca256_gated_fusion"

# 3. STGNN with GAT convolution (attention-based aggregation)
echo ""
echo ">>> Experiment 3: STGNN with GAT Convolution + Gated Fusion"
uv run python src/models/pytorch/train_stgnn.py \
    --ehr_seq_path "$PCA_DATA" \
    --cohort_path data/interim/readmit_analysis/long_los_cohort.csv \
    --graph_path data/interim/ehr_long_los/graph/graph_pyg.pt \
    --text_dir data/interim/embeddings/notes \
    --hidden_dim 128 \
    --num_gru_layers 2 \
    --dropout 0.3 \
    --lr 1e-3 \
    --epochs 100 \
    --patience 15 \
    --max_seq_len 30 \
    --conv_type gat \
    --fusion_type gated \
    --use_amp \
    --use_checkpointing \
    --run_name "stgnn_pca256_gat_gated"

echo ""
echo "======================================================================"
echo "STGNN Training Complete"
echo "Date: $(date)"
echo "======================================================================"
