#!/bin/bash
#SBATCH --job-name=stgnn512
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/stgnn_pca512_%j.log

source ~/.bashrc
cd /share/crsp/lab/pkaiser/ddlin/mids/datasci-266/mimic_nlp

echo "======================================================================"
echo "STGNN Training - PCA 512 Components"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "======================================================================"

mkdir -p logs/slurm

PCA_DATA="data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_pca512.pkl"
if [ ! -f "$PCA_DATA" ]; then
    echo "ERROR: PCA-reduced data not found at $PCA_DATA"
    exit 1
fi

# Experiment 1: STGNN EHR-only (GraphSAGE)
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
    --run_name "stgnn_pca512_ehr_only"

# Experiment 2: STGNN + Text (Gated Fusion)
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
    --run_name "stgnn_pca512_gated_fusion"

# Experiment 3: STGNN with GAT convolution
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
    --run_name "stgnn_pca512_gat_gated"

echo ""
echo "======================================================================"
echo "STGNN PCA 512 Training Complete"
echo "Date: $(date)"
echo "======================================================================"
