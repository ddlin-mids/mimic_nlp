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
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "======================================================================"

# Experiment configurations
# 1. STGNN EHR-only (no text, no graph fusion)
echo ""
echo ">>> Experiment 1: STGNN EHR-only (GConvGRU with GraphSAGE)"
uv run python src/models/pytorch/train_stgnn.py \
    --ehr_seq_path data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_cat_embedding.pkl \
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
    --run_name "stgnn_ehr_only"

# 2. STGNN + Text (Gated Fusion)
echo ""
echo ">>> Experiment 2: STGNN + Text (Gated Fusion)"
uv run python src/models/pytorch/train_stgnn.py \
    --ehr_seq_path data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_cat_embedding.pkl \
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
    --run_name "stgnn_gated_fusion"

# 3. STGNN Multimodal (Two-stream)
echo ""
echo ">>> Experiment 3: STGNN Multimodal (Two-stream)"
uv run python src/models/pytorch/train_stgnn.py \
    --ehr_seq_path data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_cat_embedding.pkl \
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
    --fusion_type multimodal \
    --run_name "stgnn_multimodal"

# 4. STGNN with GAT convolution (attention-based aggregation)
echo ""
echo ">>> Experiment 4: STGNN with GAT Convolution"
uv run python src/models/pytorch/train_stgnn.py \
    --ehr_seq_path data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_cat_embedding.pkl \
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
    --run_name "stgnn_gat_gated"

echo ""
echo "======================================================================"
echo "STGNN Training Complete"
echo "Date: $(date)"
echo "======================================================================"
