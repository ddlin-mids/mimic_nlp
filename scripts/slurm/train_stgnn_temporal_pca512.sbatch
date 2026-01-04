#!/bin/bash
#SBATCH --job-name=stgnn_pca512
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/stgnn_pca512_%j.out
#SBATCH --error=logs/stgnn_pca512_%j.err

# Temporal STGNN with PCA-512 features (double the feature dimension)

echo "=========================================="
echo "STGNN Temporal PCA-512 - Job $SLURM_JOB_ID"
echo "=========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"

cd /share/crsp/lab/pkaiser/ddlin/mids/datasci-266/mimic_nlp
source ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs

echo "Python: $(uv run which python)"

# Run with PCA-512 features
uv run python src/models/pytorch/train_stgnn_temporal.py \
    --seq_file data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_pca512.pkl \
    --features_file data/interim/ehr_long_los/integrated_features.csv \
    --output_dir data/interim/ehr_long_los/stgnn_temporal_pca512 \
    --hidden_dim 256 \
    --num_gru_layers 1 \
    --max_seq_len 9 \
    --dropout 0.3 \
    --lr 1e-3 \
    --weight_decay 1e-3 \
    --pos_weight 4.0 \
    --epochs 100 \
    --patience 15 \
    --top_perc 0.01 \
    --seed 42

echo ""
echo "End time: $(date)"
