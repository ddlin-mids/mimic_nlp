#!/bin/bash
#SBATCH --job-name=stgnn_temporal_hpo
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stgnn_temporal_hpo_%j.out
#SBATCH --error=logs/stgnn_temporal_hpo_%j.err

# Temporal STGNN Hyperparameter Optimization
# Grid search over dropout, learning rate, hidden dim, weight decay
# Focus on reducing overfitting observed in baseline model

echo "=========================================="
echo "STGNN Temporal HPO - Job $SLURM_JOB_ID"
echo "=========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"

# Setup environment
cd /share/crsp/lab/pkaiser/ddlin/mids/datasci-266/mimic_nlp
source ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"

# Create logs directory
mkdir -p logs

echo ""
echo "Python: $(uv run which python)"
echo "PyTorch version: $(uv run python -c 'import torch; print(torch.__version__)')"
echo ""

# Run HPO
echo "Starting HPO..."
uv run python src/models/pytorch/train_stgnn_temporal_hpo.py \
    --seq_file data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_pca256.pkl \
    --features_file data/interim/ehr_long_los/integrated_features.csv \
    --output_dir data/interim/ehr_long_los/stgnn_temporal_hpo \
    --max_seq_len 9 \
    --top_perc 0.01 \
    --epochs 100 \
    --patience 15 \
    --seed 42

echo ""
echo "=========================================="
echo "HPO Complete"
echo "End time: $(date)"
echo "=========================================="
