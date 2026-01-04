#!/bin/bash
#SBATCH --job-name=stgnn_temporal
#SBATCH --partition=free
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/stgnn_temporal_%j.out
#SBATCH --error=logs/stgnn_temporal_%j.err

# Temporal STGNN Training
# Matches reference MM-STGNN architecture:
# - 9-day temporal sequences
# - Top 1% edges by demographic similarity
# - Weighted GraphSAGE in GConvGRU
# - Inductive learning setup

echo "=============================================="
echo "Temporal STGNN Training - Matching Reference"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=============================================="

cd /share/crsp/lab/pkaiser/ddlin/mids/datasci-266/mimic_nlp

# Create logs directory
mkdir -p logs

# Activate virtual environment and set Python path
source .venv/bin/activate
export PYTHONPATH="${PWD}:${PYTHONPATH}"
echo "Python: $(which python)"
echo "PYTHONPATH: $PYTHONPATH"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "Starting training on CPU..."
export CUDA_VISIBLE_DEVICES=""

# Run training with unbuffered output
python -u src/models/pytorch/train_stgnn_temporal.py \
    --seq_file data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_pca256.pkl \
    --features_file data/interim/ehr_long_los/integrated_features.csv \
    --output_dir data/interim/ehr_long_los/stgnn_temporal \
    --max_seq_len 9 \
    --hidden_dim 256 \
    --num_gru_layers 1 \
    --conv_type weighted_sage \
    --dropout 0.2 \
    --lr 3e-3 \
    --weight_decay 5e-4 \
    --pos_weight 4.0 \
    --top_perc 0.01 \
    --epochs 100 \
    --patience 10 \
    --seed 42

echo ""
echo "=============================================="
echo "Training completed: $(date)"
echo "=============================================="
