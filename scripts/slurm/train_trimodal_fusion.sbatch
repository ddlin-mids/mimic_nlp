#!/bin/bash
#SBATCH --job-name=trimodal
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/trimodal_%j.log
#SBATCH --error=logs/slurm/trimodal_%j.err

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Script: Trimodal Gated Fusion (EHR + Text + Labs)"
echo "===================================================================="

cd /share/crsp/lab/pkaiser/ddlin/mids/datasci-266/mimic_nlp

# Activate environment
source .venv/bin/activate

# Force unbuffered output
export PYTHONUNBUFFERED=1

# Run multiple experiments
echo ">>> Experiment 1: Gated Fusion (default settings)"
stdbuf -oL python -u src/models/pytorch/train_trimodal_gated_fusion.py \
    --epochs 50 \
    --lr 1e-4 \
    --hidden_dim 128 \
    --dropout 0.3 \
    --pca_text 64 \
    --pca_lab 0 \
    --model_type gated 2>&1

echo ""
echo ">>> Experiment 2: Gated Fusion with Focal Loss"
python -u src/models/pytorch/train_trimodal_gated_fusion.py \
    --epochs 50 \
    --lr 1e-4 \
    --hidden_dim 128 \
    --dropout 0.3 \
    --pca_text 64 \
    --pca_lab 0 \
    --model_type gated \
    --use_focal_loss \
    --focal_alpha 0.25 \
    --focal_gamma 2.0

echo ""
echo ">>> Experiment 3: Attention-based Fusion"
python -u src/models/pytorch/train_trimodal_gated_fusion.py \
    --epochs 50 \
    --lr 1e-4 \
    --hidden_dim 128 \
    --dropout 0.3 \
    --pca_text 64 \
    --pca_lab 0 \
    --model_type attention

echo ""
echo ">>> Experiment 4: Larger Hidden Dimension"
python -u src/models/pytorch/train_trimodal_gated_fusion.py \
    --epochs 50 \
    --lr 5e-5 \
    --hidden_dim 256 \
    --dropout 0.4 \
    --pca_text 128 \
    --pca_lab 0 \
    --model_type gated

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
