#!/bin/bash
#SBATCH --job-name=late-ens
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/late_fusion_ensemble_%j.log

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Script: Late Fusion Ensemble (Lab XGBoost + Neural Gated Fusion)"
echo "===================================================================="

cd /share/crsp/lab/pkaiser/ddlin/mids/datasci-266/mimic_nlp

# Activate environment
source .venv/bin/activate

# Force unbuffered output
export PYTHONUNBUFFERED=1

# Run late fusion ensemble
python -u src/models/train_late_fusion_ensemble.py \
    --hidden_dim 64 \
    --dropout 0.3 \
    --lr 2e-4 \
    --epochs 30 \
    --pca_text 64 \
    --seed 42

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
