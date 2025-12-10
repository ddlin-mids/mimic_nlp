#!/bin/bash
#SBATCH --job-name=e2e_gated
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/e2e_gated_%j.log

module load cuda/12.1
source ~/.bashrc
cd /share/crsp/lab/pkaiser/ddlin/mids/datasci-266/mimic_nlp

echo "Starting End-to-End Gated Fusion Training..."
echo "Date: $(date)"
echo "Node: $(hostname)"

uv run python src/models/pytorch/train_e2e_gated_fusion.py \
    --ehr_seq_path data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_cat_embedding.pkl \
    --text_dir data/interim/embeddings/notes \
    --cohort_path data/interim/readmit_analysis/long_los_cohort.csv \
    --pretrained_encoder_path data/interim/ehr_long_los/embeddings_transformer/ehr_transformer_best.pt \
    --epochs 50 \
    --lr 1e-4 \
    --encoder_lr 2e-5 \
    --hidden_dim 128 \
    --pca_components 64 \
    --warmup_epochs 3 \
    --batch_size 16 \
    --accumulation_steps 8 \
    --save_dir data/interim/ehr_long_los/models_e2e

echo "Training finished."
