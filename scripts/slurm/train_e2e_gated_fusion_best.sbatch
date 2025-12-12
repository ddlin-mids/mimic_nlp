#!/bin/bash
#SBATCH --job-name=e2e_best
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm/e2e_best_%j.log

source ~/.bashrc
cd /share/crsp/lab/pkaiser/ddlin/mids/datasci-266/mimic_nlp

echo "======================================================================"
echo "E2E Gated Fusion - Training with Best HPO Parameters"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "======================================================================"

# Best params from HPO Trial 22 (Val AUC: 0.6423)
# hidden_dim: 256
# dropout: 0.371
# lr: 0.000369
# encoder_lr: 6.75e-05
# num_layers: 2
# num_heads: 8
# pca_components: 128
# warmup_ratio: 0.068
# batch_size: 16
# weight_decay: 0.046

uv run python src/models/pytorch/train_e2e_gated_fusion.py \
    --ehr_seq_path data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_cat_embedding.pkl \
    --text_dir data/interim/embeddings/notes \
    --cohort_path data/interim/readmit_analysis/long_los_cohort.csv \
    --pretrained_encoder_path data/interim/ehr_long_los/embeddings_transformer/ehr_transformer_best.pt \
    --epochs 50 \
    --lr 0.000369 \
    --encoder_lr 6.75e-05 \
    --hidden_dim 256 \
    --dropout 0.371 \
    --pca_components 128 \
    --warmup_epochs 3 \
    --batch_size 16 \
    --accumulation_steps 4 \
    --save_dir data/interim/ehr_long_los/models_hpo_best_pretrained

echo "======================================================================"
echo "Training Complete"
echo "Date: $(date)"
echo "======================================================================"
