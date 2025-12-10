#!/bin/bash
#SBATCH --job-name=e2e_hpo
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/e2e_hpo_%j.log

module load cuda/12.1
source ~/.bashrc
cd /share/crsp/lab/pkaiser/ddlin/mids/datasci-266/mimic_nlp

echo "======================================================================"
echo "E2E Gated Fusion HPO - Intensive Hyperparameter Optimization"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "======================================================================"

# HPO Search Space:
# - hidden_dim: [64, 128, 256]
# - dropout: [0.1, 0.5]
# - lr: [5e-5, 5e-3]
# - encoder_lr: [1e-6, 1e-4]
# - num_layers: [1, 3]
# - num_heads: [2, 4, 8]
# - pca_components: [32, 64, 128]
# - warmup_ratio: [0.05, 0.2]
# - batch_size: [16, 32, 64]
# - weight_decay: [1e-4, 1e-1]

uv run python src/models/pytorch/train_e2e_gated_fusion_hpo.py \
    --ehr_seq_path data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_cat_embedding.pkl \
    --text_dir data/interim/embeddings/notes \
    --cohort_path data/interim/readmit_analysis/long_los_cohort.csv \
    --n_trials 50 \
    --epochs_per_trial 15 \
    --final_epochs 50 \
    --accumulation_steps 4 \
    --save_dir data/interim/ehr_long_los/models_hpo \
    --study_name "e2e_gated_fusion_hpo_v1"

echo "======================================================================"
echo "HPO Complete"
echo "Date: $(date)"
echo "======================================================================"
