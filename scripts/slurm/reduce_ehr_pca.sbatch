#!/bin/bash
#SBATCH --job-name=ehr_pca
#SBATCH --partition=free
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm/ehr_pca_%j.log

source ~/.bashrc
cd /share/crsp/lab/pkaiser/ddlin/mids/datasci-266/mimic_nlp

echo "======================================================================"
echo "PCA Dimensionality Reduction for EHR Sequences"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "======================================================================"

mkdir -p logs/slurm

# Run PCA reduction
# Input: 6,219 features -> Output: 256 PCA components
uv run python ehr/reduce_ehr_features_pca.py \
    --input_path data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_cat_embedding.pkl \
    --cohort_path data/interim/readmit_analysis/long_los_cohort.csv \
    --output_path data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_pca256.pkl \
    --n_components 256

echo ""
echo "======================================================================"
echo "PCA Reduction Complete"
echo "Date: $(date)"
echo "======================================================================"
