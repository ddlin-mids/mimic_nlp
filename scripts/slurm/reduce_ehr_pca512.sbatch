#!/bin/bash
#SBATCH --job-name=ehr_pca512
#SBATCH --partition=free
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm/ehr_pca512_%j.log

source ~/.bashrc
cd /share/crsp/lab/pkaiser/ddlin/mids/datasci-266/mimic_nlp

echo "======================================================================"
echo "PCA Dimensionality Reduction for EHR Sequences (512 components)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "======================================================================"

mkdir -p logs/slurm

# Run PCA reduction with 512 components
uv run python ehr/reduce_ehr_features_pca.py \
    --input_path data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_cat_embedding.pkl \
    --cohort_path data/interim/readmit_analysis/long_los_cohort.csv \
    --output_path data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_pca512.pkl \
    --n_components 512

echo ""
echo "======================================================================"
echo "PCA 512 Reduction Complete"
echo "Date: $(date)"
echo "======================================================================"
