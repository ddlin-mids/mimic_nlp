#!/bin/bash
#SBATCH --job-name=prep_temporal
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/prep_temporal_%j.out
#SBATCH --error=logs/prep_temporal_%j.err

# Prepare temporal data for EHR and Radiology modalities
# This creates the npz files needed for train_temporal_attention_fusion_v2.py

echo "======================================"
echo "Preparing Temporal Data"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "======================================"

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate environment
source ~/.bashrc
cd /share/crsp/lab/pkaiser/ddlin/mids/datasci-266/mimic_nlp

# Create output directory
mkdir -p data/interim/ehr_long_los/temporal

echo ""
echo ">>> Step 1: Preparing temporal radiology data..."
uv run python src/data/prepare_temporal_radiology.py \
    --radiology_path data/interim/embeddings/notes/radiology_report.npz \
    --cohort_path data/interim/readmit_analysis/long_los_cohort.csv \
    --output_path data/interim/ehr_long_los/temporal/radiology.npz \
    --max_reports 20

echo ""
echo ">>> Step 2: Preparing temporal EHR data..."
uv run python src/data/prepare_temporal_ehr.py \
    --sequence_pkl data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_cat_embedding.pkl \
    --cohort_path data/interim/readmit_analysis/long_los_cohort.csv \
    --output_path data/interim/ehr_long_los/temporal/ehr.npz \
    --max_days 100

echo ""
echo "======================================"
echo "Data preparation complete!"
echo "End time: $(date)"
echo "======================================"

# List output files
echo ""
echo "Output files:"
ls -lh data/interim/ehr_long_los/temporal/
