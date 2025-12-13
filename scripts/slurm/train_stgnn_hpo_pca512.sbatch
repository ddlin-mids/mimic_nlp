#!/bin/bash
#SBATCH --job-name=stgnn_hpo512
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm/stgnn_hpo_pca512_%j.log

source ~/.bashrc
cd /share/crsp/lab/pkaiser/ddlin/mids/datasci-266/mimic_nlp

echo "======================================================================"
echo "STGNN HPO - PCA 512 Components"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "======================================================================"

mkdir -p logs/slurm

PCA_DATA="data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_pca512.pkl"
if [ ! -f "$PCA_DATA" ]; then
    echo "ERROR: PCA-reduced data not found at $PCA_DATA"
    exit 1
fi

uv run python src/models/pytorch/train_stgnn_hpo.py \
    --ehr_seq_path "$PCA_DATA" \
    --cohort_path data/interim/readmit_analysis/long_los_cohort.csv \
    --graph_path data/interim/ehr_long_los/graph/graph_pyg.pt \
    --text_dir data/interim/embeddings/notes \
    --n_trials 50 \
    --epochs 50 \
    --patience 10 \
    --max_seq_len 30 \
    --run_name "stgnn_hpo_pca512"

echo ""
echo "======================================================================"
echo "STGNN HPO PCA 512 Complete"
echo "Date: $(date)"
echo "======================================================================"
