#!/bin/bash
#SBATCH --job-name=stgnn_hpo
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm/stgnn_hpo_%j.log

source ~/.bashrc
cd /share/crsp/lab/pkaiser/ddlin/mids/datasci-266/mimic_nlp

echo "======================================================================"
echo "STGNN Hyperparameter Optimization with Optuna"
echo "Using PCA-reduced EHR features (256 dims)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "======================================================================"

mkdir -p logs/slurm

# Check if PCA data exists
PCA_DATA="data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_pca256.pkl"
if [ ! -f "$PCA_DATA" ]; then
    echo "ERROR: PCA-reduced data not found at $PCA_DATA"
    echo "Please run: sbatch scripts/slurm/reduce_ehr_pca.sbatch first"
    exit 1
fi

# Run HPO with text embeddings
uv run python src/models/pytorch/train_stgnn_hpo.py \
    --ehr_seq_path "$PCA_DATA" \
    --cohort_path data/interim/readmit_analysis/long_los_cohort.csv \
    --graph_path data/interim/ehr_long_los/graph/graph_pyg.pt \
    --text_dir data/interim/embeddings/notes \
    --n_trials 50 \
    --epochs 50 \
    --patience 10 \
    --max_seq_len 30 \
    --run_name "stgnn_hpo_pca256"

echo ""
echo "======================================================================"
echo "STGNN HPO Complete"
echo "Date: $(date)"
echo "======================================================================"
