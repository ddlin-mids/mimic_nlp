#!/bin/bash
#
# Run Ablation Study with Clinical Events (MIMIC-IV-Ext-22MCTS)
#
# This runs the full ablation suite including the new clinical_events modality.
# All experiments are tracked in MLflow.
#
# Usage:
#   sbatch scripts/slurm/train_ablation_clinical_events.sbatch
#

#SBATCH --job-name=ablation-ce
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/ablation_clinical_events_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${SLURM_SUBMIT_DIR}/mlruns"

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "MLflow URI: $MLFLOW_TRACKING_URI"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

# Ensure log directory exists
mkdir -p logs/slurm

echo ""
echo ">>> Running Ablation Study with Clinical Events..."
echo "Experiments:"
echo "  - Baseline (Demo)"
echo "  - Static EHR"
echo "  - Structured EHR (Transformer)"
echo "  - Discharge Notes"
echo "  - Radiology Notes"
echo "  - Clinical Events (NEW - MIMIC-IV-Ext-22MCTS)"
echo "  - All Notes"
echo "  - All Text (Notes + Events)"
echo "  - Full Fusion (Transformer)"
echo "  - Full Fusion + Events (Transformer)"
echo ""

uv run python src/models/train_fusion_xgboost.py \
    --embedding_dir "data/interim/ehr_long_los/embeddings_transformer" \
    --embedding_type Transformer

echo ""
echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
