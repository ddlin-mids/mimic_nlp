#!/bin/bash
#
# Embed clinical events using BioClinical ModernBERT.
# Produces admission-level embeddings in same format as discharge_summary.npz
#
# Prerequisites:
#   - Run extract_unique_events.sbatch first
#   - data/interim/mimic_22m/event_stats.csv must exist
#   - data/interim/mimic_22m/hadm_event_mapping.parquet must exist
#
# Usage:
#   sbatch scripts/slurm/embed_clinical_events.sbatch
#

#SBATCH --job-name=embed-events
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/embed_clinical_events_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1
export HF_HOME="/share/crsp/lab/pkaiser/ddlin/mids/datasci-266/mimic_nlp/.cache/huggingface"
export TORCH_COMPILE_DISABLE=1

# Configuration
TOP_K=10000  # Top 10K events covers ~57% of occurrences
COHORT_PATH="data/interim/readmit_analysis/long_los_cohort.csv"
OUTPUT_DIR="data/interim/embeddings/notes"

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Top-K: $TOP_K"
echo "Cohort: $COHORT_PATH"
echo "Output: $OUTPUT_DIR"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

# Ensure directories exist
mkdir -p logs/slurm
mkdir -p "$OUTPUT_DIR"

# Check prerequisites
STATS_FILE="data/interim/mimic_22m/event_stats.csv"
MAPPING_FILE="data/interim/mimic_22m/hadm_event_mapping.parquet"

if [[ ! -f "$STATS_FILE" ]]; then
    echo "ERROR: Stats file not found: $STATS_FILE"
    echo "Run extract_unique_events.sbatch first."
    exit 1
fi

if [[ ! -f "$MAPPING_FILE" ]]; then
    echo "ERROR: Mapping file not found: $MAPPING_FILE"
    echo "Run extract_unique_events.sbatch with --save_mapping first."
    exit 1
fi

# Run embedding generation with cohort filtering
uv run python ehr/embed_clinical_events.py \
    --stats-file "$STATS_FILE" \
    --mapping-file "$MAPPING_FILE" \
    --cohort-path "$COHORT_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --output-prefix "clinical_events" \
    --top-k $TOP_K \
    --model-name thomas-sounack/bioclinical-modernbert-base \
    --batch-size 64 \
    --max-length 128 \
    --dtype bfloat16

echo ""
echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
