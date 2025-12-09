#!/bin/bash
#
# Extract unique clinical events from MIMIC-IV-Ext-22MCTS dataset.
# This is a CPU-only job (no GPU needed) - just reading CSV and deduplicating.
#
# Usage:
#   sbatch scripts/slurm/extract_unique_events.sbatch
#

#SBATCH --job-name=extract-events
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/extract_unique_events_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

# Ensure log directory exists
mkdir -p logs/slurm

# Run extraction with mapping save (for later reconstruction)
uv run python ehr/extract_unique_clinical_events.py \
    --input physionet.org/files/mimic-iv-ext-22mcts/1.0.0/clinical_event_timestamp.csv \
    --output_dir data/interim/mimic_22m \
    --save_mapping

echo ""
echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
