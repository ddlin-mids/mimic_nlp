#!/bin/bash
#
# Preprocess Graph (PyG)
#
# Resources: CPU only, 8 cores, 64GB Mem.
#
# Usage:
#   sbatch scripts/slurm/preprocess_graph.sbatch
#

#SBATCH --job-name=prep-graph
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/prep_graph_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Script: ehr/preprocess_graph.py"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

INPUT_PKL="data/interim/ehr_long_los/ehr_preprocessed_all_one_hot.pkl"
COHORT="data/interim/readmit_analysis/long_los_cohort.csv"
MAPPING="data/interim/ehr_long_los/embeddings/structured_ehr_mapping.csv"
SAVE_DIR="data/interim/ehr_long_los/graph"

echo ">>> Building Patient Similarity Graph..."
uv run python ehr/preprocess_graph.py \
    --input_path "$INPUT_PKL" \
    --cohort_path "$COHORT" \
    --mapping_path "$MAPPING" \
    --save_dir "$SAVE_DIR" \
    --k 15

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
