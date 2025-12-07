#!/bin/bash
#
# Preprocess Multimodal Graph (EHR + Text) for GNN
#
# Resources: CPU only, 16 cores, 64GB memory.
#
# Usage:
#   sbatch scripts/slurm/preprocess_graph_multimodal.sbatch
#

#SBATCH --job-name=prep-graph-multi
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/prep_graph_multi_%j.log

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

EMB_PATH="data/interim/ehr_long_los/embeddings/structured_ehr_embeddings.npz"
TEXT_BASE="data/interim"
COHORT="data/interim/readmit_analysis/long_los_cohort.csv"
MAPPING="data/interim/ehr_long_los/graph/graph_node_map.csv"
SAVE_DIR="data/interim/ehr_long_los/graph_multimodal"

echo "===================================================================="
echo "Job Started: $(date)"
echo "Building Multimodal Graph (KNN on EHR + Text Embeddings)..."
echo "EHR: $EMB_PATH"
echo "Text Base: $TEXT_BASE"
echo "Output: $SAVE_DIR"
echo "===================================================================="

uv run python ehr/preprocess_graph.py \
    --use_embeddings "$EMB_PATH" \
    --text_path "$TEXT_BASE" \
    --cohort_path "$COHORT" \
    --mapping_path "$MAPPING" \
    --save_dir "$SAVE_DIR" \
    --k 15

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
