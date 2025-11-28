#!/bin/bash
#
# Generate Static EHR Embeddings (Bag-of-Embeddings + SVD)
# This creates a baseline representation that captures overall clinical burden
# but ignores temporal order.
#
# Resources: CPU-only, 16 cores, 64GB memory.
#
# Usage:
#   sbatch scripts/slurm/encode_ehr_static.sbatch
#

#SBATCH --job-name=ehr-static-enc
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/encode_ehr_static_%j.log

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# Define paths
INPUT_CSV="data/interim/ehr_long_los/ehr_combined.csv"
OUTPUT_NPZ="data/interim/ehr_long_los/embeddings/static_ehr_embeddings.npz"
SAVE_DIR="data/interim/ehr_long_los/embeddings"

echo "Starting Static EHR Encoding (SVD)..."
echo "Input: $INPUT_CSV"
echo "Timestamp: $(date)"

mkdir -p "$SAVE_DIR"

# Run the static encoder script
uv run python ehr/encode_ehr_static_boe.py \
  --input "$INPUT_CSV" \
  --output "$OUTPUT_NPZ" \
  --dims 128

echo "Static encoding complete."
