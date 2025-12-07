#!/bin/bash
#
# Generate Embeddings Visualization
#
# Resources: 4 CPUs, 16GB Memory.
#
# Usage:
#   sbatch scripts/slurm/visualize_embeddings.sbatch
#

#SBATCH --job-name=viz-emb
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/viz_emb_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Script: src/visualization/visualize_embeddings.py"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

echo ">>> Generating t-SNE Plot..."
uv run python src/visualization/visualize_embeddings.py

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
