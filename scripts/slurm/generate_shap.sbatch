#!/bin/bash
#
# Generate SHAP Values and Plots
#
# Resources: 4 CPUs, 32GB Memory.
#
# Usage:
#   sbatch scripts/slurm/generate_shap.sbatch
#

#SBATCH --job-name=shap-gen
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/shap_gen_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Script: src/visualization/generate_shap.py"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

echo ">>> Generating SHAP plots..."
uv run python src/visualization/generate_shap.py

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
