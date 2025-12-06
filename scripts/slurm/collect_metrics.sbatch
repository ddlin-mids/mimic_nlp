#!/bin/bash
#
# Collect Final Metrics
#
# Resources: 4 CPUs, 8GB Memory.
#
# Usage:
#   sbatch scripts/slurm/collect_metrics.sbatch
#

#SBATCH --job-name=collect-met
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/collect_metrics_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${SLURM_SUBMIT_DIR}/mlruns"

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Script: src/visualization/collect_metrics.py"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

echo ">>> Collecting Metrics..."
uv run python src/visualization/collect_metrics.py

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
