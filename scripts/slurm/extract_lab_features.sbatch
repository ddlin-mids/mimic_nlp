#!/bin/bash
#
# Extract Lab Trajectory Features for Readmission Prediction
#
# Processes 158M lab events from labevents.csv.gz to extract
# trajectory features for cardiorenal cohort.
#
# Resources: CPU only, 8 CPUs, 64GB Memory.
#
# Usage:
#   sbatch scripts/slurm/extract_lab_features.sbatch
#

#SBATCH --job-name=lab-feat
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/lab_feat_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Script: src/data/extract_lab_trajectory_features.py"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

# Ensure logs directory exists
mkdir -p logs/slurm

echo ">>> Extracting Lab Trajectory Features..."
uv run python src/data/extract_lab_trajectory_features.py \
    --cohort_path "data/interim/readmit_analysis/long_los_cohort.csv" \
    --labevents_path "physionet.org/files/mimiciv/3.1/hosp/labevents.csv.gz" \
    --output_dir "data/interim/ehr_long_los/lab_features"

echo "===================================================================="
echo "Job Completed: $(date)"
echo "===================================================================="
