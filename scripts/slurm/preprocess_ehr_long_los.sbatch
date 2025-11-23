#!/bin/bash
#
# Preprocess structured EHR features for the long-LOS cohort (LOS >= 15).
# Outputs are written to data/interim/ehr_long_los/.
#
# Resources: CPU only, 8 cores, 64G memory. No time limit specified.
#
# Usage:
#   sbatch scripts/slurm/preprocess_ehr_long_los.sbatch
#

#SBATCH --job-name=preproc-ehr-longlos
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/preprocess_ehr_long_los_%j.log

set -euo pipefail

cd "$(dirname "$0")/../.."

echo "Starting EHR preprocessing for long-LOS cohort (LOS >= 15)..."
echo "Working dir: $(pwd)"
echo "Timestamp: $(date)"

uv run python ehr/preprocess_ehr.py \
  --demo_file data/interim/readmit_analysis/long_los_cohort.csv \
  --icd_file physionet.org/files/mimiciv/3.1/hosp/diagnoses_icd.csv.gz \
  --lab_file physionet.org/files/mimiciv/3.1/hosp/labevents.csv.gz \
  --med_file physionet.org/files/mimiciv/3.1/hosp/prescriptions.csv.gz \
  --save_dir data/interim/ehr_long_los

echo "Completed EHR preprocessing."
