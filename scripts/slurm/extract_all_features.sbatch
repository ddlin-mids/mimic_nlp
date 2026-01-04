#!/bin/bash
#
# Extract All Clinical Features for Readmission Prediction
#
# Runs the complete feature extraction pipeline:
# 1. Admission history features
# 2. Medication trajectory features  
# 3. Procedure/ICU features
# 4. Risk scores (LACE, Charlson CCI)
# 5. Feature integration
#
# Note: Lab features should already exist (run extract_lab_features.sbatch first if needed)
#
# Resources: CPU only, 4 CPUs, 32GB Memory
# Expected runtime: ~1-2 hours
#
# Usage:
#   sbatch scripts/slurm/extract_all_features.sbatch
#

#SBATCH --job-name=feat-all
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/feat_all_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1

echo "===================================================================="
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Feature Extraction Pipeline"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

# Ensure logs and output directories exist
mkdir -p logs/slurm
mkdir -p data/interim/ehr_long_los

# Common paths
COHORT_PATH="data/interim/readmit_analysis/long_los_cohort.csv"
MIMIC_PATH="physionet.org/files"
OUTPUT_DIR="data/interim/ehr_long_los"

echo ""
echo ">>> Step 1/5: Extracting Admission History Features..."
echo "    (prior admissions, 30-day readmissions, ED visits)"
uv run python src/data/extract_admission_history.py \
    --cohort_path "$COHORT_PATH" \
    --mimic_path "$MIMIC_PATH" \
    --output_path "$OUTPUT_DIR/admission_history_features.csv"

echo ""
echo ">>> Step 2/5: Extracting Medication Features..."
echo "    (13 drug classes: diuretics, ACE/ARB, beta-blockers, etc.)"
uv run python src/data/extract_medication_features.py \
    --cohort_path "$COHORT_PATH" \
    --mimic_path "$MIMIC_PATH" \
    --output_path "$OUTPUT_DIR/medication_features.csv"

echo ""
echo ">>> Step 3/5: Extracting Procedure/ICU Features..."
echo "    (ICU stays, ventilation, dialysis, vasopressors)"
uv run python src/data/extract_procedure_icu_features.py \
    --cohort_path "$COHORT_PATH" \
    --mimic_path "$MIMIC_PATH" \
    --output_path "$OUTPUT_DIR/procedure_icu_features.csv"

echo ""
echo ">>> Step 4/5: Extracting Risk Scores..."
echo "    (LACE score, Charlson Comorbidity Index, individual comorbidities)"
uv run python src/data/extract_risk_scores.py \
    --cohort_path "$COHORT_PATH" \
    --mimic_path "$MIMIC_PATH" \
    --output_path "$OUTPUT_DIR/risk_scores.csv"

echo ""
echo ">>> Step 5/5: Integrating All Features..."
echo "    (merging demographics, history, meds, procedures, risk scores, labs)"
uv run python src/data/integrate_features.py \
    --cohort_path "$COHORT_PATH" \
    --feature_dir "$OUTPUT_DIR" \
    --output_path "$OUTPUT_DIR/integrated_features.csv"

echo ""
echo "===================================================================="
echo "Feature Extraction Complete!"
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "  No CSV files found"
echo ""
echo "Job Completed: $(date)"
echo "===================================================================="
