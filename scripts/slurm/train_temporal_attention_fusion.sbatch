#!/bin/bash
#
# Train Temporal Attention Fusion V2 (with ablation support)
#
# Resources: GPU (A30), 8 CPUs, 48GB Memory.
#
# Usage:
#   sbatch scripts/slurm/train_temporal_attention_fusion.sbatch [ablation]
#   
#   Ablation options:
#     full            - All modalities with temporal attention (default)
#     no_temporal_rad - No temporal attention on radiology  
#     no_temporal_ehr - No temporal attention on EHR
#     no_temporal     - No temporal attention at all
#     ehr_only        - EHR only (single modality baseline)
#     rad_only        - Radiology only
#     text_only       - Radiology + Discharge (no EHR)
#     no_discharge    - EHR + Radiology (no discharge)
#     ehr_disch       - EHR + Discharge (no radiology)
#     ehr_rad         - EHR + Radiology (no discharge)
#

#SBATCH --job-name=temp_fusion_v2
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/temp_fusion_v2_%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${SLURM_SUBMIT_DIR}/mlruns"

# Default ablation is "full" (all modalities, all temporal)
ABLATION=${1:-full}

echo "===================================================================="
echo "Temporal Attention Fusion V2 Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Ablation: ${ABLATION}"
echo "Start time: $(date)"
echo "===================================================================="

cd "$SLURM_SUBMIT_DIR"

# Create logs directory
mkdir -p logs/slurm

echo ""
echo ">>> Running ablation: ${ABLATION}"
echo ""

# Run training
uv run python src/models/pytorch/train_temporal_attention_fusion_v2.py \
    --ablation ${ABLATION} \
    --epochs 50 \
    --lr 1e-4 \
    --dropout 0.3 \
    --proj_dim 256 \
    --num_heads 4 \
    --num_layers 2 \
    --batch_size 32 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --patience 10 \
    --seed 42

echo ""
echo "===================================================================="
echo "Training complete!"
echo "End time: $(date)"
echo "===================================================================="
