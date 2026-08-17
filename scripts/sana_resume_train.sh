#!/bin/bash
# SANA-512 Resume Training Script
#
# Resumes SANA-600M-512px training from checkpoint-6000 to 20000 steps.
# Supports two modes:
#   1. Full 11-task (same as original) → continues all tasks
#   2. 3-task focus (depth, normal, semantic) → concentrated training
#
# Usage:
#   # Full 11-task resume (default)
#   bash scripts/sana_resume_train.sh
#
#   # 3-task focus (recommended for better per-task quality)
#   bash scripts/sana_resume_train.sh --3task
#
#   # Custom steps
#   bash scripts/sana_resume_train.sh --3task --steps 20000

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESUME_CKPT="${PROJECT_ROOT}/output/sana_512_vb/checkpoint-6000_best/lora.pt"
OUTPUT_DIR="${PROJECT_ROOT}/output/sana_512_vb"
DATA_DIR="/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v3"
STEPS=20000
TASKS_FLAG=""

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --3task)
            TASKS_FLAG="--tasks depth normal semantic"
            OUTPUT_DIR="${PROJECT_ROOT}/output/sana_512_vb_3task"
            shift
            ;;
        --steps)
            STEPS=$2
            shift 2
            ;;
        --data-dir)
            DATA_DIR=$2
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "SANA-512 Resume Training"
echo "============================================"
echo "Resume from: ${RESUME_CKPT}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Total steps: ${STEPS}"
echo "Data dir:    ${DATA_DIR}"
if [ -n "${TASKS_FLAG}" ]; then
    echo "Tasks:       depth, normal, semantic (3-task focus)"
else
    echo "Tasks:       all 11 tasks (full multi-task)"
fi
echo "============================================"

# Verify checkpoint exists
if [ ! -f "${RESUME_CKPT}" ]; then
    echo "ERROR: Resume checkpoint not found: ${RESUME_CKPT}"
    exit 1
fi

cd "${PROJECT_ROOT}"

python scripts/train_sana_512.py \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-name sana-600m-512px \
    --steps ${STEPS} \
    --lr 5e-5 \
    --batch-size 4 \
    --grad-accum 4 \
    --resolution 512 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --scheduler cosine \
    --warmup-ratio 0.05 \
    --ema-decay 0.9999 \
    --save-every 500 \
    --eval-every 500 \
    --log-every 50 \
    --resume "${RESUME_CKPT}" \
    ${TASKS_FLAG}

echo "============================================"
echo "Training complete!"
echo "Checkpoints in: ${OUTPUT_DIR}"
echo "============================================"
