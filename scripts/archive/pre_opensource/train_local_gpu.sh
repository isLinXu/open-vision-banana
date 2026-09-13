#!/usr/bin/env bash
# ============================================================================
# TinyVisionBanana - Local GPU Training Script (macOS MPS)
# ============================================================================
# This script is designed to run on your local Mac with Apple Silicon GPU (MPS).
# It uses a larger model configuration that doesn't fit in the sandbox's 2GB limit.
#
# Requirements:
#   - macOS with Apple Silicon (M1/M2/M3/M4)
#   - PyTorch >= 2.0 with MPS support
#   - Python dependencies: torch, numpy, pillow
#
# Usage:
#   chmod +x scripts/train_local_gpu.sh
#   bash scripts/train_local_gpu.sh [config_name] [steps]
#
# Config presets:
#   "large"     - w=32, 256px, 11.86M params (recommended, ~1-2h on M-series)
#   "xlarge"   - w=48, 256px, 26.6M params (if memory allows)
#   "multitask" - 5-task, w=32, 256px (full multi-task, ~3-5h)
#
# ============================================================================

set -euo pipefail

CONFIG_NAME="${1:-large}"
STEPS="${2:-5000}"
DATA_DIR="/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v3"
OUTPUT_BASE="output/local_gpu"

# ── Config presets ──────────────────────────────────────────────
case "$CONFIG_NAME" in
    large)
        # Single-task (depth) with large model
        TASKS="depth"
        BASE_WIDTH=32
        RESOLUTION=256
        GRAD_ACCUM=4
        BATCH_SIZE=2
        LR=0.0005
        OUTPUT_DIR="${OUTPUT_BASE}/depth_w32_256"
        ;;
    xlarge)
        # Single-task (depth) with extra-large model
        TASKS="depth"
        BASE_WIDTH=48
        RESOLUTION=256
        GRAD_ACCUM=2
        BATCH_SIZE=1
        LR=0.0003
        OUTPUT_DIR="${OUTPUT_BASE}/depth_w48_256"
        ;;
    multitask)
        # 5-task multi-task with large model
        TASKS="depth normal edge semantic generation"
        BASE_WIDTH=32
        RESOLUTION=256
        GRAD_ACCUM=4
        BATCH_SIZE=2
        LR=0.0005
        OUTPUT_DIR="${OUTPUT_BASE}/multitask_w32_256"
        ;;
    multitask_xlarge)
        # 5-task multi-task with extra-large model
        TASKS="depth normal edge semantic generation"
        BASE_WIDTH=48
        RESOLUTION=256
        GRAD_ACCUM=2
        BATCH_SIZE=1
        LR=0.0003
        OUTPUT_DIR="${OUTPUT_BASE}/multitask_w48_256"
        ;;
    *)
        echo "Unknown config: $CONFIG_NAME"
        echo "Available: large, xlarge, multitask, multitask_xlarge"
        exit 1
        ;;
esac

echo "╔══════════════════════════════════════════════════════════╗"
echo "║     TinyVisionBanana - Local GPU Training               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Config:        $CONFIG_NAME"
echo "Tasks:         $TASKS"
echo "Base width:    $BASE_WIDTH"
echo "Resolution:    ${RESOLUTION}px"
echo "Steps:         $STEPS"
echo "Batch size:    $BATCH_SIZE"
echo "Grad accum:    $GRAD_ACCUM"
echo "Learning rate: $LR"
echo "Output:        $OUTPUT_DIR"
echo ""
echo "Device: auto (will use MPS if available, else CPU)"
echo ""

# ── Create output dir ──────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

# ── Run training ────────────────────────────────────────────────
cd "$(dirname "$0")/.."

export PYTHONPATH=src
export PYTORCH_ENABLE_MPS_FALLBACK=1

python3 scripts/train_tiny_vision_banana.py \
    --data-dir "$DATA_DIR" \
    --tasks $TASKS \
    --resolution "$RESOLUTION" \
    --base-width "$BASE_WIDTH" \
    --num-stages 4 \
    --block-depth 2 \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --weight-decay 0.0001 \
    --warmup-steps 200 \
    --eval-every 250 \
    --max-val-samples 115 \
    --save-every 500 \
    --output-dir "$OUTPUT_DIR" \
    --device auto \
    --seed 42 \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo ""
echo "Training complete. Output saved to: $OUTPUT_DIR"
echo ""
echo "To evaluate:"
echo "  PYTHONPATH=src python3 -c \\"
echo "    \"from open_vision_banana.tiny_model import *; print('ready')\""
