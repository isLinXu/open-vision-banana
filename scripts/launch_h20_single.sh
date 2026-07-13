#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# H20 Single-GPU Training Launch Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Optimized for NVIDIA H20 (Hopper, 96GB HBM3) single-card training.
# Uses bfloat16 + AMP for maximum throughput on 96GB VRAM.
#
# Usage:
#   bash scripts/launch_h20_single.sh
#   bash scripts/launch_h20_single.sh --data-path /path/to/vision_data
#   bash scripts/launch_h20_single.sh --generation-path /path/to/gen_data --wandb
#   bash scripts/launch_h20_single.sh --resume output/checkpoints_h20_single/checkpoint-1000
#
# Environment variables:
#   CUDA_VISIBLE_DEVICES   GPU device index (default: 0)
#   OMP_NUM_THREADS        CPU threads per process (default: auto)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Environment Setup ─────────────────────────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# H20-specific CUDA optimizations
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# PyTorch / CUDA tuning for Hopper
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# CPU threading (leave cores for DataLoader)
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( NPROC / 2 ))}"

cd "$PROJECT_ROOT"

# ── Verify H20 availability ───────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo "Unknown")
    GPU_MEM=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -n1 || echo "Unknown")
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  H20 Single-GPU Training"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  GPU:           ${GPU_NAME}"
    echo "  VRAM:          ${GPU_MEM}"
    echo "  Device ID:     ${CUDA_VISIBLE_DEVICES}"
    echo "  Config:        configs/training_h20_single.yaml"
    echo "  Output:        output/checkpoints_h20_single"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
else
    echo "WARNING: nvidia-smi not found — ensure CUDA is available."
fi

# ── Launch training ───────────────────────────────────────────────────
exec python scripts/train.py \
    --config configs/training_h20_single.yaml \
    --device cuda \
    "$@"
