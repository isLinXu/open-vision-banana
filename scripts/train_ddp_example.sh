#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Multi-node DDP Training Example
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Prerequisites:
#   1. Passwordless SSH between all nodes (ssh-keygen -t rsa && ssh-copy-id)
#   2. Same Python / torch / CUDA versions on all nodes
#   3. Shared filesystem (NFS / Lustre) for checkpoints and data
#
# Usage:
#   # On master (node rank 0):
#   bash scripts/train_ddp_example.sh
#
#   # On workers:
#   bash scripts/train_ddp_example.sh --rank 1 --master-addr 10.0.0.1

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}

# Auto-detect number of GPUs
if command -v nvidia-smi &>/dev/null; then
    NGPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
elif python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    NGPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
else
    NGPUS=0
fi

# ── Launch ────────────────────────────────────────────────────────────
echo "Launching DDP training on ${NGPUS} GPU(s)..."

if [ "${NGPUS:-0}" -gt 1 ]; then
    echo "Multi-GPU DDP training"
    torchrun \
        --nnodes=${NNODES} \
        --node-rank=${NODE_RANK} \
        --rdzv-backend=c10d \
        --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        $(pwd)/scripts/train.py \
        --world-size ${NGPUS} \
        --rank ${NODE_RANK} \
        "$@"
else
    echo "Single-GPU or CPU training"
    python3 $(pwd)/scripts/train.py "$@"
fi
