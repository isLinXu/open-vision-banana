#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Distributed Data-Parallel (DDP) Training Launch Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# This script supports both torchrun (recommended) and manual launch.
#
# Prerequisites:
#   1. Passwordless SSH between all nodes
#   2. Same Python / torch / CUDA versions on all nodes
#   3. Shared filesystem (NFS / Lustre) for checkpoints and data
#
# Usage:
#   # Single-node, multi-GPU:
#   bash scripts/train_ddp.sh
#
#   # Multi-node:
#   bash scripts/train_ddp.sh --master-addr <IP> --master-port <PORT>

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────
NNODES=${NNODES:-1}               # Number of nodes (default: 1)
NODE_RANK=${NODE_RANK:-0}         # Rank of this node (default: 0)
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}
RANK=${RANK:-0}                   # Global rank (0 = master)
WORLD_SIZE=${WORLD_SIZE:-1}       # Total number of processes
LOCAL_RANK=${LOCAL_RANK:-0}       # Rank within node

# ── Auto-detect number of GPUs ───────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    NGPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
elif python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    NGPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
else
    NGPUS=0
fi

# ── Launch via torchrun (recommended) ────────────────────────────────
if command -v torchrun &>/dev/null; then
    echo "Using torchrun for distributed launch"
    torchrun \
        --nnodes=$NNODES \
        --node-rank=$NODE_RANK \
        --rdzv-backend=c10d \
        --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
        train.py \
        --world-size $WORLD_SIZE \
        --rank $RANK \
        "$@"
else
    echo "torchrun not found — falling back to manual launch"
    echo "Each process will set up its own distributed environment."

    # Master node
    if [ $RANK -eq 0 ]; then
        python3 -c "
import torch.distributed as dist
if not dist.is_initialized():
    dist.init_process_group(
        backend='gloo',
        init_method=f'tcp://{MASTER_ADDR}:{MASTER_PORT}',
        rank=$RANK,
        world_size=$WORLD_SIZE,
    )
print(f'Node {$RANK}/{$WORLD_SIZE} initialized')
"
    fi
fi
