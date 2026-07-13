#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# H20 8x FSDP Training Launch Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Fully Sharded Data Parallel (FSDP) training on 8 x NVIDIA H20.
# Uses FULL_SHARD to partition model + optimizer states across GPUs.
# Larger per-GPU batch than DDP thanks to sharded parameters.
#
# Usage (single-node, 8 GPUs):
#   bash scripts/launch_h20_fsdp.sh --data-path /path/to/vision_data
#
# Usage (multi-node, 2 nodes x 8 GPUs):
#   # Node 0 (master):
#   bash scripts/launch_h20_fsdp.sh --nodes 2 --node-rank 0 --master-addr 10.0.0.1 --data-path /path/to/data
#   # Node 1 (worker):
#   bash scripts/launch_h20_fsdp.sh --nodes 2 --node-rank 1 --master-addr 10.0.0.1 --data-path /path/to/data
#
# Usage with gradient checkpointing (saves VRAM, ~30% slower):
#   bash scripts/launch_h20_fsdp.sh --gradient-checkpointing --data-path /path/to/data
#
# Usage with custom sharding strategy:
#   bash scripts/launch_h20_fsdp.sh --sharding-strategy shard_grad_op --data-path /path/to/data
#
# Environment variables:
#   NCCL_IB_DISABLE       Disable InfiniBand if not available (default: 0)
#   NCCL_SOCKET_IFNAME    Network interface for NCCL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Distributed Configuration ─────────────────────────────────────────
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}
SHARDING_STRATEGY=${SHARDING_STRATEGY:-"full"}   # full | shard_grad_op | no_shard
MIN_WRAP_PARAMS=${MIN_WRAP_PARAMS:-1000000}

# ── Environment Setup ─────────────────────────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# H20 / Hopper-specific CUDA optimizations
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# NCCL tuning
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"

# CPU threading
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 16)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( NPROC / GPUS_PER_NODE ))}"

# ── Parse script-specific arguments ───────────────────────────────────
TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes)
            NNODES="$2"; shift 2 ;;
        --node-rank)
            NODE_RANK="$2"; shift 2 ;;
        --gpus-per-node)
            GPUS_PER_NODE="$2"; shift 2 ;;
        --master-addr)
            MASTER_ADDR="$2"; shift 2 ;;
        --master-port)
            MASTER_PORT="$2"; shift 2 ;;
        --sharding-strategy)
            SHARDING_STRATEGY="$2"; shift 2 ;;
        --min-wrap-params)
            MIN_WRAP_PARAMS="$2"; shift 2 ;;
        --help|-h)
            cat <<'EOF'
H20 8x FSDP Training Launch Script

Options:
  --nodes N              Number of nodes (default: 1)
  --node-rank R          Rank of this node (default: 0)
  --gpus-per-node G      GPUs per node (default: 8)
  --master-addr IP       Master node IP (default: 127.0.0.1)
  --master-port PORT     Master port (default: 29500)
  --sharding-strategy S  FSDP sharding: full | shard_grad_op | no_shard
  --min-wrap-params N    Minimum params for auto-wrap (default: 1M)

All other arguments are passed through to scripts/train.py.
EOF
            exit 0
            ;;
        *)
            TRAIN_ARGS+=("$1")
            shift
            ;;
    esac
done

# ── Auto-detect GPUs ──────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    DETECTED=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$DETECTED" -gt 0 && "$GPUS_PER_NODE" == "8" ]]; then
        GPUS_PER_NODE=$DETECTED
    fi
fi

WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  H20 8x FSDP Training Launch"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Nodes:         ${NNODES}"
echo "  Node rank:     ${NODE_RANK}"
echo "  GPUs/node:     ${GPUS_PER_NODE}"
echo "  World size:    ${WORLD_SIZE}"
echo "  Master:        ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Sharding:      ${SHARDING_STRATEGY}"
echo "  Min wrap:      ${MIN_WRAP_PARAMS}"
echo "  Config:        configs/training_h20_fsdp_8x.yaml"
echo "  Output:        output/checkpoints_h20_fsdp8"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Verify torchrun ───────────────────────────────────────────────────
if ! command -v torchrun &>/dev/null; then
    echo "ERROR: torchrun not found. Install PyTorch >= 2.0."
    exit 1
fi

# ── Launch via torchrun ───────────────────────────────────────────────
cd "$PROJECT_ROOT"

exec torchrun \
    --nnodes="${NNODES}" \
    --node-rank="${NODE_RANK}" \
    --nproc-per-node="${GPUS_PER_NODE}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    scripts/train.py \
    --config configs/training_h20_fsdp_8x.yaml \
    --fsdp \
    "${TRAIN_ARGS[@]}"
