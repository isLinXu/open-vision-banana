#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# H20 DeepSpeed ZeRO-3 Training Launch Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# DeepSpeed ZeRO-3 training on NVIDIA H20 cluster.
# Recommended for multi-node scaling beyond single-node NVSwitch.
# Supports activation checkpointing + CPU-offload for extreme scale.
#
# Usage (single-node, 8 GPUs):
#   bash scripts/launch_h20_deepspeed.sh --data-path /path/to/vision_data
#
# Usage (multi-node, 2 nodes x 8 GPUs):
#   # Node 0 (master):
#   bash scripts/launch_h20_deepspeed.sh --nodes 2 --node-rank 0 --master-addr 10.0.0.1 --data-path /path/to/data
#   # Node 1 (worker):
#   bash scripts/launch_h20_deepspeed.sh --nodes 2 --node-rank 1 --master-addr 10.0.0.1 --data-path /path/to/data
#
# Usage with CPU-offload (for 48GB GPUs or larger models):
#   bash scripts/launch_h20_deepspeed.sh --cpu-offload --data-path /path/to/data
#
# Usage with gradient checkpointing:
#   bash scripts/launch_h20_deepspeed.sh --gradient-checkpointing --data-path /path/to/data
#
# Environment variables:
#   DS_SKIP_CUDA_CHECK    Skip DeepSpeed CUDA version check (default: 1 for H20)
#   NCCL_IB_DISABLE       Disable InfiniBand if not available

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Distributed Configuration ─────────────────────────────────────────
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}

# ── DeepSpeed Configuration ───────────────────────────────────────────
DEEPSPEED_CONFIG="configs/deepspeed_zero3_h20.json"
CPU_OFFLOAD=false

# ── Environment Setup ─────────────────────────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# H20 / Hopper-specific CUDA optimizations
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# DeepSpeed skips strict CUDA version check (H20 needs recent CUDA)
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"

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
        --deepspeed-config)
            DEEPSPEED_CONFIG="$2"; shift 2 ;;
        --cpu-offload)
            CPU_OFFLOAD=true; shift ;;
        --help|-h)
            cat <<'EOF'
H20 DeepSpeed ZeRO-3 Training Launch Script

Options:
  --nodes N                  Number of nodes (default: 1)
  --node-rank R              Rank of this node (default: 0)
  --gpus-per-node G          GPUs per node (default: 8)
  --master-addr IP           Master node IP (default: 127.0.0.1)
  --master-port PORT         Master port (default: 29500)
  --deepspeed-config PATH    DeepSpeed JSON config (default: configs/deepspeed_zero3_h20.json)
  --cpu-offload              Enable CPU-offload for optimizer states

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

# ── DeepSpeed hostfile generation (for multi-node) ────────────────────
HOSTFILE="${PROJECT_ROOT}/.deepspeed_hostfile"
if [[ "$NNODES" -gt 1 && ! -f "$HOSTFILE" ]]; then
    echo "WARNING: Multi-node DeepSpeed requires a hostfile."
    echo "Create ${HOSTFILE} with format: hostname slots=N"
    echo "Example: node0 slots=8"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  H20 DeepSpeed ZeRO-3 Training Launch"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Nodes:         ${NNODES}"
echo "  Node rank:     ${NODE_RANK}"
echo "  GPUs/node:     ${GPUS_PER_NODE}"
echo "  World size:    ${WORLD_SIZE}"
echo "  Master:        ${MASTER_ADDR}:${MASTER_PORT}"
echo "  DeepSpeed:     ${DEEPSPEED_CONFIG}"
echo "  CPU offload:   ${CPU_OFFLOAD}"
echo "  Config:        configs/training_h20_single.yaml (base)"
echo "  Output:        output/checkpoints_h20_deepspeed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Verify deepspeed launcher ─────────────────────────────────────────
if ! command -v deepspeed &>/dev/null; then
    echo "ERROR: deepspeed not found. Install: pip install deepspeed"
    exit 1
fi

# ── Build DeepSpeed arguments ─────────────────────────────────────────
DS_ARGS=(
    --num_gpus="${GPUS_PER_NODE}"
    --num_nodes="${NNODES}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
    --node_rank="${NODE_RANK}"
)

if [[ "$CPU_OFFLOAD" == true ]]; then
    DS_ARGS+=(--offload_optimizer)
fi

if [[ -f "$HOSTFILE" ]]; then
    DS_ARGS+=(--hostfile="${HOSTFILE}")
fi

# ── Launch via DeepSpeed ──────────────────────────────────────────────
cd "$PROJECT_ROOT"

exec deepspeed "${DS_ARGS[@]}" \
    scripts/train.py \
    --config configs/training_h20_single.yaml \
    --deepspeed \
    --deepspeed-config "${DEEPSPEED_CONFIG}" \
    "${TRAIN_ARGS[@]}"
