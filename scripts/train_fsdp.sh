#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FSDP (Fully Sharded Data Parallel) Training Launch Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# For multi-GPU training of FLUX.1 (12B params) with memory-efficient
# sharding.  Recommended for 8×A100-80GB or 4×H100-80GB nodes.
#
# Prerequisites:
#   - PyTorch 2.1+ with FSDP support
#   - nvidia-container-toolkit (if running in Docker)
#   - Shared filesystem for checkpoints
#
# Usage:
#   # Single-node, 8 GPUs (most common)
#   bash scripts/train_fsdp.sh --nodes 1 --gpus-per-node 8
#
#   # Multi-node, 2 nodes × 8 GPUs
#   # Node 0 (master):
#   bash scripts/train_fsdp.sh --nodes 2 --node-rank 0 --master-addr <IP>
#   # Node 1:
#   bash scripts/train_fsdp.sh --nodes 2 --node-rank 1 --master-addr <IP>
#
#   # With custom config
#   bash scripts/train_fsdp.sh --config configs/fsdp_training_config.yaml
#
#   # With gradient checkpointing (saves VRAM, ~30% slower)
#   bash scripts/train_fsdp.sh --gradient-checkpointing
#
#   # Resume from checkpoint
#   bash scripts/train_fsdp.sh --resume output/checkpoints/epoch_5.pt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Defaults ─────────────────────────────────────────────────────────
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}
SHARDING_STRATEGY=${SHARDING_STRATEGY:-"full"}  # full | shard_grad_op | no_shard
MIN_WRAP_PARAMS=${MIN_WRAP_PARAMS:-1000000}

# Training arguments passed through to train.py
TRAIN_ARGS=()

# ── Parse script-specific arguments ────────────────────────────────────
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
FSDP Training Launch Script for Open-Vision-Banana

Options:
  --nodes N              Number of nodes (default: 1)
  --node-rank R          Rank of this node (default: 0)
  --gpus-per-node G      GPUs per node (default: 8, auto-detected)
  --master-addr IP       Master node IP (default: 127.0.0.1)
  --master-port PORT     Master port (default: 29500)
  --sharding-strategy S  FSDP sharding: full | shard_grad_op | no_shard
  --min-wrap-params N    Minimum params for auto-wrap (default: 1M)

All other arguments are passed through to scripts/train.py.
Run `python scripts/train.py --help` for training options.
EOF
            exit 0
            ;;
        *)
            TRAIN_ARGS+=("$1")
            shift
            ;;
    esac
done

# ── Auto-detect GPUs if not specified ────────────────────────────────
if [[ "$GPUS_PER_NODE" == "8" ]] && command -v nvidia-smi &>/dev/null; then
    DETECTED=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$DETECTED" -gt 0 ]]; then
        GPUS_PER_NODE=$DETECTED
        echo "Auto-detected $GPUS_PER_NODE GPUs"
    fi
fi

WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  FSDP Training Launch"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Nodes:         $NNODES"
echo "  Node rank:     $NODE_RANK"
echo "  GPUs/node:     $GPUS_PER_NODE"
echo "  World size:    $WORLD_SIZE"
echo "  Master:        $MASTER_ADDR:$MASTER_PORT"
echo "  Sharding:      $SHARDING_STRATEGY"
echo "  Min wrap:      $MIN_WRAP_PARAMS"
echo "  Train args:    ${TRAIN_ARGS[*]}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Build torchrun command ───────────────────────────────────────────
TROCHRUN_ARGS=(
    --nnodes="$NNODES"
    --node-rank="$NODE_RANK"
    --nproc-per-node="$GPUS_PER_NODE"
    --rdzv-backend=c10d
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT"
)

TRAIN_ARGS+=(
    --fsdp
    "--sharding-strategy=$SHARDING_STRATEGY"
)

if ! command -v torchrun &>/dev/null; then
    echo "ERROR: torchrun not found. Install PyTorch >= 2.0."
    exit 1
fi

cd "$PROJECT_ROOT"

exec torchrun "${TROCHRUN_ARGS[@]}" \
    scripts/train.py \
    "${TRAIN_ARGS[@]}"
