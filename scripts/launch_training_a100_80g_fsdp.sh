#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# A100-80GB FSDP (Fully Sharded Data Parallel) Training Launch Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# For very large batch sizes or multi-node scaling where model states
# must be sharded across GPUs. FSDP shards parameters, gradients, and
# optimizer states — enabling effective batch sizes beyond what DDP
# can achieve on 80GB cards.
#
# Recommended when:
#   - Batch size per GPU > 8 would OOM under DDP
#   - Training on 16+ A100s (2+ nodes) with high communication bandwidth
#   - Need to train at 1024×1024 resolution
#
# Usage:
#   # Single-node 8×A100 (full sharding)
#   bash scripts/launch_training_a100_80g_fsdp.sh
#
#   # With gradient checkpointing (saves ~40% VRAM, ~25% slower)
#   bash scripts/launch_training_a100_80g_fsdp.sh --gradient-checkpointing
#
#   # Multi-node 2×8 A100
#   bash scripts/launch_training_a100_80g_fsdp.sh --nodes 2 --node-rank 0 --master-addr 10.0.1.100
#
#   # Sharded grad-op only (lower overhead, less memory savings)
#   bash scripts/launch_training_a100_80g_fsdp.sh --sharding-strategy shard_grad_op

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Defaults ───────────────────────────────────────────────────────────
CONFIG="${PROJECT_ROOT}/configs/training_a100_80g.yaml"
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}
SHARDING_STRATEGY="full"          # full | shard_grad_op | no_shard
GRADIENT_CHECKPOINTING=""

TRAIN_ARGS=()

# ── Parse script args ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"; shift 2 ;;
        --nodes|--nnodes)
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
        --gradient-checkpointing)
            GRADIENT_CHECKPOINTING="--gradient-checkpointing"; shift ;;
        --help|-h)
            cat <<'EOF'
A100-80GB FSDP Training Launcher

Options:
  --config PATH               YAML config path
  --nodes N                   Number of nodes (default: 1)
  --node-rank R               This node's rank (default: 0)
  --gpus-per-node G           GPUs per node (default: auto-detect)
  --master-addr IP            Master node IP (default: 127.0.0.1)
  --master-port PORT          Master port (default: 29500)
  --sharding-strategy S       full | shard_grad_op | no_shard (default: full)
  --gradient-checkpointing    Enable gradient checkpointing on transformer blocks

All other arguments forwarded to scripts/train.py.
EOF
            exit 0
            ;;
        *)
            TRAIN_ARGS+=("$1")
            shift
            ;;
    esac
done

# ── Auto-detect GPUs ───────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    DETECTED=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$DETECTED" -gt 0 ]]; then
        GPUS_PER_NODE=$DETECTED
    fi
fi

WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

# ── Validate ───────────────────────────────────────────────────────────
if ! command -v torchrun &>/dev/null; then
    echo "[ERR] torchrun not found. Install PyTorch >= 2.0."
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "[ERR] Config not found: $CONFIG"
    exit 1
fi

# ── Banner ─────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     A100-80GB  FSDP  Training  Launch                            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
printf "  %-20s %s\n" "Config:"           "$CONFIG"
printf "  %-20s %s\n" "Nodes:"            "$NNODES"
printf "  %-20s %s\n" "Node rank:"        "$NODE_RANK"
printf "  %-20s %s\n" "GPUs/node:"        "$GPUS_PER_NODE"
printf "  %-20s %s\n" "World size:"       "$WORLD_SIZE"
printf "  %-20s %s\n" "Master:"           "$MASTER_ADDR:$MASTER_PORT"
printf "  %-20s %s\n" "Sharding:"         "$SHARDING_STRATEGY"
printf "  %-20s %s\n" "Grad checkpoint:"  "${GRADIENT_CHECKPOINTING:-disabled}"
printf "  %-20s %s\n" "Extra args:"       "${TRAIN_ARGS[*]}"
echo ""

# ── Launch ─────────────────────────────────────────────────────────────
cd "$PROJECT_ROOT"

exec torchrun \
    --nnodes="$NNODES" \
    --node-rank="$NODE_RANK" \
    --nproc-per-node="$GPUS_PER_NODE" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    scripts/train.py \
    --config "$CONFIG" \
    --fsdp \
    ${GRADIENT_CHECKPOINTING} \
    "${TRAIN_ARGS[@]}"
