#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# A100-80GB DeepSpeed ZeRO-3 Training Launch Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Uses DeepSpeed ZeRO-3 for maximum memory efficiency and massive
# batch scaling. Parameters, gradients, and optimizer states are
# partitioned across all GPUs; each GPU holds only a slice.
#
# Best for:
#   - Batch sizes > 128 that would OOM even with FSDP
#   - Training with CPU/NVMe offloading (not enabled by default for A100)
#   - Seamless integration with DeepSpeed features (1-bit Adam, etc.)
#
# Usage:
#   # Single-node 8×A100 with default ZeRO-3 config
#   bash scripts/launch_training_a100_80g_deepspeed.sh
#
#   # Custom DeepSpeed JSON
#   bash scripts/launch_training_a100_80g_deepspeed.sh \
#       --deepspeed-config configs/deepspeed_zero3.json
#
#   # With custom training config and data
#   bash scripts/launch_training_a100_80g_deepspeed.sh \
#       --config configs/training_a100_80g_multitask.yaml \
#       --data-path data/vb_v4_depth_2k
#
#   # Multi-node
#   bash scripts/launch_training_a100_80g_deepspeed.sh --nodes 2 --node-rank 0 --master-addr 10.0.1.100

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Defaults ───────────────────────────────────────────────────────────
CONFIG="${PROJECT_ROOT}/configs/training_a100_80g.yaml"
DEEPSPEED_CONFIG="${PROJECT_ROOT}/configs/deepspeed_zero3.json"
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}

TRAIN_ARGS=()

# ── Parse script args ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"; shift 2 ;;
        --deepspeed-config)
            DEEPSPEED_CONFIG="$2"; shift 2 ;;
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
        --help|-h)
            cat <<'EOF'
A100-80GB DeepSpeed ZeRO-3 Training Launcher

Options:
  --config PATH               YAML training config path
  --deepspeed-config PATH     DeepSpeed JSON config path (default: configs/deepspeed_zero3.json)
  --nodes N                   Number of nodes (default: 1)
  --node-rank R               This node's rank (default: 0)
  --gpus-per-node G           GPUs per node (default: auto-detect)
  --master-addr IP            Master node IP (default: 127.0.0.1)
  --master-port PORT          Master port (default: 29500)

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
if ! command -v deepspeed &>/dev/null; then
    echo "[ERR] deepspeed command not found. Install: pip install deepspeed"
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "[ERR] Training config not found: $CONFIG"
    exit 1
fi

if [[ ! -f "$DEEPSPEED_CONFIG" ]]; then
    echo "[ERR] DeepSpeed config not found: $DEEPSPEED_CONFIG"
    exit 1
fi

# ── Banner ─────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     A100-80GB  DeepSpeed ZeRO-3  Training  Launch                ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
printf "  %-20s %s\n" "Train config:"     "$CONFIG"
printf "  %-20s %s\n" "DeepSpeed config:" "$DEEPSPEED_CONFIG"
printf "  %-20s %s\n" "Nodes:"            "$NNODES"
printf "  %-20s %s\n" "Node rank:"        "$NODE_RANK"
printf "  %-20s %s\n" "GPUs/node:"        "$GPUS_PER_NODE"
printf "  %-20s %s\n" "World size:"       "$WORLD_SIZE"
printf "  %-20s %s\n" "Master:"           "$MASTER_ADDR:$MASTER_PORT"
printf "  %-20s %s\n" "Extra args:"       "${TRAIN_ARGS[*]}"
echo ""

# ── Launch ─────────────────────────────────────────────────────────────
cd "$PROJECT_ROOT"

exec deepspeed \
    --num_gpus="$GPUS_PER_NODE" \
    --num_nodes="$NNODES" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    scripts/train.py \
    --config "$CONFIG" \
    --deepspeed \
    --deepspeed-config "$DEEPSPEED_CONFIG" \
    --gradient-checkpointing \
    "${TRAIN_ARGS[@]}"
