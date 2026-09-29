#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# A100-80GB DDP (Data Parallel) Training Launch Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Optimized for 1–8×A100-80GB nodes using PyTorch DDP.
# Recommended as the default strategy for A100 clusters:
#   - Simplest setup (no FSDP sharding complexity)
#   - Fastest iteration speed for batch sizes that fit in 80GB
#   - Full model replica per GPU (12B params ~ 24GB @ bfloat16)
#
# Usage:
#   # Single-node, auto-detect GPUs
#   bash scripts/launch_training_a100_80g_ddp.sh
#
#   # With custom config / data
#   bash scripts/launch_training_a100_80g_ddp.sh \
#       --config configs/training_a100_80g_multitask.yaml \
#       --data-path data/vb_v4_depth_2k
#
#   # Multi-node (2×8 A100)
#   # Node 0 (master):
#   bash scripts/launch_training_a100_80g_ddp.sh --nodes 2 --node-rank 0 --master-addr 10.0.1.100
#   # Node 1 (worker):
#   bash scripts/launch_training_a100_80g_ddp.sh --nodes 2 --node-rank 1 --master-addr 10.0.1.100
#
#   # Resume from checkpoint
#   bash scripts/launch_training_a100_80g_ddp.sh --resume output/checkpoints_a100_80g/checkpoint-5000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/_gpus.sh"

# ── Defaults ───────────────────────────────────────────────────────────
CONFIG="${PROJECT_ROOT}/configs/training_a100_80g.yaml"
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=${GPUS_PER_NODE:-}  # empty: detect (see scripts/_gpus.sh)

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
        --master-addr)
            MASTER_ADDR="$2"; shift 2 ;;
        --master-port)
            MASTER_PORT="$2"; shift 2 ;;
        --gpus-per-node)
            GPUS_PER_NODE="$2"; shift 2 ;;
        --help|-h)
            cat <<'EOF'
A100-80GB DDP Training Launcher

Options:
  --config PATH          YAML config path
  --nodes N              Number of nodes (default: 1)
  --node-rank R          This node's rank (default: 0)
  --master-addr IP       Master node IP (default: 127.0.0.1)
  --master-port PORT     Master port (default: 29500)
  --gpus-per-node G      GPUs per node (default: auto-detect)

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
GPUS_PER_NODE=$(detect_gpus_per_node "$GPUS_PER_NODE")
if [[ "$GPUS_PER_NODE" -eq 0 ]]; then
    echo "[ERR] No GPUs detected; pass --gpus-per-node."
    exit 1
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
echo "║     A100-80GB  DDP  Training  Launch                             ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
printf "  %-20s %s\n" "Config:"      "$CONFIG"
printf "  %-20s %s\n" "Nodes:"       "$NNODES"
printf "  %-20s %s\n" "Node rank:"   "$NODE_RANK"
printf "  %-20s %s\n" "GPUs/node:"   "$GPUS_PER_NODE"
printf "  %-20s %s\n" "World size:"  "$WORLD_SIZE"
printf "  %-20s %s\n" "Master:"      "$MASTER_ADDR:$MASTER_PORT"
printf "  %-20s %s\n" "Extra args:"  "${TRAIN_ARGS[*]}"
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
    --ddp \
    "${TRAIN_ARGS[@]}"
