#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  A100 40GB DeepSpeed Training Launch Script — Production Grade
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#  Hardware target: NVIDIA A100 40GB (single-node / multi-node)
#  Model: FLUX.1-schnell (12B) + rsLoRA via DeepSpeed ZeRO-3
#
#  Supports three modes:
#    multi   — 8x A100 40GB, ZeRO-3, NO offload (recommended)
#    single  — 1x A100 40GB, ZeRO-3 + CPU offload
#    multi-offload — 2-4x A100 40GB, ZeRO-3 + CPU offload
#
#  Usage:
#    # 8x A100 single node (default)
#    bash scripts/train_a100_40g_deepspeed.sh --mode multi --data-path /data/vision
#
#    # 1x A100 with CPU offload
#    bash scripts/train_a100_40g_deepspeed.sh --mode single --data-path /data/vision
#
#    # Multi-node (2 nodes × 8 GPUs)
#    # Node 0 (master):
#    bash scripts/train_a100_40g_deepspeed.sh --mode multi --nodes 2 --node-rank 0 --master-addr <IP>
#    # Node 1:
#    bash scripts/train_a100_40g_deepspeed.sh --mode multi --nodes 2 --node-rank 1 --master-addr <IP>
#
#    # Resume from checkpoint
#    bash scripts/train_a100_40g_deepspeed.sh --mode multi --resume output/checkpoints_a100_deepspeed/checkpoint-5000
#
#    # Custom config / deepspeed JSON
#    bash scripts/train_a100_40g_deepspeed.sh --mode multi --config configs/custom.yaml --ds-config configs/custom_ds.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Defaults ─────────────────────────────────────────────────────────
MODE="multi"                              # multi | single | multi-offload
CONFIG="${PROJECT_ROOT}/configs/training_a100_40g_deepspeed.yaml"
DS_CONFIG=""
DATA_PATH=""
RESUME=""
WANDB=false
WANDB_PROJECT="open-vision-banana-a100"
WANDB_RUN_NAME=""
EXTRA_ARGS=()

# Distributed defaults
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=${GPUS_PER_NODE:-0}         # 0 = auto-detect

# ── Color output ─────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ── Helper functions ─────────────────────────────────────────────────
log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_section() { echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"; shift 2 ;;
        --config)
            CONFIG="$2"; shift 2 ;;
        --ds-config)
            DS_CONFIG="$2"; shift 2 ;;
        --data-path)
            DATA_PATH="$2"; shift 2 ;;
        --resume)
            RESUME="$2"; shift 2 ;;
        --wandb)
            WANDB=true; shift ;;
        --wandb-project)
            WANDB_PROJECT="$2"; shift 2 ;;
        --wandb-run-name)
            WANDB_RUN_NAME="$2"; shift 2 ;;
        --nodes)
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
A100 40GB DeepSpeed Training Launch Script

Options:
  --mode {multi|single|multi-offload}
                          Training mode (default: multi)
  --config PATH           Training YAML config (default: configs/training_a100_40g_deepspeed.yaml)
  --ds-config PATH        Override DeepSpeed JSON config
  --data-path PATH        Path to vision training data
  --resume PATH           Resume from checkpoint directory
  --wandb                 Enable W&B logging
  --wandb-project NAME    W&B project name
  --wandb-run-name NAME   W&B run name
  --nodes N               Number of nodes (default: 1)
  --node-rank R           Rank of this node (default: 0)
  --master-addr IP        Master node IP (default: 127.0.0.1)
  --master-port PORT      Master port (default: 29500)
  --gpus-per-node G       GPUs per node (default: auto-detect)

All other arguments are passed through to scripts/train.py.
EOF
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ── Validate mode ────────────────────────────────────────────────────
case "$MODE" in
    multi|single|multi-offload) ;;
    *) log_error "Invalid mode: $MODE. Use: multi, single, multi-offload"; exit 1 ;;
esac

# ── Auto-detect GPUs ─────────────────────────────────────────────────
if [[ "$GPUS_PER_NODE" -eq 0 ]]; then
    if command -v nvidia-smi &>/dev/null; then
        GPUS_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    else
        log_error "nvidia-smi not found and --gpus-per-node not set"
        exit 1
    fi
fi

if [[ "$GPUS_PER_NODE" -eq 0 ]]; then
    log_error "No GPUs detected. Ensure NVIDIA driver and CUDA are installed."
    exit 1
fi

WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

# ── Select DeepSpeed config by mode ──────────────────────────────────
if [[ -z "$DS_CONFIG" ]]; then
    case "$MODE" in
        multi)
            DS_CONFIG="${PROJECT_ROOT}/configs/deepspeed_a100_40g_zero3.json"
            ;;
        single|multi-offload)
            DS_CONFIG="${PROJECT_ROOT}/configs/deepspeed_a100_40g_zero3_offload.json"
            ;;
    esac
fi

# ── Validate config files exist ──────────────────────────────────────
if [[ ! -f "$CONFIG" ]]; then
    log_error "Training config not found: $CONFIG"
    exit 1
fi
if [[ ! -f "$DS_CONFIG" ]]; then
    log_error "DeepSpeed config not found: $DS_CONFIG"
    exit 1
fi

# ── Set environment variables for A100 ───────────────────────────────
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-""}  # auto-detect
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

# A100-specific: enable TF32 for matmul (speedup ~1.5-2× for compatible ops)
export NVIDIA_TF32_OVERRIDE=1

# DeepSpeed-specific: avoid fork warning
export MKL_THREADING_LAYER=GNU

# If using CPU offload, set optimal numa policy
if [[ "$MODE" == "single" || "$MODE" == "multi-offload" ]]; then
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
fi

# ── Build training arguments ─────────────────────────────────────────
TRAIN_ARGS=(
    --config "$CONFIG"
    --deepspeed
    --deepspeed-config "$DS_CONFIG"
    --gradient-checkpointing
)

if [[ -n "$DATA_PATH" ]]; then
    TRAIN_ARGS+=(--data-path "$DATA_PATH")
fi

if [[ -n "$RESUME" ]]; then
    TRAIN_ARGS+=(--resume "$RESUME")
fi

if [[ "$WANDB" == true ]]; then
    TRAIN_ARGS+=(--wandb)
    TRAIN_ARGS+=(--wandb-project "$WANDB_PROJECT")
    if [[ -n "$WANDB_RUN_NAME" ]]; then
        TRAIN_ARGS+=(--wandb-run-name "$WANDB_RUN_NAME")
    fi
fi

# Add any extra passthrough arguments
TRAIN_ARGS+=("${EXTRA_ARGS[@]}")

# ── Print launch banner ──────────────────────────────────────────────
log_section
echo -e "${GREEN}  Open-Vision-Banana  —  A100 40GB DeepSpeed Training${NC}"
log_section
echo "  Mode:            $MODE"
echo "  Nodes:           $NNODES"
echo "  Node rank:       $NODE_RANK"
echo "  GPUs/node:       $GPUS_PER_NODE"
echo "  World size:      $WORLD_SIZE"
echo "  Master:          $MASTER_ADDR:$MASTER_PORT"
echo "  DeepSpeed:       $DS_CONFIG"
echo "  Training config: $CONFIG"
[[ -n "$DATA_PATH" ]] && echo "  Data path:       $DATA_PATH"
[[ -n "$RESUME" ]] && echo "  Resume:          $RESUME"
echo "  W&B:             $WANDB"
log_section
echo ""

# ── Preflight checks ─────────────────────────────────────────────────
log_info "Running preflight checks..."

# Check Python & DeepSpeed
if ! python3 -c "import deepspeed; print(f'DeepSpeed {deepspeed.__version__}')" 2>/dev/null; then
    log_error "DeepSpeed not installed. Run: pip install deepspeed"
    exit 1
fi

# Check CUDA
cuda_version=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
log_info "PyTorch CUDA version: $cuda_version"

# Check GPU compute capability (A100 = 8.0)
if command -v nvidia-smi &>/dev/null; then
    gpu_name=$(nvidia-smi -L | head -1 | sed 's/.*: \(.*\) (UUID.*/\1/')
    log_info "Detected GPU: $gpu_name"
fi

# Check NCCL
nccl_version=$(python3 -c "import torch; print(torch.cuda.nccl.version())" 2>/dev/null || echo "unknown")
log_info "NCCL version: $nccl_version"

# Verify config JSON is valid
if ! python3 -c "import json; json.load(open('$DS_CONFIG'))" 2>/dev/null; then
    log_error "DeepSpeed config JSON is invalid: $DS_CONFIG"
    exit 1
fi

echo ""

# ── Launch training ──────────────────────────────────────────────────
cd "$PROJECT_ROOT"

if [[ "$MODE" == "single" ]]; then
    # Single GPU: use deepspeed with --include to pin to gpu:0
    log_info "Launching single-GPU DeepSpeed training with CPU offload..."
    exec deepspeed --include localhost:0 \
        scripts/train.py \
        "${TRAIN_ARGS[@]}"
else
    # Multi-GPU: deepspeed launcher handles multi-node via hostfile or SLURM
    log_info "Launching multi-GPU DeepSpeed training (ZeRO-3)..."

    if command -v deepspeed &>/dev/null; then
        # Use deepspeed launcher (recommended for DeepSpeed)
        if [[ "$NNODES" -gt 1 ]]; then
            # Multi-node: deepspeed uses hostfile or environment
            export MASTER_ADDR
            export MASTER_PORT
            exec deepspeed \
                --num_gpus "$GPUS_PER_NODE" \
                --num_nodes "$NNODES" \
                --master_addr "$MASTER_ADDR" \
                --master_port "$MASTER_PORT" \
                scripts/train.py \
                "${TRAIN_ARGS[@]}"
        else
            # Single-node multi-GPU
            exec deepspeed \
                --num_gpus "$GPUS_PER_NODE" \
                scripts/train.py \
                "${TRAIN_ARGS[@]}"
        fi
    else
        log_error "deepspeed command not found. Install with: pip install deepspeed"
        exit 1
    fi
fi
