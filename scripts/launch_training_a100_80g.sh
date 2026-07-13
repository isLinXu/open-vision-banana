#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Open-Vision-Banana A100-80GB Unified Training Launcher
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Single entry-point for all A100 80GB training modes:
#   single-GPU | DDP | FSDP | DeepSpeed ZeRO-3
#
# Auto-detects GPU count, validates environment, and dispatches to
# the appropriate distributed strategy.
#
# Usage:
#   # Interactive menu (auto-detects GPUs & recommends strategy)
#   bash scripts/launch_training_a100_80g.sh
#
#   # Explicit mode selection
#   bash scripts/launch_training_a100_80g.sh --mode ddp    --data-path data/vb_depth_pseudo_5k
#   bash scripts/launch_training_a100_80g.sh --mode fsdp   --gradient-checkpointing
#   bash scripts/launch_training_a100_80g.sh --mode ds     --deepspeed-config configs/deepspeed_zero3.json
#
#   # Override config or hyperparameters
#   bash scripts/launch_training_a100_80g.sh --mode ddp --config configs/training_a100_80g_multitask.yaml --epochs 20
#
#   # Multi-node DDP / FSDP
#   bash scripts/launch_training_a100_80g.sh --mode ddp --nodes 2 --node-rank 0 --master-addr 10.0.1.100

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Defaults ───────────────────────────────────────────────────────────
MODE="auto"                       # auto | single | ddp | fsdp | ds | deepspeed
CONFIG="${PROJECT_ROOT}/configs/training_a100_80g.yaml"
DATA_PATH=""
EXTRA_ARGS=()

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# ── Color output ───────────────────────────────────────────────────────
RED='\033[0;31m'
GRN='\033[0;32m'
YEL='\033[1;33m'
BLU='\033[0;34m'
CYN='\033[0;36m'
RST='\033[0m'

info()  { echo -e "${BLU}[INFO]${RST}  $*"; }
ok()    { echo -e "${GRN}[OK]${RST}    $*"; }
warn()  { echo -e "${YEL}[WARN]${RST}  $*"; }
err()   { echo -e "${RED}[ERR]${RST}   $*" >&2; }

# ── Environment validation ─────────────────────────────────────────────
check_env() {
    local fatal=0

    if ! command -v python3 &>/dev/null; then
        err "python3 not found in PATH"
        fatal=1
    fi

    if ! python3 -c "import torch" 2>/dev/null; then
        err "PyTorch not installed"
        fatal=1
    fi

    if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        err "CUDA not available — A100 training requires NVIDIA GPU + CUDA"
        fatal=1
    fi

    local cuda_arch
    cuda_arch=$(python3 -c "import torch; cap=torch.cuda.get_device_capability(); print(f'{cap[0]}{cap[1]}')" 2>/dev/null || echo "unknown")
    if [[ "$cuda_arch" != "80" && "$cuda_arch" != "unknown" ]]; then
        warn "Device compute capability is sm_$cuda_arch (expected sm_80 for A100)"
    fi

    if [[ ! -f "$CONFIG" ]]; then
        err "Config not found: $CONFIG"
        fatal=1
    fi

    if [[ $fatal -eq 1 ]]; then
        exit 1
    fi

    ok "Environment validated (CUDA arch: sm_${cuda_arch})"
}

# ── GPU auto-detection ─────────────────────────────────────────────────
detect_gpus() {
    if command -v nvidia-smi &>/dev/null; then
        GPUS_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    else
        GPUS_PER_NODE=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
    fi

    if [[ "$GPUS_PER_NODE" -eq 0 ]]; then
        err "No GPUs detected"
        exit 1
    fi

    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
    local vram
    vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')

    info "Detected: ${GPUS_PER_NODE} × ${gpu_name} (~${vram} MiB)"
}

# ── Parse arguments ────────────────────────────────────────────────────
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --mode)
                MODE="$2"; shift 2 ;;
            --config)
                CONFIG="$2"; shift 2 ;;
            --data-path)
                DATA_PATH="$2"; shift 2 ;;
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
Open-Vision-Banana A100-80GB Unified Launcher

Options:
  --mode MODE            Training mode: auto | single | ddp | fsdp | ds
  --config PATH          YAML config path (default: configs/training_a100_80g.yaml)
  --data-path PATH       Override dataset path
  --nodes N              Number of nodes (default: 1)
  --node-rank R          This node's rank (default: 0)
  --master-addr IP       Master node IP (default: 127.0.0.1)
  --master-port PORT     Master port (default: 29500)
  --gpus-per-node G      GPUs per node (default: auto-detect)

All remaining arguments are forwarded to scripts/train.py.
EOF
                exit 0
                ;;
            *)
                EXTRA_ARGS+=("$1")
                shift
                ;;
        esac
    done
}

# ── Recommend strategy based on GPU count ──────────────────────────────
recommend_mode() {
    if [[ "$GPUS_PER_NODE" -eq 1 ]]; then
        echo "single"
    elif [[ "$GPUS_PER_NODE" -ge 4 ]]; then
        echo "ddp"               # DDP is simplest and fastest for 4-8 A100s
    else
        echo "ddp"
    fi
}

# ── Build common train.py arguments ────────────────────────────────────
build_train_args() {
    local args=(
        --config "$CONFIG"
    )
    [[ -n "$DATA_PATH" ]] && args+=(--data-path "$DATA_PATH")
    args+=("${EXTRA_ARGS[@]}")
    printf '%s\n' "${args[@]}"
}

# ── Single-GPU launch ──────────────────────────────────────────────────
launch_single() {
    info "Mode: single-GPU training"
    ok "Launching on cuda:0"

    cd "$PROJECT_ROOT"
    exec python3 scripts/train.py \
        --config "$CONFIG" \
        ${DATA_PATH:+--data-path "$DATA_PATH"} \
        "${EXTRA_ARGS[@]}"
}

# ── DDP launch ─────────────────────────────────────────────────────────
launch_ddp() {
    local world_size=$((NNODES * GPUS_PER_NODE))
    info "Mode: DDP (Data Parallel)"
    info "Topology: ${NNODES} node(s) × ${GPUS_PER_NODE} GPU(s) = world_size ${world_size}"

    if ! command -v torchrun &>/dev/null; then
        err "torchrun not found. Install PyTorch >= 2.0."
        exit 1
    fi

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
        ${DATA_PATH:+--data-path "$DATA_PATH"} \
        "${EXTRA_ARGS[@]}"
}

# ── FSDP launch ────────────────────────────────────────────────────────
launch_fsdp() {
    local world_size=$((NNODES * GPUS_PER_NODE))
    info "Mode: FSDP (Fully Sharded Data Parallel)"
    info "Topology: ${NNODES} node(s) × ${GPUS_PER_NODE} GPU(s) = world_size ${world_size}"

    if ! command -v torchrun &>/dev/null; then
        err "torchrun not found. Install PyTorch >= 2.0."
        exit 1
    fi

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
        --gradient-checkpointing \
        ${DATA_PATH:+--data-path "$DATA_PATH"} \
        "${EXTRA_ARGS[@]}"
}

# ── DeepSpeed launch ───────────────────────────────────────────────────
launch_deepspeed() {
    local world_size=$((NNODES * GPUS_PER_NODE))
    local ds_config="${EXTRA_ARGS[*]}"
    local ds_config_path="configs/deepspeed_zero3.json"

    # Extract --deepspeed-config if provided
    for ((i=0; i<${#EXTRA_ARGS[@]}; i++)); do
        if [[ "${EXTRA_ARGS[$i]}" == "--deepspeed-config" ]]; then
            ds_config_path="${EXTRA_ARGS[$((i+1))]}"
            break
        fi
    done

    info "Mode: DeepSpeed ZeRO-3"
    info "Topology: ${NNODES} node(s) × ${GPUS_PER_NODE} GPU(s) = world_size ${world_size}"
    info "DeepSpeed config: ${ds_config_path}"

    if ! command -v deepspeed &>/dev/null; then
        err "deepspeed command not found. Install: pip install deepspeed"
        exit 1
    fi

    cd "$PROJECT_ROOT"
    exec deepspeed \
        --num_gpus="$GPUS_PER_NODE" \
        --num_nodes="$NNODES" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        --hostfile="" \
        scripts/train.py \
        --config "$CONFIG" \
        --deepspeed \
        --deepspeed-config "$ds_config_path" \
        --gradient-checkpointing \
        ${DATA_PATH:+--data-path "$DATA_PATH"} \
        "${EXTRA_ARGS[@]}"
}

# ── Print banner ───────────────────────────────────────────────────────
print_banner() {
    echo ""
    echo -e "${CYN}╔════════════════════════════════════════════════════════════════╗${RST}"
    echo -e "${CYN}║     Open-Vision-Banana  A100-80GB  Training Launcher         ║${RST}"
    echo -e "${CYN}╚════════════════════════════════════════════════════════════════╝${RST}"
    echo ""
}

# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
print_banner
parse_args "$@"
check_env
detect_gpus

# Auto-select mode
if [[ "$MODE" == "auto" ]]; then
    MODE=$(recommend_mode)
    info "Auto-selected mode: $MODE"
fi

# Validate mode
case "$MODE" in
    single|s|1)      launch_single ;;
    ddp|data_parallel) launch_ddp ;;
    fsdp|fully_sharded) launch_fsdp ;;
    ds|deepspeed|zero3) launch_deepspeed ;;
    *)
        err "Unknown mode: $MODE"
        echo "Valid modes: auto, single, ddp, fsdp, ds"
        exit 1
        ;;
esac
