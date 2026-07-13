#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Open-Vision-Banana Unified Training Launch Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Production-grade launcher supporting all training paradigms:
#   • Single-GPU (CUDA / MPS / CPU)
#   • DDP (Distributed Data Parallel)
#   • FSDP (Fully Sharded Data Parallel)
#   • DeepSpeed ZeRO-1/2/3
#   • Multi-node cluster
#
# Usage:
#   bash scripts/unified_launch.sh --mode single_gpu --config configs/training_config.yaml
#   bash scripts/unified_launch.sh --mode ddp     --config configs/training_config.yaml
#   bash scripts/unified_launch.sh --mode fsdp    --config configs/training_config.yaml
#   bash scripts/unified_launch.sh --mode deepspeed --config configs/training_config.yaml
#   bash scripts/unified_launch.sh --mode mps     --config configs/training_mps.yaml
#   bash scripts/unified_launch.sh --mode cpu     --config configs/training_config.yaml
#
#   # Auto-detect best mode from hardware:
#   bash scripts/unified_launch.sh --auto --config configs/training_config.yaml
#
#   # Multi-node (node 0 / master):
#   bash scripts/unified_launch.sh --mode ddp --nodes 2 --node-rank 0 --master-addr 192.168.1.10
#
#   # Resume from checkpoint:
#   bash scripts/unified_launch.sh --mode fsdp --resume output/checkpoints/checkpoint-epoch_5
#
# Environment Variables (all optional, override script defaults):
#   OVB_MODE              single_gpu | ddp | fsdp | deepspeed | mps | cpu | auto
#   OVB_CONFIG            path to YAML config
#   OVB_NODES             number of nodes
#   OVB_NODE_RANK         rank of this node
#   OVB_GPUS_PER_NODE     GPUs per node (auto-detected if unset)
#   OVB_MASTER_ADDR       master node IP
#   OVB_MASTER_PORT       master node port
#   OVB_OUTPUT_DIR        output directory for checkpoints
#   OVB_BATCH_SIZE        override batch size
#   OVB_LR                override learning rate
#   OVB_EPOCHS            override epochs
#   OVB_SEED              random seed
#   HF_TOKEN              Hugging Face access token
#   WANDB_API_KEY         Weights & Biases API key
#
# Exit Codes:
#   0  success
#   1  general error / argument error
#   2  hardware detection failure
#   3  dependency missing
#   4  checkpoint not found
#   5  distributed initialization failure
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -euo pipefail

# ── Script Identity ──────────────────────────────────────────────────
readonly SCRIPT_NAME="unified_launch.sh"
readonly SCRIPT_VERSION="1.0.0"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Color Output ─────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    readonly C_RED='\033[0;31m'
    readonly C_GREEN='\033[0;32m'
    readonly C_YELLOW='\033[1;33m'
    readonly C_BLUE='\033[0;34m'
    readonly C_CYAN='\033[0;36m'
    readonly C_BOLD='\033[1m'
    readonly C_RESET='\033[0m'
else
    readonly C_RED='' C_GREEN='' C_YELLOW='' C_BLUE='' C_CYAN='' C_BOLD='' C_RESET=''
fi

_log_info()  { echo -e "${C_BLUE}[INFO]${C_RESET}  $*"; }
_log_warn()  { echo -e "${C_YELLOW}[WARN]${C_RESET}  $*" >&2; }
_log_error() { echo -e "${C_RED}[ERROR]${C_RESET} $*" >&2; }
_log_success(){ echo -e "${C_GREEN}[OK]${C_RESET}    $*"; }
_log_banner() { echo -e "${C_BOLD}${C_CYAN}$*${C_RESET}"; }

# ── Default Configuration ────────────────────────────────────────────
MODE="${OVB_MODE:-auto}"
CONFIG="${OVB_CONFIG:-configs/training_config.yaml}"
NNODES="${OVB_NODES:-1}"
NODE_RANK="${OVB_NODE_RANK:-0}"
GPUS_PER_NODE="${OVB_GPUS_PER_NODE:-}"
MASTER_ADDR="${OVB_MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${OVB_MASTER_PORT:-29500}"
OUTPUT_DIR="${OVB_OUTPUT_DIR:-}"
BATCH_SIZE="${OVB_BATCH_SIZE:-}"
LR="${OVB_LR:-}"
EPOCHS="${OVB_EPOCHS:-}"
SEED="${OVB_SEED:-42}"
RESUME=""
DRY_RUN=false
VERBOSE=false

# DeepSpeed config
DEEPSPEED_CONFIG="${OVB_DEEPSPEED_CONFIG:-configs/deepspeed_zero3.json}"

# Training extras
GRADIENT_CHECKPOINTING=false
WANDB=false
TENSORBOARD=false

# ── Helper: Print Usage ──────────────────────────────────────────────
print_usage() {
    cat <<'EOF'
Usage: unified_launch.sh [OPTIONS]

Required:
  --config PATH         Path to training YAML config

Mode Selection (pick one):
  --mode MODE           Training mode:
                        • single_gpu  – single CUDA GPU
                        • ddp         – Distributed Data Parallel
                        • fsdp        – Fully Sharded Data Parallel
                        • deepspeed   – DeepSpeed ZeRO
                        • mps         – Apple Silicon MPS
                        • cpu         – CPU training
                        • auto        – auto-detect from hardware (default)

Distributed Options:
  --nodes N             Number of nodes (default: 1)
  --node-rank R         Rank of this node (default: 0)
  --gpus-per-node G     GPUs per node (default: auto-detect)
  --master-addr IP      Master node IP (default: 127.0.0.1)
  --master-port PORT    Master port (default: 29500)

Training Overrides:
  --output-dir PATH     Checkpoint output directory
  --batch-size N        Override batch size
  --lr LR               Override learning rate
  --epochs N            Override epochs
  --seed N              Random seed (default: 42)
  --resume PATH         Resume from checkpoint directory
  --gradient-checkpointing
                        Enable gradient checkpointing

Logging:
  --wandb               Enable Weights & Biases
  --tensorboard         Enable TensorBoard

Utility:
  --dry-run             Print command without executing
  --verbose             Verbose output
  --help, -h            Show this help

Examples:
  # Auto-detect + single GPU
  bash scripts/unified_launch.sh --config configs/training_config.yaml

  # DDP on 4 GPUs
  bash scripts/unified_launch.sh --mode ddp --config configs/training_config.yaml

  # FSDP with gradient checkpointing
  bash scripts/unified_launch.sh --mode fsdp --config configs/training_config.yaml --gradient-checkpointing

  # DeepSpeed ZeRO-3
  bash scripts/unified_launch.sh --mode deepspeed --config configs/training_config.yaml

  # Apple MPS
  bash scripts/unified_launch.sh --mode mps --config configs/training_mps.yaml

  # Multi-node DDP (node 0)
  bash scripts/unified_launch.sh --mode ddp --nodes 2 --node-rank 0 \
      --master-addr 192.168.1.10 --config configs/training_config.yaml

  # Resume training
  bash scripts/unified_launch.sh --mode fsdp --config configs/training_config.yaml \
      --resume output/checkpoints/checkpoint-epoch_3
EOF
}

# ── Helper: Parse Arguments ──────────────────────────────────────────
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --mode)
                MODE="$2"; shift 2 ;;
            --config)
                CONFIG="$2"; shift 2 ;;
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
            --output-dir)
                OUTPUT_DIR="$2"; shift 2 ;;
            --batch-size)
                BATCH_SIZE="$2"; shift 2 ;;
            --lr)
                LR="$2"; shift 2 ;;
            --epochs)
                EPOCHS="$2"; shift 2 ;;
            --seed)
                SEED="$2"; shift 2 ;;
            --resume)
                RESUME="$2"; shift 2 ;;
            --deepspeed-config)
                DEEPSPEED_CONFIG="$2"; shift 2 ;;
            --gradient-checkpointing)
                GRADIENT_CHECKPOINTING=true; shift ;;
            --wandb)
                WANDB=true; shift ;;
            --tensorboard)
                TENSORBOARD=true; shift ;;
            --dry-run)
                DRY_RUN=true; shift ;;
            --verbose)
                VERBOSE=true; shift ;;
            --help|-h)
                print_usage; exit 0 ;;
            *)
                _log_error "Unknown argument: $1"
                print_usage >&2
                exit 1 ;;
        esac
    done
}

# ── Helper: Check Dependencies ───────────────────────────────────────
check_dependencies() {
    local missing=()

    if ! command -v python3 &>/dev/null; then
        missing+=("python3")
    fi

    if [[ "$MODE" != "cpu" && "$MODE" != "mps" ]]; then
        if ! python3 -c "import torch" 2>/dev/null; then
            missing+=("torch")
        fi
    fi

    if [[ "$MODE" == "deepspeed" ]]; then
        if ! python3 -c "import deepspeed" 2>/dev/null; then
            missing+=("deepspeed")
        fi
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        _log_error "Missing dependencies: ${missing[*]}"
        _log_info "Install with: pip install torch"
        [[ "$MODE" == "deepspeed" ]] && _log_info "              pip install deepspeed"
        exit 3
    fi
}

# ── Helper: Detect Hardware ──────────────────────────────────────────
detect_hardware() {
    _log_info "Detecting hardware environment..."

    local has_cuda=false
    local has_mps=false
    local cuda_count=0

    # Check CUDA
    if command -v nvidia-smi &>/dev/null; then
        cuda_count=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$cuda_count" -gt 0 ]]; then
            has_cuda=true
        fi
    elif python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        cuda_count=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
        if [[ "$cuda_count" -gt 0 ]]; then
            has_cuda=true
        fi
    fi

    # Check MPS
    if python3 -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
        has_mps=true
    fi

    # Auto-detect mode
    if [[ "$MODE" == "auto" ]]; then
        if [[ "$has_cuda" == true ]]; then
            if [[ "$cuda_count" -gt 1 ]]; then
                MODE="ddp"
                _log_info "Auto-selected mode: ddp ($cuda_count GPUs detected)"
            else
                MODE="single_gpu"
                _log_info "Auto-selected mode: single_gpu (1 GPU detected)"
            fi
        elif [[ "$has_mps" == true ]]; then
            MODE="mps"
            _log_info "Auto-selected mode: mps (Apple Silicon detected)"
        else
            MODE="cpu"
            _log_info "Auto-selected mode: cpu (no GPU/MPS detected)"
        fi
    fi

    # Auto-detect GPUS_PER_NODE if not set
    if [[ -z "$GPUS_PER_NODE" ]]; then
        if [[ "$has_cuda" == true ]]; then
            GPUS_PER_NODE=$cuda_count
        elif [[ "$MODE" == "mps" ]]; then
            GPUS_PER_NODE=1
        else
            GPUS_PER_NODE=0
        fi
    fi

    # Export for child processes
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

    _log_info "Hardware: CUDA=$has_cuda (count=$cuda_count), MPS=$has_mps"
    _log_info "Mode: $MODE, GPUs/node: $GPUS_PER_NODE"
}

# ── Helper: Validate Configuration ───────────────────────────────────
validate_config() {
    local config_path="$PROJECT_ROOT/$CONFIG"
    if [[ ! -f "$config_path" ]]; then
        _log_error "Config file not found: $config_path"
        exit 1
    fi
    _log_success "Config: $CONFIG"
}

# ── Helper: Validate Resume Path ─────────────────────────────────────
validate_resume() {
    if [[ -n "$RESUME" ]]; then
        local resume_path="$PROJECT_ROOT/$RESUME"
        if [[ ! -d "$resume_path" ]]; then
            _log_error "Resume checkpoint directory not found: $resume_path"
            exit 4
        fi
        _log_success "Resume from: $RESUME"
    fi
}

# ── Helper: Setup Environment ────────────────────────────────────────
setup_environment() {
    cd "$PROJECT_ROOT"

    # Python path
    export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

    # Hugging Face token
    if [[ -f "$HOME/.huggingface/token" ]]; then
        export HF_TOKEN="$(cat "$HOME/.huggingface/token")"
        export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
        _log_info "HF token loaded from ~/.huggingface/token"
    fi

    # Distributed env vars (for non-torchrun modes)
    export MASTER_ADDR="$MASTER_ADDR"
    export MASTER_PORT="$MASTER_PORT"

    # PyTorch settings
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

    # CUDA optimizations
    if [[ "$MODE" != "mps" && "$MODE" != "cpu" ]]; then
        export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
        export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-INFO}"
    fi

    if [[ "$VERBOSE" == true ]]; then
        _log_info "PYTHONPATH=$PYTHONPATH"
        _log_info "MASTER_ADDR=$MASTER_ADDR:$MASTER_PORT"
        _log_info "OMP_NUM_THREADS=$OMP_NUM_THREADS"
    fi
}

# ── Helper: Build Training Arguments ─────────────────────────────────
build_train_args() {
    local args=()

    args+=(--config "$CONFIG")
    args+=(--seed "$SEED")

    if [[ -n "$OUTPUT_DIR" ]]; then
        args+=(--output-dir "$OUTPUT_DIR")
    fi
    if [[ -n "$BATCH_SIZE" ]]; then
        args+=(--batch-size "$BATCH_SIZE")
    fi
    if [[ -n "$LR" ]]; then
        args+=(--lr "$LR")
    fi
    if [[ -n "$EPOCHS" ]]; then
        args+=(--epochs "$EPOCHS")
    fi
    if [[ -n "$RESUME" ]]; then
        args+=(--resume "$RESUME")
    fi
    if [[ "$GRADIENT_CHECKPOINTING" == true ]]; then
        args+=(--gradient-checkpointing)
    fi
    if [[ "$WANDB" == true ]]; then
        args+=(--wandb)
    fi
    if [[ "$TENSORBOARD" == true ]]; then
        args+=(--tensorboard)
    fi

    # Mode-specific flags
    case "$MODE" in
        ddp)
            args+=(--ddp)
            ;;
        fsdp)
            args+=(--fsdp)
            ;;
        deepspeed)
            args+=(--deepspeed)
            args+=(--deepspeed-config "$DEEPSPEED_CONFIG")
            ;;
        mps)
            args+=(--device mps)
            ;;
        cpu)
            args+=(--device cpu)
            ;;
    esac

    echo "${args[@]}"
}

# ── Helper: Build torchrun Command ───────────────────────────────────
build_torchrun_cmd() {
    local world_size=$((NNODES * GPUS_PER_NODE))

    local cmd=(torchrun)
    cmd+=(--nnodes="$NNODES")
    cmd+=(--node-rank="$NODE_RANK")
    cmd+=(--nproc-per-node="$GPUS_PER_NODE")
    cmd+=(--rdzv-backend=c10d)
    cmd+=(--rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT")
    cmd+=(scripts/train.py)

    local train_args
    train_args=$(build_train_args)
    read -ra train_args_arr <<< "$train_args"
    cmd+=("${train_args_arr[@]}")

    echo "${cmd[*]}"
}

# ── Helper: Build Single-Process Command ─────────────────────────────
build_single_cmd() {
    local cmd=(python3 scripts/train.py)

    local train_args
    train_args=$(build_train_args)
    read -ra train_args_arr <<< "$train_args"
    cmd+=("${train_args_arr[@]}")

    echo "${cmd[*]}"
}

# ── Helper: Build DeepSpeed Command ──────────────────────────────────
build_deepspeed_cmd() {
    local world_size=$((NNODES * GPUS_PER_NODE))

    local cmd=(deepspeed)
    cmd+=(--num_gpus="$GPUS_PER_NODE")
    cmd+=(--num_nodes="$NNODES")
    cmd+=(--master_addr="$MASTER_ADDR")
    cmd+=(--master_port="$MASTER_PORT")
    cmd+=(scripts/train.py)

    local train_args
    train_args=$(build_train_args)
    read -ra train_args_arr <<< "$train_args"
    cmd+=("${train_args_arr[@]}")

    echo "${cmd[*]}"
}

# ── Helper: Execute Command ──────────────────────────────────────────
execute() {
    local cmd="$1"

    _log_banner ""
    _log_banner "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    _log_banner "  Open-Vision-Banana Unified Training Launcher v${SCRIPT_VERSION}"
    _log_banner "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    _log_info "Mode:        $MODE"
    _log_info "Config:      $CONFIG"
    _log_info "Nodes:       $NNODES"
    _log_info "Node rank:   $NODE_RANK"
    _log_info "GPUs/node:   $GPUS_PER_NODE"
    _log_info "Master:      $MASTER_ADDR:$MASTER_PORT"
    _log_info "Seed:        $SEED"
    [[ -n "$OUTPUT_DIR" ]] && _log_info "Output:      $OUTPUT_DIR"
    [[ -n "$RESUME" ]]   && _log_info "Resume:      $RESUME"
    _log_banner "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    _log_banner ""

    if [[ "$DRY_RUN" == true ]]; then
        _log_info "[DRY RUN] Command:"
        echo "$cmd"
        return 0
    fi

    _log_info "Launching training..."
    _log_info "Command: $cmd"
    echo ""

    # Execute
    eval "$cmd"
    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        _log_success "Training completed successfully!"
    else
        _log_error "Training exited with code $exit_code"
    fi

    return $exit_code
}

# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

main() {
    parse_args "$@"

    # Validate mode
    case "$MODE" in
        single_gpu|ddp|fsdp|deepspeed|mps|cpu|auto) ;;
        *)
            _log_error "Invalid mode: $MODE"
            _log_info "Valid modes: single_gpu, ddp, fsdp, deepspeed, mps, cpu, auto"
            exit 1
            ;;
    esac

    # Validate config
    validate_config

    # Check dependencies
    check_dependencies

    # Detect hardware & auto-select mode
    detect_hardware

    # Validate resume path
    validate_resume

    # Setup environment
    setup_environment

    # Build and execute command based on mode
    local cmd
    case "$MODE" in
        single_gpu|mps|cpu)
            cmd=$(build_single_cmd)
            ;;
        ddp|fsdp)
            if ! command -v torchrun &>/dev/null; then
                _log_error "torchrun not found. Install PyTorch >= 2.0."
                exit 3
            fi
            cmd=$(build_torchrun_cmd)
            ;;
        deepspeed)
            if ! command -v deepspeed &>/dev/null; then
                _log_error "deepspeed not found. Install with: pip install deepspeed"
                exit 3
            fi
            cmd=$(build_deepspeed_cmd)
            ;;
    esac

    execute "$cmd"
}

main "$@"
