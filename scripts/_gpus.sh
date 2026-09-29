# Sourced by the torchrun launchers. Prints how many GPUs this node should use:
#   1. an explicit count (first argument, from --gpus-per-node / GPUS_PER_NODE),
#   2. else the number of devices in CUDA_VISIBLE_DEVICES,
#   3. else every GPU nvidia-smi lists,
#   4. else torch.cuda.device_count().
# nvidia-smi ignores CUDA_VISIBLE_DEVICES, so it must not win over (2).
detect_gpus_per_node() {
    if [[ -n "${1:-}" ]]; then
        echo "$1"
    elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES"
    elif command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
        nvidia-smi -L | wc -l | tr -d ' '
    else
        python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0
    fi
}
