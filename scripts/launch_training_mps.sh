#!/bin/bash
# MPS 训练启动脚本
# ================================
# 使用方法：
#   bash scripts/launch_training_mps.sh [fluxe|sd15]
#
# 方案 A (fluxe): 使用 FLUX.1-schnell（需要 Hugging Face token）
# 方案 B (sd15):  使用 SD 1.5（公开模型，无需 token，用于验证 MPS 流程）

set -e

MODE="${1:-fluxe}"
PYTHON="/usr/local/bin/python3.11"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH}"

# 设置 HF Token（如果存在）
if [ -f "$HOME/.huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.huggingface/token")"
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "========================================"
echo "Open-Vision-Banana MPS Training Launcher"
echo "========================================"
echo "Mode: ${MODE}"
echo "Python: ${PYTHON}"
echo "Project: ${PROJECT_DIR}"
echo "MPS: $(python3 -c 'import torch; print(torch.backends.mps.is_available())' 2>/dev/null || echo 'unknown')"
echo ""

if [ "$MODE" = "sd15" ]; then
    echo ">> 使用 SD 1.5 进行 MPS 训练流程验证（无需 HF token）"
    echo ">> 注意：SD 1.5 不是 FLUX.1，训练效果不同，仅用于验证流程"
    echo ""

    $PYTHON scripts/train.py \
        --config configs/training_mps.yaml \
        --model-name "runwayml/stable-diffusion-v1-5" \
        --data-path "data/vb_depth_300" \
        --generation-path "data/vb_generation_5k" \
        --batch-size 1 \
        --device mps \
        --epochs 2 \
        --lr 5e-5 \
        --tensorboard \
        --output-dir "output/checkpoints_sd15_smoke"

elif [ "$MODE" = "fluxe" ]; then
    echo ">> 使用 FLUX.1-schnell 进行正式训练"
    echo ">> 检查 HF token..."

    HF_TOKEN_FILE="$HOME/.huggingface/token"
    if [ ! -f "$HF_TOKEN_FILE" ] && [ -z "$HF_TOKEN" ]; then
        echo ""
        echo "⚠️  FLUX.1-schnell 是 Hugging Face Gated Repository，需要 Access Token"
        echo ""
        echo "请按以下步骤获取 token："
        echo "  1. 访问 https://huggingface.co/black-forest-labs/FLUX.1-schnell"
        echo "  2. 点击 'Access repository' 并同意许可协议"
        echo "  3. 进入 Settings → Access Tokens → New Token"
        echo "  4. 创建 Read 权限的 token"
        echo "  5. 在终端运行: echo 'YOUR_TOKEN' > ~/.huggingface/token"
        echo ""
        echo "或者使用备用方案验证 MPS 训练流程："
        echo "  bash scripts/launch_training_mps.sh sd15"
        echo ""
        exit 1
    fi

    echo ">> HF token 已配置，开始训练..."
    echo ""

    $PYTHON scripts/train.py \
        --config configs/training_mps.yaml \
        --data-path "data/vb_depth_300" \
        --generation-path "data/vb_generation_5k" \
        --batch-size 1 \
        --device mps \
        --tensorboard \
        --output-dir "output/checkpoints_mps"

else
    echo "用法: bash scripts/launch_training_mps.sh [fluxe|sd15]"
    echo "  fluxe - FLUX.1-schnell 正式训练（需要 HF token）"
    echo "  sd15  - SD 1.5 流程验证（无需 HF token）"
    exit 1
fi

echo ""
echo "========================================"
echo "Training session completed or interrupted"
echo "Check logs and checkpoints in output/"
echo "========================================"
