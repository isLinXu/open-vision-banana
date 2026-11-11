#!/bin/bash
# MPS training launcher
# ================================
# Usage:
#   bash scripts/launch_training_mps.sh [fluxe|sd15]
#
# Option A (fluxe): FLUX.1-schnell (needs a Hugging Face token)
# Option B (sd15):  SD 1.5 (public model, no token needed, to check the MPS pipeline)

set -e

MODE="${1:-fluxe}"
PYTHON="/usr/local/bin/python3.11"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH}"

# Set the HF token (if present)
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
    echo ">> Using SD 1.5 to check the MPS training pipeline (no HF token needed)"
    echo ">> Note: SD 1.5 is not FLUX.1 and trains differently; this only checks the pipeline"
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
    echo ">> Using FLUX.1-schnell for the real training run"
    echo ">> Checking the HF token..."

    HF_TOKEN_FILE="$HOME/.huggingface/token"
    if [ ! -f "$HF_TOKEN_FILE" ] && [ -z "$HF_TOKEN" ]; then
        echo ""
        echo "⚠️  FLUX.1-schnell is a gated Hugging Face repository and needs an access token"
        echo ""
        echo "Get a token as follows:"
        echo "  1. Open https://huggingface.co/black-forest-labs/FLUX.1-schnell"
        echo "  2. Click 'Access repository' and accept the license"
        echo "  3. Go to Settings → Access Tokens → New Token"
        echo "  4. Create a token with Read permission"
        echo "  5. In a terminal run: echo 'YOUR_TOKEN' > ~/.huggingface/token"
        echo ""
        echo "Or check the MPS training pipeline with the fallback option:"
        echo "  bash scripts/launch_training_mps.sh sd15"
        echo ""
        exit 1
    fi

    echo ">> HF token configured, starting training..."
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
    echo "Usage: bash scripts/launch_training_mps.sh [fluxe|sd15]"
    echo "  fluxe - real FLUX.1-schnell training (needs an HF token)"
    echo "  sd15  - SD 1.5 pipeline check (no HF token needed)"
    exit 1
fi

echo ""
echo "========================================"
echo "Training session completed or interrupted"
echo "Check logs and checkpoints in output/"
echo "========================================"
