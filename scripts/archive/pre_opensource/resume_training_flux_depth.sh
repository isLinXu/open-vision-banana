#!/bin/bash
# Resume depth-only estimation training: warm start from checkpoint-50
# ================================================
# Training config: configs/training_flux_depth.yaml
# Data: data/vb_depth_pseudo_5k (5839 samples, pure depth)
# LoRA: rank=64, alpha=128, rsLoRA, target=to_k/to_v
# Resume: checkpoint-50 (loss=8.34, step=50)
#
# Fixes:
#   1. CheckpointManager.load loads the safetensors directly, avoiding PeftModel.from_pretrained double-wrapping
#   2. Key mapping: .lora_A.weight → .lora_A.default.weight
#   3. tuner.train() gains start_step/start_epoch parameters, so the LR scheduler resumes correctly
#   4. _direct_forward no longer unwraps __wrapped__, keeping the LoRA gradient path

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

cd "$PROJECT_DIR"

exec python scripts/train.py \
    --config configs/training_flux_depth.yaml \
    --model-name "/Users/gatilin/MyWork/models/ckpts/FLUX.1-schnell" \
    --data-path "data/vb_depth_pseudo_5k" \
    --batch-size 1 \
    --device mps \
    --epochs 3 \
    --lr 5e-5 \
    --log-every 1 \
    --resume "output/checkpoints_flux_depth/checkpoint-50" \
    --output-dir "output/checkpoints_flux_depth"
