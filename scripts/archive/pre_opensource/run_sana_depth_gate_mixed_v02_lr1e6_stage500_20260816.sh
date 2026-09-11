#!/bin/zsh
set -euo pipefail
cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
export PYTHONUNBUFFERED=1 PYTHONPATH=src PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1
export OPEN_VISION_BANANA_MODEL_PATH=/Users/gatilin/.cache/huggingface/hub/models--Efficient-Large-Model--Sana_600M_512px_diffusers/snapshots/83d7a190bfd1fd070570a793d2dab5c7a3231b9d
exec /Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python scripts/train_sana_512.py --config configs/experiments/sana_depth_gate_mixed_v02_lr1e6_stage500_20260816.yaml --resume output/sana_depth_gate_mixed_v02_lr1e6_stage250_20260816/checkpoint-250_final/lora.pt
