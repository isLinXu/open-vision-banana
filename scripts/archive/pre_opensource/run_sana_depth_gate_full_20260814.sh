#!/bin/zsh
set -euo pipefail
cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec /Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python scripts/train_sana_512.py \
  --data-dir /Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v5_full \
  --pair-manifest output/mar_vb_clean_pairs_v4/pairs.jsonl \
  --tasks depth \
  --model-name sana-600m-512px \
  --steps 12000 \
  --lr 5e-5 \
  --scheduler cosine \
  --warmup-steps 600 \
  --batch-size 1 \
  --grad-accum 1 \
  --resolution 512 \
  --dtype float16 \
  --device mps \
  --no-min-snr \
  --cfg-dropout-rate 0 \
  --output-dir output/sana_depth_gate_full_20260814 \
  --save-every 500 \
  --eval-every 12000 \
  --log-every 10 \
  --num-workers 0
