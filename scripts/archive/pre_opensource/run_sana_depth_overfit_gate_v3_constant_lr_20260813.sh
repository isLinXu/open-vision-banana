#!/bin/zsh
set -euo pipefail
cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
exec /Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python scripts/train_sana_512.py \
  --data-dir /Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v5_full \
  --pair-manifest output/mar_vb_clean_pairs_v4/pairs.jsonl \
  --tasks depth \
  --max-samples-per-task 4 \
  --model-name sana-600m-512px \
  --steps 400 \
  --lr 1e-4 \
  --scheduler constant \
  --warmup-steps 10 \
  --batch-size 1 \
  --grad-accum 1 \
  --resolution 512 \
  --dtype float16 \
  --device mps \
  --no-min-snr \
  --cfg-dropout-rate 0 \
  --output-dir output/sana_depth_overfit_gate_v3_constant_lr_20260813 \
  --save-every 400 \
  --eval-every 400 \
  --log-every 25 \
  --num-workers 0
