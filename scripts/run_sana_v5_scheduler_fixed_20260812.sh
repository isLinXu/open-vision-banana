#!/bin/zsh
set -euo pipefail

project_root="/Users/gatilin/PycharmProjects/open-vision-banana-git-v260722"
python_bin="/Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python"
data_root="/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v5_full"
pair_manifest="${project_root}/output/mar_vb_clean_pairs_v4/pairs.jsonl"
output_dir="${project_root}/output/sana_v5_full_scheduler_fixed_20260812"

cd "${project_root}"
exec "${python_bin}" scripts/train_sana_512.py \
  --data-dir "${data_root}" \
  --pair-manifest "${pair_manifest}" \
  --tasks depth normal segmentation structure \
  --model-name sana-600m-512px \
  --steps 6000 \
  --lr 5e-5 \
  --batch-size 1 \
  --grad-accum 1 \
  --num-workers 0 \
  --resolution 512 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --scheduler cosine \
  --warmup-ratio 0.05 \
  --ema-decay 0.9999 \
  --save-every 500 \
  --eval-every 500 \
  --log-every 50 \
  --no-min-snr \
  --output-dir "${output_dir}" \
  --device mps
