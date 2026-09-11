#!/bin/zsh
set -euo pipefail

project_root="/Users/gatilin/PycharmProjects/open-vision-banana-git-v260722"
python_bin="/Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python"
data_root="/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v5_full"
pair_manifest="${project_root}/output/sana_structure_binary_targets_v1/pairs.jsonl"
output_dir="${project_root}/output/sana_structure_binary_specialist_20260813"

cd "${project_root}"
exec "${python_bin}" scripts/train_sana_512.py \
  --data-dir "${data_root}" \
  --pair-manifest "${pair_manifest}" \
  --tasks structure \
  --model-name sana-600m-512px \
  --steps 3000 \
  --lr 1e-4 \
  --batch-size 1 \
  --grad-accum 1 \
  --num-workers 0 \
  --resolution 512 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --scheduler constant \
  --warmup-ratio 0.0 \
  --ema-decay 0.9999 \
  --save-every 250 \
  --eval-every 250 \
  --log-every 25 \
  --task-weights uniform \
  --task-loss-weights structure=2 \
  --no-min-snr \
  --cfg-dropout-rate 0.05 \
  --output-dir "${output_dir}" \
  --device mps
