#!/bin/zsh
set -euo pipefail

project_root="/Users/gatilin/PycharmProjects/open-vision-banana-git-v260722"
python_bin="/Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python"
data_root="/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v5_full"
pair_manifest="${project_root}/output/mar_vb_clean_pairs_v4/pairs.jsonl"
sana_out="${project_root}/output/sana_v5_full_lora_20260812"
mar_out="${project_root}/output/mar_v5_full_mar_large_palette_v4_20260812"

cd "${project_root}"

"${python_bin}" scripts/train_sana_512.py \
  --data-dir "${data_root}" \
  --pair-manifest "${pair_manifest}" \
  --tasks depth normal segmentation structure \
  --model-name sana-600m-512px \
  --steps 6000 \
  --lr 5e-5 \
  --batch-size 1 \
  --grad-accum 8 \
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
  --resume output/sana_512_vb/checkpoint-6000_best/lora.pt \
  --resume-weights-only \
  --output-dir "${sana_out}" \
  --device mps

env \
  VB_DATA_ROOT="${data_root}" \
  VB_OUT="${mar_out}" \
  VB_PAIR_MANIFEST="${pair_manifest}" \
  VB_RESUME="${project_root}/output/mar_vb_v5_full_clean_mar_large_384_20260811_run2/ckpt_best.pt" \
  VB_RESET_OPT=1 \
  VB_RESET_EMA=1 \
  VB_RESET_BEST=1 \
  VB_RESET_TASKS=seg \
  VB_STEPS=6000 \
  VB_TOTAL_STEPS=3000 \
  VB_IMG_SIZE=384 \
  VB_BATCH=1 \
  VB_LR=2e-6 \
  VB_WARMUP=100 \
  VB_VAL_EVERY=500 \
  VB_SAMPLE_EVERY=1000 \
  VB_SAMPLE_ITER=32 \
  VB_SAVE_LATEST_EVERY=0 \
  VB_NUM_WORKERS=2 \
  "${python_bin}" scripts/mar_vb_train_v5_multitask.py
