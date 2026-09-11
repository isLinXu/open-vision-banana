#!/bin/zsh
set -euo pipefail

# Checkpoint selection: KITTI step 300 had the lowest train loss (0.0855).
# Sequential MPS. Do not start while another SANA job is running.

cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OPEN_VISION_BANANA_MODEL_PATH=/Users/gatilin/.cache/huggingface/hub/models--Efficient-Large-Model--Sana_600M_512px_diffusers/snapshots/83d7a190bfd1fd070570a793d2dab5c7a3231b9d

OUT=output/sana_kitti_metric_depth_lr1e5_20260823
PY=/Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python
CKPT="$OUT/checkpoint-300/lora_adapter"

mkdir -p "$OUT/logs" "$OUT/evals"
echo "[kitti-metric] eval checkpoint-300 ETH3D + NYU n=32" | tee -a "$OUT/logs/runner.log"

"$PY" scripts/evaluate_paper_benchmarks.py \
  --benchmark depth \
  --eth3d-root /Users/gatilin/MyWork/datasets/eth3d_eval \
  --model-name sana-600m-512px \
  --lora-path "$CKPT" \
  --device mps \
  --dtype float16 \
  --resolution 512 \
  --num-inference-steps 20 \
  --output-json "$OUT/evals/eth3d_delta1_step300_scenes83_capped.json" \
  >"$OUT/logs/eval_eth3d_300.log" 2>&1

"$PY" scripts/evaluate_paper_benchmarks.py \
  --benchmark depth \
  --nyu-root data/benchmarks/nyu/extracted \
  --model-name sana-600m-512px \
  --lora-path "$CKPT" \
  --device mps \
  --dtype float16 \
  --resolution 512 \
  --num-inference-steps 20 \
  --max-samples 32 \
  --output-json "$OUT/evals/nyu_delta1_step300_n32.json" \
  >"$OUT/logs/eval_nyu_300_n32.log" 2>&1

echo "[kitti-metric] ckpt-300 evals written" | tee -a "$OUT/logs/runner.log"
