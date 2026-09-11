#!/bin/zsh
set -euo pipefail

cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OPEN_VISION_BANANA_MODEL_PATH=/Users/gatilin/MyWork/models/ckpts/FLUX.1-schnell

OUT=output/flux_schnell_paper_gate_lr1e5_20260824
CKPT="$OUT/checkpoint-300"
DATA=data/flux_paper_gate_512
PY=/Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python
LOG="$OUT/logs/runner_cityscapes_viz_ckpt300.log"

mkdir -p "$OUT/logs" "$OUT/evals"
echo "[ckpt300-eval] $(date '+%F %T') Cityscapes n=16 then viz" | tee -a "$LOG"

"$PY" scripts/evaluate_paper_benchmarks.py \
  --benchmark semantic \
  --cityscapes-root data/benchmarks/cityscapes \
  --model-name flux-schnell \
  --lora-path "$CKPT" \
  --device mps \
  --dtype bfloat16 \
  --resolution 512 \
  --num-inference-steps 4 \
  --max-samples 16 \
  --output-json "$OUT/evals/cityscapes_miou_n16_bs4_latest.json" \
  >"$OUT/logs/eval_cityscapes_bs4_latest.log" 2>&1

echo "[ckpt300-eval] $(date '+%F %T') Cityscapes done, starting viz" | tee -a "$LOG"

"$PY" scripts/viz_flux_paper_gate.py \
  --checkpoint "$CKPT" \
  --data-dir "$DATA" \
  --output-dir "$OUT/viz_checkpoint-300_bs4" \
  --n-per-task 2 \
  --resolution 512 \
  --num-inference-steps 4 \
  --device mps \
  --dtype bfloat16 \
  >"$OUT/logs/viz_bs4_latest.log" 2>&1

echo "[ckpt300-eval] $(date '+%F %T') done" | tee -a "$LOG"
