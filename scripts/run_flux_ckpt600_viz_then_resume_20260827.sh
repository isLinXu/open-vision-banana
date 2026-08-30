#!/bin/zsh
set -euo pipefail

# Pause vision-boost training, viz checkpoint-600, resume train → 750.
cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OPEN_VISION_BANANA_MODEL_PATH=/Users/gatilin/MyWork/models/ckpts/FLUX.1-schnell

OUT=output/flux_schnell_paper_gate_lr1e5_20260824
DATA=data/flux_paper_gate_512
CKPT="$OUT/checkpoint-600"
VIZ="$OUT/viz_checkpoint-600_vision_boost_r05"
PY=/Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python
LOG="$OUT/logs/runner_ckpt600_viz_then_resume.log"

mkdir -p "$OUT/logs"

echo "[ckpt600-viz] $(date '+%F %T') stopping MPS jobs for viz" | tee -a "$LOG"
pkill -f "scripts/train_sdxl_mps.py.*flux_schnell_paper_gate" 2>/dev/null || true
pkill -f "run_flux_schnell_vision_boost_r05" 2>/dev/null || true
sleep 5
if ps aux | awk '/train_sdxl_mps|evaluate_paper_benchmarks|viz_flux_paper_gate/ && !/awk/ {print}' | grep -q .; then
  echo "[ckpt600-viz] MPS still busy" | tee -a "$LOG"
  ps aux | awk '/train_sdxl_mps|evaluate_paper|viz_flux/ && !/awk/ {print}' | tee -a "$LOG"
  exit 1
fi

if [ ! -d "$CKPT" ]; then
  echo "[ckpt600-viz] missing $CKPT" | tee -a "$LOG"
  exit 1
fi

echo "[ckpt600-viz] $(date '+%F %T') viz $CKPT → $VIZ" | tee -a "$LOG"
"$PY" scripts/viz_flux_paper_gate.py \
  --checkpoint "$CKPT" \
  --data-dir "$DATA" \
  --output-dir "$VIZ" \
  --n-per-task 2 \
  --resolution 512 \
  --num-inference-steps 4 \
  --device mps \
  --dtype bfloat16 \
  >"$OUT/logs/viz_checkpoint-600_vision_boost_r05.log" 2>&1

echo "[ckpt600-viz] $(date '+%F %T') resume train 600 → 750" | tee -a "$LOG"
exec "$PY" scripts/train_sdxl_mps.py \
  --model flux-schnell \
  --data-dir "$DATA" \
  --output-dir "$OUT" \
  --resume-from "$CKPT" \
  --lr 1e-5 \
  --max-steps 750 \
  --save-every 10 \
  --max-keep 20 \
  --log-every 10 \
  --batch-size 4 \
  --grad-accum 1 \
  --homogeneous-batches \
  --vision-ratio 0.5 \
  --rank 64 \
  --alpha 128 \
  --resolution 512 \
  --max-per-task 400 \
  --max-generation 2000 \
  --no-ema \
  --seed 42 \
  >>"$OUT/logs/train_vision_boost_r05.log" 2>&1
