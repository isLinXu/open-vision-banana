#!/bin/zsh
set -euo pipefail

# Wait until the FLUX gate trainer/eval releases MPS, then visualize ckpt-10.
# Do not start this if you intend to run another MPS job yourself.

cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OPEN_VISION_BANANA_MODEL_PATH=/Users/gatilin/MyWork/models/ckpts/FLUX.1-schnell

OUT=output/flux_schnell_paper_gate_lr1e5_20260824
CKPT="$OUT/checkpoint-10"
PY=/Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python
VIZ="$OUT/viz_checkpoint-10"

mkdir -p "$OUT/logs" "$VIZ"
echo "[flux-viz] waiting for MPS to be free before visualizing $CKPT" | tee -a "$OUT/logs/viz_waiter.log"

mps_busy() {
  pgrep -f 'scripts/train_sdxl_mps.py' >/dev/null \
    || pgrep -f 'scripts/evaluate_paper_benchmarks.py' >/dev/null \
    || pgrep -f 'scripts/run_flux_schnell_paper_gate_20260824.sh' >/dev/null
}

while mps_busy; do
  echo "[flux-viz] $(date '+%H:%M:%S') trainer/eval still on MPS" >>"$OUT/logs/viz_waiter.log"
  sleep 60
done

sleep 15
echo "[flux-viz] MPS free, starting checkpoint-10 visualization" | tee -a "$OUT/logs/viz_waiter.log"
"$PY" scripts/viz_flux_paper_gate.py \
  --checkpoint "$CKPT" \
  --data-dir data/flux_paper_gate_512 \
  --output-dir "$VIZ" \
  --n-per-task 2 \
  --resolution 512 \
  --num-inference-steps 4 \
  --device mps \
  --dtype bfloat16 \
  >"$OUT/logs/viz_checkpoint-10.log" 2>&1

echo "[flux-viz] panels written under $VIZ" | tee -a "$OUT/logs/viz_waiter.log"
