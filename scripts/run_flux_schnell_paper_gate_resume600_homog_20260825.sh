#!/bin/zsh
set -euo pipefail

# Wait until Cityscapes/viz (or any other MPS job) is done, then resume
# FLUX paper-gate 300 → 600 with homogeneous micro-batches.

cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OPEN_VISION_BANANA_MODEL_PATH=/Users/gatilin/MyWork/models/ckpts/FLUX.1-schnell

OUT=output/flux_schnell_paper_gate_lr1e5_20260824
DATA=data/flux_paper_gate_512
PY=/Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python
LOG="$OUT/logs/runner_resume600_homog.log"

mkdir -p "$OUT/logs" "$OUT/evals"

mps_busy() {
  ps aux | awk '/evaluate_paper_benchmarks|viz_flux_paper_gate|train_sdxl_mps|train_sana/ && !/awk|resume600_homog|wait_then/ {print}' | grep -q .
}

echo "[flux-homog600] $(date '+%F %T') waiting for free MPS" | tee -a "$LOG"
while mps_busy; do
  sleep 30
done
echo "[flux-homog600] $(date '+%F %T') MPS free; resume 300 → 600 homogeneous batches" | tee -a "$LOG"

latest_ckpt() {
  local best="" bestn=-1
  for d in "$OUT"/checkpoint-*; do
    [ -d "$d" ] || continue
    n="${d##*-}"
    case "$n" in
      ''|*[!0-9]*) continue ;;
    esac
    if [ "$n" -gt "$bestn" ]; then
      bestn="$n"
      best="$d"
    fi
  done
  printf '%s' "$best"
}

ckpt="$(latest_ckpt)"
if [ -z "$ckpt" ]; then
  echo "[flux-homog600] no checkpoint" | tee -a "$LOG"
  exit 1
fi

attempt=0
max_attempts=8
target_step=600
while [ "$attempt" -lt "$max_attempts" ]; do
  ckpt="$(latest_ckpt)"
  step="$("$PY" -c "import json; print(json.load(open('$ckpt/metadata.json')).get('step',0))")"
  echo "[flux-homog600] $(date '+%F %T') attempt=$((attempt+1)) ckpt=$ckpt step=$step" | tee -a "$LOG"
  if [ "$step" -ge "$target_step" ]; then
    break
  fi
  set +e
  "$PY" scripts/train_sdxl_mps.py \
    --model flux-schnell \
    --data-dir "$DATA" \
    --output-dir "$OUT" \
    --resume-from "$ckpt" \
    --lr 1e-5 \
    --max-steps "$target_step" \
    --save-every 10 \
    --max-keep 20 \
    --log-every 10 \
    --batch-size 4 \
    --grad-accum 1 \
    --homogeneous-batches \
    --rank 64 \
    --alpha 128 \
    --resolution 512 \
    --max-per-task 400 \
    --max-generation 2000 \
    --vision-ratio 0.1 \
    --no-ema \
    --seed 42 \
    >>"$OUT/logs/train_to_600_homog.log" 2>&1
  rc=$?
  set -e
  ckpt="$(latest_ckpt)"
  step="$("$PY" -c "import json; print(json.load(open('$ckpt/metadata.json')).get('step',0))")"
  echo "[flux-homog600] $(date '+%F %T') train_exit=$rc step=$step" | tee -a "$LOG"
  if [ "$rc" -eq 0 ] || [ "$step" -ge "$target_step" ]; then
    break
  fi
  attempt=$((attempt + 1))
  sleep 30
done

ckpt="$(latest_ckpt)"
echo "[flux-homog600] $(date '+%F %T') evaluating $ckpt" | tee -a "$LOG"

"$PY" scripts/evaluate_paper_benchmarks.py \
  --benchmark depth \
  --nyu-root data/benchmarks/nyu/extracted \
  --model-name flux-schnell \
  --lora-path "$ckpt" \
  --device mps \
  --dtype bfloat16 \
  --resolution 512 \
  --num-inference-steps 4 \
  --max-samples 32 \
  --output-json "$OUT/evals/nyu_delta1_n32_homog600.json" \
  >"$OUT/logs/eval_nyu_homog600.log" 2>&1

"$PY" scripts/evaluate_paper_benchmarks.py \
  --benchmark semantic \
  --cityscapes-root data/benchmarks/cityscapes \
  --model-name flux-schnell \
  --lora-path "$ckpt" \
  --device mps \
  --dtype bfloat16 \
  --resolution 512 \
  --num-inference-steps 4 \
  --max-samples 16 \
  --output-json "$OUT/evals/cityscapes_miou_n16_homog600.json" \
  >"$OUT/logs/eval_cityscapes_homog600.log" 2>&1

"$PY" scripts/viz_flux_paper_gate.py \
  --checkpoint "$ckpt" \
  --data-dir "$DATA" \
  --output-dir "$OUT/viz_$(basename "$ckpt")_homog" \
  --n-per-task 2 \
  --resolution 512 \
  --num-inference-steps 4 \
  --device mps \
  --dtype bfloat16 \
  >"$OUT/logs/viz_homog600.log" 2>&1

echo "[flux-homog600] $(date '+%F %T') done ckpt=$ckpt" | tee -a "$LOG"
