#!/bin/zsh
set -euo pipefail

# Vision-boost: vision_ratio=0.5, homogeneous batch=4, resume latest → 750.
# Not paper 9:1. Do not start while another MPS job is using the GPU.

cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OPEN_VISION_BANANA_MODEL_PATH=/Users/gatilin/MyWork/models/ckpts/FLUX.1-schnell

OUT=output/flux_schnell_paper_gate_lr1e5_20260824
DATA=data/flux_paper_gate_512
PY=/Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python
LOG="$OUT/logs/runner_vision_boost_r05.log"
TARGET=750

mkdir -p "$OUT/logs" "$OUT/evals"

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

ckpt_step() {
  "$PY" -c "import json; print(json.load(open('$1/metadata.json')).get('step', 0))"
}

echo "[vision-boost-r05] $(date '+%F %T') resume latest → $TARGET, vision_ratio=0.5 homog" | tee -a "$LOG"

attempt=0
max_attempts=8
while [ "$attempt" -lt "$max_attempts" ]; do
  ckpt="$(latest_ckpt)"
  if [ -z "$ckpt" ]; then
    echo "[vision-boost-r05] no checkpoint" | tee -a "$LOG"
    exit 1
  fi
  step="$(ckpt_step "$ckpt")"
  echo "[vision-boost-r05] $(date '+%F %T') attempt=$((attempt+1)) ckpt=$ckpt step=$step" | tee -a "$LOG"
  if [ "$step" -ge "$TARGET" ]; then
    break
  fi

  set +e
  "$PY" scripts/train_sdxl_mps.py \
    --model flux-schnell \
    --data-dir "$DATA" \
    --output-dir "$OUT" \
    --resume-from "$ckpt" \
    --lr 1e-5 \
    --max-steps "$TARGET" \
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
  rc=$?
  set -e

  ckpt="$(latest_ckpt)"
  step="$(ckpt_step "$ckpt")"
  echo "[vision-boost-r05] $(date '+%F %T') train_exit=$rc step=$step" | tee -a "$LOG"
  if [ "$rc" -eq 0 ] || [ "$step" -ge "$TARGET" ]; then
    break
  fi
  attempt=$((attempt + 1))
  sleep 30
done

ckpt="$(latest_ckpt)"
echo "[vision-boost-r05] $(date '+%F %T') evaluating $ckpt" | tee -a "$LOG"

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
  --output-json "$OUT/evals/nyu_delta1_n32_vision_boost_r05.json" \
  >"$OUT/logs/eval_nyu_vision_boost_r05.log" 2>&1

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
  --output-json "$OUT/evals/cityscapes_miou_n16_vision_boost_r05.json" \
  >"$OUT/logs/eval_cityscapes_vision_boost_r05.log" 2>&1

"$PY" scripts/viz_flux_paper_gate.py \
  --checkpoint "$ckpt" \
  --data-dir "$DATA" \
  --output-dir "$OUT/viz_$(basename "$ckpt")_vision_boost_r05" \
  --n-per-task 2 \
  --resolution 512 \
  --num-inference-steps 4 \
  --device mps \
  --dtype bfloat16 \
  >"$OUT/logs/viz_vision_boost_r05.log" 2>&1

echo "[vision-boost-r05] $(date '+%F %T') done ckpt=$ckpt" | tee -a "$LOG"
