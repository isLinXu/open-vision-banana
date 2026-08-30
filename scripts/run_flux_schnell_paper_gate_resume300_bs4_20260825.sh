#!/bin/zsh
set -euo pipefail

# Sequential MPS: resume latest FLUX paper-gate ckpt to step 300 (batch 4),
# retry if the process is killed, then NYU n=32 + Cityscapes n=16 + viz.
# Do not start while another MPS job is using the GPU.

cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OPEN_VISION_BANANA_MODEL_PATH=/Users/gatilin/MyWork/models/ckpts/FLUX.1-schnell

OUT=output/flux_schnell_paper_gate_lr1e5_20260824
DATA=data/flux_paper_gate_512
PY=/Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python
LOG="$OUT/logs/runner_resume300_bs4.log"

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
  local meta="$1/metadata.json"
  if [ -f "$meta" ]; then
    "$PY" -c "import json; print(json.load(open('$meta')).get('step', 0))"
  else
    echo "${1##*-}"
  fi
}

echo "[flux-resume300-bs4] $(date '+%F %T') resume latest → 300, batch=4 accum=1" | tee -a "$LOG"

attempt=0
max_attempts=8
target_step=300
while [ "$attempt" -lt "$max_attempts" ]; do
  ckpt="$(latest_ckpt)"
  if [ -z "$ckpt" ]; then
    echo "[flux-resume300-bs4] no checkpoint to resume" | tee -a "$LOG"
    exit 1
  fi
  step="$(ckpt_step "$ckpt")"
  echo "[flux-resume300-bs4] $(date '+%F %T') attempt=$((attempt+1)) latest=$ckpt step=$step" | tee -a "$LOG"
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
    --rank 64 \
    --alpha 128 \
    --resolution 512 \
    --max-per-task 400 \
    --max-generation 2000 \
    --vision-ratio 0.1 \
    --no-ema \
    --seed 42 \
    >>"$OUT/logs/train_to_300_bs4.log" 2>&1
  rc=$?
  set -e
  ckpt="$(latest_ckpt)"
  step="$(ckpt_step "$ckpt")"
  echo "[flux-resume300-bs4] $(date '+%F %T') train_exit=$rc latest=$ckpt step=$step" | tee -a "$LOG"
  if [ "$rc" -eq 0 ] || [ "$step" -ge "$target_step" ]; then
    break
  fi
  attempt=$((attempt + 1))
  echo "[flux-resume300-bs4] retrying in 30s ($attempt/$max_attempts)" | tee -a "$LOG"
  sleep 30
done

ckpt="$(latest_ckpt)"
if [ -z "$ckpt" ]; then
  echo "[flux-resume300-bs4] no checkpoint after resume training" | tee -a "$LOG"
  exit 1
fi
echo "[flux-resume300-bs4] $(date '+%F %T') evaluating $ckpt" | tee -a "$LOG"

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
  --output-json "$OUT/evals/nyu_delta1_n32_bs4_latest.json" \
  >"$OUT/logs/eval_nyu_bs4_latest.log" 2>&1

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
  --output-json "$OUT/evals/cityscapes_miou_n16_bs4_latest.json" \
  >"$OUT/logs/eval_cityscapes_bs4_latest.log" 2>&1

viz_dir="$OUT/viz_$(basename "$ckpt")_bs4"
"$PY" scripts/viz_flux_paper_gate.py \
  --checkpoint "$ckpt" \
  --data-dir "$DATA" \
  --output-dir "$viz_dir" \
  --n-per-task 2 \
  --resolution 512 \
  --num-inference-steps 4 \
  --device mps \
  --dtype bfloat16 \
  >"$OUT/logs/viz_bs4_latest.log" 2>&1

echo "[flux-resume300-bs4] $(date '+%F %T') done. ckpt=$ckpt viz=$viz_dir" | tee -a "$LOG"
