#!/bin/zsh
set -euo pipefail

# Sequential MPS: eval ckpt-170 baseline → resume train to 300 → eval + viz.
# Do not start while another MPS job is using the GPU.

cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OPEN_VISION_BANANA_MODEL_PATH=/Users/gatilin/MyWork/models/ckpts/FLUX.1-schnell

OUT=output/flux_schnell_paper_gate_lr1e5_20260824
DATA=data/flux_paper_gate_512
CKPT170="$OUT/checkpoint-170"
PY=/Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python

mkdir -p "$OUT/logs" "$OUT/evals"
echo "[flux-resume300] $(date '+%F %T') baseline eval ckpt-170, then train to 300" | tee -a "$OUT/logs/runner_resume300.log"

if [ ! -d "$CKPT170" ]; then
  echo "[flux-resume300] missing $CKPT170" | tee -a "$OUT/logs/runner_resume300.log"
  exit 1
fi

"$PY" scripts/evaluate_paper_benchmarks.py \
  --benchmark depth \
  --nyu-root data/benchmarks/nyu/extracted \
  --model-name flux-schnell \
  --lora-path "$CKPT170" \
  --device mps \
  --dtype bfloat16 \
  --resolution 512 \
  --num-inference-steps 4 \
  --max-samples 32 \
  --output-json "$OUT/evals/nyu_delta1_n32_ckpt170.json" \
  >"$OUT/logs/eval_nyu_ckpt170.log" 2>&1

"$PY" scripts/evaluate_paper_benchmarks.py \
  --benchmark semantic \
  --cityscapes-root data/benchmarks/cityscapes \
  --model-name flux-schnell \
  --lora-path "$CKPT170" \
  --device mps \
  --dtype bfloat16 \
  --resolution 512 \
  --num-inference-steps 4 \
  --max-samples 16 \
  --output-json "$OUT/evals/cityscapes_miou_n16_ckpt170.json" \
  >"$OUT/logs/eval_cityscapes_ckpt170.log" 2>&1

echo "[flux-resume300] $(date '+%F %T') resume train 170 → 300" | tee -a "$OUT/logs/runner_resume300.log"

"$PY" scripts/train_sdxl_mps.py \
  --model flux-schnell \
  --data-dir "$DATA" \
  --output-dir "$OUT" \
  --resume-from "$CKPT170" \
  --lr 1e-5 \
  --max-steps 300 \
  --save-every 10 \
  --log-every 10 \
  --batch-size 1 \
  --grad-accum 4 \
  --rank 64 \
  --alpha 128 \
  --resolution 512 \
  --max-per-task 400 \
  --max-generation 2000 \
  --vision-ratio 0.1 \
  --no-ema \
  --seed 42 \
  >"$OUT/logs/train_to_300.log" 2>&1

ckpt=""
for cand in 300 290 280 270 260 250 240 230 220 210 200 190 180 170; do
  if [ -d "$OUT/checkpoint-$cand" ]; then
    ckpt="$OUT/checkpoint-$cand"
    break
  fi
done
if [ -z "$ckpt" ]; then
  echo "[flux-resume300] no checkpoint after resume training" | tee -a "$OUT/logs/runner_resume300.log"
  exit 1
fi
echo "[flux-resume300] $(date '+%F %T') evaluating $ckpt" | tee -a "$OUT/logs/runner_resume300.log"

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
  --output-json "$OUT/evals/nyu_delta1_n32_latest.json" \
  >"$OUT/logs/eval_nyu_latest.log" 2>&1

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
  --output-json "$OUT/evals/cityscapes_miou_n16_latest.json" \
  >"$OUT/logs/eval_cityscapes_latest.log" 2>&1

viz_dir="$OUT/viz_$(basename "$ckpt")"
"$PY" scripts/viz_flux_paper_gate.py \
  --checkpoint "$ckpt" \
  --data-dir "$DATA" \
  --output-dir "$viz_dir" \
  --n-per-task 2 \
  --resolution 512 \
  --num-inference-steps 4 \
  --device mps \
  --dtype bfloat16 \
  >"$OUT/logs/viz_latest.log" 2>&1

echo "[flux-resume300] $(date '+%F %T') done. ckpt=$ckpt viz=$viz_dir" | tee -a "$OUT/logs/runner_resume300.log"
