#!/bin/zsh
set -euo pipefail

# FLUX.1-schnell shared-LoRA paper-method gate. Sequential MPS.
# Do not start while another training/eval job is using the GPU.

cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OPEN_VISION_BANANA_MODEL_PATH=/Users/gatilin/MyWork/models/ckpts/FLUX.1-schnell

OUT=output/flux_schnell_paper_gate_lr1e5_20260824
DATA=data/flux_paper_gate_512
PY=/Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python

mkdir -p "$OUT/logs" "$OUT/evals"
echo "[flux-paper-gate] 100-step shared LoRA on schnell, then NYU n=32 + Cityscapes n=16" | tee -a "$OUT/logs/runner.log"

"$PY" scripts/prepare_flux_paper_gate_data.py --output "$DATA" \
  >"$OUT/logs/prepare_data.log" 2>&1

"$PY" scripts/train_sdxl_mps.py \
  --model flux-schnell \
  --data-dir "$DATA" \
  --output-dir "$OUT" \
  --lr 1e-5 \
  --max-steps 100 \
  --save-every 50 \
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
  >"$OUT/logs/train_to_100.log" 2>&1

ckpt=""
for cand in 100 50; do
  if [ -d "$OUT/checkpoint-$cand" ]; then
    ckpt="$OUT/checkpoint-$cand"
    break
  fi
done
if [ -z "$ckpt" ]; then
  echo "[flux-paper-gate] no checkpoint after training" | tee -a "$OUT/logs/runner.log"
  exit 1
fi
echo "[flux-paper-gate] evaluating $ckpt" | tee -a "$OUT/logs/runner.log"

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
  --output-json "$OUT/evals/nyu_delta1_n32.json" \
  >"$OUT/logs/eval_nyu.log" 2>&1

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
  --output-json "$OUT/evals/cityscapes_miou_n16.json" \
  >"$OUT/logs/eval_cityscapes.log" 2>&1

echo "[flux-paper-gate] evals written under $OUT/evals" | tee -a "$OUT/logs/runner.log"
