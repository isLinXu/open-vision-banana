#!/bin/zsh
set -euo pipefail

# Re-evaluate Hypersim metric-depth LoRAs under Eigen NYU protocol:
# crop + GT in (1e-3, 10] + pred/gt clip to [1e-3, 10].
# Sequential: do not run two MPS jobs at once.

cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OPEN_VISION_BANANA_MODEL_PATH=/Users/gatilin/.cache/huggingface/hub/models--Efficient-Large-Model--Sana_600M_512px_diffusers/snapshots/83d7a190bfd1fd070570a793d2dab5c7a3231b9d

OUT=output/sana_hypersim_metric_depth_lr1e5_20260822
PY=/Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python

mkdir -p "$OUT/logs" "$OUT/evals"
echo "[hypersim-metric] NYU Eigen 10m-cap re-eval: step 500 then 1000" | tee -a "$OUT/logs/runner.log"

"$PY" scripts/evaluate_paper_benchmarks.py \
  --benchmark depth \
  --nyu-root data/benchmarks/nyu/extracted \
  --model-name sana-600m-512px \
  --lora-path "$OUT/checkpoint-500/lora_adapter" \
  --device mps \
  --dtype float16 \
  --resolution 512 \
  --num-inference-steps 20 \
  --max-samples 32 \
  --output-json "$OUT/evals/nyu_delta1_step500_capped.json" \
  >"$OUT/logs/eval_nyu_500_capped.log" 2>&1

echo "[hypersim-metric] step500 capped written" | tee -a "$OUT/logs/runner.log"

"$PY" scripts/evaluate_paper_benchmarks.py \
  --benchmark depth \
  --nyu-root data/benchmarks/nyu/extracted \
  --model-name sana-600m-512px \
  --lora-path "$OUT/checkpoint-1000/lora_adapter" \
  --device mps \
  --dtype float16 \
  --resolution 512 \
  --num-inference-steps 20 \
  --max-samples 32 \
  --output-json "$OUT/evals/nyu_delta1_step1000_capped.json" \
  >"$OUT/logs/eval_nyu_1000_capped.log" 2>&1

echo "[hypersim-metric] step1000 capped written" | tee -a "$OUT/logs/runner.log"
