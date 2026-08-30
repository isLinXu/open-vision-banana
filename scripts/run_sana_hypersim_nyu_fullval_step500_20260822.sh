#!/bin/zsh
set -euo pipefail

# Full NYU Eigen protocol eval for the selected metric-depth specialist.
# n=654 val, 512px, 20 steps, crop + 10m cap. Single MPS job.

cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OPEN_VISION_BANANA_MODEL_PATH=/Users/gatilin/.cache/huggingface/hub/models--Efficient-Large-Model--Sana_600M_512px_diffusers/snapshots/83d7a190bfd1fd070570a793d2dab5c7a3231b9d

OUT=output/sana_hypersim_metric_depth_lr1e5_20260822
PY=/Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python
CKPT="$OUT/checkpoint-500/lora_adapter"

mkdir -p "$OUT/logs" "$OUT/evals"
echo "[hypersim-metric] NYU Eigen 10m-cap FULL val n=654 on checkpoint-500" | tee -a "$OUT/logs/runner.log"

"$PY" scripts/evaluate_paper_benchmarks.py \
  --benchmark depth \
  --nyu-root data/benchmarks/nyu/extracted \
  --model-name sana-600m-512px \
  --lora-path "$CKPT" \
  --device mps \
  --dtype float16 \
  --resolution 512 \
  --num-inference-steps 20 \
  --output-json "$OUT/evals/nyu_delta1_step500_capped_fullval.json" \
  >"$OUT/logs/eval_nyu_500_capped_fullval.log" 2>&1

echo "[hypersim-metric] full NYU val written to $OUT/evals/nyu_delta1_step500_capped_fullval.json" | tee -a "$OUT/logs/runner.log"
