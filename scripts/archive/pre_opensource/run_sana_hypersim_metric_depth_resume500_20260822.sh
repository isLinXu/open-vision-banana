#!/bin/zsh
set -euo pipefail

# Continue metric-depth specialist on the expanded Hypersim split.
# Resume full trainer state from checkpoint-500. Train with the NYU-eval
# metric prompt (pairs.jsonl task=metric_depth).

cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OPEN_VISION_BANANA_MODEL_PATH=/Users/gatilin/.cache/huggingface/hub/models--Efficient-Large-Model--Sana_600M_512px_diffusers/snapshots/83d7a190bfd1fd070570a793d2dab5c7a3231b9d

CFG=configs/experiments/sana_hypersim_metric_depth_lr1e5_20260822.yaml
OUT=output/sana_hypersim_metric_depth_lr1e5_20260822
PY=/Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python
RESUME="$OUT/checkpoint-500/lora.pt"

mkdir -p "$OUT/logs" "$OUT/evals"
echo "[hypersim-metric] resume step 500 -> 1000 on expanded metric pairs, then NYU delta1" | tee -a "$OUT/logs/runner.log"

"$PY" scripts/train_sana_512.py --config "$CFG" --stop-after-step 1000 --save-every 100 \
  --resume "$RESUME" \
  >"$OUT/logs/train_500_to_1000.log" 2>&1

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
  --output-json "$OUT/evals/nyu_delta1_step1000.json" \
  >"$OUT/logs/eval_nyu_1000.log" 2>&1

echo "[hypersim-metric] NYU eval written to $OUT/evals/nyu_delta1_step1000.json" | tee -a "$OUT/logs/runner.log"
