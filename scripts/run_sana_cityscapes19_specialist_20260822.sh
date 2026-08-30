#!/bin/zsh
set -euo pipefail

cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OPEN_VISION_BANANA_MODEL_PATH=/Users/gatilin/.cache/huggingface/hub/models--Efficient-Large-Model--Sana_600M_512px_diffusers/snapshots/83d7a190bfd1fd070570a793d2dab5c7a3231b9d

CFG=configs/experiments/sana_cityscapes19_specialist_lr1e5_20260822.yaml
OUT=output/sana_cityscapes19_specialist_lr1e5_20260822
PY=/Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python

mkdir -p "$OUT/logs"
echo "[cityscapes19] train to 500 then paper semantic eval" | tee -a "$OUT/logs/runner.log"

"$PY" scripts/train_sana_512.py --config "$CFG" --stop-after-step 500 --save-every 100 \
  >"$OUT/logs/train_to_500.log" 2>&1

"$PY" scripts/evaluate_paper_benchmarks.py \
  --benchmark semantic \
  --cityscapes-root data/benchmarks/cityscapes \
  --model-name sana-600m-512px \
  --lora-path "$OUT/checkpoint-500/lora_adapter" \
  --device mps \
  --dtype float16 \
  --resolution 512 \
  --num-inference-steps 20 \
  --max-samples 16 \
  --output-json "$OUT/evals/paper_semantic_step500.json" \
  >"$OUT/logs/eval_paper_500.log" 2>&1

echo "[cityscapes19] step 500 paper eval written to $OUT/evals/paper_semantic_step500.json" | tee -a "$OUT/logs/runner.log"
