#!/bin/zsh
set -euo pipefail

# Continue the mixed indoor+KITTI replay from checkpoint-100 (same run).
# Full --resume keeps optimizer / scheduler / step. Then ETH3D + NYU n=32.
# Sequential MPS. Do not start while another SANA job is running.

cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OPEN_VISION_BANANA_MODEL_PATH=/Users/gatilin/.cache/huggingface/hub/models--Efficient-Large-Model--Sana_600M_512px_diffusers/snapshots/83d7a190bfd1fd070570a793d2dab5c7a3231b9d

CFG=configs/experiments/sana_metric_mixed_replay_lr1e5_20260823.yaml
OUT=output/sana_metric_mixed_replay_lr1e5_20260823
PY=/Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python
RESUME="$OUT/checkpoint-100/lora.pt"

mkdir -p "$OUT/logs" "$OUT/evals"
echo "[mixed-replay] resume 100→200, then ETH3D + NYU n=32" | tee -a "$OUT/logs/runner.log"

"$PY" scripts/train_sana_512.py --config "$CFG" --stop-after-step 200 --save-every 100 \
  --log-every 10 --resume "$RESUME" \
  >"$OUT/logs/train_100_to_200.log" 2>&1

"$PY" scripts/evaluate_paper_benchmarks.py \
  --benchmark depth \
  --eth3d-root /Users/gatilin/MyWork/datasets/eth3d_eval \
  --model-name sana-600m-512px \
  --lora-path "$OUT/checkpoint-200/lora_adapter" \
  --device mps \
  --dtype float16 \
  --resolution 512 \
  --num-inference-steps 20 \
  --output-json "$OUT/evals/eth3d_delta1_step200_scenes83_capped.json" \
  >"$OUT/logs/eval_eth3d_200.log" 2>&1

"$PY" scripts/evaluate_paper_benchmarks.py \
  --benchmark depth \
  --nyu-root data/benchmarks/nyu/extracted \
  --model-name sana-600m-512px \
  --lora-path "$OUT/checkpoint-200/lora_adapter" \
  --device mps \
  --dtype float16 \
  --resolution 512 \
  --num-inference-steps 20 \
  --max-samples 32 \
  --output-json "$OUT/evals/nyu_delta1_step200_n32.json" \
  >"$OUT/logs/eval_nyu_200_n32.log" 2>&1

echo "[mixed-replay] evals written under $OUT/evals" | tee -a "$OUT/logs/runner.log"
