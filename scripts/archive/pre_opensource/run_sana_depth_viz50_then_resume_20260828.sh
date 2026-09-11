#!/bin/zsh
set -euo pipefail

# Pause Stage A train, viz intermediate checkpoint-50, resume 50→300.
# Sequential MPS only.

cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OPEN_VISION_BANANA_MODEL_PATH=/Users/gatilin/.cache/huggingface/hub/models--Efficient-Large-Model--Sana_600M_512px_diffusers/snapshots/83d7a190bfd1fd070570a793d2dab5c7a3231b9d

OUT=output/sana_depth_consolidate_from_hypersim500_20260828
CFG=configs/experiments/sana_depth_consolidate_from_hypersim500_20260828.yaml
PY=/Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python
CKPT="$OUT/checkpoint-50"
VIZ="$OUT/viz_checkpoint-50"
VIZ_DATA=data/flux_paper_gate_512

mkdir -p "$OUT/logs"

echo "[depth-viz50] $(date '+%F %T') stopping Stage A train for mid-ckpt viz" | tee -a "$OUT/logs/runner.log"
pkill -f "scripts/train_sana_512.py.*sana_depth_consolidate_from_hypersim500" 2>/dev/null || true
pkill -f "run_sana_depth_consolidate_20260828" 2>/dev/null || true
sleep 5
if ps aux | awk '/train_sana_512|evaluate_paper_benchmarks|viz_sana_paper_gate/ && !/awk/ {print}' | grep -q .; then
  echo "[depth-viz50] MPS still busy" | tee -a "$OUT/logs/runner.log"
  ps aux | awk '/train_sana|evaluate_paper|viz_sana/ && !/awk/ {print}' | tee -a "$OUT/logs/runner.log"
  exit 1
fi

if [ ! -d "$CKPT/lora_adapter" ]; then
  echo "[depth-viz50] missing $CKPT/lora_adapter" | tee -a "$OUT/logs/runner.log"
  exit 1
fi

echo "[depth-viz50] $(date '+%F %T') viz $CKPT → $VIZ" | tee -a "$OUT/logs/runner.log"
"$PY" scripts/viz_sana_paper_gate.py \
  --checkpoint "$CKPT" \
  --data-dir "$VIZ_DATA" \
  --output-dir "$VIZ" \
  --n-per-task 2 \
  --resolution 512 \
  --num-inference-steps 20 \
  --device mps \
  --dtype float16 \
  >"$OUT/logs/viz_checkpoint-50.log" 2>&1

echo "[depth-viz50] $(date '+%F %T') resume train 50 → 300" | tee -a "$OUT/logs/runner.log"
"$PY" scripts/train_sana_512.py \
  --config "$CFG" \
  --stop-after-step 300 \
  --save-every 50 \
  --log-every 10 \
  --resume "$CKPT/lora.pt" \
  >>"$OUT/logs/train_to_300.log" 2>&1

# After resume train finishes, run the same post steps as the main runner.
CKPT300=""
for step in 300 250 200 150 100 50; do
  if [ -d "$OUT/checkpoint-${step}" ]; then
    CKPT300="$OUT/checkpoint-${step}"
    break
  fi
done
LORA="$CKPT300/lora_adapter"
echo "[depth-viz50] $(date '+%F %T') evaluating $CKPT300" | tee -a "$OUT/logs/runner.log"

"$PY" scripts/evaluate_paper_benchmarks.py \
  --benchmark depth \
  --nyu-root data/benchmarks/nyu/extracted \
  --model-name sana-600m-512px \
  --lora-path "$LORA" \
  --device mps \
  --dtype float16 \
  --resolution 512 \
  --num-inference-steps 20 \
  --max-samples 32 \
  --output-json "$OUT/evals/nyu_delta1_n32_step300.json" \
  >"$OUT/logs/eval_nyu_n32_step300.log" 2>&1

echo "[depth-viz50] $(date '+%F %T') viz $CKPT300" | tee -a "$OUT/logs/runner.log"
"$PY" scripts/viz_sana_paper_gate.py \
  --checkpoint "$CKPT300" \
  --data-dir "$VIZ_DATA" \
  --output-dir "$OUT/viz_${CKPT300##*/}" \
  --n-per-task 2 \
  --resolution 512 \
  --num-inference-steps 20 \
  --device mps \
  --dtype float16 \
  >"$OUT/logs/viz_${CKPT300##*/}.log" 2>&1

echo "[depth-viz50] $(date '+%F %T') done" | tee -a "$OUT/logs/runner.log"
