#!/bin/zsh
set -euo pipefail

# Stage A: depth-only consolidation from Hypersim metric ckpt-500 → 300.
# Then NYU n=32 + enhanced depth viz. Sequential MPS only.

cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OPEN_VISION_BANANA_MODEL_PATH=/Users/gatilin/.cache/huggingface/hub/models--Efficient-Large-Model--Sana_600M_512px_diffusers/snapshots/83d7a190bfd1fd070570a793d2dab5c7a3231b9d

CFG=configs/experiments/sana_depth_consolidate_from_hypersim500_20260828.yaml
OUT=output/sana_depth_consolidate_from_hypersim500_20260828
PY=/Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python
RESUME=output/sana_hypersim_metric_depth_lr1e5_20260822/checkpoint-500/lora.pt
VIZ_DATA=data/flux_paper_gate_512

mkdir -p "$OUT/logs" "$OUT/evals"
echo $$ > "$OUT/logs/runner.pid"

if ps aux | awk '/train_sana_512|train_sdxl_mps|evaluate_paper_benchmarks|viz_sana_paper_gate/ && !/awk/ {print}' | grep -q .; then
  echo "[depth-consolidate] $(date '+%F %T') MPS busy; refuse to start" | tee -a "$OUT/logs/runner.log"
  exit 1
fi

if [ ! -f "$RESUME" ]; then
  echo "[depth-consolidate] missing baseline $RESUME" | tee -a "$OUT/logs/runner.log"
  exit 1
fi

echo "[depth-consolidate] $(date '+%F %T') weights-only from Hypersim-500 → stop 300" | tee -a "$OUT/logs/runner.log"

"$PY" scripts/train_sana_512.py \
  --config "$CFG" \
  --stop-after-step 300 \
  --save-every 50 \
  --log-every 10 \
  --resume "$RESUME" \
  --resume-weights-only \
  >>"$OUT/logs/train_to_300.log" 2>&1

CKPT=""
for step in 300 250 200 150 100 50; do
  if [ -d "$OUT/checkpoint-${step}" ]; then
    CKPT="$OUT/checkpoint-${step}"
    break
  fi
done
if [ -z "$CKPT" ]; then
  echo "[depth-consolidate] no checkpoint after train" | tee -a "$OUT/logs/runner.log"
  exit 1
fi
LORA="$CKPT/lora_adapter"
echo "[depth-consolidate] $(date '+%F %T') evaluating $CKPT" | tee -a "$OUT/logs/runner.log"

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

echo "[depth-consolidate] $(date '+%F %T') viz $CKPT" | tee -a "$OUT/logs/runner.log"
"$PY" scripts/viz_sana_paper_gate.py \
  --checkpoint "$CKPT" \
  --data-dir "$VIZ_DATA" \
  --output-dir "$OUT/viz_${CKPT##*/}" \
  --n-per-task 2 \
  --resolution 512 \
  --num-inference-steps 20 \
  --device mps \
  --dtype float16 \
  >"$OUT/logs/viz_${CKPT##*/}.log" 2>&1

echo "[depth-consolidate] $(date '+%F %T') done. ckpt=$CKPT" | tee -a "$OUT/logs/runner.log"
