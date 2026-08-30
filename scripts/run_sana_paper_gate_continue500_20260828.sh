#!/bin/zsh
set -euo pipefail

cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OPEN_VISION_BANANA_MODEL_PATH=/Users/gatilin/.cache/huggingface/hub/models--Efficient-Large-Model--Sana_600M_512px_diffusers/snapshots/83d7a190bfd1fd070570a793d2dab5c7a3231b9d

CFG=configs/experiments/sana_paper_gate_from_hypersim500_20260828.yaml
OUT=output/sana_paper_gate_from_hypersim500_20260828
PY=/Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python
VIZ_DATA=data/flux_paper_gate_512

mkdir -p "$OUT/logs" "$OUT/evals"
echo $$ > "$OUT/logs/runner_continue.pid"

# Pick newest existing checkpoint <=500
RESUME=""
for step in 450 400 350 300 250 200 150 100 50 1; do
  if [ -f "$OUT/checkpoint-${step}/lora.pt" ]; then
    RESUME="$OUT/checkpoint-${step}/lora.pt"
    break
  fi
done
if [ -z "$RESUME" ]; then
  RESUME=output/sana_hypersim_metric_depth_lr1e5_20260822/checkpoint-500/lora.pt
  EXTRA=(--resume-weights-only)
  echo "[sana-paper-gate] $(date '+%F %T') no local ckpt; weights-only from Hypersim-500" | tee -a "$OUT/logs/runner.log"
else
  EXTRA=()
  echo "[sana-paper-gate] $(date '+%F %T') continue $RESUME → stop 500" | tee -a "$OUT/logs/runner.log"
fi

"$PY" scripts/train_sana_512.py \
  --config "$CFG" \
  --stop-after-step 500 \
  --save-every 50 \
  --log-every 10 \
  --resume "$RESUME" \
  "${EXTRA[@]}" \
  >>"$OUT/logs/train_to_500.log" 2>&1

pick_ckpt() {
  local step="$1"
  if [ -d "$OUT/checkpoint-${step}" ]; then
    echo "$OUT/checkpoint-${step}"
    return 0
  fi
  ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1
}

CKPT="$(pick_ckpt 500)"
LORA="$CKPT/lora_adapter"
echo "[sana-paper-gate] $(date '+%F %T') evaluating $CKPT" | tee -a "$OUT/logs/runner.log"

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
  --output-json "$OUT/evals/nyu_delta1_n32_step500.json" \
  >"$OUT/logs/eval_nyu_n32_step500.log" 2>&1

"$PY" scripts/evaluate_paper_benchmarks.py \
  --benchmark semantic \
  --cityscapes-root data/benchmarks/cityscapes \
  --model-name sana-600m-512px \
  --lora-path "$LORA" \
  --device mps \
  --dtype float16 \
  --resolution 512 \
  --num-inference-steps 20 \
  --max-samples 16 \
  --output-json "$OUT/evals/cityscapes_miou_n16_step500.json" \
  >"$OUT/logs/eval_cityscapes_n16_step500.log" 2>&1

echo "[sana-paper-gate] $(date '+%F %T') viz $CKPT" | tee -a "$OUT/logs/runner.log"
"$PY" scripts/viz_sana_paper_gate.py \
  --checkpoint "$CKPT" \
  --data-dir "$VIZ_DATA" \
  --output-dir "$OUT/viz_checkpoint-500" \
  --n-per-task 2 \
  --resolution 512 \
  --num-inference-steps 20 \
  --device mps \
  --dtype float16 \
  >"$OUT/logs/viz_checkpoint-500.log" 2>&1

echo "[sana-paper-gate] $(date '+%F %T') done. ckpt=$CKPT" | tee -a "$OUT/logs/runner.log"
