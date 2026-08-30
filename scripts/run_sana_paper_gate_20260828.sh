#!/bin/zsh
set -euo pipefail

# SANA paper-method gate from Hypersim metric-depth ckpt-500.
# Shared LoRA, RGB depth+Cityscapes-19, vision_ratio=0.1 generation mix.
# Sequential MPS. Do not start while another GPU job is running.
# Gate checkpoints: 250 (mid) and 500 (primary). Then NYU / Cityscapes / viz.

cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OPEN_VISION_BANANA_MODEL_PATH=/Users/gatilin/.cache/huggingface/hub/models--Efficient-Large-Model--Sana_600M_512px_diffusers/snapshots/83d7a190bfd1fd070570a793d2dab5c7a3231b9d

CFG=configs/experiments/sana_paper_gate_from_hypersim500_20260828.yaml
OUT=output/sana_paper_gate_from_hypersim500_20260828
PY=/Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python
RESUME=output/sana_hypersim_metric_depth_lr1e5_20260822/checkpoint-500/lora.pt
VIZ_DATA=data/flux_paper_gate_512

mkdir -p "$OUT/logs" "$OUT/evals"

if ps aux | awk '/train_sana_512|train_sdxl_mps|evaluate_paper_benchmarks|viz_sana_paper_gate|viz_flux_paper_gate/ && !/awk/ {print}' | grep -q .; then
  echo "[sana-paper-gate] $(date '+%F %T') MPS busy; refuse to start" | tee -a "$OUT/logs/runner.log"
  ps aux | awk '/train_sana|train_sdxl|evaluate_paper|viz_sana|viz_flux/ && !/awk/ {print}' | tee -a "$OUT/logs/runner.log"
  exit 1
fi

if [ ! -f "$RESUME" ]; then
  echo "[sana-paper-gate] missing baseline $RESUME" | tee -a "$OUT/logs/runner.log"
  exit 1
fi

echo "[sana-paper-gate] $(date '+%F %T') resume weights-only from Hypersim-500 → stop 500" | tee -a "$OUT/logs/runner.log"

"$PY" scripts/train_sana_512.py \
  --config "$CFG" \
  --stop-after-step 500 \
  --save-every 50 \
  --log-every 10 \
  --resume "$RESUME" \
  --resume-weights-only \
  >"$OUT/logs/train_to_500.log" 2>&1

pick_ckpt() {
  local step="$1"
  if [ -d "$OUT/checkpoint-${step}" ]; then
    echo "$OUT/checkpoint-${step}"
    return 0
  fi
  local cand
  for cand in $(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -r); do
    echo "$cand"
    return 0
  done
  return 1
}

CKPT="$(pick_ckpt 500)" || {
  echo "[sana-paper-gate] no checkpoint after train" | tee -a "$OUT/logs/runner.log"
  exit 1
}
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

echo "[sana-paper-gate] $(date '+%F %T') done. ckpt=$CKPT evals+viz under $OUT" | tee -a "$OUT/logs/runner.log"
