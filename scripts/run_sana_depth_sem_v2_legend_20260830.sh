#!/bin/zsh
set -euo pipefail

# Stage B v2: depth + Cityscapes-19 from Hypersim-500, unified legend prompt,
# EMA-correct checkpoints. Run in your own Terminal (long MPS job):
#   nohup zsh scripts/run_sana_depth_sem_v2_legend_20260830.sh \
#     > output/sana_depth_sem_v2_legend_20260830/logs/runner.log 2>&1 &
# Resume after interruption (export VB_RESUME before rerunning):
#   export VB_RESUME=output/sana_depth_sem_v2_legend_20260830/checkpoint-400/lora.pt

cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OPEN_VISION_BANANA_MODEL_PATH=/Users/gatilin/.cache/huggingface/hub/models--Efficient-Large-Model--Sana_600M_512px_diffusers/snapshots/83d7a190bfd1fd070570a793d2dab5c7a3231b9d

PY=/Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python
BASE=output/sana_hypersim_metric_depth_lr1e5_20260822/checkpoint-500
BASE_LORA="$BASE/lora.pt"
OUT=output/sana_depth_sem_v2_legend_20260830
CFG=configs/experiments/sana_depth_sem_v2_legend_20260830.yaml
VIZ_DATA=data/flux_paper_gate_512

mkdir -p "$OUT/logs" "$OUT/evals"

if ps aux | awk '/train_sana_512|evaluate_paper_benchmarks|viz_sana_paper_gate/ && !/awk/ {print}' | grep -q .; then
  echo "[v2] $(date '+%F %T') MPS busy; refuse to start" | tee -a "$OUT/logs/runner.log"
  exit 1
fi

# Optional continuation from a previous v2 checkpoint.
RESUME_ARGS=(--resume "$BASE_LORA" --resume-weights-only)
if [[ -n "${VB_RESUME:-}" && -f "$VB_RESUME" ]]; then
  RESUME_ARGS=(--resume "$VB_RESUME")
  echo "[v2] $(date '+%F %T') continuing from $VB_RESUME" | tee -a "$OUT/logs/runner.log"
fi

echo "[v2] $(date '+%F %T') train depth+semantic (legend prompt, lr 1e-6) -> 1000" | tee -a "$OUT/logs/runner.log"
"$PY" scripts/train_sana_512.py \
  --config "$CFG" \
  --save-every 200 \
  --log-every 10 \
  "${RESUME_ARGS[@]}" \
  >>"$OUT/logs/train.log" 2>&1

CKPT=""
for step in 1000 800 600 400 200; do
  if [ -d "$OUT/checkpoint-${step}" ]; then
    CKPT="$OUT/checkpoint-${step}"
    break
  fi
done
if [ -z "$CKPT" ]; then
  echo "[v2] no checkpoint after train" | tee -a "$OUT/logs/runner.log"
  exit 1
fi
LORA="$CKPT/lora_adapter"
echo "[v2] $(date '+%F %T') evaluating $CKPT" | tee -a "$OUT/logs/runner.log"

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
  --output-json "$OUT/evals/nyu_delta1_n32.json" \
  >"$OUT/logs/eval_nyu_n32.log" 2>&1

set +e
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
  --output-json "$OUT/evals/cityscapes_miou_n16.json" \
  >"$OUT/logs/eval_cityscapes_n16.log" 2>&1
CITY_RC=$?
set -e
if [ "$CITY_RC" -ne 0 ]; then
  echo "[v2] cityscapes eval failed rc=$CITY_RC (see log)" | tee -a "$OUT/logs/runner.log"
fi

echo "[v2] $(date '+%F %T') viz $CKPT" | tee -a "$OUT/logs/runner.log"
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

echo "[v2] $(date '+%F %T') done. ckpt=$CKPT" | tee -a "$OUT/logs/runner.log"
