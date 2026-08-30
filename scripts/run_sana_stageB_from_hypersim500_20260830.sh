#!/bin/zsh
set -euo pipefail

# 1) Re-validate Hypersim-500 with current inference stack.
# 2) If δ1 >= 0.25, Stage B: depth+Cityscapes from Hypersim-500 (lr 5e-6, vr=1.0).
# 3) Eval NYU n=32 + Cityscapes n=16 + enhanced viz.
# Sequential MPS only. Do not continue consolidate-300 (failed gate).

cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OPEN_VISION_BANANA_MODEL_PATH=/Users/gatilin/.cache/huggingface/hub/models--Efficient-Large-Model--Sana_600M_512px_diffusers/snapshots/83d7a190bfd1fd070570a793d2dab5c7a3231b9d

PY=/Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python
BASE=output/sana_hypersim_metric_depth_lr1e5_20260822/checkpoint-500
BASE_LORA="$BASE/lora_adapter"
OUT=output/sana_depth_sem_from_hypersim500_20260830
CFG=configs/experiments/sana_depth_sem_from_hypersim500_20260830.yaml
VIZ_DATA=data/flux_paper_gate_512
CHECK_DIR=output/sana_hypersim_metric_depth_lr1e5_20260822

mkdir -p "$OUT/logs" "$OUT/evals" "$CHECK_DIR/evals" "$CHECK_DIR/logs"

if ps aux | awk '/train_sana_512|train_sdxl_mps|evaluate_paper_benchmarks|viz_sana_paper_gate/ && !/awk/ {print}' | grep -q .; then
  echo "[stageB] $(date '+%F %T') MPS busy; refuse to start" | tee -a "$OUT/logs/runner.log"
  exit 1
fi

if [ ! -d "$BASE_LORA" ]; then
  echo "[stageB] missing baseline $BASE_LORA" | tee -a "$OUT/logs/runner.log"
  exit 1
fi

echo "[stageB] $(date '+%F %T') re-eval Hypersim-500 NYU n=32 (current code)" | tee -a "$OUT/logs/runner.log"
"$PY" scripts/evaluate_paper_benchmarks.py \
  --benchmark depth \
  --nyu-root data/benchmarks/nyu/extracted \
  --model-name sana-600m-512px \
  --lora-path "$BASE_LORA" \
  --device mps \
  --dtype float16 \
  --resolution 512 \
  --num-inference-steps 20 \
  --max-samples 32 \
  --output-json "$CHECK_DIR/evals/nyu_delta1_n32_ckpt500_reeval_20260830.json" \
  >"$CHECK_DIR/logs/eval_nyu_n32_ckpt500_reeval_20260830.log" 2>&1

echo "[stageB] $(date '+%F %T') viz Hypersim-500" | tee -a "$OUT/logs/runner.log"
"$PY" scripts/viz_sana_paper_gate.py \
  --checkpoint "$BASE" \
  --data-dir "$VIZ_DATA" \
  --output-dir "$CHECK_DIR/viz_checkpoint-500_reeval_20260830" \
  --n-per-task 2 \
  --resolution 512 \
  --num-inference-steps 20 \
  --device mps \
  --dtype float16 \
  >"$CHECK_DIR/logs/viz_checkpoint-500_reeval_20260830.log" 2>&1

DELTA="$("$PY" - <<'PY'
import json
from pathlib import Path
p = Path("output/sana_hypersim_metric_depth_lr1e5_20260822/evals/nyu_delta1_n32_ckpt500_reeval_20260830.json")
d = json.loads(p.read_text())
def find(obj):
    if isinstance(obj, dict):
        if "delta_1.25" in obj and isinstance(obj["delta_1.25"], (int, float)):
            return float(obj["delta_1.25"])
        for v in obj.values():
            r = find(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find(v)
            if r is not None:
                return r
    return None
print(f"{find(d):.6f}")
PY
)"
echo "[stageB] $(date '+%F %T') Hypersim-500 reeval δ1=$DELTA" | tee -a "$OUT/logs/runner.log"

"$PY" - <<PY
delta = float("$DELTA")
if delta < 0.25:
    raise SystemExit(
        f"Baseline Hypersim-500 δ1={delta:.4f} < 0.25 under current inference; "
        "refuse Stage B until loading/inference is fixed."
    )
print(f"baseline ok δ1={delta:.4f}; starting Stage B")
PY

echo "[stageB] $(date '+%F %T') train depth+semantic from Hypersim-500 → 400" | tee -a "$OUT/logs/runner.log"
"$PY" scripts/train_sana_512.py \
  --config "$CFG" \
  --stop-after-step 400 \
  --save-every 50 \
  --log-every 10 \
  --resume "$BASE/lora.pt" \
  --resume-weights-only \
  >>"$OUT/logs/train_to_400.log" 2>&1

CKPT=""
for step in 400 350 300 250 200 150 100 50; do
  if [ -d "$OUT/checkpoint-${step}" ]; then
    CKPT="$OUT/checkpoint-${step}"
    break
  fi
done
if [ -z "$CKPT" ]; then
  echo "[stageB] no checkpoint after train" | tee -a "$OUT/logs/runner.log"
  exit 1
fi
LORA="$CKPT/lora_adapter"
echo "[stageB] $(date '+%F %T') evaluating $CKPT" | tee -a "$OUT/logs/runner.log"

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
  --output-json "$OUT/evals/nyu_delta1_n32_latest.json" \
  >"$OUT/logs/eval_nyu_n32.log" 2>&1

# Cityscapes may emit nan in per-class IoU; allow failure without aborting viz.
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
  --output-json "$OUT/evals/cityscapes_miou_n16_latest.json" \
  >"$OUT/logs/eval_cityscapes_n16.log" 2>&1
CITY_RC=$?
set -e
if [ "$CITY_RC" -ne 0 ]; then
  echo "[stageB] cityscapes json write failed (rc=$CITY_RC); salvaging from log" | tee -a "$OUT/logs/runner.log"
  "$PY" - <<'PY'
import json, math, re
from pathlib import Path
from datetime import datetime
out = Path("output/sana_depth_sem_from_hypersim500_20260830")
log = (out / "logs/eval_cityscapes_n16.log").read_text(errors="ignore")
m = re.search(r"semantic/cityscapes: (\{.*\})", log)
if not m:
    raise SystemExit("no cityscapes metrics in log")
metrics = eval(m.group(1), {"nan": None, "__builtins__": {}})
def clean(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, list):
        return [clean(v) for v in x]
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    return x
metrics = clean(metrics)
payload = {
    "metadata": {
        "evaluation_mode": "decoded_task_metrics",
        "model_name": "sana-600m-512px",
        "note": "Salvaged after JSON nan failure",
        "salvaged_at": datetime.now().isoformat(timespec="seconds"),
    },
    "benchmarks": {"semantic/cityscapes": {"decoded_metrics": metrics}},
}
(out / "evals/cityscapes_miou_n16_latest.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
)
print("salvaged mIoU", metrics.get("mIoU"))
PY
fi

echo "[stageB] $(date '+%F %T') viz $CKPT" | tee -a "$OUT/logs/runner.log"
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

echo "[stageB] $(date '+%F %T') done. ckpt=$CKPT baseline_delta=$DELTA" | tee -a "$OUT/logs/runner.log"
