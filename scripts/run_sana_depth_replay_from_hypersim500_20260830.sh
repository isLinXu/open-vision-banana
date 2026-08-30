#!/bin/zsh
set -euo pipefail

# Depth-only indoor+outdoor replay from Hypersim-500 with NYU gates.
# Design: docs/plans/2026-08-30-sana-depth-indoor-outdoor-replay-design.md
# Dual gate: δ1≥0.55 + qualitative. Early-stop if δ1<0.33.
# Sequential MPS only.

cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OPEN_VISION_BANANA_MODEL_PATH=/Users/gatilin/.cache/huggingface/hub/models--Efficient-Large-Model--Sana_600M_512px_diffusers/snapshots/83d7a190bfd1fd070570a793d2dab5c7a3231b9d

PY=/Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python
CFG=configs/experiments/sana_depth_replay_from_hypersim500_20260830.yaml
OUT=output/sana_depth_replay_from_hypersim500_20260830
BASE=output/sana_hypersim_metric_depth_lr1e5_20260822/checkpoint-500
BASE_LORA="$BASE/lora.pt"
DATA=data/sana_depth_indoor_outdoor_replay_512
BASELINE_DELTA=0.357017
EARLY_STOP_DELTA=0.33
SUCCESS_DELTA=0.55
GATE_EVERY=100
MAX_STEP=500

mkdir -p "$OUT/logs" "$OUT/evals"

if ps aux | awk '/train_sana_512|train_sdxl_mps|evaluate_paper_benchmarks|viz_sana/ && !/awk/ {print}' | grep -q .; then
  echo "[depth-replay] $(date '+%F %T') MPS busy; refuse to start" | tee -a "$OUT/logs/runner.log"
  exit 1
fi

if [ ! -f "$BASE_LORA" ]; then
  echo "[depth-replay] missing baseline $BASE_LORA" | tee -a "$OUT/logs/runner.log"
  exit 1
fi

if [ ! -f "$DATA/pairs.jsonl" ]; then
  echo "[depth-replay] $(date '+%F %T') building mix manifest" | tee -a "$OUT/logs/runner.log"
  "$PY" scripts/build_sana_depth_indoor_outdoor_replay.py \
    --output "$DATA" \
    --kitti-count 200 \
    --seed 42 \
    | tee -a "$OUT/logs/runner.log"
fi

extract_delta() {
  local json_path="$1"
  "$PY" - <<PY
import json
from pathlib import Path
p = Path("$json_path")
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
v = find(d)
if v is None:
    raise SystemExit(f"no delta_1.25 in {p}")
print(f"{v:.6f}")
PY
}

write_gate_summary() {
  local best_step="$1"
  local best_delta="$2"
  local gate_status="$3"
  GATE_STATUS_ARG="$gate_status" BEST_STEP_ARG="$best_step" BEST_DELTA_ARG="$best_delta" \
  BASELINE_DELTA_ARG="$BASELINE_DELTA" EARLY_STOP_DELTA_ARG="$EARLY_STOP_DELTA" \
  SUCCESS_DELTA_ARG="$SUCCESS_DELTA" OUT_ARG="$OUT" \
  "$PY" - <<'PY'
import json
import os
from datetime import datetime
from pathlib import Path
out = Path(os.environ["OUT_ARG"])
best_step = os.environ["BEST_STEP_ARG"]
best_delta = os.environ["BEST_DELTA_ARG"]
gate_status = os.environ["GATE_STATUS_ARG"]
payload = {
    "updated_at": datetime.now().isoformat(timespec="seconds"),
    "baseline_delta": float(os.environ["BASELINE_DELTA_ARG"]),
    "early_stop_delta": float(os.environ["EARLY_STOP_DELTA_ARG"]),
    "success_delta": float(os.environ["SUCCESS_DELTA_ARG"]),
    "best_step": int(best_step) if best_step.isdigit() else None,
    "best_delta": float(best_delta) if best_delta not in ("", "None") else None,
    "status": gate_status,
    "dual_gate_passed": False,
}
prev = out / "gate_summary.json"
if prev.is_file():
    old = json.loads(prev.read_text())
    payload["history"] = old.get("history", [])
else:
    payload["history"] = []
payload["history"].append({
    "at": payload["updated_at"],
    "best_step": payload["best_step"],
    "best_delta": payload["best_delta"],
    "status": payload["status"],
})
# Quantitative flag only; qualitative still required for dual gate.
if payload["best_delta"] is not None and payload["best_delta"] >= float(os.environ["SUCCESS_DELTA_ARG"]):
    payload["quantitative_gate_passed"] = True
else:
    payload["quantitative_gate_passed"] = False
prev.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(payload, indent=2))
PY
}

BEST_STEP=0
BEST_DELTA="$BASELINE_DELTA"
GATE_STATUS="running"

echo "[depth-replay] $(date '+%F %T') start from Hypersim-500; baseline δ1=$BASELINE_DELTA" | tee -a "$OUT/logs/runner.log"
write_gate_summary "$BEST_STEP" "$BEST_DELTA" "baseline_recorded" >/dev/null

RESUME_ARGS=(--resume "$BASE_LORA" --resume-weights-only)
TARGET=0

while [ "$TARGET" -lt "$MAX_STEP" ]; do
  TARGET=$((TARGET + GATE_EVERY))
  if [ "$TARGET" -gt "$MAX_STEP" ]; then
    TARGET=$MAX_STEP
  fi

  echo "[depth-replay] $(date '+%F %T') train → step $TARGET" | tee -a "$OUT/logs/runner.log"
  "$PY" scripts/train_sana_512.py \
    --config "$CFG" \
    --stop-after-step "$TARGET" \
    --save-every 50 \
    --log-every 10 \
    "${RESUME_ARGS[@]}" \
    >>"$OUT/logs/train.log" 2>&1

  CKPT="$OUT/checkpoint-${TARGET}"
  if [ ! -d "$CKPT" ]; then
    # fall back to highest available ≤ TARGET
    CKPT=""
    for step in $(seq "$TARGET" -50 50); do
      if [ -d "$OUT/checkpoint-${step}" ]; then
        CKPT="$OUT/checkpoint-${step}"
        TARGET=$step
        break
      fi
    done
  fi
  if [ -z "$CKPT" ]; then
    echo "[depth-replay] no checkpoint after train to $TARGET" | tee -a "$OUT/logs/runner.log"
    write_gate_summary "$BEST_STEP" "$BEST_DELTA" "missing_checkpoint"
    exit 1
  fi

  LORA="$CKPT/lora_adapter"
  EVAL_JSON="$OUT/evals/nyu_delta1_n32_step${TARGET}.json"
  echo "[depth-replay] $(date '+%F %T') NYU n=32 on $CKPT" | tee -a "$OUT/logs/runner.log"
  "$PY" scripts/evaluate_paper_benchmarks.py \
    --benchmark depth \
    --nyu-root data/benchmarks/nyu/extracted \
    --model-name sana-600m-512px \
    --lora-path "$LORA" \
    --device mps \
    --dtype float16 \
    --allow-mps-reduced-dtype \
    --resolution 512 \
    --num-inference-steps 20 \
    --max-samples 32 \
    --output-json "$EVAL_JSON" \
    >"$OUT/logs/eval_nyu_n32_step${TARGET}.log" 2>&1

  DELTA="$(extract_delta "$EVAL_JSON")"
  echo "[depth-replay] $(date '+%F %T') step=$TARGET δ1=$DELTA" | tee -a "$OUT/logs/runner.log"

  # Confirm LoRA path once (first gate)
  if [ "$TARGET" -eq "$GATE_EVERY" ]; then
    if ! rg -q "Manual LoRA merge:.*scale=8" "$OUT/logs/eval_nyu_n32_step${TARGET}.log"; then
      echo "[depth-replay] WARNING: expected manual rsLoRA merge scale=8 not found in eval log" | tee -a "$OUT/logs/runner.log"
    else
      echo "[depth-replay] LoRA load OK (manual rsLoRA scale=8)" | tee -a "$OUT/logs/runner.log"
    fi
  fi

  echo "[depth-replay] $(date '+%F %T') NYU n=8 viz step $TARGET" | tee -a "$OUT/logs/runner.log"
  "$PY" scripts/viz_sana_nyu_depth.py \
    --checkpoint "$CKPT" \
    --nyu-root data/benchmarks/nyu/extracted \
    --output-dir "$OUT/viz_nyu_n8_step${TARGET}" \
    --max-samples 8 \
    --device mps \
    --dtype float16 \
    >"$OUT/logs/viz_nyu_n8_step${TARGET}.log" 2>&1

  # Track best δ1
  read -r BEST_STEP BEST_DELTA <<< "$("$PY" - <<PY
best = float("$BEST_DELTA")
cur = float("$DELTA")
step = int("$TARGET")
best_step = int("$BEST_STEP")
if cur >= best:
    print(f"{step} {cur:.6f}")
else:
    print(f"{best_step} {best:.6f}")
PY
)"

  # Early stop / success
  STOP=0
  if "$PY" -c "import sys; sys.exit(0 if float('$DELTA') < float('$EARLY_STOP_DELTA') else 1)"; then
    GATE_STATUS="early_stop_delta_below_${EARLY_STOP_DELTA}"
    echo "[depth-replay] EARLY STOP δ1=$DELTA < $EARLY_STOP_DELTA" | tee -a "$OUT/logs/runner.log"
    STOP=1
  elif "$PY" -c "import sys; sys.exit(0 if float('$DELTA') >= float('$SUCCESS_DELTA') else 1)"; then
    GATE_STATUS="quantitative_success_delta_ge_${SUCCESS_DELTA}"
    echo "[depth-replay] QUANT GATE PASS δ1=$DELTA ≥ $SUCCESS_DELTA (qualitative still required)" | tee -a "$OUT/logs/runner.log"
    STOP=1
  else
    GATE_STATUS="running"
  fi

  write_gate_summary "$BEST_STEP" "$BEST_DELTA" "$GATE_STATUS" >/dev/null

  if [ "$STOP" -eq 1 ]; then
    break
  fi

  # Continue with full resume (optimizer + step)
  RESUME_ARGS=(--resume "$CKPT/lora.pt")
done

if [ "$GATE_STATUS" = "running" ]; then
  GATE_STATUS="completed_max_steps"
  write_gate_summary "$BEST_STEP" "$BEST_DELTA" "$GATE_STATUS" >/dev/null
fi

echo "[depth-replay] $(date '+%F %T') done status=$GATE_STATUS best_step=$BEST_STEP best_δ1=$BEST_DELTA" | tee -a "$OUT/logs/runner.log"
