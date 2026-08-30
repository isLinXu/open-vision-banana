#!/bin/zsh
set -euo pipefail

cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OPEN_VISION_BANANA_MODEL_PATH=/Users/gatilin/.cache/huggingface/hub/models--Efficient-Large-Model--Sana_600M_512px_diffusers/snapshots/83d7a190bfd1fd070570a793d2dab5c7a3231b9d

CFG=configs/experiments/sana_multitask_v03_no_normal_weighted_20260821.yaml
OUT=output/sana_multitask_v03_no_normal_weighted_20260821
PY=/Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python
MANIFEST=output/mar_vb_clean_pairs_v4/pairs.jsonl
DATA=/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v5_full

mkdir -p "$OUT/segment_logs" "$OUT/evals"
prev=0
stale=0

# Gate at step 500: do not lengthen unless eval metrics pass.
# A watcher writes gate_step500.json; continue only on an explicit pass.
for stop in 500; do
  if (( stop <= prev )); then continue; fi
  echo "[segment] target=$stop/4000 resume=$prev"
  if (( prev == 0 )); then
    "$PY" scripts/train_sana_512.py --config "$CFG" --stop-after-step "$stop" \
      --save-every 100 \
      >"$OUT/segment_logs/train_to_$stop.log" 2>&1
  else
    "$PY" scripts/train_sana_512.py --config "$CFG" --stop-after-step "$stop" \
      --save-every 100 \
      --resume "$OUT/checkpoint-$prev/lora.pt" \
      >"$OUT/segment_logs/train_to_$stop.log" 2>&1
  fi

  ckpt="$OUT/checkpoint-$stop/lora_adapter"
  eval_out="$OUT/evals/val_n8_step_$stop"
  "$PY" scripts/eval_sana_unified.py \
    --data-dir "$DATA" --pair-manifest "$MANIFEST" --checkpoint "$ckpt" \
    --output-dir "$eval_out" --split val --n-per-task 8 --batch-size 1 \
    --resolution 512 --num-inference-steps 20 --guidance-scale 1.0 --seed 42 \
    --device mps --dtype float16 --tasks depth,semantic,structure --save-panels \
    >"$OUT/segment_logs/eval_$stop.log" 2>&1

  "$PY" - "$eval_out/metrics.json" "$OUT/selection_state.json" <<'PY'
import json, sys
metrics = json.load(open(sys.argv[1]))["per_task"]
values = {
    "depth": metrics["depth"]["relative_position_spearman_r"],
    "semantic": metrics["semantic"]["semantic_miou"],
    "structure": metrics["structure"]["structure_edge_f1_ods"],
}
try:
    state = json.load(open(sys.argv[2]))
except FileNotFoundError:
    state = {"best": {"depth": -999.0, "semantic": -999.0, "structure": -999.0}, "stale": 0}
changed = {k: v > state["best"].get(k, -999.0) + 0.002 for k, v in values.items()}
state["best"] = {k: max(v, state["best"].get(k, -999.0)) for k, v in values.items()}
state["last"] = values
state["improved"] = changed
state["stale"] = 0 if any(changed.values()) else state.get("stale", 0) + 1
json.dump(state, open(sys.argv[2], "w"), indent=2)
print(json.dumps({"values": values, "improved": changed, "stale": state["stale"]}))
PY
  echo "[segment] step=$stop metrics recorded in $eval_out/metrics.json" | tee -a "$OUT/selection.log"

  stale=$("$PY" - "$OUT/selection_state.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["stale"])
PY
)
  if (( stop >= 1000 && stale >= 2 )); then
    echo "[early-stop] no task metric improved for two consecutive evaluations" | tee -a "$OUT/selection.log"
    break
  fi
  prev=$stop
done

echo "[complete] segmented weighted no-normal experiment finished" | tee -a "$OUT/selection.log"
