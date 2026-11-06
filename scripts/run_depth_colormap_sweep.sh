#!/usr/bin/env bash
# P0 diagnostic — depth prompt × colormap sweep on an existing checkpoint (no retraining).
#
# Question it answers: did multi-colormap training teach the model per-palette
# bindings that the bare "metric depth" prompt fails to select? Method:
#   1. baseline   : canonical prompt + cube_edge decode (the E1/E2 eval protocol)
#   2. named      : "...using the {cmap} colormap." prompt + matching decode, for
#                   every colormap in --colormaps (the trained named bindings)
#   3. oracle     : canonical prompt + per-image best-roundtrip colormap decode
#                   (--depth-oracle-colormaps). If the canonical output is a
#                   consistent non-cube_edge palette, oracle recovers it.
#
# Read-out: eval_*.json under OUT_DIR plus a summary CSV. If a named or oracle
# reading beats baseline δ1.25 by a wide margin, the depth failure is a
# prompt-binding issue, not a capacity issue.
#
#   bash scripts/run_depth_colormap_sweep.sh --lora-path <ckpt> \
#       --nyu-root data/raw/nyu_depth_v2 --out-dir output/depth_sweep
#
# Env knobs: MODEL_NAME (qwen-image-2.1) INFER_STEPS (50) GUIDANCE (4.0)
#            EVAL_SEED (42) MAX_SAMPLES (100) RESOLUTION (512) DTYPE (bfloat16)
#            COLORMAPS ("cube_edge plasma inferno viridis grayscale") PYTHON (python)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:+:$PYTHONPATH}"

MODEL_NAME="${MODEL_NAME:-qwen-image-2.1}"
INFER_STEPS="${INFER_STEPS:-50}"; GUIDANCE="${GUIDANCE:-4.0}"; EVAL_SEED="${EVAL_SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"; RESOLUTION="${RESOLUTION:-512}"; DTYPE="${DTYPE:-bfloat16}"
COLORMAPS="${COLORMAPS:-cube_edge plasma inferno viridis grayscale}"
PYTHON="${PYTHON:-python}"

LORA_PATH=""; NYU_ROOT=""; OUT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lora-path) LORA_PATH="$2"; shift ;;
    --nyu-root) NYU_ROOT="$2"; shift ;;
    --out-dir) OUT_DIR="$2"; shift ;;
    --model-name) MODEL_NAME="$2"; shift ;;
    --max-samples) MAX_SAMPLES="$2"; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
[[ -n $LORA_PATH && -n $NYU_ROOT && -n $OUT_DIR ]] || { echo "need --lora-path --nyu-root --out-dir" >&2; exit 2; }
mkdir -p "$OUT_DIR"

CANONICAL="Generate a metric depth visualization of this image."
run_eval() { # name extra-args...
  local name="$1"; shift
  echo "[sweep] $name"
  "$PYTHON" scripts/evaluate_paper_benchmarks.py --model-name "$MODEL_NAME" --lora-path "$LORA_PATH" \
    --num-inference-steps "$INFER_STEPS" --guidance-scale "$GUIDANCE" --seed "$EVAL_SEED" \
    --resolution "$RESOLUTION" --native-resolution --dtype "$DTYPE" \
    --nyu-root "$NYU_ROOT" --max-samples "$MAX_SAMPLES" \
    --output-json "$OUT_DIR/eval_${name}.json" "$@"
}

# 1. baseline: canonical prompt, cube_edge decode — the protocol used so far
run_eval baseline --depth-prompt "$CANONICAL" --depth-colormap cube_edge

# 2. named bindings: prompt names the palette, decode matches it
for cmap in $COLORMAPS; do
  run_eval "named_${cmap}" \
    --depth-prompt "Generate a metric depth visualization of this image using the ${cmap} colormap." \
    --depth-colormap "$cmap"
done

# 3. oracle decode of the canonical-prompt output (single inference pass)
ORACLE_LIST="$(echo "$COLORMAPS" | tr ' ' ',')"
run_eval oracle --depth-prompt "$CANONICAL" --depth-colormap cube_edge --depth-oracle-colormaps "$ORACLE_LIST"

# summary
"$PYTHON" - "$OUT_DIR" <<'EOF'
import csv, glob, json, sys, os
out_dir = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(out_dir, "eval_*.json"))):
    name = os.path.basename(path)[5:-5]
    b = json.load(open(path))["benchmarks"]["depth/nyu"]["decoded_metrics"]
    oracle = b.get("oracle") or {}
    rows.append({
        "run": name,
        "delta_1.25": b.get("delta_1.25"),
        "AbsRel": b.get("AbsRel"),
        "delta_1.25_aligned": b.get("delta_1.25_aligned"),
        "oracle_delta_1.25": oracle.get("delta_1.25"),
        "oracle_hist": json.dumps(b.get("oracle_colormap_hist") or ""),
    })
with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['run']:>16}  δ1.25={r['delta_1.25']:.4f}  aligned={r['delta_1.25_aligned']:.4f}"
          + (f"  oracle_δ1.25={r['oracle_delta_1.25']:.4f}  {r['oracle_hist']}" if r['oracle_delta_1.25'] else ""))
print("wrote", os.path.join(out_dir, "summary.csv"))
EOF
