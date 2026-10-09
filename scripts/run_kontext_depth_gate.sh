#!/usr/bin/env bash
# Pre-registered depth gate for the Kontext (in-context conditioning) experiment.
#
#   1. Score trivial baselines on NYU under the exact benchmark protocol.
#   2. Train depth-only LoRA on FLUX.1-Kontext (configs/experiments/flux_kontext_depth_gate.yaml).
#   3. At each checkpoint, evaluate NYU (native resolution, full multi-step sampling).
#   4. Apply the metric + structure gates (src/open_vision_banana/evaluation/gates.py).
#
# Requires: CUDA GPU, data/hypersim_depth_512, NYU val at $NYU_ROOT, and (optionally) a
# synthetic prior set for the mean-depth-map baseline. Usage:
#   NYU_ROOT=/data/nyu bash scripts/run_kontext_depth_gate.sh [N_EVAL_SAMPLES]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

NYU_ROOT="${NYU_ROOT:?set NYU_ROOT to the NYU depth val root}"
N="${1:-32}"                    # samples per gated eval (use 654 for the final report)
STEPS="${GATE_EVAL_STEPS:-28}"  # sampling steps; never 1 (single-step x0 is not the paper protocol)
CFG=configs/experiments/flux_kontext_depth_gate.yaml
OUT=output/flux_kontext_depth_gate
EVAL="$OUT/evals"
mkdir -p "$EVAL"

echo "== 1/4 trivial baselines (NYU, n=$N) =="
python scripts/depth_trivial_baselines.py \
  --dataset-key nyu_depth --dataset-name nyu --root "$NYU_ROOT" --native-resolution \
  --prior-key depth --prior-root data/hypersim_depth_512 --prior-split train \
  --max-samples "$N" --output-json "$EVAL/nyu_trivial_baselines.json"

echo "== 2/4 training =="
python scripts/train.py --config "$CFG"

echo "== 3/4 + 4/4 evaluate and gate every checkpoint =="
status=1
for ckpt in "$OUT"/checkpoint-*/; do
  [ -d "$ckpt" ] || continue
  name="$(basename "$ckpt")"
  python scripts/evaluate_paper_benchmarks.py \
    --benchmark depth --nyu-root "$NYU_ROOT" --split val \
    --model-name flux-kontext --lora-path "$ckpt" --dtype bfloat16 \
    --native-resolution --num-inference-steps "$STEPS" --batch-size 1 \
    --max-samples "$N" --output-json "$EVAL/nyu_${name}.json"
  if python scripts/check_depth_gate.py \
      --model-json "$EVAL/nyu_${name}.json" --benchmark depth/nyu \
      --baseline-json "$EVAL/nyu_trivial_baselines.json" \
      --output-json "$EVAL/gate_${name}.json"; then
    echo ">>> $name PASSED both gates"; status=0
  else
    echo ">>> $name failed"
  fi
done
exit $status
