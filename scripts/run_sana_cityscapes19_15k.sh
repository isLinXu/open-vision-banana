#!/usr/bin/env bash
# Continue the official CS-19 specialist: reload LoRA only, fresh 15000-step cosine.
# Do not resume optimizer/scheduler from the finished 3000-step run (lr was 5e-7).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OPEN_VISION_BANANA_MODEL_PATH="${OPEN_VISION_BANANA_MODEL_PATH:?set OPEN_VISION_BANANA_MODEL_PATH to the local SANA snapshot}"

OUT_DATA="${CITYSCAPES19_OUT:-data/cityscapes19}"
RESUME="${CITYSCAPES19_RESUME:-output/sana_1600m_cityscapes19/checkpoint-3000_best/lora.pt}"
RUN_OUT="${CITYSCAPES19_15K_OUT:-output/sana_1600m_cityscapes19_15k}"

python scripts/train.py \
  --config configs/experiments/sana_1600m_cityscapes19_15k.yaml \
  --data-dir "$OUT_DATA" \
  --pair-manifest "$OUT_DATA/pairs.jsonl" \
  --output-dir "$RUN_OUT" \
  --resume "$RESUME" \
  --resume-weights-only
