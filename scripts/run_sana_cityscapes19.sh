#!/usr/bin/env bash
# Extract official Cityscapes train+val, paint the 19-color targets, and train
# a SANA-1600M semantic specialist. Do not use cityscapes8 or ADE-150 SNVC segs.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OPEN_VISION_BANANA_MODEL_PATH="${OPEN_VISION_BANANA_MODEL_PATH:?set OPEN_VISION_BANANA_MODEL_PATH to the local SANA snapshot}"

CS_ROOT="${CITYSCAPES_ROOT:-data/paper_benchmarks/cityscapes}"
GTFINE_ZIP="${CITYSCAPES_GTFINE_ZIP:-$CS_ROOT/zips/gtFine_trainvaltest.zip}"
LEFTIMG_ZIP="${CITYSCAPES_LEFTIMG_ZIP:-$CS_ROOT/zips/leftImg8bit_trainvaltest.zip}"
OUT_DATA="${CITYSCAPES19_OUT:-data/cityscapes19}"
RUN_OUT="${CITYSCAPES19_RUN_OUT:-output/sana_1600m_cityscapes19}"

python scripts/setup_cityscapes_official.py \
  --gtfine-zip "$GTFINE_ZIP" \
  --leftimg-zip "$LEFTIMG_ZIP" \
  --output "$CS_ROOT" \
  --splits train val

python scripts/build_cityscapes19_manifest.py \
  --cityscapes-root "$CS_ROOT" \
  --output "$OUT_DATA" \
  --resolution 512 \
  --splits train val

python scripts/train.py \
  --config configs/experiments/sana_1600m_cityscapes19.yaml \
  --data-dir "$OUT_DATA" \
  --pair-manifest "$OUT_DATA/pairs.jsonl" \
  --output-dir "$RUN_OUT"
