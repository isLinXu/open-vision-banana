#!/usr/bin/env bash
# Download a few Hypersim scenes, encode metric depth as cube-edge RGB, and
# train a depth-only SANA-1600M LoRA. NYU is not part of this training set.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OPEN_VISION_BANANA_MODEL_PATH="${OPEN_VISION_BANANA_MODEL_PATH:?set OPEN_VISION_BANANA_MODEL_PATH to the local SANA snapshot}"

RAW="${HYPERSIM_RAW:-$HOME/data/hypersim_raw}"
SCENES="${HYPERSIM_SCENES:-$HOME/data/hypersim_scenes}"
OUT_DATA="${HYPERSIM_OUT:-$HOME/data/hypersim_depth_512}"
RUN_OUT="${HYPERSIM_RUN_OUT:-output/sana_1600m_hypersim_metric_depth}"
MAX_SAMPLES="${HYPERSIM_MAX_SAMPLES:-1000}"
SCENES_LIST="${HYPERSIM_SCENE_LIST:-ai_001_001 ai_001_002 ai_001_003 ai_001_004}"

mkdir -p "$RAW" "$SCENES"
for scene in $SCENES_LIST; do
  zip="$RAW/${scene}.zip"
  url="https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/${scene}.zip"
  if [ ! -s "$zip" ]; then
    curl -L -C - --fail -o "$zip" "$url"
  fi
  if [ ! -d "$SCENES/$scene" ]; then
    unzip -q "$zip" -d "$SCENES"
  fi
done

python scripts/prepare_hypersim.py \
  --source "$SCENES" \
  --output "$OUT_DATA" \
  --resolution 512 \
  --max-samples "$MAX_SAMPLES" \
  --overwrite

python scripts/train.py \
  --config configs/experiments/sana_1600m_hypersim_metric_depth.yaml \
  --data-dir "$OUT_DATA" \
  --pair-manifest "$OUT_DATA/pairs.jsonl" \
  --output-dir "$RUN_OUT"
