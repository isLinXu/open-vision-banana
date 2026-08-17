#!/bin/bash
# generate_depth_normal_mps.sh - Generate COCO and Object365 depth/normal targets
# using Depth-Anything-Small on macOS MPS (Apple Silicon)
#
# Usage: bash scripts/generate_depth_normal_mps.sh [--coco-only | --object365-only | --all]
#
# Prerequisites (macOS):
#   pip install torch transformers pillow
#   # MPS is automatically used on Apple Silicon Macs
#
# Output:
#   coco2017/depth/000000XXXXXX_depth.png     (grayscale, 0-255, closer=brighter)
#   coco2017/normal/000000XXXXXX_normal.png   (RGB, XYZ channels)
#   object365/depth/patchXX_objects365_v2_xxx_depth.png

set -e

DATASETS_ROOT="${DATASETS_ROOT:-/Users/gatilin/MyWork/datasets}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:---all}"

echo "============================================================"
echo "Depth/Normal Target Generation (macOS MPS)"
echo "============================================================"
echo "Datasets root: $DATASETS_ROOT"
echo "Mode: $MODE"
echo ""

# Run Python script
python3 "$SCRIPT_DIR/generate_depth_normal.py" \
    --datasets-root "$DATASETS_ROOT" \
    --mode "$MODE"
