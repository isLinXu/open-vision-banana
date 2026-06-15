#!/usr/bin/env python3
"""Model export script for Open-Vision-Banana.

Exports the trained LoRA-augmented FLUX.1 transformer and RGB mapping
decoders to ONNX format for deployment.

Usage:
    python scripts/export.py --config configs/training_config.yaml
    python scripts/export.py --model-dir output/checkpoints --output-dir output/onnx
"""

import argparse
import logging
from pathlib import Path

import torch

from open_vision_banana import VisionBananaConfig
from open_vision_banana.export import (
    DepthRGBMapperONNX,
    NormalRGBMapperONNX,
    export_mapper_onnx,
    export_full_pipeline_onnx,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--model-dir", type=str, default="output/checkpoints",
                        help="Directory containing trained model checkpoints")
    parser.add_argument("--output-dir", type=str, default="output/onnx",
                        help="Output directory for ONNX files")
    parser.add_argument("--opset-version", type=int, default=17,
                        help="ONNX opset version")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export depth RGB mapper
    logger.info("Exporting depth RGB mapper to ONNX...")
    export_mapper_onnx("depth_to_rgb", str(output_dir / "depth_to_rgb.onnx"), args.opset_version)

    # Export normal RGB mapper
    logger.info("Exporting normal RGB mapper to ONNX...")
    export_mapper_onnx("normal_to_rgb", str(output_dir / "normal_to_rgb.onnx"), args.opset_version)

    logger.info(f"Models exported to {output_dir}")


if __name__ == "__main__":
    main()
