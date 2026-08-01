#!/usr/bin/env python3
"""RGB mapper export script for Open-Vision-Banana.

Exports standalone RGB mapping components to ONNX. It does not package a
diffusion backbone, VAE, text encoders, tokenizer, or scheduler.

Usage:
    python scripts/export.py --config configs/training_config.yaml
    python scripts/export.py --model-dir output/checkpoints --output-dir output/onnx
"""

import argparse
import logging
from pathlib import Path

from open_vision_banana.export import (
    export_mapper_onnx,
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
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["int8"],
        help="Apply post-export quantization to RGB mappers (int8 dynamic)",
    )
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

    if args.quantization == "int8":
        import torch

        from open_vision_banana.export.onnx_export import DepthRGBMapperONNX, NormalRGBMapperONNX
        from open_vision_banana.inference.optimization import quantize_mapper_int8

        logger.info("Exporting INT8 TorchScript mappers...")
        depth_q = quantize_mapper_int8(DepthRGBMapperONNX())
        normal_q = quantize_mapper_int8(NormalRGBMapperONNX())
        torch.jit.save(torch.jit.trace(depth_q, torch.rand(1, 1, 64, 64)), str(output_dir / "depth_mapper_int8.pt"))
        torch.jit.save(torch.jit.trace(normal_q, torch.rand(1, 3, 64, 64)), str(output_dir / "normal_mapper_int8.pt"))
        logger.info("INT8 mapper scripts saved")

    logger.info(f"Models exported to {output_dir}")


if __name__ == "__main__":
    main()
