#!/usr/bin/env python3
"""ONNX export script for Open-Vision-Banana.

Exports trained RGB mappers and/or the full pipeline to ONNX format.

Usage:
    python scripts/export_onnx.py --mapper depth_to_rgb --output models/depth_to_rgb.onnx
    python scripts/export_onnx.py --mapper normal_to_rgb --output models/normal_to_rgb.onnx
    python scripts/export_onnx.py --all-mappers --output-dir models/
"""

import argparse
import logging
from pathlib import Path

from open_vision_banana.export import export_mapper_onnx

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Export Vision Banana to ONNX")
    parser.add_argument("--mapper", type=str, default=None,
                        choices=["depth_to_rgb", "normal_to_rgb"],
                        help="Which mapper to export")
    parser.add_argument("--all-mappers", action="store_true",
                        help="Export all mappers")
    parser.add_argument("--output", type=str, default=None,
                        help="Output ONNX file path")
    parser.add_argument("--output-dir", type=str, default="models/",
                        help="Output directory for --all-mappers")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.all_mappers:
        mappers = ["depth_to_rgb", "normal_to_rgb"]
        for mapper in mappers:
            output_path = str(Path(args.output_dir) / f"{mapper}.onnx")
            export_mapper_onnx(mapper, output_path, opset_version=args.opset)
    elif args.mapper:
        if not args.output:
            args.output = str(Path(args.output_dir) / f"{args.mapper}.onnx")
        export_mapper_onnx(args.mapper, args.output, opset_version=args.opset)
    else:
        logger.error("Specify --mapper or --all-mappers")
        return

    logger.info("Export complete!")


if __name__ == "__main__":
    main()
