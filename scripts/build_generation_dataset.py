#!/usr/bin/env python3
"""Build a generation preservation dataset for 9:1 training mix.

Creates ``images/`` + ``prompts.json`` + ``metadata.jsonl`` compatible with
``GenerationDataset`` and ``MixedVisionDataset``.

Usage:
    # Fast synthetic dataset (no GPU)
    python scripts/build_generation_dataset.py --output-dir gen_data/ --num-samples 200

    # From custom prompts file
    python scripts/build_generation_dataset.py \\
        --output-dir gen_data/ --prompts-file prompts.txt --mode synthetic

    # Generate with FLUX.1 (requires GPU + ~24GB VRAM)
    python scripts/build_generation_dataset.py \\
        --output-dir gen_data/ --prompts-file prompts.txt --mode flux
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from open_vision_banana.data.generation import build_generation_dataset
from open_vision_banana.data.generation_builder import (
    build_flux_generation_dataset,
    build_synthetic_generation_dataset,
    load_prompts_file,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build generation preservation dataset for Vision Banana training"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write images/ + prompts.json",
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "flux"],
        default="synthetic",
        help="synthetic=fast placeholders; flux=FLUX.1 text-to-image (GPU)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples for synthetic mode (default: 100)",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Optional .txt or .json prompts file",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--validate", action="store_true", help="Verify output after build")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    prompts = None
    if args.prompts_file:
        prompts = load_prompts_file(args.prompts_file)
        logger.info("Loaded %d prompts from %s", len(prompts), args.prompts_file)

    if args.mode == "synthetic":
        build_synthetic_generation_dataset(
            output_dir,
            num_samples=args.num_samples,
            resolution=args.resolution,
            seed=args.seed,
            prompts=prompts,
        )
    else:
        if not prompts:
            logger.error("FLUX mode requires --prompts-file")
            return 1
        build_flux_generation_dataset(
            output_dir,
            prompts=prompts,
            model_name=args.model_name,
            resolution=args.resolution,
            device=args.device,
            seed=args.seed,
        )

    ds = build_generation_dataset(output_dir, resolution=args.resolution)
    logger.info("GenerationDataset reports %d loadable samples", len(ds))

    if args.validate:
        img_dir = output_dir / "images"
        if not img_dir.is_dir() or not any(img_dir.iterdir()):
            logger.error("Validation failed: no images written")
            return 1
        if not (output_dir / "prompts.json").exists():
            logger.error("Validation failed: prompts.json missing")
            return 1
        logger.info("Basic generation dataset validation passed")

    print(str(output_dir.resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
