#!/usr/bin/env python3
"""Evaluation script for Open-Vision-Banana.

Runs evaluation on a trained model for any supported vision task.

Usage:
    python scripts/evaluate.py --task depth --data-path /path/to/data
    python scripts/evaluate.py --task normal --checkpoint output/best.pt
    python scripts/evaluate.py --task semantic --num-classes 19
"""

import argparse
import logging
from pathlib import Path

import torch

from open_vision_banana.evaluation import VisionEvaluator
from open_vision_banana.data import build_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Vision Banana model")
    parser.add_argument("--task", type=str, required=True,
                        choices=["depth", "normal", "semantic", "instance", "referring"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model checkpoint")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to evaluation dataset")
    parser.add_argument("--num-classes", type=int, default=19,
                        help="Number of classes for semantic segmentation")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limit evaluation to N batches")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Evaluating {args.task} on {args.data_path}")
    logger.info(f"Device: {device}")

    # Build dataset
    dataset = build_dataset(args.task, args.data_path, split="val")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # Load model — use VisionBanana.from_pretrained for LoRA checkpoints
    if args.checkpoint:
        from open_vision_banana import VisionBanana
        model = VisionBanana.from_pretrained(
            lora_path=args.checkpoint,
            device=device,
        )
    else:
        model = None

    evaluator = VisionEvaluator(model, device=device)
    metrics = evaluator.evaluate(
        task=args.task,
        num_classes=args.num_classes,
    )

    # Print results
    logger.info("=" * 50)
    logger.info(f"Evaluation Results ({args.task})")
    logger.info("=" * 50)
    for name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {name}: {value:.4f}")
        elif isinstance(value, list):
            logger.info(f"  {name}: {[f'{v:.4f}' for v in value]}")


if __name__ == "__main__":
    main()
