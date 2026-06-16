#!/usr/bin/env python3
"""Evaluation script for Open-Vision-Banana.

Runs evaluation on a trained model for any supported vision task.

Usage:
    python scripts/evaluate.py --task depth --data-path /path/to/data
    python scripts/evaluate.py --task normal --lora-path output/lora
    python scripts/evaluate.py --task semantic --num-classes 19 --data-path /cityscapes
"""

import argparse
import json
import logging

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Vision Banana model")
    parser.add_argument(
        "--task", type=str, required=True,
        choices=["depth", "normal", "semantic", "instance", "referring"],
    )
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Path to evaluation dataset root directory",
    )
    parser.add_argument(
        "--lora-path", type=str, default=None,
        help="Path to trained LoRA checkpoint directory",
    )
    parser.add_argument(
        "--model-name", type=str,
        default="black-forest-labs/FLUX.1-schnell",
    )
    parser.add_argument(
        "--num-classes", type=int, default=19,
        help="Number of classes for semantic segmentation",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="Limit evaluation to N batches (useful for quick checks)",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="If set, write metric dict to this JSON file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(f"Task: {args.task} | Device: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────
    from open_vision_banana.data import build_dataset
    dataset = build_dataset(
        args.task, args.data_path, split="val", resolution=args.resolution
    )
    if len(dataset) == 0:
        logger.error("Dataset is empty — check --data-path")
        return

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    logger.info(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches")

    # ── Model ─────────────────────────────────────────────────────────────
    if args.lora_path:
        logger.info(f"Loading FLUX.1 + LoRA from {args.lora_path} …")
        from open_vision_banana.inference.generator import BaseGenerator
        from open_vision_banana.inference.pipeline import VisionBanana
        BaseGenerator(
            model_name=args.model_name,
            lora_path=args.lora_path,
            device=device,
        )
        model = VisionBanana.from_pretrained.__func__(
            VisionBanana, model_name=args.model_name, lora_path=args.lora_path, device=device
        )
    else:
        logger.warning(
            "No --lora-path given — using a dummy identity model for metric smoke-test"
        )
        import torch.nn as nn
        class _Identity(nn.Module):
            def forward(self, x): return x
        model = _Identity().to(device)

    # ── Evaluate ──────────────────────────────────────────────────────────
    from open_vision_banana.evaluation.evaluator import VisionEvaluator

    # Build class_color_map for semantic task
    class_color_map = None
    if args.task == "semantic":
        from open_vision_banana.data.datasets import CityscapesSegDataset
        class_color_map = CityscapesSegDataset.CITYSCAPES_COLORS

    evaluator = VisionEvaluator(model, device=device)
    metrics = evaluator.evaluate(
        dataloader=dataloader,
        task=args.task,
        num_classes=args.num_classes,
        max_batches=args.max_batches,
        class_color_map=class_color_map,
    )

    # ── Results ───────────────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info(f"Evaluation Results ({args.task})")
    logger.info("=" * 50)
    for name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {name}: {value:.4f}")
        elif isinstance(value, list):
            logger.info(f"  {name}: {[f'{v:.4f}' if isinstance(v, float) else v for v in value]}")

    if args.output_json:
        out = {k: (v if not isinstance(v, float) else round(v, 6)) for k, v in metrics.items()}
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Metrics written to {args.output_json}")


if __name__ == "__main__":
    main()

