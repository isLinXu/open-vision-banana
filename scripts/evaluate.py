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
import os

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
        choices=[
        "depth", "normal", "semantic", "instance", "referring",
        "edge", "flow", "keypoint", "object_detection", "depth_edge",
        "generation_preservation",
    ],
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
    parser.add_argument(
        "--compare-sota", action="store_true",
        help="Print comparison table against published SOTA baselines",
    )
    parser.add_argument(
        "--ddp", action="store_true",
        help="Distributed evaluation (launch with torchrun)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_ddp = args.ddp and torch.cuda.is_available()

    if use_ddp:
        torch.distributed.init_process_group(backend="nccl")
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    elif args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    is_main = (not use_ddp) or local_rank == 0
    if is_main:
        logger.info(f"Task: {args.task} | Device: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────
    from open_vision_banana.data import build_dataset

    # Map CLI task names to build_dataset registry keys
    dataset_task = args.task
    if dataset_task == "object_detection":
        dataset_task = "coco_detection"
    elif dataset_task == "flow":
        dataset_task = "sintel"
    elif dataset_task == "edge":
        dataset_task = "coco_stuff"
    elif dataset_task == "keypoint":
        dataset_task = "coco_detection"
    elif dataset_task == "depth_edge" or dataset_task == "generation_preservation":
        dataset_task = "depth"

    dataset = build_dataset(
        dataset_task, args.data_path, split="val", resolution=args.resolution
    )
    if len(dataset) == 0:
        if is_main:
            logger.error("Dataset is empty — check --data-path")
        return

    sampler = None
    if use_ddp:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=False)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
    )
    if is_main:
        logger.info(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches")

    # ── Model ─────────────────────────────────────────────────────────────
    if args.lora_path:
        if is_main:
            logger.info(f"Loading FLUX.1 + LoRA from {args.lora_path} …")
        from open_vision_banana.inference.pipeline import VisionBanana
        model = VisionBanana.from_pretrained(
            model_name=args.model_name, lora_path=args.lora_path, device=device
        )
    else:
        if is_main:
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

    if use_ddp:
        metrics = VisionEvaluator.reduce_metrics(metrics, device=device)
        torch.distributed.barrier()

    if not is_main:
        if use_ddp:
            torch.distributed.destroy_process_group()
        return

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

    if args.compare_sota:
        from open_vision_banana.evaluation.baselines import format_sota_comparison
        print()
        print(format_sota_comparison(args.task, metrics))

    if use_ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

