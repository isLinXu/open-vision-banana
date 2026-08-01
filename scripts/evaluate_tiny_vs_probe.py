#!/usr/bin/env python3
"""Evaluate TinyVisionBanana against frozen-feature probe baseline.

Compares the tiny standalone model with the SSD-1B frozen-feature depth
probe on the v3 official validation split, producing a structured report.

Usage:

    python scripts/evaluate_tiny_vs_probe.py \\
        --tiny-checkpoint output/tiny_vb_depth_256/final_model \\
        --data-dir /path/to/vb_snvc_train_v3 \\
        --tasks depth \\
        --output output/tiny_vs_probe_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.data.tiny_dataset import _load_image  # noqa: E402
from open_vision_banana.tiny_model import (  # noqa: E402
    TinyVisionBanana,
    evaluate_tiny_predictions,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate TinyVisionBanana")
    p.add_argument("--tiny-checkpoint", required=True,
                   help="Path to tiny model directory (with config.json + model.safetensors)")
    p.add_argument("--data-dir", required=True, help="Path to v3 dataset")
    p.add_argument("--tasks", nargs="+", default=["depth"],
                   help="Tasks to evaluate")
    p.add_argument("--split", default="val", choices=["train", "val"],
                   help="Data split to evaluate on")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max-samples", type=int, default=0,
                   help="Max samples per task (0=all)")
    p.add_argument("--output", default="", help="Output JSON path")
    p.add_argument("--device", default="auto", help="Device: auto, mps, cpu")
    return p.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def evaluate_task(
    model: TinyVisionBanana,
    data_dir: str,
    task: str,
    split: str,
    resolution: int,
    device: torch.device,
    max_samples: int,
) -> dict[str, float]:
    """Evaluate the tiny model on a single task."""
    from open_vision_banana.data.tiny_dataset import DIR_TO_TASK, TinyMultiTaskDataset

    model.eval()
    ds = TinyMultiTaskDataset(
        data_dir, split,
        tasks=[task],
        resolution=resolution,
        balance=False,
    )

    samples = ds.samples
    if max_samples > 0 and len(samples) > max_samples:
        samples = samples[:max_samples]

    if not samples:
        logger.warning(f"No samples for task {task!r} in split {split!r}")
        return {"rgb_mae": 1.0, "rgb_rmse": 1.0, "pearson": 0.0, "count": 0}

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for sample in samples:
            image = _load_image(sample.image_path, resolution, (-1.0, 1.0))
            target = _load_image(sample.target_path, resolution, (0.0, 1.0))
            image = image.unsqueeze(0).to(device)
            target = target.unsqueeze(0).to(device)

            task_idx = torch.tensor(
                [model.task_index[task]], device=device, dtype=torch.long
            )
            pred = model(image=image, task=task_idx)
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    metrics = evaluate_tiny_predictions(preds, targets)
    metrics["count"] = len(samples)
    return metrics


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    device = select_device(args.device)
    logger.info(f"Device: {device}")

    # Load model
    model = TinyVisionBanana.load(args.tiny_checkpoint, device=str(device))
    logger.info(f"Loaded model with {model.num_parameters():,} params")
    logger.info(f"Tasks: {model.task_names}")

    # Evaluate each task
    results: dict[str, dict[str, float]] = {}
    for task in args.tasks:
        logger.info(f"Evaluating task: {task}")
        metrics = evaluate_task(
            model, args.data_dir, task, args.split,
            args.resolution, device, args.max_samples,
        )
        results[task] = metrics
        logger.info(
            f"  {task}: rgb_mae={metrics['rgb_mae']:.4f} "
            f"rgb_rmse={metrics['rgb_rmse']:.4f} "
            f"pearson={metrics['pearson']:.4f} "
            f"(n={metrics['count']})"
        )

    # Summary
    report = {
        "model": "tiny_vision_banana",
        "checkpoint": args.tiny_checkpoint,
        "data_dir": args.data_dir,
        "split": args.split,
        "resolution": args.resolution,
        "num_parameters": model.num_parameters(),
        "results": results,
        # Baseline reference (from production refit experiment)
        "frozen_probe_baseline": {
            "depth": {
                "rgb_mae": 0.1296,
                "pearson": 0.7715,
                "spearman": 0.7565,
                "note": "SSD-1B frozen-feature probe @ 512px, 3-seed mean, 115 val",
            }
        },
    }

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Task':<12} {'RGB MAE':<12} {'Pearson':<12} {'Count':<8} {'vs Probe':<12}")
    print("-" * 70)
    for task, m in results.items():
        probe_mae = report["frozen_probe_baseline"].get(task, {}).get("rgb_mae")
        comparison = ""
        if probe_mae is not None:
            delta = m["rgb_mae"] - probe_mae
            comparison = f"{'+' if delta >= 0 else ''}{delta:.4f}"
        print(f"{task:<12} {m['rgb_mae']:<12.4f} {m['pearson']:<12.4f} {m['count']:<8} {comparison:<12}")
    print("=" * 70)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info(f"Report saved to: {out_path}")


if __name__ == "__main__":
    main()
