#!/usr/bin/env python3
"""Evaluate TinyVisionBanana checkpoints — local GPU version.

Usage:
    # Evaluate a single checkpoint
    PYTHONPATH=src python3 scripts/evaluate_local_gpu.py \\
        --checkpoint output/local_gpu/depth_w32_256/best.pt \\
        --data-dir /path/to/vb_snvc_train_v3 \\
        --base-width 32 --resolution 256

    # Compare multiple checkpoints (bar chart + table)
    PYTHONPATH=src python3 scripts/evaluate_local_gpu.py \\
        --checkpoint-dir output/local_gpu/depth_w32_256 \\
        --data-dir /path/to/vb_snvc_train_v3 \\
        --base-width 32 --resolution 256 \\
        --compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.tiny_model import (
    TinyVisionBanana,
    TinyVisionBananaConfig,
    evaluate_tiny_predictions,
    TINY_TASKS,
)
from open_vision_banana.data.tiny_dataset import TinyMultiTaskDataset, _load_image


def eval_checkpoint(
    ckpt_path: str,
    data_dir: str,
    base_width: int,
    resolution: int,
    tasks: list[str],
    max_val: int = 0,
) -> dict:
    """Evaluate a single checkpoint on val set."""
    cfg = TinyVisionBananaConfig(
        base_width=base_width, num_stages=4, block_depth=2, resolution=resolution
    )
    model = TinyVisionBanana(cfg, tasks=tasks)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_ds = TinyMultiTaskDataset(
        data_dir, "val", tasks=tasks, resolution=resolution, balance=False
    )

    # Per-task evaluation
    results = {}
    all_preds, all_targets = [], []

    for task in tasks:
        task_samples = [s for s in val_ds.samples if s.task == task]
        if max_val > 0:
            task_samples = task_samples[:max_val]

        if not task_samples:
            continue

        preds, targets = [], []
        with torch.no_grad():
            for sample in task_samples:
                img = _load_image(sample.image_path, resolution, (-1.0, 1.0)).unsqueeze(0)
                tgt = _load_image(sample.target_path, resolution, (0.0, 1.0)).unsqueeze(0)
                task_idx = torch.tensor([model.task_index[sample.task]])
                pred = model(img, task_idx)
                preds.append(pred)
                targets.append(tgt)
                del img, tgt, pred

        p = torch.cat(preds)
        t = torch.cat(targets)
        m = evaluate_tiny_predictions(p, t)
        results[task] = m
        all_preds.append(p)
        all_targets.append(t)

    # Overall
    if all_preds:
        all_p = torch.cat(all_preds)
        all_t = torch.cat(all_targets)
        results["overall"] = evaluate_tiny_predictions(all_p, all_t)

    return results


def print_table(results: dict, title: str = ""):
    """Print results in a formatted table."""
    if title:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}")

    print(f"\n{'Task':<15} | {'MAE':>8s} | {'RMSE':>8s} | {'Pearson':>8s}")
    print("-" * 50)
    for task, m in results.items():
        label = task if task == "overall" else task
        print(f"{label:<15} | {m['rgb_mae']:.4f}   | {m['rgb_rmse']:.4f}   | {m['pearson']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate TinyVisionBanana checkpoints")
    parser.add_argument("--checkpoint", type=str, default="", help="Single checkpoint path")
    parser.add_argument("--checkpoint-dir", type=str, default="", help="Directory with checkpoints")
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset path")
    parser.add_argument("--base-width", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--tasks", nargs="+", default=list(TINY_TASKS))
    parser.add_argument("--max-val", type=int, default=50, help="Max val per task (0=all)")
    parser.add_argument("--compare", action="store_true", help="Compare all checkpoints in dir")
    args = parser.parse_args()

    if args.compare and args.checkpoint_dir:
        # Evaluate all checkpoints in directory
        ckpt_dir = Path(args.checkpoint_dir)
        checkpoints = sorted(
            ckpt_dir.glob("checkpoint-*.pt"), key=lambda p: int(p.stem.split("-")[1])
        )

        all_results = {}
        for ckpt_path in checkpoints:
            step = int(ckpt_path.stem.split("-")[1])
            print(f"\nEvaluating step {step}...", flush=True)
            results = eval_checkpoint(
                str(ckpt_path),
                args.data_dir,
                args.base_width,
                args.resolution,
                args.tasks,
                args.max_val,
            )
            all_results[step] = results
            overall = results.get("overall", {})
            print(
                f"  Overall: mae={overall.get('rgb_mae', 0):.4f} pearson={overall.get('pearson', 0):.4f}",
                flush=True,
            )

        # Summary table
        print(f"\n\n{'=' * 70}")
        print(f"  COMPARISON TABLE — {args.checkpoint_dir}")
        print(f"{'=' * 70}")
        print(f"\n{'Step':>6s}", end="")
        for task in args.tasks + ["overall"]:
            print(f" | {task:>10s} MAE", end="")
        print()
        print("-" * (8 + len(args.tasks) * 2 * 14))

        for step, results in sorted(all_results.items()):
            print(f"{step:6d}", end="")
            for task in args.tasks + ["overall"]:
                m = results.get(task, {})
                print(f" | {m.get('rgb_mae', 0):.4f}     ", end="")
            print()

        print(f"\n{'Step':>6s}", end="")
        for task in args.tasks + ["overall"]:
            print(f" | {task:>10s} Cor", end="")
        print()
        print("-" * (8 + len(args.tasks) * 2 * 14))

        for step, results in sorted(all_results.items()):
            print(f"{step:6d}", end="")
            for task in args.tasks + ["overall"]:
                m = results.get(task, {})
                print(f" | {m.get('pearson', 0):.4f}     ", end="")
            print()

    elif args.checkpoint:
        results = eval_checkpoint(
            args.checkpoint,
            args.data_dir,
            args.base_width,
            args.resolution,
            args.tasks,
            args.max_val,
        )
        print_table(results, f"Checkpoint: {args.checkpoint}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
