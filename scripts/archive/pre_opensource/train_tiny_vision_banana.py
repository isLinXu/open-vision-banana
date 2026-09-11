#!/usr/bin/env python3
"""Train TinyVisionBanana — standalone lightweight vision model.

Supports multi-task training on depth/normal/edge/semantic/generation
with config-driven hyperparameters and MPS/CUDA/CPU device selection.

Usage examples:

    # Overfit gate (8 samples, verify model can memorize)
    python scripts/train_tiny_vision_banana.py \\
        --data-dir /path/to/vb_snvc_train_v3 \\
        --tasks depth \\
        --num-samples 8 \\
        --steps 300 \\
        --gate

    # Full multi-task training
    python scripts/train_tiny_vision_banana.py \\
        --data-dir /path/to/vb_snvc_train_v3 \\
        --tasks depth normal edge semantic generation \\
        --steps 3000 \\
        --eval-every 250 \\
        --output-dir output/tiny_vb_multitask_256

    # Resume from checkpoint
    python scripts/train_tiny_vision_banana.py \\
        --data-dir /path/to/vb_snvc_train_v3 \\
        --resume output/tiny_vb_depth_256/checkpoint_latest.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.data.tiny_dataset import (  # noqa: E402
    TINY_TASK_ORDER,
    TinyMultiTaskDataset,
    collate_tiny_batch,
)
from open_vision_banana.tiny_model import (  # noqa: E402
    TinyVisionBanana,
    TinyVisionBananaConfig,
    evaluate_tiny_predictions,
    tiny_overfit_gate_passed,
    tiny_vision_loss,
)

logger = logging.getLogger(__name__)


# ── Argument parsing ─────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train TinyVisionBanana standalone model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    p.add_argument("--data-dir", required=True, help="Path to v3 dataset root")
    p.add_argument("--tasks", nargs="+", default=list(TINY_TASK_ORDER),
                   help="Tasks to train on")
    p.add_argument("--resolution", type=int, default=256,
                   help="Training resolution (default: 256)")
    p.add_argument("--num-samples", type=int, default=0,
                   help="Limit samples per task (0=all, for gate experiments)")

    # Model
    p.add_argument("--base-width", type=int, default=32,
                   help="Base feature width (default: 32)")
    p.add_argument("--num-stages", type=int, default=4,
                   help="Encoder stages (default: 4)")
    p.add_argument("--block-depth", type=int, default=2,
                   help="Conv blocks per stage (default: 2)")
    p.add_argument("--task-embed-dim", type=int, default=32,
                   help="Task embedding dimension (default: 32)")

    # Training
    p.add_argument("--steps", type=int, default=3000, help="Max training steps")
    p.add_argument("--batch-size", type=int, default=4, help="Batch size")
    p.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    p.add_argument("--warmup-steps", type=int, default=100, help="LR warmup steps")
    p.add_argument("--gradient-weight", type=float, default=0.5,
                   help="Spatial gradient loss weight")
    p.add_argument("--pearson-weight", type=float, default=0.1,
                   help="Pearson correlation loss weight")
    p.add_argument("--no-balance", action="store_true",
                   help="Disable task balancing (use raw frequencies)")

    # Evaluation
    p.add_argument("--eval-every", type=int, default=250, help="Evaluate every N steps")
    p.add_argument("--max-val-samples", type=int, default=64,
                   help="Max val samples per task during eval")

    # Checkpointing
    p.add_argument("--output-dir", type=str, default="output/tiny_vb",
                   help="Output directory")
    p.add_argument("--save-every", type=int, default=500, help="Save checkpoint every N steps")
    p.add_argument("--resume", type=str, default="", help="Resume from checkpoint path")

    # Device
    p.add_argument("--device", type=str, default="auto",
                   help="Device: auto, mps, cuda, cpu")

    # Gate mode
    p.add_argument("--gate", action="store_true",
                   help="Run overfit gate (8 samples, stricter thresholds)")

    # Seed
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    return p.parse_args()


# ── Device selection ─────────────────────────────────────────────


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


# ── Learning rate schedule ───────────────────────────────────────


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    # Cosine decay to 1% of peak
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress)) * 0.99 + 0.01


# ── Training ────────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info(f"Args: {json.dumps(vars(args), indent=2)}")

    # Set seed
    torch.manual_seed(args.seed)

    # Device
    device = select_device(args.device)
    logger.info(f"Device: {device}")

    # Build model
    config = TinyVisionBananaConfig(
        base_width=args.base_width,
        num_stages=args.num_stages,
        block_depth=args.block_depth,
        task_embed_dim=args.task_embed_dim,
        resolution=args.resolution,
    )
    tasks = tuple(args.tasks)
    model = TinyVisionBanana(config, tasks=tasks)
    model = model.to(device)

    num_params = model.num_parameters()
    logger.info(f"Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    # Build datasets
    train_ds = TinyMultiTaskDataset(
        args.data_dir, "train",
        tasks=tasks,
        resolution=args.resolution,
        balance=not args.no_balance,
    )
    val_ds = TinyMultiTaskDataset(
        args.data_dir, "val",
        tasks=tasks,
        resolution=args.resolution,
        balance=False,
    )
    logger.info(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
    logger.info(f"Task counts: {train_ds.task_counts()}")

    # Limit samples for gate
    if args.num_samples > 0:
        # Subsample for gate experiment
        gate_samples = []
        for task in tasks:
            task_samples = [s for s in train_ds.samples if s.task == task]
            gate_samples.extend(task_samples[:args.num_samples])
        train_ds.samples = gate_samples
        logger.info(f"Gate mode: limited to {len(train_ds.samples)} samples")

    # DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_tiny_batch,
        num_workers=0,  # MPS safe
        drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_lambda(step, args.warmup_steps, args.steps),
    )

    # Resume
    start_step = 0
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.is_file():
            ckpt = torch.load(str(ckpt_path), map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_step = ckpt["step"]
            logger.info(f"Resumed from step {start_step}")
        else:
            logger.warning(f"Checkpoint not found: {ckpt_path}")

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_meta = {
        "model_config": asdict(config),
        "tasks": list(tasks),
        "num_parameters": num_params,
        "args": vars(args),
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(config_meta, indent=2), encoding="utf-8"
    )

    # Training loop
    model.train()
    step = start_step
    accum_loss = 0.0
    accum_count = 0
    best_val_mae = float("inf")
    train_iter = _inf_iter(train_loader)

    logger.info(f"Starting training for {args.steps} steps...")
    start_time = time.time()

    while step < args.steps:
        batch = next(train_iter)
        images = batch["image"].to(device)
        targets = batch["target_rgb"].to(device)
        tasks_batch = batch["task"]

        # Forward
        # Use first task in batch (simplified: single-task per batch)
        task_name = tasks_batch[0] if isinstance(tasks_batch[0], str) else tasks_batch[0]
        # Convert task names to indices
        task_indices = torch.tensor(
            [model.task_index[t] for t in tasks_batch],
            device=device, dtype=torch.long,
        )

        predictions = model(image=images, task=task_indices)

        loss = tiny_vision_loss(
            predictions, targets,
            gradient_weight=args.gradient_weight,
            pearson_weight=args.pearson_weight,
        )
        loss_scaled = loss / args.grad_accum

        # Backward
        loss_scaled.backward()
        accum_loss += loss.item()
        accum_count += 1

        # Step
        if accum_count >= args.grad_accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step += 1
            if step % 10 == 0:
                elapsed = time.time() - start_time
                avg_loss = accum_loss / accum_count
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Step {step}/{args.steps} | loss={avg_loss:.6f} | lr={lr:.2e} | "
                    f"{elapsed:.1f}s ({step / elapsed:.1f} step/s)"
                )

            # Evaluation
            if step % args.eval_every == 0:
                val_metrics = evaluate(model, val_ds, device, args)
                logger.info(
                    f"  VAL @ step {step}: "
                    f"rgb_mae={val_metrics['rgb_mae']:.4f} "
                    f"pearson={val_metrics['pearson']:.4f}"
                )
                if val_metrics["rgb_mae"] < best_val_mae:
                    best_val_mae = val_metrics["rgb_mae"]
                    save_checkpoint(model, optimizer, step, output_dir, "best.pt")
                    logger.info(f"  New best val MAE: {best_val_mae:.4f}")

                # Gate check
                if args.gate:
                    gate_metrics = evaluate(model, train_ds, device, args, is_train=True)
                    passed = tiny_overfit_gate_passed(gate_metrics)
                    logger.info(
                        f"  GATE @ step {step}: "
                        f"rgb_mae={gate_metrics['rgb_mae']:.4f} "
                        f"pearson={gate_metrics['pearson']:.4f} "
                        f"passed={passed}"
                    )
                    if passed:
                        logger.info("  *** GATE PASSED ***")
                        save_checkpoint(model, optimizer, step, output_dir, "gate_passed.pt")
                        break

                model.train()

            # Save checkpoint
            if step % args.save_every == 0:
                save_checkpoint(model, optimizer, step, output_dir, f"checkpoint-{step}.pt")
                save_checkpoint(model, optimizer, step, output_dir, "checkpoint_latest.pt")

            accum_loss = 0.0
            accum_count = 0

    # Final save
    save_checkpoint(model, optimizer, step, output_dir, "checkpoint_final.pt")
    model.save(str(output_dir / "final_model"))

    elapsed = time.time() - start_time
    logger.info(f"Training complete. {step} steps in {elapsed:.1f}s")
    logger.info(f"Best val MAE: {best_val_mae:.4f}")
    logger.info(f"Output: {output_dir}")


def _inf_iter(loader: DataLoader) -> Iterator:
    """Infinite iterator over a dataloader."""
    while True:
        for batch in loader:
            yield batch


def evaluate(
    model: TinyVisionBanana,
    dataset: TinyMultiTaskDataset,
    device: torch.device,
    args: argparse.Namespace,
    *,
    is_train: bool = False,
) -> dict[str, float]:
    """Evaluate model on a dataset, return aggregate metrics."""
    model.eval()
    all_preds = []
    all_targets = []

    # Subsample for speed
    samples = dataset.samples
    max_eval = args.max_val_samples if not is_train else args.num_samples
    if max_eval > 0 and len(samples) > max_eval * len(args.tasks):
        # Subsample per task
        subsampled = []
        for task in args.tasks:
            task_samples = [s for s in samples if s.task == task]
            subsampled.extend(task_samples[:max_eval])
        samples = subsampled

    with torch.no_grad():
        for sample in samples:
            from open_vision_banana.data.tiny_dataset import _load_image

            image = _load_image(sample.image_path, args.resolution, (-1.0, 1.0))
            target = _load_image(sample.target_path, args.resolution, (0.0, 1.0))

            image = image.unsqueeze(0).to(device)
            target = target.unsqueeze(0).to(device)

            task_idx = torch.tensor(
                [model.task_index[sample.task]], device=device, dtype=torch.long
            )
            pred = model(image=image, task=task_idx)
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

    if not all_preds:
        return {"rgb_mae": 1.0, "rgb_rmse": 1.0, "pearson": 0.0}

    preds = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)
    return evaluate_tiny_predictions(preds, targets_cat)


def save_checkpoint(
    model: TinyVisionBanana,
    optimizer: torch.optim.Optimizer,
    step: int,
    output_dir: Path,
    name: str,
) -> None:
    """Save a training checkpoint."""
    path = output_dir / name
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "num_parameters": model.num_parameters(),
    }, str(path))
    logger.info(f"Saved checkpoint: {path}")


if __name__ == "__main__":
    train(parse_args())
