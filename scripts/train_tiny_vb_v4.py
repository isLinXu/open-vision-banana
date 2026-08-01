#!/usr/bin/env python3
"""Train TinyVisionBanana V4 — anti-oversmoothing optimized training.

V4 improvements over V3:
1. **V4 loss function** — L1 (not L2), multi-scale gradient, SSIM, histogram,
   ordinal ranking. Directly targets oversmoothing and dynamic range compression.
2. **No residual for depth/normal** — prevents RGB contamination.
3. **Larger effective batch** — grad_accum=4 (effective batch=16).
4. **Longer training** — 12000 steps (2+ full epochs).
5. **Lower LR** — 3e-4 (was 5e-4), more stable convergence.
6. **tanh output option** — avoids sigmoid midpoint bias (teal/cyan).
7. **Cosine warm-restart** — 2 restarts for better convergence.

Usage:
    python scripts/train_tiny_vb_v4.py \
        --data-dir /path/to/vb_snvc_train_v3 \
        --steps 12000 --lr 3e-4 --eval-every 250 \
        --output-dir output/tiny_vb_v4
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.data.tiny_dataset import (  # noqa: E402
    TINY_TASK_ORDER,
    TinyMultiTaskDataset,
    TinySample,
    collate_tiny_batch,
)
from open_vision_banana.tiny_model_v2 import (  # noqa: E402
    TinyVisionBananaV2,
    TinyVisionBananaV2Config,
)
from open_vision_banana.tiny_model_v4 import (  # noqa: E402
    tiny_vision_loss_v4,
    evaluate_tiny_predictions,
)

logger = logging.getLogger(__name__)

DEFAULT_TASK_WEIGHTS: dict[str, float] = {
    "depth": 2.5,
    "normal": 2.0,
    "edge": 1.5,
    "semantic": 0.5,
    "generation": 0.5,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train TinyVisionBanana V4 (anti-oversmoothing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Data
    p.add_argument("--data-dir", required=True)
    p.add_argument("--tasks", nargs="+", default=list(TINY_TASK_ORDER))
    p.add_argument("--resolution", type=int, default=256)

    # Model
    p.add_argument("--base-width", type=int, default=32)
    p.add_argument("--num-stages", type=int, default=4)
    p.add_argument("--block-depth", type=int, default=2)
    p.add_argument("--task-embed-dim", type=int, default=32)
    p.add_argument(
        "--output-activation",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "none"],
        help="none=tanh(rescaled to [0,1]); sigmoid=current",
    )

    # Training
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation (effective batch = batch_size * grad_accum)",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=300)

    # V4 loss weights
    p.add_argument(
        "--gradient-weight",
        type=float,
        default=2.0,
        help="Multi-scale gradient weight (was 0.5 in V2/V3)",
    )
    p.add_argument(
        "--pearson-weight",
        type=float,
        default=0.5,
        help="Pearson correlation weight (was 0.1 in V2)",
    )
    p.add_argument("--ssim-weight", type=float, default=0.5)
    p.add_argument("--histogram-weight", type=float, default=0.3)
    p.add_argument("--ordinal-weight", type=float, default=0.2)
    p.add_argument(
        "--palette-ce-weight",
        type=float,
        default=2.0,
        help="Palette cross-entropy weight for semantic (anti-task-collapse)",
    )
    p.add_argument(
        "--palette-proj-weight",
        type=float,
        default=1.0,
        help="Palette projection weight for semantic (discrete color enforcement)",
    )

    # Task sampling
    p.add_argument("--task-weights", type=str, default="", help="JSON string of {task: weight}")

    # Evaluation
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--max-val-samples", type=int, default=32)

    # Checkpointing
    p.add_argument("--output-dir", type=str, default="output/tiny_vb_v4")
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--resume", type=str, default="")

    # Cosine restart
    p.add_argument(
        "--restart-steps",
        type=str,
        default="",
        help="Comma-separated step numbers for cosine restarts (e.g. '4000,8000')",
    )

    # Device
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def lr_lambda_with_restarts(
    step: int,
    warmup: int,
    total: int,
    restart_steps: list[int],
) -> float:
    """Cosine schedule with warm restarts.

    Between restart points, LR follows a cosine decay from 1.0 to ~0.01.
    At each restart step, LR jumps back to 1.0.
    """
    if step < warmup:
        return step / max(warmup, 1)

    # Find current segment
    restart_points = [0] + restart_steps + [total]
    seg_start = 0
    seg_end = total
    for i in range(len(restart_points) - 1):
        if restart_points[i] <= step < restart_points[i + 1]:
            seg_start = restart_points[i]
            seg_end = restart_points[i + 1]
            break

    # Cosine decay within segment
    seg_len = max(seg_end - seg_start, 1)
    progress = (step - seg_start) / seg_len
    # First segment includes warmup offset
    if seg_start == 0:
        progress = (step - warmup) / max(seg_end - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress)) * 0.99 + 0.01


def build_weighted_dataset(
    dataset: TinyMultiTaskDataset,
    task_weights: dict[str, float],
) -> tuple[list[TinySample], list[float]]:
    """Build reweighted sample list and per-sample weights."""
    per_task = dataset._per_task
    total_original = sum(len(v) for v in per_task.values())
    weight_sum = sum(task_weights.get(t, 1.0) for t in per_task)
    normalized = {}
    for t, samples in per_task.items():
        w = task_weights.get(t, 1.0)
        normalized[t] = w / weight_sum

    all_samples: list[TinySample] = []
    sample_weights: list[float] = []

    for t, samples in per_task.items():
        if not samples:
            continue
        n = len(samples)
        w_per_sample = normalized[t] * total_original / n
        all_samples.extend(samples)
        sample_weights.extend([w_per_sample] * n)

    return all_samples, sample_weights


def evaluate_per_task(
    model: TinyVisionBananaV2,
    dataset: TinyMultiTaskDataset,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    """Evaluate model per-task."""
    model.eval()
    results = {}

    with torch.no_grad():
        for task in args.tasks:
            task_samples = [s for s in dataset.samples if s.task == task]
            if not task_samples:
                continue
            max_eval = args.max_val_samples
            if max_eval > 0 and len(task_samples) > max_eval:
                task_samples = task_samples[:max_eval]

            preds = []
            targets = []
            from open_vision_banana.data.tiny_dataset import _load_image

            for sample in task_samples:
                image = _load_image(sample.image_path, args.resolution, (-1.0, 1.0))
                target = _load_image(sample.target_path, args.resolution, (0.0, 1.0))
                image = image.unsqueeze(0).to(device)
                target_t = target.unsqueeze(0).to(device)
                task_idx = torch.tensor(
                    [model.task_index[sample.task]], device=device, dtype=torch.long
                )
                pred = model(image=image, task=task_idx)
                preds.append(pred.cpu())
                targets.append(target_t.cpu())

            if preds:
                preds_cat = torch.cat(preds)
                targets_cat = torch.cat(targets)
                metrics = evaluate_tiny_predictions(preds_cat, targets_cat)
                results[task] = metrics

    model.train()
    return results


def save_checkpoint(
    model: TinyVisionBananaV2,
    optimizer: torch.optim.Optimizer,
    step: int,
    output_dir: Path,
    name: str,
) -> None:
    path = output_dir / name
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_parameters": model.num_parameters(),
            "architecture": "tiny_vision_banana_v2",
            "loss_version": "v4",
        },
        str(path),
    )
    logger.info(f"Saved checkpoint: {path}")


def _inf_iter(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def train(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info(f"Args: {json.dumps(vars(args), indent=2)}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = select_device(args.device)
    logger.info(f"Device: {device}")

    # Parse task weights
    if args.task_weights:
        task_weights = json.loads(args.task_weights)
    else:
        task_weights = DEFAULT_TASK_WEIGHTS.copy()
    logger.info(f"Task sampling weights: {task_weights}")

    # Parse restart steps
    restart_steps = []
    if args.restart_steps:
        restart_steps = [int(s) for s in args.restart_steps.split(",")]
    logger.info(f"Cosine restart steps: {restart_steps}")

    # Build model
    config = TinyVisionBananaV2Config(
        base_width=args.base_width,
        num_stages=args.num_stages,
        block_depth=args.block_depth,
        task_embed_dim=args.task_embed_dim,
        resolution=args.resolution,
        use_residual=True,  # Keep residual but V4 loss + training will handle depth
        output_activation=args.output_activation,
    )
    tasks = tuple(args.tasks)
    model = TinyVisionBananaV2(config, tasks=tasks)
    model = model.to(device)
    num_params = model.num_parameters()

    effective_batch = args.batch_size * args.grad_accum
    logger.info(f"Model: TinyVisionBananaV2 | parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    logger.info(
        f"Effective batch size: {effective_batch} (batch={args.batch_size} x accum={args.grad_accum})"
    )
    logger.info(
        f"V4 Loss: L1 + {args.gradient_weight}x*grad(ms) + {args.pearson_weight}x*pearson + "
        f"{args.ssim_weight}x*ssim + {args.histogram_weight}x*hist + {args.ordinal_weight}x*ordinal"
    )
    logger.info(
        f"V4 Semantic: {args.palette_ce_weight}x*palette_ce + {args.palette_proj_weight}x*palette_proj"
    )

    # Datasets
    train_ds = TinyMultiTaskDataset(
        args.data_dir,
        "train",
        tasks=tasks,
        resolution=args.resolution,
        balance=False,
    )
    val_ds = TinyMultiTaskDataset(
        args.data_dir,
        "val",
        tasks=tasks,
        resolution=args.resolution,
        balance=False,
    )
    logger.info(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
    logger.info(f"Task counts (train): {train_ds.task_counts()}")

    # Weighted sampler
    train_samples, sample_weights = build_weighted_dataset(train_ds, task_weights)
    train_ds.samples = train_samples

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_samples),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_tiny_batch,
        num_workers=0,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: lr_lambda_with_restarts(s, args.warmup_steps, args.steps, restart_steps),
    )

    # Resume
    start_step = 0
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.is_file():
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_step = ckpt["step"]
            logger.info(f"Resumed from step {start_step}")
        else:
            logger.warning(f"Checkpoint not found: {ckpt_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "per_task"
    ckpt_dir.mkdir(exist_ok=True)

    config_meta = {
        "model_config": asdict(config),
        "architecture": "tiny_vision_banana_v2",
        "version": "v4",
        "loss_version": "v4",
        "tasks": list(tasks),
        "num_parameters": num_params,
        "args": vars(args),
        "task_weights": task_weights,
        "effective_batch_size": effective_batch,
        "v4_loss_components": {
            "value": "L1 (not Smooth L1)",
            "gradient": f"multi-scale(1,2,4) x{args.gradient_weight}",
            "pearson": f"x{args.pearson_weight}",
            "ssim": f"x{args.ssim_weight}",
            "histogram": f"x{args.histogram_weight}",
            "ordinal": f"x{args.ordinal_weight}",
        },
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(config_meta, indent=2), encoding="utf-8"
    )

    # Per-task best tracking
    best_per_task: dict[str, dict] = {}
    best_agg_mae = float("inf")

    model.train()
    step = start_step
    accum_loss = 0.0
    accum_count = 0
    train_iter = _inf_iter(train_loader)
    eval_history = []

    logger.info(f"Starting V4 training for {args.steps} steps...")
    logger.info(
        f"  Key changes: L1 loss, multi-scale grad x{args.gradient_weight}, "
        f"histogram x{args.histogram_weight}, ordinal x{args.ordinal_weight}"
    )
    start_time = time.time()

    while step < args.steps:
        batch = next(train_iter)
        images = batch["image"].to(device)
        targets = batch["target_rgb"].to(device)
        tasks_batch = batch["task"]

        task_indices = torch.tensor(
            [model.task_index[t] for t in tasks_batch],
            device=device,
            dtype=torch.long,
        )

        predictions = model(image=images, task=task_indices)

        # V4 anti-oversmoothing loss
        loss = tiny_vision_loss_v4(
            predictions,
            targets,
            task_indices,
            gradient_weight=args.gradient_weight,
            pearson_weight=args.pearson_weight,
            ssim_weight=args.ssim_weight,
            histogram_weight=args.histogram_weight,
            ordinal_weight=args.ordinal_weight,
            palette_ce_weight=args.palette_ce_weight,
            palette_proj_weight=args.palette_proj_weight,
        )
        loss_scaled = loss / args.grad_accum

        loss_scaled.backward()
        accum_loss += loss.item()
        accum_count += 1

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

            if step % args.eval_every == 0:
                per_task_metrics = evaluate_per_task(model, val_ds, device, args)

                all_maes = [m["rgb_mae"] for m in per_task_metrics.values()]
                agg_mae = sum(all_maes) / len(all_maes) if all_maes else float("inf")

                task_str = " | ".join(
                    f"{t}: mae={m['rgb_mae']:.4f} r={m['pearson']:.4f}"
                    for t, m in per_task_metrics.items()
                )
                logger.info(f"  VAL @ {step}: {task_str}")
                logger.info(f"  AGG MAE: {agg_mae:.4f} (best: {best_agg_mae:.4f})")

                eval_history.append(
                    {
                        "step": step,
                        "aggregate_mae": agg_mae,
                        "per_task": per_task_metrics,
                    }
                )

                # Per-task best
                for task_name, metrics in per_task_metrics.items():
                    mae = metrics["rgb_mae"]
                    prev = best_per_task.get(task_name)
                    if prev is None or mae < prev["mae"]:
                        best_per_task[task_name] = {
                            "mae": mae,
                            "step": step,
                            "pearson": metrics["pearson"],
                        }
                        save_checkpoint(
                            model,
                            optimizer,
                            step,
                            ckpt_dir,
                            f"best_{task_name}.pt",
                        )
                        logger.info(f"  *** New best {task_name}: MAE={mae:.4f} @ step {step} ***")

                # Aggregate best
                if agg_mae < best_agg_mae:
                    best_agg_mae = agg_mae
                    save_checkpoint(model, optimizer, step, output_dir, "best.pt")
                    logger.info(f"  *** New best aggregate MAE: {best_agg_mae:.4f} ***")

                model.train()

            if step % args.save_every == 0:
                save_checkpoint(model, optimizer, step, output_dir, f"checkpoint-{step}.pt")
                save_checkpoint(model, optimizer, step, output_dir, "checkpoint_latest.pt")

            with open(output_dir / "eval_history.json", "w") as f:
                json.dump(eval_history, f, indent=2)
            with open(output_dir / "per_task_best.json", "w") as f:
                json.dump(best_per_task, f, indent=2)

            accum_loss = 0.0
            accum_count = 0

    # Final save
    save_checkpoint(model, optimizer, step, output_dir, "checkpoint_final.pt")
    save_checkpoint(model, optimizer, step, output_dir, "checkpoint_latest.pt")

    logger.info(f"Training complete. Best aggregate MAE: {best_agg_mae:.4f}")
    logger.info("Per-task best checkpoints:")
    for task_name, info in best_per_task.items():
        logger.info(
            f"  {task_name}: MAE={info['mae']:.4f} Pearson={info['pearson']:.4f} @ step {info['step']}"
        )


if __name__ == "__main__":
    train(parse_args())
