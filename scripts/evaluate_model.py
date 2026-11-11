#!/usr/bin/env python3
"""
Open-Vision-Banana model evaluation script (evaluate_model.py)

Features:
    - End-to-end evaluation of a trained Vision Banana model on every supported vision task
    - Resume: an interrupted evaluation continues from its checkpoint without repeating inference
    - DDP distributed evaluation (launched with torchrun)
    - Per-batch error handling, optionally skipping a failing batch and continuing
    - Visualization export, SOTA baseline comparison, automatic W&B upload

Usage:
    # single-GPU depth evaluation
    python scripts/evaluate_model.py --task depth --data-path /data/nyu --lora-path output/checkpoint-best

    # multi-GPU distributed evaluation (DDP)
    torchrun --nproc_per_node=4 scripts/evaluate_model.py --task semantic --data-path /data/cityscapes --ddp

    # resume an interrupted evaluation
    python scripts/evaluate_model.py --task normal --data-path /data/scannet --resume-state output/eval_normal_state.json

    # quick smoke test with a limited number of batches
    python scripts/evaluate_model.py --task depth --data-path /data/nyu --max-batches 10 --compare-sota
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler

# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Task constants and mappings
# ------------------------------------------------------------------
SUPPORTED_TASKS: list[str] = [
    "depth",
    "normal",
    "semantic",
    "instance",
    "referring",
    "edge",
    "flow",
    "keypoint",
    "object_detection",
    "depth_edge",
    "generation_preservation",
]

# Mapping from CLI task name to build_dataset registry key
TASK_TO_DATASET_KEY: dict[str, str] = {
    "object_detection": "coco_detection",
    "flow": "sintel",
    "edge": "coco_stuff",
    "keypoint": "coco_detection",
    "depth_edge": "depth",
    "generation_preservation": "depth",
}


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments and return the Namespace."""
    parser = argparse.ArgumentParser(
        description="Evaluate Vision Banana model on vision tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --task depth --data-path ./data/nyu --lora-path ./output/checkpoint-best
  %(prog)s --task semantic --data-path ./data/cityscapes --num-classes 19 --ddp
  %(prog)s --task normal --data-path ./data/scannet --resume-state ./eval_state.json
        """,
    )

    # basic settings
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=SUPPORTED_TASKS,
        help="Evaluation task",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Root directory of the evaluation dataset",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Trained LoRA checkpoint path (directory or hf:repo_id)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="black-forest-labs/FLUX.1-schnell",
        help="Base FLUX.1 model name (default FLUX.1-schnell)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=19,
        help="Number of classes for semantic segmentation (default 19, Cityscapes)",
    )

    # inference hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Inference batch size (default 4)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Input image resolution (default 512)",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=1,
        help="Diffusion denoising steps (default 1, recommended for schnell)",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.85,
        help="img2img noise strength; lower stays closer to the input image (default 0.85)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of batches to evaluate (for smoke tests)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (cuda / mps / cpu / auto)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model weight dtype (default bfloat16)",
    )

    # output settings
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output JSON path for the evaluation metrics",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/evaluation",
        help="Directory for visualizations and the state file (default output/evaluation)",
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save prediction vs ground-truth visualizations",
    )
    parser.add_argument(
        "--viz-freq",
        type=int,
        default=50,
        help="Save a visualization every N batches (default 50)",
    )
    parser.add_argument(
        "--compare-sota",
        action="store_true",
        help="Print a comparison table against SOTA baselines",
    )

    # resume and fault tolerance
    parser.add_argument(
        "--resume-state",
        type=str,
        default=None,
        help="Resume state JSON path; batches already evaluated are skipped if it exists",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Skip a failing batch and continue instead of exiting",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=10,
        help="Maximum number of consecutive errors allowed (default 10)",
    )

    # distributed
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Enable DDP distributed evaluation (launch with torchrun)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader num_workers (default 0; keep 0 on MPS/CPU)",
    )

    # W&B
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log metrics to Weights & Biases automatically",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="open-vision-banana",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (generated automatically by default)",
    )

    return parser.parse_args()


def _resolve_device(device_arg: str | None) -> str:
    """Pick the best device from the arguments and the environment."""
    if device_arg and device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    """Map a dtype string to a torch.dtype."""
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_str]


def _load_state(state_path: str | None) -> dict[str, Any]:
    """Load the resume state; return an empty state if the file does not exist."""
    if state_path is None:
        return {}
    path = Path(state_path)
    if not path.exists():
        logger.info("Resume state file not found, evaluating from the start: %s", path)
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        logger.info(
            "Loaded resume state: %d / %d batches done",
            state.get("completed_batches", 0),
            state.get("total_batches", 0),
        )
        return state
    except Exception as exc:
        logger.warning("Failed to read the resume state (%s), starting from scratch", exc)
        return {}


def _save_state(state: dict[str, Any], state_path: str | None) -> None:
    """Persist the current evaluation state to disk for resume."""
    if state_path is None:
        return
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _advance_dataloader(dataloader: DataLoader, skip_batches: int, device: str) -> None:
    """Skip already evaluated batches by draining the iterator (resume)."""
    if skip_batches <= 0:
        return
    logger.info("Resuming: skipping the first %d evaluated batches …", skip_batches)
    iterator = iter(dataloader)
    skipped = 0
    for _ in range(skip_batches):
        try:
            next(iterator)
            skipped += 1
        except StopIteration:
            logger.warning("DataLoader ended early while skipping (skipped %d / %d)", skipped, skip_batches)
            break
    logger.info("Skipped %d batches", skipped)


def _save_visualization(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    batch_idx: int,
    output_dir: Path,
    task: str,
    sample_idx: int = 0,
) -> None:
    """Save one prediction and its ground truth side by side as a PNG."""
    try:
        import numpy as np
        from PIL import Image

        pred = pred_rgb[sample_idx].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        gt = gt_rgb[sample_idx].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        # concatenate: pred | gt
        canvas = np.concatenate([pred, gt], axis=1)
        canvas = (canvas * 255).astype(np.uint8)
        img = Image.fromarray(canvas)

        viz_dir = output_dir / "visualizations" / task
        viz_dir.mkdir(parents=True, exist_ok=True)
        save_path = viz_dir / f"batch_{batch_idx:06d}_sample_{sample_idx}.png"
        img.save(save_path)
    except Exception as exc:
        logger.warning("Failed to save the visualization (batch %d): %s", batch_idx, exc)


def _init_wandb(args: argparse.Namespace, is_main: bool) -> Any:
    """Initialize the W&B logger (main process only)."""
    if not args.wandb or not is_main:
        return None
    try:
        import wandb

        run_name = args.wandb_run_name or f"eval-{args.task}-{int(time.time())}"
        run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "task": args.task,
                "model_name": args.model_name,
                "lora_path": args.lora_path,
                "batch_size": args.batch_size,
                "resolution": args.resolution,
                "num_inference_steps": args.num_inference_steps,
                "strength": args.strength,
                "max_batches": args.max_batches,
            },
        )
        logger.info("W&B run initialized: %s", run.name)
        return run
    except ImportError:
        logger.warning("wandb is not installed, skipping W&B logging")
        return None
    except Exception as exc:
        logger.warning("W&B initialization failed: %s", exc)
        return None


def _log_to_wandb(wandb_run: Any, metrics: dict[str, float], step: int | None = None) -> None:
    """Upload scalar metrics to W&B."""
    if wandb_run is None:
        return
    try:
        import wandb

        # drop non-scalar values
        scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        wandb.log(scalar_metrics, step=step)
    except Exception as exc:
        logger.debug("W&B log failed (non-fatal): %s", exc)


def _print_metrics(metrics: dict[str, Any], task: str) -> None:
    """Print the formatted metrics table to the terminal."""
    logger.info("=" * 60)
    logger.info("Evaluation Results — Task: %s", task)
    logger.info("=" * 60)
    for name, value in sorted(metrics.items()):
        if isinstance(value, float):
            logger.info("  %-30s : %.4f", name, value)
        elif isinstance(value, list):
            formatted = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in value]
            logger.info("  %-30s : %s", name, formatted)
        elif isinstance(value, dict):
            logger.info("  %-30s :", name)
            for sub_k, sub_v in sorted(value.items()):
                if isinstance(sub_v, float):
                    logger.info("    %-28s : %.4f", sub_k, sub_v)
                else:
                    logger.info("    %-28s : %s", sub_k, sub_v)
        else:
            logger.info("  %-30s : %s", name, value)


def main() -> int:
    """Main entry point; returns the process exit code (0 on success)."""
    args = parse_args()

    # ── DDP setup ─────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_ddp = args.ddp and torch.cuda.is_available()

    if use_ddp:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    else:
        device = _resolve_device(args.device)

    is_main = (not use_ddp) or (local_rank == 0)
    world_size = torch.distributed.get_world_size() if use_ddp else 1

    if is_main:
        logger.info("=" * 60)
        logger.info("Vision Banana — Evaluate Model")
        logger.info("=" * 60)
        logger.info("Task        : %s", args.task)
        logger.info("Model       : %s", args.model_name)
        logger.info("LoRA path   : %s", args.lora_path or "(none — base model)")
        logger.info("Device      : %s", device)
        logger.info("Dtype       : %s", args.dtype)
        logger.info("Batch size  : %d", args.batch_size)
        logger.info("Resolution  : %d", args.resolution)
        logger.info("DDP         : %s", f"enabled (world_size={world_size})" if use_ddp else "disabled")
        logger.info("Resume      : %s", args.resume_state or "no")

    # ── Load the model ────────────────────────────────────────────────
    dtype = _resolve_dtype(args.dtype)

    if args.lora_path:
        if is_main:
            logger.info("Loading VisionBanana (FLUX.1 + LoRA) …")
        from open_vision_banana.inference.pipeline import VisionBanana

        try:
            model = VisionBanana.from_pretrained(
                model_name=args.model_name,
                lora_path=args.lora_path,
                device=device,
                dtype=dtype,
            )
        except Exception as exc:
            logger.error("Model loading failed: %s", exc)
            traceback.print_exc()
            return 1
    else:
        if is_main:
            logger.warning("No --lora-path given, running a zero-shot evaluation of base FLUX.1")
        from open_vision_banana.inference.pipeline import VisionBanana

        try:
            model = VisionBanana.from_pretrained(
                model_name=args.model_name,
                device=device,
                dtype=dtype,
            )
        except Exception as exc:
            logger.error("Model loading failed: %s", exc)
            traceback.print_exc()
            return 1

    # ── Build the Dataset & DataLoader ────────────────────────────────
    from open_vision_banana.data import build_dataset

    dataset_task = TASK_TO_DATASET_KEY.get(args.task, args.task)
    try:
        dataset = build_dataset(
            dataset_task,
            args.data_path,
            split="val",
            resolution=args.resolution,
        )
    except Exception as exc:
        logger.error("Dataset construction failed: %s", exc)
        traceback.print_exc()
        return 1

    if len(dataset) == 0:
        if is_main:
            logger.error("The dataset is empty, check --data-path: %s", args.data_path)
        return 1

    if is_main:
        logger.info("Dataset loaded: %d samples", len(dataset))

    sampler = None
    if use_ddp:
        sampler = DistributedSampler(dataset, shuffle=False)

    num_workers = args.num_workers
    if device in ("mps", "cpu") and num_workers > 0:
        if is_main:
            logger.warning("num_workers>0 may hang on %s, forcing it to 0", device)
        num_workers = 0

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    total_batches = len(dataloader)

    # ── Restore the resume state ──────────────────────────────────────
    state: dict[str, Any] = {}
    state_path = args.resume_state
    if state_path is None:
        # generate the state file path automatically
        state_path = str(Path(args.output_dir) / f"eval_{args.task}_state.json")

    if is_main:
        state = _load_state(state_path)
    if use_ddp:
        # broadcast the state to every rank
        state_list = [state] if is_main else [None]
        torch.distributed.broadcast_object_list(state_list, src=0)
        state = state_list[0] if state_list[0] is not None else {}

    completed_batches: set[int] = set(state.get("completed_batch_indices", []))
    skip_batches = max(completed_batches) + 1 if completed_batches else 0

    if skip_batches > 0:
        _advance_dataloader(dataloader, skip_batches, device)

    # ── Evaluator setup ───────────────────────────────────────────────
    from open_vision_banana.evaluation.evaluator import VisionEvaluator

    class_color_map = None
    if args.task == "semantic":
        from open_vision_banana.data.datasets import CityscapesSegDataset
        class_color_map = CityscapesSegDataset.CITYSCAPES_COLORS

    evaluator = VisionEvaluator(model, device=device)

    # ── W&B setup ─────────────────────────────────────────────────────
    wandb_run = _init_wandb(args, is_main)

    # ── Main evaluation loop ──────────────────────────────────────────
    if is_main:
        logger.info("Starting evaluation, total_batches=%d, skipped %d", total_batches, skip_batches)

    all_metrics: dict[str, Any] = {}
    n_batches_evaluated = 0
    n_errors = 0
    start_time = time.time()

    # live state for resume
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch in enumerate(dataloader, start=skip_batches):
        # max_batches limit
        if args.max_batches is not None and batch_idx >= (skip_batches + args.max_batches):
            if is_main:
                logger.info("Reached the --max-batches limit, stopping early")
            break

        # under DDP keep the ranks in step (a barrier is optional; the iterator synchronizes here)
        if batch_idx in completed_batches:
            continue

        try:
            image = batch["image"].to(device)
            target_rgb = batch["target_rgb"].to(device)

            # build the extra kwargs
            eval_kwargs: dict[str, Any] = {
                "num_inference_steps": args.num_inference_steps,
                "strength": args.strength,
            }
            if class_color_map is not None:
                eval_kwargs["class_color_map"] = class_color_map
            if "expression" in batch:
                eval_kwargs["expression"] = batch["expression"]
            if "object_name" in batch:
                eval_kwargs["object_name"] = batch["object_name"]

            # run evaluate_batch (single-batch mode, finer grained)
            batch_metrics = evaluator.evaluate_batch(
                image=image,
                target_rgb=target_rgb,
                task=args.task,
                **eval_kwargs,
            )

            # accumulate metrics
            for k, v in batch_metrics.items():
                if isinstance(v, (int, float)):
                    all_metrics[k] = all_metrics.get(k, 0.0) + v
                elif isinstance(v, list):
                    if k not in all_metrics:
                        all_metrics[k] = []
                    all_metrics[k].append(v)
                elif isinstance(v, dict):
                    if k not in all_metrics:
                        all_metrics[k] = {}
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, (int, float)):
                            all_metrics[k][sub_k] = all_metrics[k].get(sub_k, 0.0) + sub_v

            n_batches_evaluated += 1
            n_errors = 0  # reset the error count
            completed_batches.add(batch_idx)

            # visualization export
            if args.save_visualizations and is_main and (batch_idx % args.viz_freq == 0):
                # pred_rgb needs another inference (evaluate_batch does not return the rgb)
                # a quick forward with predict, used only for the visualization
                with torch.no_grad():
                    predict_kwargs = {"task": args.task, **eval_kwargs}
                    # drop the keys evaluate_batch does not accept
                    predict_kwargs.pop("class_color_map", None)
                    if args.task == "semantic" and class_color_map is not None:
                        predict_kwargs["class_color_map"] = class_color_map
                    result = model.predict(image=image, **predict_kwargs)
                    pred_rgb = result.get("rgb_output", image)
                _save_visualization(pred_rgb, target_rgb, batch_idx, output_dir, args.task, sample_idx=0)

            # save the state every 10 batches (resume granularity)
            if is_main and (batch_idx % 10 == 0):
                state["completed_batch_indices"] = sorted(completed_batches)
                state["total_batches"] = total_batches
                state["task"] = args.task
                state["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                _save_state(state, state_path)

            # progress output
            if is_main and (batch_idx % 10 == 0 or batch_idx == total_batches - 1):
                elapsed = time.time() - start_time
                throughput = (batch_idx - skip_batches + 1) / (elapsed + 1e-6)
                logger.info(
                    "Progress: batch %d/%d | throughput %.2f batch/s | elapsed %.1fs",
                    batch_idx,
                    total_batches,
                    throughput,
                    elapsed,
                )

        except Exception as exc:
            n_errors += 1
            logger.error("Batch %d evaluation failed (%d/%d errors): %s", batch_idx, n_errors, args.max_errors, exc)
            if not args.continue_on_error:
                traceback.print_exc()
                if is_main:
                    logger.error("Use --continue-on-error to skip failing batches and continue")
                return 1
            if n_errors >= args.max_errors:
                logger.error("Consecutive errors reached the limit %d, stopping evaluation", args.max_errors)
                return 1
            continue

    # ── Normalize the metrics ─────────────────────────────────────────
    if n_batches_evaluated == 0:
        if is_main:
            logger.error("No batch was evaluated successfully")
        return 1

    final_metrics: dict[str, Any] = {}
    for k, v in all_metrics.items():
        if isinstance(v, float):
            final_metrics[k] = v / n_batches_evaluated
        elif isinstance(v, list) and v and isinstance(v[0], list):
            # list of lists → per-element mean
            elem_len = len(v[0])
            result = []
            import math

            for i in range(elem_len):
                vals = [
                    batch[i]
                    for batch in v
                    if not (isinstance(batch[i], float) and math.isnan(batch[i]))
                ]
                result.append(sum(vals) / len(vals) if vals else float("nan"))
            final_metrics[k] = result
        elif isinstance(v, dict):
            final_metrics[k] = {}
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, float):
                    final_metrics[k][sub_k] = sub_v / n_batches_evaluated
                else:
                    final_metrics[k][sub_k] = sub_v
        else:
            final_metrics[k] = v

    # ── DDP aggregation ───────────────────────────────────────────────
    if use_ddp:
        final_metrics = VisionEvaluator.reduce_metrics(final_metrics, device=device)
        torch.distributed.barrier()

    # non-main processes exit here
    if not is_main:
        if use_ddp:
            torch.distributed.destroy_process_group()
        return 0

    # ── Output the results ────────────────────────────────────────────
    total_time = time.time() - start_time
    final_metrics["eval_time_seconds"] = round(total_time, 2)
    final_metrics["batches_evaluated"] = n_batches_evaluated
    final_metrics["samples_evaluated"] = n_batches_evaluated * args.batch_size
    final_metrics["world_size"] = world_size

    _print_metrics(final_metrics, args.task)

    # JSON output
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # round floats for readability
        json_safe = {}
        for k, v in final_metrics.items():
            if isinstance(v, float):
                json_safe[k] = round(v, 6)
            elif isinstance(v, list):
                json_safe[k] = [round(x, 6) if isinstance(x, float) else x for x in v]
            else:
                json_safe[k] = v
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(json_safe, f, indent=2, ensure_ascii=False)
        logger.info("Metrics saved to: %s", out_path)

    # compare with SOTA
    if args.compare_sota:
        from open_vision_banana.evaluation.baselines import format_sota_comparison

        print()  # blank separator line
        print(format_sota_comparison(args.task, final_metrics))

    # W&B upload
    _log_to_wandb(wandb_run, final_metrics)
    if wandb_run is not None:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass

    # clean up the resume state (after a successful evaluation)
    if state_path and Path(state_path).exists():
        # it could be kept or deleted; here it is renamed to .completed as an archive
        completed_path = str(Path(state_path).with_suffix(".completed.json"))
        Path(state_path).rename(completed_path)
        logger.info("Resume state archived: %s", completed_path)

    if use_ddp:
        torch.distributed.destroy_process_group()

    logger.info("Evaluation done, total time %.2f s", total_time)
    return 0


if __name__ == "__main__":
    sys.exit(main())
