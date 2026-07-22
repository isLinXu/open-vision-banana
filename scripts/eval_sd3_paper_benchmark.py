#!/usr/bin/env python3
"""SD3-Medium paper benchmark evaluation on 4 tasks × 100 samples.

Tasks:
  - Semantic segmentation (Cityscapes val)
  - Referring segmentation (RefCOCOg UMD val)
  - Reasoning segmentation (ReasonSeg val)
  - Generation preservation (GenAI-Bench)

Usage:
  python scripts/eval_sd3_paper_benchmark.py --lora-path output/sd3_checkpoints/lora_final.pt
  python scripts/eval_sd3_paper_benchmark.py --lora-path output/sd3_checkpoints/lora_final.pt --task semantic --max-batches 10
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.inference.generator import BaseGenerator  # noqa: E402
from open_vision_banana.inference.pipeline import VisionBanana  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_sd3")

EVAL_SAMPLES_DIR = PROJECT_ROOT / "output" / "08662d03-9ca5-4394-b0c8-e019adb06a5d" / "paper_eval_samples"

TASKS = ["semantic", "referring", "reasoning", "generation"]


CITYSCAPES_COLORS = {
    "road": (128, 64, 128),
    "sidewalk": (244, 35, 232),
    "building": (70, 70, 70),
    "wall": (102, 102, 156),
    "fence": (190, 153, 153),
    "pole": (153, 153, 153),
    "traffic_light": (250, 170, 30),
    "traffic_sign": (220, 220, 0),
    "vegetation": (107, 142, 35),
    "terrain": (152, 251, 152),
    "sky": (70, 130, 180),
    "person": (220, 20, 60),
    "rider": (255, 0, 0),
    "car": (0, 0, 142),
    "truck": (0, 0, 70),
    "bus": (0, 60, 100),
    "train": (0, 80, 100),
    "motorcycle": (0, 0, 230),
    "bicycle": (119, 11, 32),
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_eval_data(task: str) -> list[dict]:
    """Load evaluation samples for a task."""
    task_dir = EVAL_SAMPLES_DIR / task
    labels_file = task_dir / "labels.json"
    with open(labels_file) as f:
        labels = json.load(f)

    samples = []
    img_dir = task_dir / "images"
    tgt_dir = task_dir / "targets"
    image_files = sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ) if img_dir.exists() else []
    target_files = sorted(
        p for p in tgt_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ) if tgt_dir.exists() else []

    for position, item in enumerate(labels):
        idx = item["index"]
        # Samples are exported as 0.png, 1.png, ... while labels keep the
        # source dataset index; fall back to positional alignment.
        img_path = next((img_dir / f"{idx}{ext}" for ext in [".jpg", ".png", ".jpeg", ".webp"]
                         if (img_dir / f"{idx}{ext}").exists()), None)
        if img_path is None and position < len(image_files):
            img_path = image_files[position]
        target_path = next((tgt_dir / f"{idx}{ext}" for ext in [".jpg", ".png", ".jpeg", ".webp"]
                            if (tgt_dir / f"{idx}{ext}").exists()), None)
        if target_path is None and position < len(target_files):
            target_path = target_files[position]

        samples.append({
            "index": idx,
            "prompt": item.get("prompt", ""),
            "image_path": str(img_path) if img_path is not None else None,
            "target_path": str(target_path) if target_path else None,
            "raw": item,
        })

    return samples


def compute_metrics(pred: np.ndarray, target: np.ndarray, task: str) -> dict:
    """Compute metrics used by the paper's task tables.

    RGB reconstruction scores are intentionally excluded: they do not measure
    the decoded task output and were previously reported as misleading proxies.
    """
    if pred.shape[:2] != target.shape[:2]:
        # Keep the evaluator robust to older generators that return the
        # pipeline's native canvas instead of the input raster size.
        pred_img = Image.fromarray((pred.clip(0, 1) * 255).astype(np.uint8))
        pred = np.asarray(
            pred_img.resize((target.shape[1], target.shape[0]), Image.Resampling.NEAREST),
            dtype=np.float32,
        ) / 255.0

    pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0)
    target_t = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0)

    if task == "semantic":
        from open_vision_banana.core.segmentation import SegmentationDecoder
        from open_vision_banana.evaluation.semantic import evaluate_semantic

        decoder = SegmentationDecoder()
        labels_pred = decoder.decode_semantic(pred_t, CITYSCAPES_COLORS)
        labels_target = decoder.decode_semantic(target_t, CITYSCAPES_COLORS)
        return evaluate_semantic(labels_pred, labels_target, len(CITYSCAPES_COLORS))

    if task in ("referring", "reasoning"):
        # Evaluation samples use a fixed foreground color per task.
        channel = 0 if task == "referring" else 1
        pred_mask = (pred[:, :, channel] > 0.5) & (
            pred[:, :, channel] > np.delete(pred, channel, axis=2).max(axis=2) + 0.05
        )
        target_mask = target[:, :, channel] > 0.5
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        metric_name = "cIoU" if task == "referring" else "gIoU"
        return {metric_name: float(intersection / max(union, 1))}

    if task == "generation":
        return {}

    raise ValueError(f"Unsupported paper benchmark task: {task}")


def run_evaluation(
    model: VisionBanana,
    task: str,
    samples: list[dict],
    device: torch.device,
    resolution: int = 384,
    max_batches: int | None = None,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
) -> list[dict]:
    """Run evaluation for a single task."""
    results = []
    n = len(samples) if max_batches is None else min(max_batches, len(samples))

    for i, sample in enumerate(samples[:n]):
        t0 = time.time()
        idx = sample["index"]

        # Load input image
        if task != "generation" and sample["image_path"] is None:
            logger.warning(f"  [{task}] Sample {idx}: no image, skipping")
            continue

        img_tensor = None
        if sample["image_path"] is not None:
            img = Image.open(sample["image_path"]).convert("RGB")
            img = img.resize((resolution, resolution), Image.LANCZOS)
            img_tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
            img_tensor = img_tensor * 2 - 1  # [-1, 1]
            img_tensor = img_tensor.unsqueeze(0).to(device)

        # Load target
        target_rgb = None
        if sample["target_path"]:
            tgt = Image.open(sample["target_path"]).convert("RGB")
            tgt = tgt.resize((resolution, resolution), Image.LANCZOS)
            target_rgb = np.array(tgt, dtype=np.float32) / 255.0

        # Run inference
        prompt = sample["prompt"]
        try:
            # Use img2img for perception tasks
            if task != "generation":
                output = model.generator.generate(
                    image=img_tensor,
                    prompt=prompt,
                    strength=0.85,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )
            else:
                # Generation: text-to-image
                output = model.generator.generate(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=resolution,
                    width=resolution,
                )

            # Extract output image
            if isinstance(output, torch.Tensor):
                pred = output.squeeze(0).cpu().float()
            elif isinstance(output, Image.Image):
                pred = torch.from_numpy(np.array(output, dtype=np.float32) / 255.0).permute(2, 0, 1)
            elif isinstance(output, list) and len(output) > 0:
                if isinstance(output[0], torch.Tensor):
                    pred = output[0].squeeze(0).cpu().float()
                elif isinstance(output[0], Image.Image):
                    pred = torch.from_numpy(np.array(output[0], dtype=np.float32) / 255.0).permute(2, 0, 1)
            else:
                logger.warning(f"  [{task}] Sample {idx}: unexpected output type {type(output)}")
                continue

            # Convert to numpy [0, 1]
            pred_np = pred.permute(1, 2, 0).numpy()
            # BaseGenerator returns [0, 1]; accept legacy [-1, 1] tensors too.
            if pred_np.min() < 0.0:
                pred_np = pred_np * 0.5 + 0.5
            pred_np = pred_np.clip(0, 1)

            # Compute metrics
            metrics = {}
            if target_rgb is not None:
                metrics = compute_metrics(pred_np, target_rgb, task)

            elapsed = time.time() - t0
            results.append({
                "index": idx,
                "task": task,
                "metrics": metrics,
                "time": elapsed,
            })

            if (i + 1) % 10 == 0 or i == 0:
                numeric_metrics = {
                    k: v for k, v in metrics.items()
                    if isinstance(v, (int, float, np.integer, np.floating))
                }
                m_str = ", ".join(f"{k}={v:.4f}" for k, v in numeric_metrics.items()) if numeric_metrics else "no target"
                logger.info(f"  [{task}] {i+1}/{n} | {m_str} | {elapsed:.1f}s")

        except Exception as e:
            logger.error(f"  [{task}] Sample {idx}: {e}")
            results.append({"index": idx, "task": task, "error": str(e), "metrics": {}})

    return results


def summarize_results(results: list[dict]) -> dict:
    """Compute summary statistics."""
    valid = [r for r in results if "metrics" in r and r["metrics"] and "error" not in r]
    if not valid:
        return {"n_valid": 0, "n_total": len(results)}

    all_metrics = {}
    for r in valid:
        for k, v in r["metrics"].items():
            if isinstance(v, (int, float)):
                all_metrics.setdefault(k, []).append(v)

    summary = {"n_valid": len(valid), "n_total": len(results)}
    for k, vals in all_metrics.items():
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "median": float(np.median(vals)),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="SD3 paper benchmark evaluation")
    parser.add_argument("--lora-path", type=str, required=True, help="Path to LoRA weights (.pt or dir)")
    parser.add_argument("--model-path", type=str,
                        default=str(Path.home() / "MyWork/models/sd3-medium-diffusers"),
                        help="Model path or HF repo ID (e.g. segmind/SSD-1B)")
    parser.add_argument("--task", type=str, default="all", choices=["all"] + TASKS)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--max-batches", type=int, default=None, help="Limit samples per task")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = get_device()
    logger.info(f"Device: {device}")

    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    t0 = time.time()
    generator = BaseGenerator(
        model_name=args.model_path,
        device=device,
        dtype=torch.bfloat16,
    )
    logger.info(f"Model loaded in {time.time()-t0:.1f}s")

    # Load LoRA. Training checkpoints use PEFT's base_model.model.* keys,
    # which diffusers may accept without attaching them to SD3 transformer.
    # Prefer the explicit merge path and report its coverage.
    lora_path = Path(args.lora_path)
    logger.info(f"Loading LoRA from {lora_path}...")
    if not lora_path.exists():
        logger.error(f"LoRA path not found: {lora_path}")
        sys.exit(1)

    cleanup_dir = None
    adapter_dir = lora_path if lora_path.is_dir() else None
    if lora_path.is_file() and lora_path.suffix == ".pt":
        from safetensors.torch import save_file

        cleanup_dir = tempfile.mkdtemp(prefix="sd3_eval_adapter_")
        adapter_dir = Path(cleanup_dir)
        state = torch.load(str(lora_path), map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise TypeError(f"Expected LoRA state dict, got {type(state).__name__}")
        save_file(state, str(adapter_dir / "adapter_model.safetensors"))
        cfg_src = lora_path.parent / "adapter_config.json"
        if cfg_src.exists():
            shutil.copy2(cfg_src, adapter_dir / "adapter_config.json")
        else:
            # Infer target modules from model family to avoid SD3/SDXL mismatch
            model_short = str(args.model_path).lower()
            if "ssd" in model_short or "sdxl" in model_short:
                _lora_targets = ["to_q", "to_k", "to_v", "to_out.0",
                                 "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2"]
            else:
                _lora_targets = ["to_q", "to_k", "to_v", "to_out.0",
                                 "to_q_context", "to_k_context", "to_v_context",
                                 "to_out_context", "context_embedder", "proj_out",
                                 "linear_1", "linear_2"]
            with open(adapter_dir / "adapter_config.json", "w") as f:
                json.dump({
                    "peft_type": "LORA", "r": 64, "lora_alpha": 128,
                    "target_modules": _lora_targets,
                    "lora_dropout": 0.0, "bias": "none",
                }, f)

    if adapter_dir is None:
        raise ValueError(f"Unsupported LoRA path: {lora_path}")
    merged = generator._load_and_merge_lora(generator._t2i_pipe, str(adapter_dir))
    if not merged:
        raise RuntimeError(f"LoRA merge matched zero layers: {lora_path}")
    logger.info("LoRA merged into model: %d layers", merged)

    model = VisionBanana(base_generator=generator)

    # Output directory
    model_short = str(args.model_path).lower()
    if "ssd" in model_short:
        eval_dir_name = "ssd-1b_eval_results"
    elif "sdxl" in model_short:
        eval_dir_name = "sdxl_eval_results"
    else:
        eval_dir_name = "sd3_eval_results"
    output_dir = Path(args.output_dir) if args.output_dir else \
        PROJECT_ROOT / "output" / eval_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    tasks_to_eval = TASKS if args.task == "all" else [args.task]
    all_results = {}

    for task in tasks_to_eval:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {task}")
        logger.info(f"{'='*60}")

        samples = load_eval_data(task)
        logger.info(f"Loaded {len(samples)} samples for {task}")

        results = run_evaluation(
            model=model,
            task=task,
            samples=samples,
            device=device,
            resolution=args.resolution,
            max_batches=args.max_batches,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )

        summary = summarize_results(results)
        all_results[task] = {"summary": summary, "results": results}

        # Save per-task results
        with open(output_dir / f"{task}_results.json", "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2, default=str)

        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)

        logger.info(f"\n{task} Summary:")
        for k, v in summary.items():
            if isinstance(v, dict):
                logger.info(f"  {k}: mean={v['mean']:.4f} ± {v['std']:.4f} (median={v['median']:.4f})")
            else:
                logger.info(f"  {k}: {v}")

    # Save combined results
    combined_summary = {task: data["summary"] for task, data in all_results.items()}
    with open(output_dir / "eval_summary.json", "w") as f:
        json.dump(combined_summary, f, indent=2, default=str)

    logger.info(f"\n{'='*60}")
    logger.info("All evaluations complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'='*60}")

    # Print comparison table
    logger.info("\nSummary Table:")
    logger.info(f"{'Task':<15} {'Paper metrics':<40}")
    for task, data in all_results.items():
        s = data["summary"]
        paper_metrics = ", ".join(
            f"{key}={value.get('mean', value):.4f}"
            if isinstance(value, dict) and isinstance(value.get("mean", value), (int, float))
            else f"{key}={value}"
            for key, value in s.items()
            if key not in {"n_valid", "n_total"}
        ) or "not computed (no reference target)"
        logger.info(f"{task:<15} {paper_metrics:<40}")


if __name__ == "__main__":
    main()
