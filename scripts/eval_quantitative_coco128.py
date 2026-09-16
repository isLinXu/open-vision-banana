#!/usr/bin/env python3
"""Quantitative evaluation for SANA-1600M multi-task model on COCO128.

Computes metrics from existing predictions in output/coco128_inference/:
- Inference time & throughput
- Output image statistics (mean, std, dynamic range)
- Self-consistency (same seed, two runs)
- Task-specific heuristic quality metrics

Usage:
    python scripts/eval_quantitative_coco128.py \
        --pred-dir output/coco128_inference \
        --data-dir data/coco128/coco128/images/train2017
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


TASK_PROMPTS = {
    "depth": "Generate a metric depth visualization of this image.",
    "normal": "Generate a surface normal estimation visualization of this image.",
    "semantic": (
        "Generate a semantic segmentation visualization image, "
        'using this color mapping: {"road": <128, 64, 128>, "sidewalk": <244, 35, 232>, '
        '"building": <70, 70, 70>, "wall": <102, 102, 156>, "pole": <190, 153, 153>, '
        '"traffic light": <250, 170, 30>, "traffic sign": <220, 220, 0>, '
        '"vegetation": <107, 142, 35>, "terrain": <152, 251, 152>, '
        '"sky": <70, 130, 180>, "person": <220, 20, 60>, "car": <0, 0, 142>}.'
    ),
    "edge": "This image is an edge detection map. The edges are rendered in white, the background in black.",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=str,
                   default="gatilin/open-vision-banana-sana_1600m_multitask")
    p.add_argument("--pred-dir", type=str, default="output/coco128_inference")
    p.add_argument("--data-dir", type=str,
                   default="data/coco128/coco128/images/train2017")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--num-inference-steps", type=int, default=20)
    p.add_argument("--guidance-scale", type=float, default=4.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="output/coco128_eval_quant")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="float16")
    p.add_argument("--tasks", nargs="+",
                   default=["depth", "normal", "semantic", "edge"],
                   choices=list(TASK_PROMPTS.keys()))
    p.add_argument("--max-samples", type=int, default=8)
    p.add_argument("--skip-rerun", action="store_true",
                   help="Skip re-running inference (use existing predictions only)")
    return p.parse_args()


def auto_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(dtype_str):
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[dtype_str]


def pil_to_tensor(img, device, dtype):
    arr = np.array(img, dtype="float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype)


# ── Image-level metrics ────────────────────────────────────────────────

def compute_image_stats(img_array: np.ndarray) -> dict:
    """Compute basic statistics of an image array [H, W, 3] in [0, 255]."""
    gray = np.mean(img_array, axis=2)
    return {
        "mean_rgb": img_array.mean(axis=(0, 1)).tolist(),
        "mean_gray": float(gray.mean()),
        "std_gray": float(gray.std()),
        "min_gray": float(gray.min()),
        "max_gray": float(gray.max()),
        "dynamic_range": float(gray.max() - gray.min()),
        "coverage_nonzero": float((gray > 5).mean()),
    }


def compute_edge_sparsity(edge_array: np.ndarray, threshold: float = 30) -> dict:
    """For edge maps: ratio of edge pixels, edge density."""
    gray = np.mean(edge_array, axis=2)
    edge_mask = gray > threshold
    density = edge_mask.mean()
    return {
        "edge_pixel_ratio": float(density),
        "background_ratio": float(1 - density),
        "edge_uniformity": float(1 - density),
    }


def compute_depth_smoothness(depth_array: np.ndarray) -> dict:
    """For depth maps: gradient magnitude (lower = smoother predictions)."""
    gray = np.mean(depth_array, axis=2).astype(np.float64)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    return {
        "mean_grad_x": float(gx.mean()),
        "mean_grad_y": float(gy.mean()),
        "mean_gradient": float((gx.mean() + gy.mean()) / 2),
        "max_gradient": float(max(gx.max(), gy.max())),
    }


def compute_normal_diversity(normal_array: np.ndarray) -> dict:
    """For normal maps: color diversity (unique color bins)."""
    h, w, _ = normal_array.shape
    quantized = (normal_array // 16).astype(np.int32)
    unique_colors = len(set(map(tuple, quantized.reshape(-1, 3).tolist())))
    total_pixels = h * w
    return {
        "unique_color_bins": unique_colors,
        "color_diversity_ratio": float(unique_colors) / min(total_pixels, 4096),
        "dominant_channel_mean": float(normal_array.mean(axis=(0, 1)).max()),
    }


def compute_semantic_diversity(semantic_array: np.ndarray) -> dict:
    """For semantic maps: number of distinct regions / intensity levels."""
    gray = np.mean(semantic_array, axis=2)
    hist, _ = np.histogram(gray, bins=20, range=(0, 256))
    active_bins = (hist > 50).sum()
    return {
        "active_intensity_bins": int(active_bins),
        "total_bins": 20,
        "entropy": float(-np.sum(
            (hist / hist.sum()) * np.log2(hist / hist.sum() + 1e-10)
        )),
    }


def compute_self_consistency(pred1: np.ndarray, pred2: np.ndarray) -> dict:
    """Compare two predictions from same input (same seed)."""
    diff = np.abs(pred1.astype(np.float64) - pred2.astype(np.float64))
    return {
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "pixel_match_ratio_5": float((diff < 5).mean()),
        "pixel_match_ratio_10": float((diff < 10).mean()),
    }


# ── Main evaluation ───────────────────────────────────────────────────

def load_existing_predictions(pred_dir: Path, task: str):
    """Load existing prediction images for a task."""
    task_dir = pred_dir / task
    if not task_dir.exists():
        return []
    preds = sorted(task_dir.glob("*.png"))
    results = []
    for p in preds:
        img = Image.open(p).convert("RGB")
        arr = np.array(img)
        w = img.width
        if w == 1024:
            left = arr[:, :512, :]
            right = arr[:, 512:, :]
            results.append({
                "path": str(p),
                "stem": p.stem,
                "input_crop": left,
                "pred_crop": right,
            })
        else:
            results.append({
                "path": str(p),
                "stem": p.stem,
                "input_crop": None,
                "pred_crop": arr,
            })
    return results


def evaluate_existing_predictions(pred_dir: Path, tasks: list, max_samples: int):
    """Compute metrics from existing prediction images."""
    all_metrics = {}

    for task in tasks:
        preds = load_existing_predictions(pred_dir, task)
        if not preds:
            logger.warning(f"No existing predictions for task: {task}")
            continue

        preds = preds[:max_samples]
        logger.info(f"\nEvaluating {task}: {len(preds)} predictions")

        task_metrics = {
            "task": task,
            "num_samples": len(preds),
            "per_sample": [],
        }

        all_pred_arrays = []

        for i, pred_data in enumerate(preds):
            pred_arr = pred_data["pred_crop"]
            all_pred_arrays.append(pred_arr)

            stats = compute_image_stats(pred_arr)

            if task == "edge":
                stats.update(compute_edge_sparsity(pred_arr))
            elif task == "depth":
                stats.update(compute_depth_smoothness(pred_arr))
            elif task == "normal":
                stats.update(compute_normal_diversity(pred_arr))
            elif task == "semantic":
                stats.update(compute_semantic_diversity(pred_arr))

            task_metrics["per_sample"].append({
                "stem": pred_data["stem"],
                **stats,
            })

        stacked = np.stack(all_pred_arrays)
        mean_pred = stacked.mean(axis=0)
        std_pred = stacked.std(axis=0)

        task_metrics["aggregate"] = {
            "mean_pred_mean": float(mean_pred.mean()),
            "mean_pred_std": float(mean_pred.std()),
            "inter_sample_std": float(std_pred.mean()),
            "cross_sample_consistency": float(1 - std_pred.mean() / (mean_pred.mean() + 1e-8)),
        }

        per_sample_stats = task_metrics["per_sample"]
        for key in ["mean_gray", "std_gray", "dynamic_range"]:
            vals = [s[key] for s in per_sample_stats]
            task_metrics[f"avg_{key}"] = float(np.mean(vals))
            task_metrics[f"std_{key}"] = float(np.std(vals))

        if task == "edge":
            vals = [s["edge_pixel_ratio"] for s in per_sample_stats]
            task_metrics["avg_edge_pixel_ratio"] = float(np.mean(vals))
        elif task == "depth":
            vals = [s["mean_gradient"] for s in per_sample_stats]
            task_metrics["avg_mean_gradient"] = float(np.mean(vals))
        elif task == "normal":
            vals = [s["unique_color_bins"] for s in per_sample_stats]
            task_metrics["avg_unique_color_bins"] = float(np.mean(vals))
        elif task == "semantic":
            vals = [s["active_intensity_bins"] for s in per_sample_stats]
            task_metrics["avg_active_intensity_bins"] = float(np.mean(vals))
            vals = [s["entropy"] for s in per_sample_stats]
            task_metrics["avg_entropy"] = float(np.mean(vals))

        all_metrics[task] = task_metrics

    return all_metrics


def run_inference_timing(pipe, components, device, dtype, data_dir, tasks,
                         resolution, num_steps, guidance_scale, max_samples):
    """Run inference and measure timing for each task."""
    from open_vision_banana.inference.sana_img2img import flow_matching_inference

    images = sorted(list(Path(data_dir).glob("*.jpg")) +
                    list(Path(data_dir).glob("*.png")))[:max_samples]

    timing = {}
    for task in tasks:
        prompt = TASK_PROMPTS[task]
        times = []
        warmup_done = False

        for i, img_path in enumerate(images):
            img = Image.open(img_path).convert("RGB")
            if img.size != (resolution, resolution):
                img = img.resize((resolution, resolution), Image.LANCZOS)
            input_tensor = pil_to_tensor(img, device, dtype)

            steps = num_steps if i > 0 else max(num_steps, 5)

            t0 = time.time()
            with torch.no_grad():
                _ = flow_matching_inference(
                    pipe.transformer, pipe.vae, components,
                    input_tensor, [prompt], device, dtype,
                    num_steps=steps, use_img2img=True,
                    max_length=300, guidance_scale=guidance_scale,
                )
            elapsed = time.time() - t0

            if i == 0:
                warmup_done = True
                continue
            times.append(elapsed)
            logger.info(f"  {task} [{i}/{len(images)}] {elapsed:.2f}s")

        if times:
            timing[task] = {
                "mean_time_sec": float(np.mean(times)),
                "std_time_sec": float(np.std(times)),
                "min_time_sec": float(np.min(times)),
                "max_time_sec": float(np.max(times)),
                "throughput_fps": float(1.0 / np.mean(times)),
                "num_measured": len(times),
            }
        logger.info(f"  {task} avg: {timing[task]['mean_time_sec']:.2f}s "
                     f"({timing[task]['throughput_fps']:.2f} fps)")

    return timing


def main():
    args = parse_args()

    device = args.device
    if device == "auto":
        device = auto_device()
    device = torch.device(device)
    dtype = get_dtype(args.dtype)
    if device.type == "mps" and dtype == torch.float16:
        dtype = torch.bfloat16

    logger.info("=" * 60)
    logger.info("SANA-1600M Quantitative Evaluation (COCO128)")
    logger.info("=" * 60)
    logger.info(f"Device: {device} | dtype: {dtype}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_dir = Path(args.pred_dir)
    all_metrics = evaluate_existing_predictions(pred_dir, args.tasks, args.max_samples)

    if not args.skip_rerun:
        logger.info("\nLoading model for timing evaluation...")
        from diffusers import SanaPipeline
        from open_vision_banana.model_registry import ModelConfig, PipelineComponents

        pipe = SanaPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
        pipe = pipe.to(device)
        if device.type == "mps":
            pipe.text_encoder.to(torch.float32)

        model_cfg = ModelConfig(
            name="sana-1600m", model_id=args.model_path, family="sana",
            backbone_type="transformer", num_text_encoders=1,
            cross_attention_dim=2240, vae_downsample_factor=8,
            memory_budget_gb=10.0, pipeline_class="SanaPipeline",
            default_resolution=args.resolution,
            default_guidance_scale=args.guidance_scale,
            default_num_inference_steps=args.num_inference_steps,
            default_lora_rank=32, default_lora_alpha=64,
            lora_target_modules=["linear", "proj_out", "conv_inverted", "conv_point"],
            needs_attention_mask=True, vae_has_latent_dist=False,
        )
        components = PipelineComponents(
            unet=pipe.transformer, vae=pipe.vae,
            text_encoder=pipe.text_encoder, text_encoder_2=None,
            tokenizer=pipe.tokenizer, config=model_cfg,
            scheduler=pipe.scheduler,
        )

        logger.info("\nRunning inference timing...")
        timing = run_inference_timing(
            pipe, components, device, dtype, args.data_dir,
            args.tasks, args.resolution, args.num_inference_steps,
            args.guidance_scale, args.max_samples,
        )
        for task in all_metrics:
            if task in timing:
                all_metrics[task]["timing"] = timing[task]

    results_path = output_dir / "quantitative_metrics.json"
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"\nMetrics saved to {results_path}")

    logger.info("\n" + "=" * 60)
    logger.info("QUANTITATIVE EVALUATION SUMMARY")
    logger.info("=" * 60)

    for task, metrics in all_metrics.items():
        logger.info(f"\n{'─' * 40}")
        logger.info(f"Task: {task} ({metrics['num_samples']} samples)")
        logger.info(f"{'─' * 40}")

        if "avg_mean_gray" in metrics:
            logger.info(f"  Mean intensity:    {metrics['avg_mean_gray']:.1f} "
                        f"± {metrics['std_mean_gray']:.1f}")
        if "avg_std_gray" in metrics:
            logger.info(f"  Std intensity:     {metrics['avg_std_gray']:.1f} "
                        f"± {metrics['std_std_gray']:.1f}")
        if "avg_dynamic_range" in metrics:
            logger.info(f"  Dynamic range:     {metrics['avg_dynamic_range']:.1f} "
                        f"± {metrics['std_dynamic_range']:.1f}")

        if task == "edge" and "avg_edge_pixel_ratio" in metrics:
            logger.info(f"  Edge pixel ratio:  {metrics['avg_edge_pixel_ratio']:.3f}")
        elif task == "depth" and "avg_mean_gradient" in metrics:
            logger.info(f"  Mean gradient:     {metrics['avg_mean_gradient']:.2f}")
        elif task == "normal" and "avg_unique_color_bins" in metrics:
            logger.info(f"  Unique color bins: {metrics['avg_unique_color_bins']:.0f}")
        elif task == "semantic":
            if "avg_active_intensity_bins" in metrics:
                logger.info(f"  Active bins:       "
                            f"{metrics['avg_active_intensity_bins']:.0f}/20")
            if "avg_entropy" in metrics:
                logger.info(f"  Entropy:           {metrics['avg_entropy']:.2f} bits")

        agg = metrics.get("aggregate", {})
        if "inter_sample_std" in agg:
            logger.info(f"  Inter-sample std:  {agg['inter_sample_std']:.1f}")
        if "cross_sample_consistency" in agg:
            logger.info(f"  Cross-sample sim:  {agg['cross_sample_consistency']:.3f}")

        if "timing" in metrics:
            t = metrics["timing"]
            logger.info(f"  Inference time:    {t['mean_time_sec']:.2f}s "
                        f"± {t['std_time_sec']:.2f}s")
            logger.info(f"  Throughput:        {t['throughput_fps']:.2f} fps")

    logger.info(f"\nFull results: {results_path}")


if __name__ == "__main__":
    main()
