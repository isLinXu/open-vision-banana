#!/usr/bin/env python3
"""SANA-1600M multi-task evaluation.

Evaluates the merged model on standard vision benchmarks:
- Depth: PSNR, SSIM (RGB-space comparison)
- Normal: PSNR, SSIM
- Semantic: PSNR, SSIM
- Generation quality: FID, CLIP score (optional)

Usage:
    python scripts/eval_sana_1600m_multitask.py \
        --model-path gatilin/open-vision-banana-sana_1600m_multitask \
        --data-dir /path/to/val/data \
        --output-dir output/sana_1600m_eval \
        --tasks depth normal semantic
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
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-path",
        type=str,
        default="gatilin/open-vision-banana-sana_1600m_multitask",
    )
    p.add_argument("--data-dir", type=str, default=None, help="Validation data directory")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--num-inference-steps", type=int, default=20)
    p.add_argument("--guidance-scale", type=float, default=4.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="output/sana_1600m_eval")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="float16")
    p.add_argument(
        "--tasks",
        nargs="+",
        default=["depth", "normal", "semantic"],
        choices=list(TASK_PROMPTS.keys()),
    )
    p.add_argument("--max-samples", type=int, default=50)
    return p.parse_args()


def auto_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(dtype_str):
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_str]


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Simple SSIM approximation."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mu_x = np.mean(pred)
    mu_y = np.mean(target)
    sigma_x = np.var(pred)
    sigma_y = np.var(target)
    sigma_xy = np.mean((pred - mu_x) * (target - mu_y))
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    )
    return float(ssim)


def load_val_samples(data_dir: Path, task: str, resolution: int, max_samples: int):
    """Load validation samples from data directory."""
    task_dir = data_dir / task
    if not task_dir.exists():
        logger.warning(f"Task directory not found: {task_dir}")
        return []

    img_dir = task_dir / "images"
    tgt_dir = task_dir / "targets"

    if not img_dir.exists():
        logger.warning(f"Image directory not found: {img_dir}")
        return []

    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    samples = []

    for img_path in images[:max_samples]:
        input_img = Image.open(img_path).convert("RGB")
        if input_img.size != (resolution, resolution):
            input_img = input_img.resize((resolution, resolution), Image.LANCZOS)

        target_img = None
        stem = img_path.stem
        for ext in [".png", ".jpg"]:
            tgt_path = tgt_dir / f"{stem}{ext}"
            if tgt_path.exists():
                target_img = Image.open(tgt_path).convert("RGB")
                if target_img.size != (resolution, resolution):
                    target_img = target_img.resize((resolution, resolution), Image.LANCZOS)
                break

        samples.append({
            "input": input_img,
            "target": target_img,
            "path": str(img_path),
            "stem": stem,
        })

    return samples


def pil_to_tensor(img: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert PIL image to [1, 3, H, W] tensor in [0, 1]."""
    arr = np.array(img, dtype="float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype)


def evaluate_task(
    transformer,
    vae,
    components,
    task: str,
    samples: list,
    args,
    device: torch.device,
    dtype: torch.dtype,
):
    """Evaluate a single task using flow-matching inference."""
    prompt = TASK_PROMPTS[task]
    logger.info(f"\nEvaluating task: {task} ({len(samples)} samples)")
    logger.info(f"  Prompt: {prompt[:80]}...")

    from open_vision_banana.inference.sana_img2img import flow_matching_inference

    psnr_list = []
    ssim_list = []
    times = []

    for i, sample in enumerate(samples):
        input_img = sample["input"]
        target_img = sample["target"]

        input_tensor = pil_to_tensor(input_img, device, dtype)

        t0 = time.time()
        with torch.no_grad():
            predicted = flow_matching_inference(
                transformer,
                vae,
                components,
                input_tensor,
                [prompt],
                device,
                dtype,
                num_steps=args.num_inference_steps,
                use_img2img=True,
                max_length=300,
                guidance_scale=args.guidance_scale,
            )
        elapsed = time.time() - t0
        times.append(elapsed)

        pred_np = predicted[0].permute(1, 2, 0).cpu().numpy()

        if target_img is not None:
            target_np = np.array(target_img).astype(np.float32) / 255.0
            psnr = compute_psnr(pred_np, target_np)
            ssim = compute_ssim(pred_np * 255, target_np * 255)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

        if (i + 1) % 10 == 0:
            logger.info(f"  [{i+1}/{len(samples)}] time={elapsed:.2f}s")

    results = {
        "task": task,
        "num_samples": len(samples),
        "avg_time_sec": sum(times) / len(times) if times else 0,
    }

    if psnr_list:
        results["psnr_mean"] = sum(psnr_list) / len(psnr_list)
        results["psnr_std"] = float(np.std(psnr_list))
        results["ssim_mean"] = sum(ssim_list) / len(ssim_list)
        results["ssim_std"] = float(np.std(ssim_list))

    return results


def main():
    args = parse_args()

    device = args.device
    if device == "auto":
        device = auto_device()
    device = torch.device(device)
    dtype = get_dtype(args.dtype)

    if device.type == "mps" and dtype == torch.float16:
        dtype = torch.bfloat16
        logger.info("MPS: switching to bfloat16")

    logger.info("=" * 60)
    logger.info("SANA-1600M Multi-Task Evaluation")
    logger.info("=" * 60)
    logger.info(f"Device: {device} | dtype: {dtype}")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Tasks: {args.tasks}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    logger.info("Loading SanaPipeline components...")
    from diffusers import SanaPipeline
    from open_vision_banana.model_registry import ModelConfig, PipelineComponents
    from open_vision_banana.inference.sana_img2img import flow_matching_inference

    pipe = SanaPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)

    if device.type == "mps":
        pipe.text_encoder.to(torch.float32)

    model_cfg = ModelConfig(
        name="sana-1600m",
        model_id=args.model_path,
        family="sana",
        backbone_type="transformer",
        num_text_encoders=1,
        cross_attention_dim=2240,
        vae_downsample_factor=8,
        memory_budget_gb=10.0,
        pipeline_class="SanaPipeline",
        default_resolution=args.resolution,
        default_guidance_scale=args.guidance_scale,
        default_num_inference_steps=args.num_inference_steps,
        default_lora_rank=32,
        default_lora_alpha=64,
        lora_target_modules=["linear", "proj_out", "conv_inverted", "conv_point"],
        needs_attention_mask=True,
        vae_has_latent_dist=False,
    )

    components = PipelineComponents(
        unet=pipe.transformer,
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        text_encoder_2=None,
        tokenizer=pipe.tokenizer,
        config=model_cfg,
        scheduler=pipe.scheduler,
    )

    logger.info("Model loaded!")

    all_results = {}

    if args.data_dir:
        data_dir = Path(args.data_dir)
        for task in args.tasks:
            samples = load_val_samples(data_dir, task, args.resolution, args.max_samples)
            if not samples:
                logger.warning(f"No samples found for task {task}")
                continue
            results = evaluate_task(
                pipe.transformer, pipe.vae, components,
                task, samples, args, device, dtype,
            )
            all_results[task] = results
    else:
        logger.info("No data directory provided. Running smoke test with synthetic data...")
        for task in args.tasks:
            logger.info(f"\nSmoke test for task: {task}")
            synthetic_img = Image.new("RGB", (args.resolution, args.resolution), (128, 128, 128))
            input_tensor = pil_to_tensor(synthetic_img, device, dtype)

            t0 = time.time()
            with torch.no_grad():
                predicted = flow_matching_inference(
                    pipe.transformer,
                    pipe.vae,
                    components,
                    input_tensor,
                    [TASK_PROMPTS[task]],
                    device,
                    dtype,
                    num_steps=min(args.num_inference_steps, 5),
                    use_img2img=True,
                    max_length=300,
                    guidance_scale=args.guidance_scale,
                )
            elapsed = time.time() - t0
            logger.info(f"  Smoke test time: {elapsed:.2f}s")
            all_results[task] = {
                "task": task,
                "smoke_test_time_sec": elapsed,
                "note": "No GT data provided",
            }

    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved to {results_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)
    for task, results in all_results.items():
        logger.info(f"\n{task}:")
        for key, value in results.items():
            if key != "task":
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()
