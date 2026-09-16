#!/usr/bin/env python3
"""COCO128 inference validation for SANA-1600M.

Runs multi-task inference on COCO128 dataset images and saves visualizations.

Usage:
    python scripts/infer_coco128.py \
        --model-path gatilin/open-vision-banana-sana_1600m_multitask \
        --data-dir data/coco128/coco128/images/train2017 \
        --output-dir output/coco128_inference \
        --tasks depth normal semantic \
        --max-samples 8
"""

from __future__ import annotations

import argparse
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
    "object_detection": "This image shows object detection results. Each detected object is outlined by a colored bounding box on a black background.",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-path",
        type=str,
        default="gatilin/open-vision-banana-sana_1600m_multitask",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default="data/coco128/coco128/images/train2017",
    )
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--num-inference-steps", type=int, default=20)
    p.add_argument("--guidance-scale", type=float, default=4.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="output/coco128_inference")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="float16")
    p.add_argument(
        "--tasks",
        nargs="+",
        default=["depth", "normal"],
        choices=list(TASK_PROMPTS.keys()),
    )
    p.add_argument("--max-samples", type=int, default=8)
    return p.parse_args()


def auto_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(dtype_str):
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_str]


def load_images(data_dir: Path, resolution: int, max_samples: int):
    """Load images from data directory."""
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return []

    images = sorted(list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png")))
    logger.info(f"Found {len(images)} images in {data_dir}")

    samples = []
    for img_path in images[:max_samples]:
        img = Image.open(img_path).convert("RGB")
        if img.size != (resolution, resolution):
            img = img.resize((resolution, resolution), Image.LANCZOS)
        samples.append({
            "input": img,
            "path": str(img_path),
            "stem": img_path.stem,
        })

    return samples


def pil_to_tensor(img: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert PIL image to [1, 3, H, W] tensor in [0, 1]."""
    arr = np.array(img, dtype="float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype)


def save_comparison(input_img: Image.Image, output_tensor: torch.Tensor, save_path: Path):
    """Save input and output side-by-side."""
    output_img = Image.fromarray(
        (output_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    )

    total_width = input_img.width + output_img.width
    max_height = max(input_img.height, output_img.height)

    comparison = Image.new("RGB", (total_width, max_height), (255, 255, 255))
    comparison.paste(input_img, (0, 0))
    comparison.paste(output_img, (input_img.width, 0))

    comparison.save(save_path)
    logger.info(f"Saved: {save_path}")


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
    logger.info("COCO128 Inference Validation")
    logger.info("=" * 60)
    logger.info(f"Device: {device} | dtype: {dtype}")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Tasks: {args.tasks}")
    logger.info(f"Max samples: {args.max_samples}")

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

    samples = load_images(Path(args.data_dir), args.resolution, args.max_samples)
    if not samples:
        logger.error("No images found!")
        return

    logger.info(f"\nLoaded {len(samples)} samples")

    for task in args.tasks:
        prompt = TASK_PROMPTS[task]
        logger.info(f"\n{'='*60}")
        logger.info(f"Task: {task}")
        logger.info(f"Prompt: {prompt[:80]}...")
        logger.info(f"{'='*60}")

        task_output_dir = output_dir / task
        task_output_dir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(samples):
            logger.info(f"\n[{i+1}/{len(samples)}] Processing: {sample['stem']}")

            input_tensor = pil_to_tensor(sample["input"], device, dtype)

            t0 = time.time()
            with torch.no_grad():
                predicted = flow_matching_inference(
                    pipe.transformer,
                    pipe.vae,
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
            logger.info(f"  Inference time: {elapsed:.2f}s")

            save_path = task_output_dir / f"{sample['stem']}_{task}.png"
            save_comparison(sample["input"], predicted, save_path)

    logger.info(f"\n{'='*60}")
    logger.info(f"All results saved to {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
