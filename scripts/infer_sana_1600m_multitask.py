#!/usr/bin/env python3
"""SANA-1600M multi-task inference visualization.

Loads the merged model (base + LoRA) from HuggingFace and runs
multi-task inference with side-by-side visualization:
  [Input | Model Output | Decoded Result]

Supports depth, normal, and semantic segmentation tasks.

Usage:
    python scripts/infer_sana_1600m_multitask.py \
        --model-path gatilin/open-vision-banana-sana_1600m_multitask \
        --image input.jpg \
        --output-dir output/sana_1600m_viz \
        --tasks depth normal semantic
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
from PIL import Image, ImageDraw

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
    p.add_argument(
        "--model-path",
        type=str,
        default="gatilin/open-vision-banana-sana_1600m_multitask",
        help="HuggingFace model path or local directory",
    )
    p.add_argument("--image", type=str, default=None, help="Input image path")
    p.add_argument(
        "--tasks",
        nargs="+",
        default=["depth", "normal", "semantic"],
        choices=list(TASK_PROMPTS.keys()),
    )
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--num-inference-steps", type=int, default=20)
    p.add_argument("--guidance-scale", type=float, default=4.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="output/sana_1600m_viz")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    return p.parse_args()


def auto_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(dtype_str):
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_str]


def load_image(path: str, resolution: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), Image.LANCZOS)
    return img


def make_side_by_side(images: list[Image.Image], labels: list[str]) -> Image.Image:
    w, h = images[0].size
    label_h = 30
    total_w = w * len(images)
    total_h = h + label_h
    canvas = Image.new("RGB", (total_w, total_h), (40, 40, 40))
    draw = ImageDraw.Draw(canvas)
    for i, (img, label) in enumerate(zip(images, labels)):
        canvas.paste(img, (i * w, label_h))
        draw.text((i * w + 5, 5), label, fill=(255, 255, 255))
    return canvas


def create_demo_image(resolution: int) -> Image.Image:
    """Create a simple demo image for testing when no input is provided."""
    img = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    for y in range(resolution):
        for x in range(resolution):
            img[y, x] = [
                int(255 * x / resolution),
                int(255 * y / resolution),
                128,
            ]
    return Image.fromarray(img)


def main():
    args = parse_args()

    device = args.device
    if device == "auto":
        device = auto_device()
    device = torch.device(device)
    dtype = get_dtype(args.dtype)

    if device == "mps" and dtype == torch.float16:
        dtype = torch.bfloat16
        logger.info("MPS: switching to bfloat16")

    logger.info("=" * 60)
    logger.info("SANA-1600M Multi-Task Inference Visualization")
    logger.info("=" * 60)
    logger.info(f"Device: {device} | dtype: {dtype}")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Tasks: {args.tasks}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    logger.info("Loading SanaPipeline components...")
    from diffusers import SanaPipeline

    pipe = SanaPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)

    if device == "mps":
        pipe.text_encoder.to(torch.float32)

    logger.info("Model loaded successfully!")

    # Build PipelineComponents for flow-matching inference
    from open_vision_banana.model_registry import ModelConfig, PipelineComponents
    from open_vision_banana.inference.sana_img2img import (
        flow_matching_inference,
        encode_text_sana,
        vae_encode_sana,
    )

    model_cfg = ModelConfig(
        name="sana-1600m",
        model_id="gatilin/open-vision-banana-sana_1600m_multitask",
        family="sana",
        backbone_type="transformer",
        num_text_encoders=1,
        cross_attention_dim=2240,
        vae_downsample_factor=8,
        memory_budget_gb=10.0,
        pipeline_class="SanaPipeline",
        default_resolution=512,
        default_guidance_scale=4.5,
        default_num_inference_steps=20,
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

    if args.image:
        input_img = load_image(args.image, args.resolution)
        img_name = Path(args.image).stem
    else:
        logger.info("No input image provided, using gradient demo image")
        input_img = create_demo_image(args.resolution)
        img_name = "demo_gradient"

    # Convert PIL to tensor [-1, 1]
    input_np = np.array(input_img).astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(input_np).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    input_tensor = input_tensor.to(device=device, dtype=dtype)

    for task in args.tasks:
        prompt = TASK_PROMPTS[task]
        logger.info(f"\nRunning task: {task}")
        logger.info(f"  Prompt: {prompt[:80]}...")

        t0 = time.time()

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

        # Convert to PIL
        pred_np = (predicted[0].cpu().float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        output_img = Image.fromarray(pred_np)

        panels = [input_img, output_img]
        labels = ["Input", f"Output ({task})"]

        comparison = make_side_by_side(panels, labels)
        save_path = output_dir / f"{img_name}_{task}_comparison.png"
        comparison.save(str(save_path))
        logger.info(f"  Saved: {save_path}")

        output_img.save(output_dir / f"{img_name}_{task}_output.png")

    logger.info(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
