#!/usr/bin/env python3
"""SD3-Medium LoRA 推理可视化脚本

SD3 uses StableDiffusion3Pipeline with native img2img support via
StableDiffusion3Img2ImgPipeline. Unlike Z-Image, SD3:
  - Has triple text encoders (CLIP-L + CLIP-G + T5-XXL)
  - Returns prompt_embeds (tensor) + pooled_prompt_embeds (tensor), not list
  - Uses FlowMatchEulerDiscreteScheduler with shift=3.0
  - Supports native img2img via StableDiffusion3Img2ImgPipeline

Usage:
  python scripts/visualize_sd3_results.py
  python scripts/visualize_sd3_results.py --checkpoint output/sd3_checkpoints/checkpoint-500
  python scripts/visualize_sd3_results.py --num-samples 3 --steps 28
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.inference.generator import BaseGenerator
from open_vision_banana.inference.prompt import PromptBuilder
from open_vision_banana.model_registry import vae_encode, vae_decode, resolve_model_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("viz_sd3")

SD3_PATH = "stabilityai/stable-diffusion-3-medium"
DATA_DIRS = [
    PROJECT_ROOT / "data" / "vb_zimage_train",
    PROJECT_ROOT / "data" / "vb_snvc_train",
]
OUTPUT_DIR = PROJECT_ROOT / "output" / "sd3_visualizations"

# P0-3 fix: for segmentation, prefer the regenerated Cityscapes color data
# (vb_snvc_train/segmentation) over the excluded ADE20K class-ID split.
SEGMENTATION_DIR_OVERRIDE = PROJECT_ROOT / "data" / "vb_snvc_train" / "segmentation"


def get_latest_checkpoint(model_path: str = SD3_PATH) -> Path | None:
    """Find the latest checkpoint directory for the given model family."""
    model_short = Path(model_path).name.lower()
    if "ssd" in model_short:
        ckpt_root = PROJECT_ROOT / "output" / "ssd-1b_checkpoints"
    elif "sdxl" in model_short:
        ckpt_root = PROJECT_ROOT / "output" / "sdxl_checkpoints"
    else:
        ckpt_root = PROJECT_ROOT / "output" / "sd3_checkpoints"
    if not ckpt_root.exists():
        return None
    dirs = sorted(ckpt_root.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return dirs[-1] if dirs else None


def load_image(path: Path, resolution: int = 256) -> Image.Image:
    """Load and resize an image."""
    return Image.open(path).convert("RGB").resize(
        (resolution, resolution), Image.LANCZOS
    )


def pil_to_tensor(img: Image.Image, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Convert PIL to [1, 3, H, W] in [-1, 1] for VAE encoding."""
    arr = torch.from_numpy(
        (np.array(img.convert("RGB")) / 255.0).astype("float32")
    ).permute(2, 0, 1).unsqueeze(0)
    return arr.to(device=device, dtype=dtype) * 2.0 - 1.0


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert [C,H,W] or [1,C,H,W] tensor in [0,1] or [-1,1] to PIL."""
    t = t.float().cpu()
    if t.dim() == 4:
        t = t[0]
    if t.min() < 0:
        t = (t + 1.0) / 2.0
    t = t.clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def make_triptych(
    input_img: Image.Image,
    pred_img: Image.Image,
    target_img: Image.Image,
    task: str,
    sample_idx: int,
    output_dir: Path,
) -> Path:
    """Create input | prediction | target side-by-side image with labels."""
    w, h = input_img.size
    label_h = 30
    canvas_w = w * 3 + 20
    canvas_h = h + label_h + 10

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()

    labels = ["Input", f"Pred ({task})", "GT"]
    images = [input_img, pred_img, target_img]
    x_offsets = [0, w + 10, 2 * (w + 10)]

    for label, img, x in zip(labels, images, x_offsets):
        draw.text((x + w // 2 - 30, 5), label, fill=(0, 0, 0), font=font)
        canvas.paste(img, (x, label_h))

    save_path = output_dir / f"{task}_sample_{sample_idx:03d}.png"
    canvas.save(save_path, quality=95)
    return save_path


def make_grid(all_images: list[dict], output_dir: Path, ckpt_name: str) -> Path | None:
    """Create a grid of all task results."""
    if not all_images:
        return None

    n = len(all_images)
    cols = 3
    rows = n

    cell_w, cell_h = 256, 256
    label_h = 35
    margin = 5

    grid_w = cols * (cell_w + margin) + margin
    grid_h = rows * (cell_h + label_h + margin) + margin + 50

    grid = Image.new("RGB", (grid_w, grid_h), (240, 240, 240))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_big = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_big = font

    draw.text(
        (margin, 8),
        f"LoRA Inference — {ckpt_name}",
        fill=(0, 0, 0),
        font=font_big,
    )

    for i, item in enumerate(all_images):
        y = 50 + i * (cell_h + label_h + margin)
        task_label = f"[{item['task']}] sample {item['sample_idx']}"
        draw.text((margin, y), task_label, fill=(50, 50, 50), font=font)

        labels = ["Input", "Prediction", "Ground Truth"]
        for j, (label, img) in enumerate(zip(labels, item["images"])):
            x = margin + j * (cell_w + margin)
            draw.text((x, y + 15), label, fill=(0, 0, 0), font=font)
            grid.paste(
                img.resize((cell_w, cell_h), Image.LANCZOS),
                (x, y + label_h),
            )

    grid_path = output_dir / "sd3_results_grid.png"
    grid.save(grid_path, quality=95)
    return grid_path


def run_img2img_inference(
    generator: BaseGenerator,
    prompt: str,
    input_pil: Image.Image,
    num_steps: int,
    guidance_scale: float,
    strength: float,
    seed: int,
) -> Image.Image:
    """Img2img inference via BaseGenerator.generate().

    Uses the generator's built-in img2img pipeline which shares components
    with the t2i pipeline (memory-efficient, no duplication).
    """
    with torch.no_grad():
        result = generator.generate(
            prompt=prompt,
            image=input_pil,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed,
        )
    return tensor_to_pil(result[0])


def run_t2i_inference(
    generator: BaseGenerator,
    prompt: str,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    height: int = 256,
    width: int = 256,
) -> Image.Image:
    """Text-to-image inference (for generation task)."""
    with torch.no_grad():
        result = generator.generate(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            seed=seed,
        )
    return tensor_to_pil(result[0])


def main():
    parser = argparse.ArgumentParser(description="Visualize SD3 LoRA inference results")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint dir (default: latest)")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Samples per task")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Inference resolution")
    parser.add_argument("--steps", type=int, default=None,
                        help="Inference steps (SSD-1B defaults to 4)")
    parser.add_argument("--guidance-scale", type=float, default=None,
                        help="CFG guidance scale (SSD-1B defaults to 1.0)")
    parser.add_argument("--strength", type=float, default=0.85,
                        help="Img2img strength (higher=more creative)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--model-path", type=str, default=SD3_PATH,
                        help="Model path or HF repo ID")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: auto from model name)")
    parser.add_argument("--tasks", type=str, default="all",
                        help="Comma-separated tasks: depth,normal,segmentation,edge,generation")
    args = parser.parse_args()

    # SSD-1B is distilled and must be sampled with its low-step, low-CFG
    # regime; SDXL/SD3 keep their conventional defaults.
    model_short = Path(args.model_path).name.lower()
    if args.steps is None:
        args.steps = 4 if "ssd" in model_short else 28
    if args.guidance_scale is None:
        args.guidance_scale = 1.0 if "ssd" in model_short else 7.0

    # Resolve checkpoint
    ckpt_path = Path(args.checkpoint) if args.checkpoint else get_latest_checkpoint(args.model_path)
    if ckpt_path is None or not (ckpt_path / "adapter_model.safetensors").exists():
        logger.error(f"No valid checkpoint found at {ckpt_path}")
        sys.exit(1)
    ckpt_name = ckpt_path.name
    logger.info(f"Using checkpoint: {ckpt_path}")

    # Setup output
    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    else:
        # Auto: infer from model name to avoid overwriting SD3 results
        model_short = Path(args.model_path).name.lower()
        if "ssd" in model_short or "sdxl" in model_short:
            OUTPUT_DIR = PROJECT_ROOT / "output" / f"{model_short}_visualizations"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "triptychs").mkdir(parents=True, exist_ok=True)

    # Load model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Loading SD3-Medium on {device} (bfloat16)...")
    t0 = time.time()
    generator = BaseGenerator(
        model_name=args.model_path,
        lora_path=str(ckpt_path),
        device=device,
        dtype=torch.bfloat16,
    )
    logger.info(f"Model + LoRA loaded in {time.time()-t0:.1f}s")

    # Verify config
    mc = generator.model_config
    logger.info(f"Family: {mc.family}, VAE scale: {mc.vae_scaling_factor}, shift: {mc.vae_shift_factor}")
    logger.info(f"VAE latent channels: {mc.vae_latent_channels}")

    tasks = (
        args.tasks.split(",")
        if args.tasks != "all"
        else ["depth", "normal", "segmentation", "edge", "generation"]
    )
    res = args.resolution
    all_results = []

    for task in tasks:
        # Search all data directories for this task
        # P0-3: segmentation uses the regenerated Cityscapes color split
        task_dir = None
        if task == "segmentation" and (SEGMENTATION_DIR_OVERRIDE / "images").exists():
            task_dir = SEGMENTATION_DIR_OVERRIDE
        if task_dir is None:
            for d in DATA_DIRS:
                candidate = d / task
                if (candidate / "images").exists():
                    task_dir = candidate
                    break
        if task_dir is None or not task_dir.exists():
            logger.warning(f"Task dir not found for '{task}'")
            continue

        img_dir = task_dir / "images"
        tgt_dir = task_dir / "targets"
        if not img_dir.exists():
            logger.warning(f"No images dir for task '{task}'")
            continue

        images = sorted(img_dir.glob("*.jpg"))[: args.num_samples]
        if not images:
            images = sorted(img_dir.glob("*.png"))[: args.num_samples]
        if not images:
            logger.warning(f"No images found for task '{task}' in {img_dir}")
            continue

        # Build prompt
        if task == "depth":
            prompt = PromptBuilder.DEPTH_PROMPT
        elif task == "normal":
            prompt = PromptBuilder.NORMAL_PROMPT
        elif task == "segmentation":
            prompt = "Generate a semantic segmentation visualization of the input image."
        elif task == "edge":
            prompt = PromptBuilder.EDGE_PROMPT
        elif task == "generation":
            prompts_file = task_dir / "prompts.jsonl"
            prompt = "a photograph"
            if prompts_file.exists():
                with open(prompts_file) as f:
                    first = json.loads(f.readline())
                    prompt = first.get("prompt", "a photograph")
        else:
            prompt = f"Generate a {task} visualization of this image."

        logger.info(f"\n{'='*60}")
        logger.info(f"Task: {task} | Samples: {len(images)} | Prompt: {prompt[:80]}")
        logger.info(f"{'='*60}")

        for i, img_path in enumerate(images):
            logger.info(f"  [{task}] sample {i}: {img_path.name}")

            input_pil = load_image(img_path, res)

            # Load target (skip for generation task — targets are .txt prompts)
            target_pil = None
            if task != "generation" and tgt_dir.exists():
                tgt_candidates = list(tgt_dir.glob(img_path.stem + "*"))
                if not tgt_candidates:
                    tgt_candidates = sorted(tgt_dir.glob("*"))
                if i < len(tgt_candidates):
                    target_pil = load_image(tgt_candidates[i], res)

            if target_pil is None:
                target_pil = input_pil

            # Run inference
            t1 = time.time()
            try:
                if task == "generation":
                    # Text-to-image: no input image conditioning
                    pred_pil = run_t2i_inference(
                        generator,
                        prompt,
                        num_steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        seed=args.seed + i,
                        height=res,
                        width=res,
                    )
                else:
                    # Img2img: condition on input image
                    pred_pil = run_img2img_inference(
                        generator,
                        prompt,
                        input_pil,
                        num_steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        strength=args.strength,
                        seed=args.seed + i,
                    )
                elapsed = time.time() - t1
                logger.info(f"    Inference: {elapsed:.1f}s")

                # Save triptych
                triptych_path = make_triptych(
                    input_pil,
                    pred_pil,
                    target_pil,
                    task,
                    i,
                    OUTPUT_DIR / "triptychs",
                )
                logger.info(f"    Saved: {triptych_path}")

                all_results.append(
                    {
                        "task": task,
                        "sample_idx": i,
                        "images": [input_pil, pred_pil, target_pil],
                    }
                )

            except Exception as e:
                logger.error(f"    Inference failed: {e}")
                import traceback

                traceback.print_exc()
                continue

    # Generate grid
    if all_results:
        logger.info(f"\nGenerating result grid ({len(all_results)} samples)...")
        grid_path = make_grid(all_results, OUTPUT_DIR, ckpt_name)
        if grid_path:
            logger.info(f"Grid saved: {grid_path}")

    logger.info(f"\nDone! All results in: {OUTPUT_DIR}")
    logger.info(f"  - Triptychs: {OUTPUT_DIR / 'triptychs'}")
    logger.info(f"  - Grid: {OUTPUT_DIR / 'sd3_results_grid.png'}")


if __name__ == "__main__":
    main()
