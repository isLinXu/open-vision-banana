#!/usr/bin/env python3
"""Z-Image LoRA 推理可视化脚本

Z-Image pipeline 不支持 img2img 的 image 参数，需要手动实现：
1. VAE 编码输入图像 → latents
2. 根据 strength 加噪 → starting latents
3. 传入 ZImagePipeline 带自定义 sigmas 进行去噪
4. VAE 解码 → 输出图像

Usage:
  python scripts/visualize_zimage_results.py
  python scripts/visualize_zimage_results.py --checkpoint output/zimage_checkpoints/checkpoint-23550
  python scripts/visualize_zimage_results.py --num-samples 3 --steps 25
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
logger = logging.getLogger("viz_zimage")

ZIMAGE_PATH = "/Users/gatilin/MyWork/models/ms_cache/models/models/Tongyi-MAI--Z-Image/snapshots/master"
DATA_DIR = PROJECT_ROOT / "data" / "vb_zimage_train"
OUTPUT_DIR = PROJECT_ROOT / "output" / "zimage_visualizations"


def get_latest_checkpoint() -> Path | None:
    """Find the latest checkpoint directory."""
    ckpt_root = PROJECT_ROOT / "output" / "zimage_checkpoints"
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
        f"Z-Image LoRA Inference — {ckpt_name}",
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

    grid_path = output_dir / "zimage_results_grid.png"
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
    """Custom img2img inference for Z-Image (no native img2img support).

    Z-Image uses FlowMatchEulerDiscreteScheduler + multi-encoder prompt_embeds (list).
    Pipeline accepts `latents` and `sigmas` for custom starting point.

    Steps:
    1. VAE encode input image → input_latents
    2. Create noisy starting latents: z_start = (1-strength)*input + strength*noise
    3. Build custom sigmas from strength → 0
    4. Encode prompt using pipeline's native encode_prompt (returns list of tensors)
    5. Call ZImagePipeline with latents + sigmas + prompt_embeds
    6. Pipeline handles VAE decode internally when output_type="pil"
    """
    pipe = generator._t2i_pipe  # ZImagePipeline
    device = generator.device
    dtype = generator.dtype
    mc = generator.model_config

    # 1. Encode input image to latents
    input_tensor = pil_to_tensor(input_pil, device, dtype)
    with torch.no_grad():
        input_latents = vae_encode(pipe.vae, input_tensor, mc)

    logger.info(f"    Input latents shape: {input_latents.shape}")

    # 2. Create noisy starting latents
    generator_obj = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        input_latents.shape,
        generator=generator_obj,
        device="cpu",
        dtype=input_latents.dtype,
    ).to(device)
    # Flow matching: z_t = (1-sigma)*z_clean + sigma*noise
    z_start = (1.0 - strength) * input_latents + strength * noise

    # 3. Build custom sigmas: from strength → 0 (num_steps+1 points for Euler)
    # Must be on CPU as numpy array — scheduler does np.array(sigmas) internally
    sigmas = torch.linspace(
        strength, 0.0, num_steps + 1, device="cpu", dtype=torch.float32
    )

    # 4. Encode prompt using pipeline's native encode_prompt
    do_cfg = guidance_scale > 1.0
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            do_classifier_free_guidance=do_cfg,
            negative_prompt="" if do_cfg else None,
            max_sequence_length=512,
        )

    logger.info(f"    Prompt embeds: {len(prompt_embeds)} tensors, shapes: {[pe.shape for pe in prompt_embeds]}")

    # 5. Call pipeline with custom latents + sigmas + prompt_embeds
    h, w = input_latents.shape[2] * 8, input_latents.shape[3] * 8  # latent→pixel
    with torch.no_grad():
        result = pipe(
            prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            height=h,
            width=w,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            sigmas=sigmas,
            latents=z_start.to(torch.float32),
            generator=torch.Generator(device=device).manual_seed(seed),
            output_type="pil",
            return_dict=True,
        )

    images = result.images if hasattr(result, "images") else result[0]
    if isinstance(images, list):
        return images[0]
    return images


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
    parser = argparse.ArgumentParser(description="Visualize Z-Image LoRA inference results")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint dir (default: latest)")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Samples per task")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Inference resolution")
    parser.add_argument("--steps", type=int, default=25,
                        help="Inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                        help="CFG guidance scale")
    parser.add_argument("--strength", type=float, default=0.85,
                        help="Img2img strength (higher=more creative)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--tasks", type=str, default="all",
                        help="Comma-separated tasks: depth,segmentation,edge,generation")
    args = parser.parse_args()

    # Resolve checkpoint
    ckpt_path = Path(args.checkpoint) if args.checkpoint else get_latest_checkpoint()
    if ckpt_path is None or not (ckpt_path / "adapter_model.safetensors").exists():
        logger.error(f"No valid checkpoint found at {ckpt_path}")
        sys.exit(1)
    ckpt_name = ckpt_path.name
    logger.info(f"Using checkpoint: {ckpt_path}")

    # Setup output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "triptychs").mkdir(parents=True, exist_ok=True)

    # Load model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Loading Z-Image on {device} (bfloat16)...")
    t0 = time.time()
    generator = BaseGenerator(
        model_name=ZIMAGE_PATH,
        lora_path=str(ckpt_path),
        device=device,
        dtype=torch.bfloat16,
    )
    logger.info(f"Model + LoRA loaded in {time.time()-t0:.1f}s")

    # Verify config
    mc = generator.model_config
    logger.info(f"Family: {mc.family}, VAE scale: {mc.vae_scaling_factor}, shift: {mc.vae_shift_factor}")

    tasks = (
        args.tasks.split(",")
        if args.tasks != "all"
        else ["depth", "segmentation", "edge", "generation"]
    )
    res = args.resolution
    all_results = []

    for task in tasks:
        task_dir = DATA_DIR / task
        if not task_dir.exists():
            logger.warning(f"Task dir not found: {task_dir}")
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
    logger.info(f"  - Grid: {OUTPUT_DIR / 'zimage_results_grid.png'}")


if __name__ == "__main__":
    main()
