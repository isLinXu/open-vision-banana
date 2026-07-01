#!/usr/bin/env python3
"""Generate synthetic training data using SDXL/SSD-1B for Vision Banana.

Generates high-quality images with captions to satisfy the 9:1 generation:vision
ratio required by the paper. Uses the base model (without LoRA) to generate
diverse images that preserve the model's original generation capabilities.

Usage:
  python scripts/generate_synthetic_data.py --model ssd-1b --num-images 1000 --output-dir data/vb_v6_synthetic
  python scripts/generate_synthetic_data.py --model sdxl --num-images 5000 --batch-size 4 --output-dir data/vb_v6_synthetic_5k
  python scripts/generate_synthetic_data.py --prompts-file prompts/coco_captions.txt --num-images 10000

The script uses the base model (no LoRA checkpoint) to ensure generation quality
is not degraded by vision task tuning.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from open_vision_banana.model_registry import (
    get_model_config,
    list_models,
    load_pipeline,
    vae_decode,
    vae_encode,
    encode_text,
    backbone_forward,
    build_added_cond_kwargs,
    prepare_position_ids,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_synthetic")


# ─── Default prompt pools ─────────────────────────────────────────────────────

DEFAULT_PROMPTS = [
    "A photorealistic image of a modern city street at sunset, with cars and pedestrians",
    "A cozy living room interior with warm lighting, wooden furniture, and a fireplace",
    "A lush green forest path with sunlight filtering through the trees",
    "A busy kitchen with stainless steel appliances and fresh ingredients on the counter",
    "A serene mountain lake reflecting snow-capped peaks at dawn",
    "A modern office workspace with multiple monitors, plants, and city view",
    "A tropical beach with crystal clear water, palm trees, and white sand",
    "A historic European street with cobblestones, old buildings, and outdoor cafes",
    "A spacious warehouse with industrial lighting and organized shelves",
    "A colorful flower garden with a winding path and a wooden bench",
    "A rainy city street at night with neon signs reflecting on wet pavement",
    "A sunny backyard with a swimming pool, patio furniture, and barbecue",
    "A bustling market with fresh produce, vendors, and shoppers",
    "A minimalist bedroom with white walls, natural light, and simple decor",
    "A dramatic cliffside overlooking the ocean with waves crashing below",
    "A snowy village with wooden cabins, smoke from chimneys, and a frozen lake",
    "A desert landscape with sand dunes, cacti, and a clear blue sky",
    "A train station platform with passengers, luggage, and a arriving train",
    "A vineyard with rows of grapevines, rolling hills, and a rustic farmhouse",
    "A modern art gallery with white walls, sculptures, and abstract paintings",
    "A rooftop terrace with city skyline views, string lights, and potted plants",
    "A Japanese zen garden with raked gravel, moss, and a stone lantern",
    "A construction site with cranes, scaffolding, and workers in hard hats",
    "A library with tall bookshelves, reading desks, and warm ambient lighting",
    "A skate park with ramps, rails, and graffiti art on the walls",
    "A botanical garden greenhouse with tropical plants, glass walls, and mist",
    "A harbor with fishing boats, seagulls, and a lighthouse in the distance",
    "A suburban neighborhood with lawns, mailboxes, and children playing",
    "A concert hall with an orchestra on stage, chandeliers, and red velvet seats",
    "A hiking trail through autumn forest with fallen leaves and a wooden bridge",
    "A futuristic laboratory with clean white surfaces, holographic displays, and robotic arms",
    "A vintage diner with red leather booths, neon signs, and a jukebox",
    "A parking garage with concrete pillars, fluorescent lights, and parked cars",
    "A wedding reception venue with decorated tables, floral arrangements, and a dance floor",
    "A basketball court with players, bleachers, and a scoreboard",
    "A sushi restaurant interior with a counter, chefs, and fresh fish displays",
    "A camping site with tents, a campfire, and a starry night sky",
    "A hospital corridor with medical equipment, nurses, and patient rooms",
    "A skate rink with ice skaters, colored lights, and a disco ball",
    "A bakery shop with bread, pastries, and a display case",
    "A soccer stadium with a green field, goalposts, and cheering fans",
    "A yoga studio with wooden floors, mirrors, and natural light from large windows",
    "A car showroom with luxury vehicles, polished floors, and spotlights",
    "A cemetery with tombstones, mature trees, and a wrought iron fence",
    "A factory floor with assembly lines, conveyor belts, and industrial robots",
    "A outdoor playground with slides, swings, and a sandbox",
    "A wine cellar with oak barrels, stone walls, and dim candlelight",
    "A rooftop garden with raised beds, vegetables, and a compost bin",
    "A movie theater with rows of seats, a large screen, and popcorn stands",
    "A ski resort with ski lifts, snow-covered slopes, and a lodge",
]


def load_prompts_file(path: Path) -> list[str]:
    """Load prompts from a text file, one per line."""
    if not path.exists():
        logger.warning(f"Prompts file not found: {path}")
        return []
    with open(path) as f:
        prompts = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    logger.info(f"Loaded {len(prompts)} prompts from {path}")
    return prompts


def generate_image(
    components,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
    resolution: int = 512,
    num_steps: int = 20,
    guidance_scale: float = 7.5,
    seed: int | None = None,
) -> Image.Image:
    """Generate a single image from a text prompt using the base model."""
    config = components.config
    backbone = components.unet
    vae = components.vae
    backbone.eval()

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Encode text
    with torch.no_grad():
        encoder_hidden_states, aux_output = encode_text(components, [prompt], device, dtype)

    attention_mask = None
    pooled_embeds = None
    if isinstance(aux_output, torch.Tensor):
        if config.needs_attention_mask:
            attention_mask = aux_output
        else:
            pooled_embeds = aux_output

    added_cond_kwargs = build_added_cond_kwargs(
        config, pooled_embeds, (resolution, resolution), device, dtype, batch_size=1
    )

    # CFG setup
    uncond_hidden = None
    uncond_added_cond = None
    uncond_mask = None
    if guidance_scale > 1.0:
        uncond_hidden, uncond_aux = encode_text(components, [""], device, dtype, negative=True)
        if isinstance(uncond_aux, torch.Tensor):
            if config.needs_attention_mask:
                uncond_mask = uncond_aux
            else:
                uncond_added_cond = build_added_cond_kwargs(
                    config, uncond_aux, (resolution, resolution), device, dtype, batch_size=1
                )

    # Initialize latents from noise (standard text-to-image, not img2img)
    latent_shape = config.get_latent_shape(1, resolution, resolution)
    latents = torch.randn(latent_shape, device=device, dtype=dtype)

    txt_ids, img_ids = prepare_position_ids(latents, encoder_hidden_states, config, device, dtype)

    # Denoising loop (DDIM-style for speed)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        timestep_long = torch.tensor([int(t * 999)], device=device).clamp(0, 999)

        if config.needs_position_ids:
            txt_ids, img_ids = prepare_position_ids(latents, encoder_hidden_states, config, device, dtype)

        if guidance_scale > 1.0 and uncond_hidden is not None:
            v_cond = backbone_forward(
                backbone, latents, timestep_long, encoder_hidden_states, config,
                added_cond_kwargs=added_cond_kwargs, guidance=guidance_scale,
                attention_mask=attention_mask, pooled_embeds=pooled_embeds,
                txt_ids=txt_ids, img_ids=img_ids,
            )
            v_uncond = backbone_forward(
                backbone, latents, timestep_long, uncond_hidden, config,
                added_cond_kwargs=uncond_added_cond, guidance=guidance_scale,
                attention_mask=uncond_mask, pooled_embeds=pooled_embeds,
                txt_ids=txt_ids, img_ids=img_ids,
            )
            pred_v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            pred_v = backbone_forward(
                backbone, latents, timestep_long, encoder_hidden_states, config,
                added_cond_kwargs=added_cond_kwargs, guidance=guidance_scale,
                attention_mask=attention_mask, pooled_embeds=pooled_embeds,
                txt_ids=txt_ids, img_ids=img_ids,
            )

        # Euler step
        dt = 1.0 / num_steps
        latents = latents + dt * pred_v

    # Decode
    with torch.no_grad():
        image = vae_decode(vae, latents, config)
        image = (image / 2 + 0.5).clamp(0, 1)
        image_np = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    return Image.fromarray(image_np)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data for Vision Banana")
    parser.add_argument("--model", type=str, default="ssd-1b", help="Base model for generation")
    parser.add_argument("--num-images", type=int, default=1000, help="Number of images to generate")
    parser.add_argument("--output-dir", type=str, default="data/vb_v6_synthetic", help="Output directory")
    parser.add_argument("--prompts-file", type=str, default=None, help="Text file with prompts (one per line)")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--num-steps", type=int, default=20, help="Diffusion steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (not yet batched, always 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-every", type=int, default=100, help="Save metadata every N images")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument(
        "--source", type=str, default="generate", choices=["generate", "coco", "laion"],
        help="Data source: 'generate' (diffusion), 'coco' (COCO val images+captions), 'laion' (LAION subset)",
    )
    parser.add_argument(
        "--coco-dir", type=str, default=None,
        help="Path to COCO dataset root (for --source coco). Expected: val2017/ and annotations/captions_val2017.json",
    )
    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable models:")
        for name in list_models():
            cfg = get_model_config(name)
            print(f"  {name:15s} | {cfg.description}")
        return

    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    captions = []

    # ── COCO fallback mode: use real COCO images + captions ──────────────
    if args.source == "coco":
        if not args.coco_dir:
            logger.error("--coco-dir is required for --source coco")
            return
        coco_root = Path(args.coco_dir)
        captions_file = coco_root / "annotations" / "captions_val2017.json"
        images_root = coco_root / "val2017"
        if not captions_file.exists() or not images_root.exists():
            logger.error(f"COCO not found at {coco_root}. Expected val2017/ and annotations/captions_val2017.json")
            return

        import json as json_mod
        with open(captions_file) as f:
            coco_data = json_mod.load(f)

        # Build image_id → captions mapping
        img_id_to_captions: dict[int, list[str]] = {}
        for ann in coco_data["annotations"]:
            img_id_to_captions.setdefault(ann["image_id"], []).append(ann["caption"])

        img_id_to_file: dict[int, str] = {
            img["id"]: img["file_name"] for img in coco_data["images"]
        }

        all_img_ids = list(img_id_to_file.keys())
        random.seed(args.seed)
        random.shuffle(all_img_ids)

        logger.info(f"COCO fallback: {len(all_img_ids)} images available, selecting {args.num_images}")
        successful = 0
        for img_id in all_img_ids[:args.num_images]:
            filename = img_id_to_file[img_id]
            src_path = images_root / filename
            dst_path = images_dir / f"coco_{img_id:012d}.jpg"
            if not src_path.exists():
                continue
            import shutil
            shutil.copy2(src_path, dst_path)
            # Use first caption (or random if multiple)
            caps = img_id_to_captions.get(img_id, [""])
            caption = random.choice(caps)
            captions.append({"filename": dst_path.name, "caption": caption, "source": "coco", "coco_id": img_id})
            successful += 1
            if successful % args.save_every == 0:
                with open(output_dir / "captions.json", "w") as f:
                    json.dump(captions, f, indent=2)
                logger.info(f"[{successful}/{args.num_images}] Copied COCO images")

        with open(output_dir / "captions.json", "w") as f:
            json.dump(captions, f, indent=2)
        metadata = {
            "source": "coco", "num_images": successful, "coco_dir": str(coco_root),
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"COCO fallback complete: {successful} images saved to {output_dir}")
        return

    # ── LAION fallback mode: download from LAION subset ──────────────────
    if args.source == "laion":
        logger.info("LAION mode: attempting to download from LAION-5B subset via img2dataset")
        try:
            import subprocess
            subprocess.run(
                ["img2dataset", "--url_list", "laion2B-en-processed",
                 "--output_folder", str(output_dir),
                 "--processes_count", "1", "--thread_count", "64",
                 "--resize_mode", "keep_ratio",
                 "--min_image_size", "256", "--max_image_size", "512",
                 "--enable_subset", str(args.num_images)],
                check=True, capture_output=True, text=True,
            )
            logger.info(f"LAION download complete: {output_dir}")
        except FileNotFoundError:
            logger.error("img2dataset not installed. Install with: pip install img2dataset")
            logger.error("Falling back to generation mode")
            args.source = "generate"
        except subprocess.CalledProcessError as e:
            logger.error(f"img2dataset failed: {e.stderr}")
            logger.error("Falling back to generation mode")
            args.source = "generate"

    # ── Generation mode (original path) ──────────────────────────────────
    # Load prompts
    if args.prompts_file:
        prompts = load_prompts_file(Path(args.prompts_file))
    else:
        prompts = DEFAULT_PROMPTS.copy()
    
    if not prompts:
        logger.error("No prompts available!")
        return

    metadata = {
        "model": args.model,
        "num_images": args.num_images,
        "resolution": args.resolution,
        "num_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "prompts_source": args.prompts_file or "default_pool",
        "num_unique_prompts": len(prompts),
    }

    # Load model (base model, no LoRA)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16
    model_cfg = get_model_config(args.model)

    logger.info("=" * 60)
    logger.info("Synthetic Data Generation")
    logger.info("=" * 60)
    logger.info(f"Model: {model_cfg.model_id}")
    logger.info(f"Device: {device}, Resolution: {args.resolution}")
    logger.info(f"Generating {args.num_images} images with {len(prompts)} unique prompts")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    components = load_pipeline(model_cfg, device, dtype, enable_gradient_checkpointing=False)
    logger.info(f"Pipeline loaded: {model_cfg.name}")

    import time
    t_start = time.time()
    successful = 0
    failed = 0

    for i in range(args.num_images):
        prompt = random.choice(prompts)
        seed = args.seed + i
        filename = f"gen_{i:06d}_{seed}.jpg"
        filepath = images_dir / filename

        if filepath.exists():
            logger.info(f"[{i+1}/{args.num_images}] Already exists: {filename}")
            captions.append({"filename": filename, "caption": prompt, "seed": seed})
            successful += 1
            continue

        try:
            img = generate_image(
                components, prompt, device, dtype,
                resolution=args.resolution,
                num_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                seed=seed,
            )
            img.save(filepath, quality=95)
            captions.append({"filename": filename, "caption": prompt, "seed": seed})
            successful += 1

            elapsed = time.time() - t_start
            avg_time = elapsed / successful if successful > 0 else 0
            eta = avg_time * (args.num_images - i - 1)
            logger.info(
                f"[{i+1}/{args.num_images}] {filename} | "
                f"Avg: {avg_time:.1f}s/img | ETA: {eta/60:.0f}m"
            )

            # Periodic save
            if (i + 1) % args.save_every == 0:
                with open(output_dir / "captions.json", "w") as f:
                    json.dump(captions, f, indent=2)
                logger.info(f"Saved {len(captions)} captions to captions.json")

        except Exception as e:
            logger.error(f"[{i+1}/{args.num_images}] Failed: {e}")
            failed += 1
            continue

    # Final save
    with open(output_dir / "captions.json", "w") as f:
        json.dump(captions, f, indent=2)

    metadata["successful"] = successful
    metadata["failed"] = failed
    metadata["total_time_s"] = time.time() - t_start
    metadata["avg_time_per_image_s"] = metadata["total_time_s"] / successful if successful > 0 else 0

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=" * 60)
    logger.info("Generation Complete")
    logger.info("=" * 60)
    logger.info(f"Successful: {successful}/{args.num_images}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total time: {metadata['total_time_s']/60:.1f} min")
    logger.info(f"Avg time/image: {metadata['avg_time_per_image_s']:.1f}s")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Captions: {output_dir / 'captions.json'}")


if __name__ == "__main__":
    main()
