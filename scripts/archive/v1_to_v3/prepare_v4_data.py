#!/usr/bin/env python3
"""Step 1 — Data preparation for Vision Banana v4 MPS verification.

This script produces two datasets:

  1. **Pseudo depth labels** — uses Depth-Anything-V2-Small (already in HF cache)
     to generate depth maps for COCO images.  We combine the existing 428 distilled
     samples with newly labelled COCO val2017 images to reach ~2K paired samples.

  2. **Synthetic generation data** — uses the base model (SDXL or SSD-1B) to
     self-generate 10K-20K text-to-image samples, ensuring the 9:1
     generation-to-vision ratio is truly satisfied.

Key design decisions (from the v4 strategy):
  - Depth-Anything V2-Small runs comfortably on MPS (only ~25M params)
  - SDXL @ float16 ≈ 7 GB — fits in 32 GB unified memory
  - Output: structured directory consumable by the v4 training script

Usage:
  python scripts/prepare_v4_data.py --num-depth 2000 --num-synthetic 10000
  python scripts/prepare_v4_data.py --skip-synthetic   # only depth labels
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
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vb_v4_data")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Existing distilled dataset (428 samples, already paired)
EXISTING_DISTILL_DIR = DATA_DIR / "vb_distill_428"
# COCO val2017 raw images (need to unzip first)
COCO_VAL_RAW = DATA_DIR / "coco_val2017_raw"
# COCO128 (128 images with captions, used for generation data source)
COCO128_DIR = DATA_DIR / "coco128"

# Output directories
DEPTH_OUTPUT_DIR = DATA_DIR / "vb_v4_depth_2k"
SYNTHETIC_OUTPUT_DIR = DATA_DIR / "vb_v4_synthetic"

RESOLUTION = 256  # Step 2 resolution; will be increased to 512 for Step 3


# ─── Depth-Anything V2 pseudo-labelling ───────────────────────────────────

def load_depth_anything(device: str):
    """Load Depth-Anything-V2-Small from HF cache."""
    from transformers import AutoModelForDepthEstimation, AutoProcessor

    model_name = "LiheYoung/depth-anything-small-hf"
    logger.info(f"Loading Depth-Anything-V2-Small from {model_name} ...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).to(device)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Depth-Anything loaded: {param_count:.1f}M params on {device}")
    return model, processor


@torch.no_grad()
def generate_depth_map(
    model, processor, image: Image.Image, device: str
) -> Image.Image:
    """Run Depth-Anything inference, return RGB depth visualization."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    # Depth-Anything outputs predicted_depth in [0, 1] (relative depth)
    depth = outputs.predicted_depth.squeeze().cpu().numpy()
    # Normalize to [0, 255] for visualization
    depth_min, depth_max = depth.min(), depth.max()
    if depth_max - depth_min > 1e-6:
        depth_norm = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = np.zeros_like(depth)
    # Apply spectral colormap (dark = near, bright = far, matching paper convention)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    Image.fromarray(depth_uint8)
    # Apply 'nipy_spectral' colormap via matplotlib-free approach
    depth_rgb = apply_spectral_colormap(depth_uint8)
    return Image.fromarray(depth_rgb)


def apply_spectral_colormap(depth_uint8: np.ndarray) -> np.ndarray:
    """Apply a spectral colormap to a uint8 depth map → RGB [H, W, 3]."""
    # Simple but effective: use a precomputed lookup table
    # This mimics 'nipy_spectral' without matplotlib dependency
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        # Spectral-like: blue → cyan → green → yellow → red
        if t < 0.25:
            r, g, b = 0, int(255 * t * 4), int(255 * (1 - t * 4) * 0.5)
        elif t < 0.5:
            r, g, b = 0, 255, int(255 * (0.5 - t) * 4)
        elif t < 0.75:
            r, g, b = int(255 * (t - 0.5) * 4), 255, 0
        else:
            r, g, b = 255, int(255 * (1 - (t - 0.75) * 4)), 0
        lut[i] = [r, g, b]
    return lut[depth_uint8]


def prepare_depth_dataset(
    device: str,
    num_samples: int = 2000,
    resolution: int = 256,
) -> None:
    """Generate pseudo depth labels using Depth-Anything-V2."""
    logger.info(f"=== Step 1a: Depth label generation (target: {num_samples}) ===")

    output_dir = DEPTH_OUTPUT_DIR
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "targets").mkdir(parents=True, exist_ok=True)

    # 1) Copy existing 428 distilled samples (already clean)
    existing_count = 0
    if EXISTING_DISTILL_DIR.exists():
        import shutil

        for img_path in sorted((EXISTING_DISTILL_DIR / "images").glob("*.jpg")):
            tgt_path = EXISTING_DISTILL_DIR / "targets" / (img_path.stem + ".png")
            if not tgt_path.exists():
                tgt_path = EXISTING_DISTILL_DIR / "targets" / img_path.name
            if tgt_path.exists():
                dst_img = output_dir / "images" / img_path.name
                dst_tgt = output_dir / "targets" / (img_path.stem + ".png")
                if not dst_img.exists():
                    shutil.copy2(img_path, dst_img)
                    shutil.copy2(tgt_path, dst_tgt)
                    existing_count += 1
        logger.info(f"Copied {existing_count} existing distilled samples")

    if existing_count >= num_samples:
        logger.info(f"Already have {existing_count} >= {num_samples}, skipping generation")
        return

    # 2) Generate pseudo labels for additional COCO images
    model, processor = load_depth_anything(device)

    # Collect candidate images from COCO val2017 + COCO128
    candidate_images = []
    # COCO128 images
    if COCO128_DIR.exists():
        for p in sorted((COCO128_DIR / "images").glob("*.jpg")):
            candidate_images.append(("coco128", p))
    # COCO val2017 (if unzipped)
    val2017_img_dir = COCO_VAL_RAW / "val2017"
    if not val2017_img_dir.exists():
        # Try unzipping
        zip_path = COCO_VAL_RAW / "val2017.zip"
        if zip_path.exists():
            logger.info(f"Unzipping {zip_path} ...")
            import zipfile

            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(COCO_VAL_RAW)
    if val2017_img_dir.exists():
        for p in sorted(val2017_img_dir.glob("*.jpg")):
            candidate_images.append(("val2017", p))

    logger.info(f"Found {len(candidate_images)} candidate images for pseudo-labelling")

    remaining = num_samples - existing_count
    selected = candidate_images[:remaining]
    logger.info(f"Will generate {len(selected)} pseudo depth maps")

    success_count = 0
    t_start = time.time()

    for i, (source, img_path) in enumerate(selected):
        try:
            img = Image.open(img_path).convert("RGB")
            img_resized = img.resize((resolution, resolution), Image.LANCZOS)

            # Generate depth map
            depth_rgb = generate_depth_map(model, processor, img, device)
            depth_resized = depth_rgb.resize((resolution, resolution), Image.LANCZOS)

            # Save with consistent naming
            name = f"{source}_{img_path.stem}"
            img_resized.save(output_dir / "images" / f"{name}.jpg", quality=95)
            depth_resized.save(output_dir / "targets" / f"{name}.png")

            success_count += 1
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_start
                eta = elapsed / (i + 1) * (len(selected) - i - 1)
                logger.info(
                    f"  Depth: {i+1}/{len(selected)} | "
                    f"{success_count} ok | {elapsed:.0f}s | ETA: {eta:.0f}s"
                )
        except Exception as e:
            logger.warning(f"  Failed {img_path.name}: {e}")

    total = existing_count + success_count
    elapsed = time.time() - t_start
    logger.info(
        f"Depth dataset ready: {total} samples "
        f"({existing_count} existing + {success_count} new) in {elapsed:.0f}s"
    )


# ─── Synthetic generation data ─────────────────────────────────────────────


def prepare_synthetic_data(
    device: str,
    num_samples: int = 10000,
    resolution: int = 256,
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    use_ssd1b: bool = False,
) -> None:
    """Self-generate synthetic image-caption pairs using SDXL or SSD-1B."""
    from diffusers import StableDiffusionXLPipeline

    logger.info(f"=== Step 1b: Synthetic data generation (target: {num_samples}) ===")

    output_dir = SYNTHETIC_OUTPUT_DIR
    (output_dir / "images").mkdir(parents=True, exist_ok=True)

    # Check how many already exist
    existing = list((output_dir / "images").glob("*.jpg"))
    if len(existing) >= num_samples:
        logger.info(f"Already have {len(existing)} >= {num_samples}, skipping")
        return

    # Load SDXL pipeline
    dtype = torch.float16
    logger.info(f"Loading {model_id} @ {dtype} ...")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)

    # Load diverse prompts (we use COCO captions as a base, plus generic prompts)
    prompts = load_generation_prompts()
    logger.info(f"Loaded {len(prompts)} diverse prompts for generation")

    start_idx = len(existing)
    batch_size = 4  # Conservative for 32GB MPS
    t_start = time.time()

    for batch_start in range(start_idx, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        actual_batch = batch_end - batch_start

        # Sample diverse prompts
        batch_prompts = [prompts[np.random.randint(len(prompts))] for _ in range(actual_batch)]

        try:
            images = pipe(
                prompt=batch_prompts,
                num_inference_steps=4,  # SDXL turbo-like fast generation
                guidance_scale=0.0,     # schnell-style: no CFG
                height=resolution,
                width=resolution,
            ).images

            for j, (img, _prompt) in enumerate(zip(images, batch_prompts, strict=False)):
                idx = batch_start + j
                img.save(output_dir / "images" / f"synth_{idx:05d}.jpg", quality=95)

            elapsed = time.time() - t_start
            done = batch_end - start_idx
            eta = elapsed / max(done, 1) * (num_samples - batch_end)
            if (batch_start - start_idx) % 20 == 0:
                logger.info(
                    f"  Synthetic: {batch_end}/{num_samples} | "
                    f"{elapsed:.0f}s | ETA: {eta/60:.1f}m"
                )
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"  OOM at batch {batch_start}, reducing batch size")
            batch_size = max(1, batch_size // 2)
        except Exception as e:
            logger.warning(f"  Batch {batch_start} failed: {e}")

    # Save metadata
    final_count = len(list((output_dir / "images").glob("*.jpg")))
    elapsed = time.time() - t_start
    logger.info(
        f"Synthetic dataset ready: {final_count} images in {elapsed:.0f}s"
    )

    # Write captions.json (prompt for each generated image)
    captions = {}
    for i in range(final_count):
        captions[f"synth_{i:05d}.jpg"] = prompts[i % len(prompts)]
    with open(output_dir / "captions.json", "w") as f:
        json.dump(
            [{"filename": k, "caption": v} for k, v in captions.items()],
            f,
            indent=2,
        )


def load_generation_prompts() -> list[str]:
    """Load diverse text prompts for synthetic generation."""
    # Load COCO128 captions
    prompts = []
    coco_captions_path = COCO128_DIR / "captions.json"
    if coco_captions_path.exists():
        with open(coco_captions_path) as f:
            data = json.load(f)
            prompts.extend([d["caption"] for d in data])

    # Add generic diverse prompts
    generic_prompts = [
        "A high quality photograph of a landscape with mountains",
        "A portrait of a person in natural lighting",
        "An indoor scene of a modern living room",
        "A street view with buildings and people",
        "A close-up of a flower in a garden",
        "A scenic view of a lake at sunset",
        "A kitchen with modern appliances",
        "A park with trees and a walking path",
        "An office workspace with a desk and computer",
        "A city skyline at night",
        "A forest path with sunlight filtering through trees",
        "A beach scene with waves and sand",
        "A restaurant interior with tables and chairs",
        "A bedroom with a large bed and window",
        "A garden with various plants and flowers",
        "A snowy mountain landscape",
        "A desert scene with sand dunes",
        "A waterfall in a tropical forest",
        "A bridge over a river in a city",
        "A farm with fields and a barn",
    ]
    prompts.extend(generic_prompts)
    return prompts


# ─── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Prepare Vision Banana v4 data")
    parser.add_argument("--num-depth", type=int, default=2000,
                        help="Number of depth-labeled samples (default: 2000)")
    parser.add_argument("--num-synthetic", type=int, default=10000,
                        help="Number of synthetic generation samples (default: 10000)")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Image resolution (default: 256)")
    parser.add_argument("--skip-depth", action="store_true",
                        help="Skip depth label generation")
    parser.add_argument("--skip-synthetic", action="store_true",
                        help="Skip synthetic data generation")
    parser.add_argument("--use-ssd1b", action="store_true",
                        help="Use SSD-1B instead of SDXL (lower memory)")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"Targets: {args.num_depth} depth + {args.num_synthetic} synthetic")

    if not args.skip_depth:
        prepare_depth_dataset(device, args.num_depth, args.resolution)
    else:
        logger.info("Skipping depth label generation")

    if not args.skip_synthetic:
        model_id = (
            "segmind/SSD-1B" if args.use_ssd1b
            else "stabilityai/stable-diffusion-xl-base-1.0"
        )
        prepare_synthetic_data(
            device, args.num_synthetic, args.resolution, model_id, args.use_ssd1b
        )
    else:
        logger.info("Skipping synthetic data generation")

    logger.info("=== Data preparation complete ===")
    logger.info(f"  Depth dataset: {DEPTH_OUTPUT_DIR}")
    logger.info(f"  Synthetic dataset: {SYNTHETIC_OUTPUT_DIR}")

    # Print dataset stats
    depth_count = len(list((DEPTH_OUTPUT_DIR / "images").glob("*.jpg")))
    synth_count = len(list((SYNTHETIC_OUTPUT_DIR / "images").glob("*.jpg")))
    actual_ratio = depth_count / max(depth_count + synth_count, 1)
    logger.info(f"  Depth samples: {depth_count}")
    logger.info(f"  Synthetic samples: {synth_count}")
    logger.info(f"  Actual vision ratio: {actual_ratio:.2%} (target: 10%)")


if __name__ == "__main__":
    main()
