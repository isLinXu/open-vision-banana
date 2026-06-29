#!/usr/bin/env python3
"""Prepare (image, depth_rgb) paired dataset for Vision Banana training.

Uses DPT/Depth-Anything via transformers to generate relative depth maps
from COCO128 images, then converts to RGB visualisations using the project's
DepthRGBMapper (cube-edge colormap + power transform).

Output layout (matches VisionTaskDataset expectations):
  data/vision_banana_depth/
    images/    ← 128 input RGB images (256×256)
    targets/   ← 128 depth RGB visualisations (256×256)
    metadata.json
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# HuggingFace mirror
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_depth")

# ─── Config ──────────────────────────────────────────────────────────────
COCO_DIR = Path("/Users/gatilin/PycharmProjects/open-vision-banana/data/coco128/images")
OUTPUT_DIR = Path("/Users/gatilin/PycharmProjects/open-vision-banana/data/vision_banana_depth")
RESOLUTION = 256
DEPTH_MODEL = "LiheYoung/depth-anything-small-hf"  # ~100MB, MPS-compatible
SEED = 42


def load_depth_model():
    """Load Depth-Anything-Small via transformers."""
    from transformers import DPTForDepthEstimation, DPTImageProcessor

    logger.info(f"Loading depth model: {DEPTH_MODEL}")
    model = DPTForDepthEstimation.from_pretrained(
        DEPTH_MODEL, torch_dtype=torch.float32
    )
    processor = DPTImageProcessor.from_pretrained(DEPTH_MODEL)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device).eval()
    logger.info(f"Depth model loaded on {device}")
    return model, processor, device


def estimate_depth(
    image: Image.Image,
    model,
    processor,
    device: torch.device,
) -> np.ndarray:
    """Run depth estimation on a single image.

    Returns relative depth map [H, W] in meters (normalised to [0.1, 10.0]).
    """
    # Prepare input
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # DPT returns predicted_depth [1, 1, H, W]
        depth = outputs.predicted_depth

    # Interpolate to target resolution
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(0) if depth.dim() == 3 else depth,
        size=(RESOLUTION, RESOLUTION),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    # Normalise to metric range [0.1, 10.0] metres
    depth_np = depth.cpu().numpy()
    d_min, d_max = depth_np.min(), depth_np.max()
    if d_max - d_min > 1e-8:
        depth_np = (depth_np - d_min) / (d_max - d_min)  # [0, 1]
    else:
        depth_np = np.zeros_like(depth_np)
    depth_np = 0.1 + depth_np * 9.9  # → [0.1, 10.0] metres

    return depth_np.astype(np.float32)


def depth_to_rgb(depth_np: np.ndarray) -> Image.Image:
    """Convert depth map to RGB visualisation using DepthRGBMapper."""
    from open_vision_banana.core.depth_mapper import DepthRGBMapper

    mapper = DepthRGBMapper(colormap="cube_edge", lam=-3.0, c=10.0 / 3.0)

    # [H, W] → [1, 1, H, W]
    depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)
    rgb_tensor = mapper.depth_to_rgb(depth_tensor)  # [1, 3, H, W] in [0, 1]

    # Convert to PIL
    rgb_np = (rgb_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(rgb_np)


def main():
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "images").mkdir(exist_ok=True)
    (OUTPUT_DIR / "targets").mkdir(exist_ok=True)

    # Load depth model
    model, processor, device = load_depth_model()

    # Get all COCO128 images
    image_paths = sorted(COCO_DIR.glob("*.jpg"))
    logger.info(f"Found {len(image_paths)} images in {COCO_DIR}")

    metadata = []
    t_start = time.time()

    for i, img_path in enumerate(image_paths):
        t0 = time.time()

        # Load and resize input image
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)

        # Estimate depth
        depth_np = estimate_depth(img_resized, model, processor, device)

        # Convert depth to RGB visualisation
        depth_rgb = depth_to_rgb(depth_np)

        # Save pair
        stem = img_path.stem
        img_resized.save(OUTPUT_DIR / "images" / f"{stem}.jpg", quality=95)
        depth_rgb.save(OUTPUT_DIR / "targets" / f"{stem}.jpg", quality=95)

        elapsed = time.time() - t0
        logger.info(
            f"[{i+1}/{len(image_paths)}] {stem} | "
            f"depth range: [{depth_np.min():.2f}, {depth_np.max():.2f}]m | "
            f"{elapsed:.1f}s"
        )

        metadata.append({
            "filename": f"{stem}.jpg",
            "depth_min": float(depth_np.min()),
            "depth_max": float(depth_np.max()),
        })

    total_time = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset prepared: {len(metadata)} pairs")
    logger.info(f"  Images:  {OUTPUT_DIR / 'images'}")
    logger.info(f"  Targets: {OUTPUT_DIR / 'targets'}")
    logger.info(f"  Total time: {total_time:.0f}s ({total_time/len(metadata):.1f}s/image)")

    # Save metadata
    meta_path = OUTPUT_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "task": "depth",
            "resolution": RESOLUTION,
            "depth_model": DEPTH_MODEL,
            "colormap": "cube_edge",
            "lam": -3.0,
            "c": 3.333,
            "num_samples": len(metadata),
            "samples": metadata,
        }, f, indent=2)
    logger.info(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
