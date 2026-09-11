#!/usr/bin/env python3
"""
为 COCO 图片生成伪深度目标图（替换灰度占位符）。

使用 Depth-Anything-V2-Small-hf（通过 transformers pipeline）。
如果模型不可用，回退到 Marigold（通过 diffusers）。

将 data/vb_snvc_train/depth/targets/ 中的灰度占位符替换为真实深度图。
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("depth_gen")

DATA_DIR = Path("/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train")
DEPTH_IMG_DIR = DATA_DIR / "depth" / "images"
DEPTH_TGT_DIR = DATA_DIR / "depth" / "targets"
RESOLUTION = 384


def load_depth_anything():
    """Load Depth-Anything-V2-Small-hf via transformers pipeline."""
    from transformers import pipeline
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device,
    )
    logger.info(f"Depth-Anything loaded on {device}")
    return pipe, "depth_anything"


def load_marigold():
    """Load Marigold depth pipeline via diffusers."""
    from diffusers import MarigoldDepthPipeline
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-depth",
        torch_dtype=torch.float32,
    ).to(device)
    logger.info(f"Marigold loaded on {device}")
    return pipe, "marigold"


def depth_to_rgb(depth: np.ndarray) -> Image.Image:
    """Convert depth array to RGB visualization (turbo-like colormap)."""
    # Normalize to 0-255
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min < 1e-6:
        return Image.new("RGB", (depth.shape[1], depth.shape[0]), (128, 128, 128))

    normalized = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)

    # Simple colormap: blue (far) -> green -> red (near)
    rgb = np.zeros((*normalized.shape, 3), dtype=np.uint8)
    for i in range(256):
        mask = normalized == i
        # Colormap: blue -> cyan -> green -> yellow -> red
        t = i / 255.0
        if t < 0.25:
            r, g, b = 0, int(t * 4 * 255), 255
        elif t < 0.5:
            r, g, b = 0, 255, int(255 - (t - 0.25) * 4 * 255)
        elif t < 0.75:
            r, g, b = int((t - 0.5) * 4 * 255), 255, 0
        else:
            r, g, b = 255, int(255 - (t - 0.75) * 4 * 255), 0
        rgb[mask] = [r, g, b]

    return Image.fromarray(rgb)


def generate_with_depth_anything(pipe, images: list[Path], targets: list[Path]):
    """Generate depth maps using Depth-Anything pipeline."""
    from PIL import Image as PILImage

    for i, (img_path, tgt_path) in enumerate(zip(images, targets)):
        if i % 100 == 0:
            logger.info(f"  Processing {i}/{len(images)}...")

        try:
            img = PILImage.open(img_path).convert("RGB")
            result = pipe(img)

            # Get depth as numpy array
            if hasattr(result, "predicted_depth"):
                depth = result.predicted_depth.cpu().numpy()
            elif isinstance(result, dict) and "depth" in result:
                # depth is a PIL image
                depth = np.array(result["depth"]).astype(np.float32)
            else:
                depth = np.array(result).astype(np.float32)

            # Resize depth to target resolution
            depth_img = depth_to_rgb(depth)
            depth_img = depth_img.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
            depth_img.save(tgt_path, quality=95)
        except Exception as e:
            logger.warning(f"  Failed {img_path.name}: {e}")
            # Keep existing placeholder


def generate_with_marigold(pipe, images: list[Path], targets: list[Path]):
    """Generate depth maps using Marigold pipeline."""
    import torch

    for i, (img_path, tgt_path) in enumerate(zip(images, targets)):
        if i % 50 == 0:
            logger.info(f"  Processing {i}/{len(images)}...")

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)

            with torch.no_grad():
                result = pipe(img, num_inference_steps=10)

            # Marigold returns depth in [0, 1] range
            depth = np.array(result.depth[0]).astype(np.float32)
            if depth.max() <= 1.0:
                depth = depth * 255

            depth_rgb = depth_to_rgb(depth.squeeze())
            depth_rgb.save(tgt_path, quality=95)
        except Exception as e:
            logger.warning(f"  Failed {img_path.name}: {e}")


def main():
    print("=" * 60)
    print("Depth Target Generation (Pseudo-Labels)")
    print("=" * 60)

    # Collect all depth image/target pairs
    images = sorted(DEPTH_IMG_DIR.glob("*.jpg"))
    targets = [DEPTH_TGT_DIR / f.name for f in images]

    logger.info(f"Found {len(images)} depth images to process")

    if not images:
        logger.error("No depth images found!")
        return

    # Try Depth-Anything first, fall back to Marigold
    pipe = None
    method = None

    try:
        pipe, method = load_depth_anything()
    except Exception as e:
        logger.warning(f"Depth-Anything failed: {e}")
        logger.info("Falling back to Marigold...")
        try:
            pipe, method = load_marigold()
        except Exception as e2:
            logger.error(f"Marigold also failed: {e2}")
            logger.error("No depth model available. Cannot generate depth targets.")
            return

    logger.info(f"Using method: {method}")
    t0 = time.time()

    if method == "depth_anything":
        generate_with_depth_anything(pipe, images, targets)
    else:
        generate_with_marigold(pipe, images, targets)

    elapsed = time.time() - t0
    logger.info(f"Done! Processed {len(images)} images in {elapsed:.0f}s ({elapsed/len(images):.1f}s/image)")


if __name__ == "__main__":
    main()
