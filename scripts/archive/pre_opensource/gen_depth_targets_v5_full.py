#!/usr/bin/env python3
"""为 v5_full depth 图像批量生成 depth targets（伪标签）。

使用 Depth-Anything-V2-Small-hf (transformers pipeline)。
Fallback: Marigold (diffusers)。

输出: targets/depth/ 下与 images/depth/ 同名的 .png depth colormap 图。
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("depth_gen_v5")

DATA_ROOT = Path("/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v5_full")
DEPTH_IMG_DIR = DATA_ROOT / "images" / "depth"
DEPTH_TGT_DIR = DATA_ROOT / "targets" / "depth"
RESOLUTION = 256  # MAR-VB 默认 img_size=256


def load_depth_anything():
    from transformers import pipeline
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
    logger.info(f"Depth-Anything loaded on {device}")
    return pipe, "depth_anything"


def load_marigold():
    from diffusers import MarigoldDepthPipeline
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = MarigoldDepthPipeline.from_pretrained("prs-eth/marigold-depth", torch_dtype=torch.float32).to(device)
    logger.info(f"Marigold loaded on {device}")
    return pipe, "marigold"


def depth_to_rgb(depth: np.ndarray) -> Image.Image:
    """Normalized depth → turbo-like colormap RGB."""
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min < 1e-6:
        return Image.new("RGB", (depth.shape[1], depth.shape[0]), (128, 128, 128))
    normalized = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    rgb = np.zeros((*normalized.shape, 3), dtype=np.uint8)
    for i in range(256):
        mask = normalized == i
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


def generate(pipe, images: list[Path], targets: list[Path], method: str):
    import torch
    for i, (img_path, tgt_path) in enumerate(zip(images, targets)):
        if tgt_path.exists():
            continue  # skip already generated
        if i % 200 == 0:
            logger.info(f"  [{i}/{len(images)}] ({(i/len(images)*100):.1f}%)")

        try:
            img = Image.open(img_path).convert("RGB")
            if method == "depth_anything":
                result = pipe(img)
                if hasattr(result, "predicted_depth"):
                    depth = result.predicted_depth  # tensor
                    if hasattr(depth, "cpu"):
                        depth = depth.cpu().numpy()
                elif isinstance(result, dict) and "depth" in result:
                    depth = np.array(result["depth"]).astype(np.float32)
                else:
                    depth = np.array(result).astype(np.float32)
            else:  # marigold
                img_resized = img.resize((384, 384), Image.LANCZOS)
                with torch.no_grad():
                    result = pipe(img_resized, num_inference_steps=10)
                depth = np.array(result.depth[0]).astype(np.float32)
                if depth.max() <= 1.0:
                    depth = depth * 255

            depth_rgb = depth_to_rgb(depth)
            depth_rgb = depth_rgb.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
            depth_rgb.save(tgt_path, quality=95)
        except Exception as e:
            logger.warning(f"  FAIL {img_path.name}: {e}")


def main():
    logger.info(f"Loading depth images from {DEPTH_IMG_DIR}")
    # Support both .jpg and .png
    images = sorted(list(DEPTH_IMG_DIR.glob("*.jpg")) + list(DEPTH_IMG_DIR.glob("*.png")))
    logger.info(f"Found {len(images)} depth images")

    DEPTH_TGT_DIR.mkdir(parents=True, exist_ok=True)
    targets = [DEPTH_TGT_DIR / f"{img.stem}.png" for img in images]

    # Count already done
    done = sum(1 for t in targets if t.exists())
    logger.info(f"Already generated: {done}, to process: {len(images) - done}")

    if done == len(images):
        logger.info("All targets already exist!")
        return

    # Load model
    pipe, method = None, None
    try:
        pipe, method = load_depth_anything()
    except Exception as e:
        logger.warning(f"Depth-Anything failed: {e}, trying Marigold...")
        try:
            pipe, method = load_marigold()
        except Exception as e2:
            logger.error(f"Marigold also failed: {e2}")
            return

    t0 = time.time()
    generate(pipe, images, targets, method)
    elapsed = time.time() - t0
    processed = len(images) - done
    logger.info(f"Done: {processed} images in {elapsed:.0f}s ({elapsed/max(1,processed):.1f}s/img)")


if __name__ == "__main__":
    main()
