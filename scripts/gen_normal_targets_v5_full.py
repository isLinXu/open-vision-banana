#!/usr/bin/env python3
"""Generate normal targets for v5_full by computing surface normals from depth.

Uses Depth-Anything-V2-Small to estimate depth, then computes surface normals
via finite-difference gradients. Saves as RGB PNG.

Supports checkpointing: skips already-generated targets.
"""
import os, time, sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
from PIL import Image

DATA_ROOT = Path(os.environ.get("VB_DATA_ROOT",
    "/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v5_full"))
OUT_DIR = Path(os.environ.get("VB_NORMAL_OUT", str(DATA_ROOT / "targets" / "normal")))
IMG_SIZE = int(os.environ.get("VB_IMG_SIZE", "256"))
BATCH = int(os.environ.get("VB_NORMAL_BATCH", "1"))

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_model():
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    print(f"Loading Depth-Anything-V2-Small on {DEVICE}...")
    model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf").to(DEVICE).eval()
    processor = AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf")
    return model, processor


def depth_to_normals(depth_np):
    """Compute surface normals from depth map via finite differences.
    depth_np: [H, W] float32, closer=larger value
    Returns: [H, W, 3] uint8 RGB (nx,ny,nz in [0,255])
    """
    H, W = depth_np.shape

    # Normalize depth
    d_min, d_max = depth_np.min(), depth_np.max()
    if d_max - d_min > 1e-8:
        depth_norm = (depth_np - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_np)

    # Gradients (central diff)
    dzdx = np.zeros_like(depth_norm)
    dzdy = np.zeros_like(depth_norm)
    dzdx[:, 1:-1] = (depth_norm[:, 2:] - depth_norm[:, :-2]) / 2.0
    dzdy[1:-1, :] = (depth_norm[2:, :] - depth_norm[:-2, :]) / 2.0

    # Surface normal: N = (-dz/dx, -dz/dy, 1) normalized
    norm = np.sqrt(dzdx**2 + dzdy**2 + 1.0)
    nx = -dzdx / norm
    ny = -dzdy / norm
    nz = 1.0 / norm

    # Encode to RGB: [-1, 1] → [0, 255]
    rgb = np.stack([
        ((nx + 1.0) * 127.5).astype(np.uint8),
        ((ny + 1.0) * 127.5).astype(np.uint8),
        ((nz + 1.0) * 127.5).astype(np.uint8),
    ], axis=-1)
    return rgb


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model, processor = load_model()

    # Collect depth images (COCO photos) as input
    img_dir = DATA_ROOT / "images" / "depth"
    image_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    print(f"Input images: {len(image_paths)}")

    # Filter: skip already generated
    to_process = []
    for p in image_paths:
        out_path = OUT_DIR / f"{p.stem}.png"
        if not out_path.exists():
            to_process.append((p, out_path))

    if not to_process:
        print("All normal targets already exist!")
        return

    print(f"To generate: {len(to_process)}")
    t0 = time.time()

    for i, (img_path, out_path) in enumerate(to_process):
        try:
            im = Image.open(img_path).convert("RGB")
            h_orig, w_orig = im.height, im.width

            # Depth estimation
            inputs = processor(images=im, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                depth = model(**inputs).predicted_depth  # [1, H, W]

            # Resize to original
            if depth.dim() == 3:
                depth = depth.unsqueeze(1)  # [1,1,H,W]
            depth_orig = torch.nn.functional.interpolate(
                depth, size=(h_orig, w_orig), mode="bilinear",
                align_corners=False).squeeze().cpu().numpy()

            # Compute normals → RGB
            normal_rgb = depth_to_normals(depth_orig)

            # Resize & save
            normal_img = Image.fromarray(normal_rgb).resize(
                (IMG_SIZE, IMG_SIZE), Image.BICUBIC)
            normal_img.save(out_path)

        except Exception as e:
            print(f"  ERROR [{img_path.name}]: {e}", flush=True)
            continue

        if (i + 1) % 200 == 0 or i < 3:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(to_process) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:>5d}/{len(to_process)}] "
                  f"{elapsed:.0f}s, {rate:.1f}img/s, ETA {eta/60:.0f}min",
                  flush=True)

    total = time.time() - t0
    n_done = sum(1 for _ in to_process)
    print(f"Done! Generated {n_done}/{len(to_process)} normal targets in {total:.0f}s "
          f"({total/60:.1f}min)")


if __name__ == "__main__":
    run()
