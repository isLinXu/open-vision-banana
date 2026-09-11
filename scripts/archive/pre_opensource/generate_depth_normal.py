#!/usr/bin/env python3
"""Generate COCO and Object365 depth/normal targets using Depth-Anything-Small.

Optimized for macOS MPS (Apple Silicon) - ~10x faster than CPU.
Also works on CUDA and CPU (fallback).

Usage:
    python generate_depth_normal.py --datasets-root /path/to/datasets --mode all
    python generate_depth_normal.py --mode coco-only
    python generate_depth_normal.py --mode object365-only
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# HuggingFace mirror (in case of China network)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

MODEL_NAME = "LiheYoung/depth-anything-small-hf"
BATCH_SIZE = 4  # MPS can handle small batches


def load_model():
    """Load Depth-Anything-Small model."""
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor

    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    # Select device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("WARNING: Using CPU - will be ~10x slower")

    model = model.to(device).eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, processor, device


def estimate_depth(model, processor, device, image):
    """Run depth estimation on a single image.

    Returns depth map [H, W] as numpy float32.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        depth = outputs.predicted_depth  # [1, H_proc, W_proc]

    # Interpolate to original image size
    w, h = image.size  # PIL: (width, height)
    if depth.dim() == 3:
        depth = depth.unsqueeze(0)  # [1, 1, H, W]
    elif depth.dim() == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)

    depth_resized = torch.nn.functional.interpolate(
        depth, size=(h, w), mode="bilinear", align_corners=False
    ).squeeze().cpu().numpy()

    return depth_resized


def depth_to_grayscale(depth_np):
    """Convert depth to grayscale PNG (0-255, closer=brighter).

    Args:
        depth_np: [H, W] float32 depth values

    Returns:
        PIL Image in L mode (grayscale)
    """
    d_min, d_max = depth_np.min(), depth_np.max()
    if d_max - d_min > 1e-8:
        depth_norm = (depth_np - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_np)

    # Closer (lower depth) = brighter (higher pixel value)
    depth_vis = ((1 - depth_norm) * 255).astype(np.uint8)
    return Image.fromarray(depth_vis, mode='L')


def depth_to_normal_rgb(depth_np):
    """Compute surface normals from depth and encode as RGB.

    Args:
        depth_np: [H, W] float32 depth values (0-1 normalized, 0=far, 1=close)

    Returns:
        PIL Image in RGB mode
    """
    # Compute gradients
    dzdx = np.zeros_like(depth_np)
    dzdy = np.zeros_like(depth_np)
    dzdx[:, 1:-1] = (depth_np[:, 2:] - depth_np[:, :-2]) / 2
    dzdy[1:-1, :] = (depth_np[2:, :] - depth_np[:-2, :]) / 2

    # Surface normal: N = (-dz/dx, -dz/dy, 1) normalized
    norm = np.sqrt(dzdx**2 + dzdy**2 + 1.0)
    nx = -dzdx / norm
    ny = -dzdy / norm
    nz = 1.0 / norm

    # Encode to RGB: [-1, 1] -> [0, 255]
    normal_rgb = np.stack([
        ((nx + 1) * 127.5).astype(np.uint8),
        ((ny + 1) * 127.5).astype(np.uint8),
        ((nz + 1) * 127.5).astype(np.uint8)
    ], axis=-1)

    return Image.fromarray(normal_rgb, mode='RGB')


def generate_coco_depth_normal(model, processor, device, datasets_root):
    """Generate COCO depth and normal targets."""
    coco_dir = Path(datasets_root) / "coco2017"
    snvc_dir = Path(datasets_root) / "snvc50m"

    depth_dir = coco_dir / "depth"
    normal_dir = coco_dir / "normal"
    depth_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)

    # Collect all unique input images from both JSONLs
    all_images = set()
    for jsonl_name in ["coco_depth_sampled.jsonl", "coco_normal_sampled.jsonl"]:
        jsonl_path = snvc_dir / jsonl_name
        if not jsonl_path.exists():
            continue
        with open(jsonl_path) as f:
            for line in f:
                s = json.loads(line)
                for img in s.get('image', []):
                    if 'depth' not in img and 'normal' not in img:
                        all_images.add(img)

    print(f"\n=== COCO Depth/Normal ===")
    print(f"Unique input images: {len(all_images)}")

    # Filter to images that need processing
    to_process = []
    for img_path in sorted(all_images):
        img_id = Path(img_path).stem  # e.g., "000000109454"
        depth_path = depth_dir / f"{img_id}_depth.png"
        normal_path = normal_dir / f"{img_id}_normal.png"

        # Need to generate if either depth or normal is missing
        if not depth_path.exists() or not normal_path.exists():
            full_path = Path(datasets_root) / img_path
            if full_path.exists():
                to_process.append((img_path, depth_path, normal_path, full_path))

    print(f"To process: {len(to_process)}")
    if not to_process:
        print("All depth/normal targets already exist!")
        return

    t_start = time.time()
    for i, (img_path, depth_path, normal_path, full_path) in enumerate(to_process):
        try:
            img = Image.open(full_path).convert("RGB")
            depth_np = estimate_depth(model, processor, device, img)

            # Save depth (grayscale, closer=brighter)
            if not depth_path.exists():
                depth_img = depth_to_grayscale(depth_np)
                depth_img.save(depth_path)

            # Save normal (RGB, XYZ channels)
            if not normal_path.exists():
                # Normalize depth for normal computation
                d_min, d_max = depth_np.min(), depth_np.max()
                if d_max - d_min > 1e-8:
                    depth_norm = (depth_np - d_min) / (d_max - d_min)
                else:
                    depth_norm = np.zeros_like(depth_np)
                normal_img = depth_to_normal_rgb(depth_norm)
                normal_img.save(normal_path)

        except Exception as e:
            print(f"  ERROR on {img_path}: {e}")
            continue

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(to_process) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(to_process)}] {elapsed:.0f}s, {rate:.1f} img/s, ETA {eta:.0f}s")

    total = time.time() - t_start
    print(f"Done! Generated {len(to_process)} depth+normal pairs in {total:.0f}s")


def generate_object365_depth(model, processor, device, datasets_root):
    """Generate Object365 depth targets for available input images."""
    obj365_dir = Path(datasets_root) / "Objects365"
    snvc_dir = Path(datasets_root) / "snvc50m"
    depth_output_dir = Path(datasets_root) / "object365" / "depth"
    depth_output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = snvc_dir / "object365_total_unified_depth_edit_regenerated_sampled.jsonl"
    if not jsonl_path.exists():
        print("Object365 depth JSONL not found, skipping")
        return

    with open(jsonl_path) as f:
        samples = [json.loads(line) for line in f]

    # Collect images to process
    to_process = []
    for s in samples:
        img_path = s['image'][0]  # "patchXX/objects365_v2_xxx.jpg"
        depth_target = s['image'][1]  # "object365/depth/patchXX*xxx_depth.png"

        # Convert * to _ in filename
        depth_filename = Path(depth_target).name.replace('*', '_')
        depth_path = depth_output_dir / depth_filename

        # Check if input image exists
        patch, filename = img_path.split('/')
        input_path = obj365_dir / patch / filename

        if input_path.exists() and not depth_path.exists():
            to_process.append((input_path, depth_path))

    print(f"\n=== Object365 Depth ===")
    print(f"Input images available: {len(to_process)}/{len(samples)}")
    if not to_process:
        print("All available depth targets already exist!")
        return

    t_start = time.time()
    for i, (input_path, depth_path) in enumerate(to_process):
        try:
            img = Image.open(input_path).convert("RGB")
            depth_np = estimate_depth(model, processor, device, img)
            depth_img = depth_to_grayscale(depth_np)
            depth_img.save(depth_path)
        except Exception as e:
            print(f"  ERROR on {input_path}: {e}")
            continue

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(to_process) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(to_process)}] {elapsed:.0f}s, {rate:.1f} img/s, ETA {eta:.0f}s")

    total = time.time() - t_start
    print(f"Done! Generated {len(to_process)} depth targets in {total:.0f}s")


def main():
    parser = argparse.ArgumentParser(description="Generate depth/normal targets")
    parser.add_argument("--datasets-root", default="/Users/gatilin/MyWork/datasets",
                        help="Root directory of datasets")
    parser.add_argument("--mode", choices=["all", "coco-only", "object365-only"],
                        default="all", help="What to generate")
    args = parser.parse_args()

    model, processor, device = load_model()

    if args.mode in ("all", "coco-only"):
        generate_coco_depth_normal(model, processor, device, args.datasets_root)

    if args.mode in ("all", "object365-only"):
        generate_object365_depth(model, processor, device, args.datasets_root)

    print("\n============================================================")
    print("All generation complete!")
    print("============================================================")


if __name__ == "__main__":
    main()
