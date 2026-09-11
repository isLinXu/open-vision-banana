#!/usr/bin/env python3
"""Generate depth and normal targets for COCO + Object365 datasets.

Generates:
  - COCO depth:   coco2017/depth/XXXXXX_depth.png
  - COCO normal:  coco2017/normal/XXXXXX_normal.png
  - Object365 depth:  object365/depth/patchXX_objects365_v2_XXXXX_depth.png
  - Object365 normal: object365/normal/patchXX_objects365_v2_XXXXX_normal.png

Usage:
  python3 scripts/generate_all_depth_normal.py [--device cpu|cuda|mps] [--limit N]
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForDepthEstimation, AutoImageProcessor

DATASETS_ROOT = Path("/Users/gatilin/MyWork/datasets")
SNVC_ROOT = DATASETS_ROOT / "snvc50m"

def load_model(device):
    """Load Depth-Anything-Small model."""
    model_name = "LiheYoung/depth-anything-small-hf"
    print(f"Loading {model_name}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params on {device}")
    return model, processor


def generate_depth(model, processor, image_path, device):
    """Generate depth map (0-255 uint8, near=bright) from input image."""
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # Depth-Anything returns predicted_depth
        depth = outputs.predicted_depth  # (1, H, W)

    # Interpolate to original size
    h, w = img.size[1], img.size[0]
    depth = F.interpolate(depth.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze()

    # Normalize to 0-255 (near = bright = high value)
    depth = depth.cpu().numpy()
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)
    else:
        depth = np.zeros_like(depth)
    depth = (depth * 255).astype(np.uint8)

    return Image.fromarray(depth, mode="L")


def generate_normal_from_depth(depth_path):
    """Compute surface normals from depth map using gradient.

    Returns RGB image where R=dx, G=dy, B=depth (normalized).
    """
    depth_img = Image.open(depth_path).convert("L")
    d = np.array(depth_img, dtype=np.float32) / 255.0

    # Compute gradients
    dy, dx = np.gradient(d)

    # Normal vector: (-dx, -dy, 1) normalized
    nz = np.ones_like(d)
    norm = np.sqrt(dx**2 + dy**2 + 1)
    nx = -dx / norm
    ny = -dy / norm
    nz = nz / norm

    # Map to 0-255: (n + 1) / 2 * 255
    r = ((nx + 1) / 2 * 255).astype(np.uint8)
    g = ((ny + 1) / 2 * 255).astype(np.uint8)
    b = ((nz + 1) / 2 * 255).astype(np.uint8)

    normal = np.stack([r, g, b], axis=-1)
    return Image.fromarray(normal, mode="RGB")


def process_coco(model, processor, device, limit=None):
    """Generate COCO depth + normal targets."""
    # Find COCO depth JSONL (try multiple name patterns)
    jsonl_path = None
    for name in ["coco_depth_sampled.jsonl", "coco_total_unified_depth_edit_regenerated_sampled.jsonl"]:
        candidate = SNVC_ROOT / name
        if candidate.exists():
            jsonl_path = candidate
            break
    if not jsonl_path:
        # Fallback: search for any coco depth jsonl
        for name in os.listdir(SNVC_ROOT):
            if "coco" in name.lower() and "depth" in name.lower() and name.endswith("_sampled.jsonl"):
                jsonl_path = SNVC_ROOT / name
                break
    if not jsonl_path:
        print("COCO depth JSONL not found, skipping")
        return

    depth_dir = DATASETS_ROOT / "coco2017" / "depth"
    normal_dir = DATASETS_ROOT / "coco2017" / "normal"
    depth_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)

    # Load all COCO depth samples
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            imgs = d.get("image", [])
            if isinstance(imgs, str):
                imgs = [imgs]
            if len(imgs) >= 2:
                input_path = imgs[0]
                depth_target = imgs[1]
                # Extract image ID
                img_id = os.path.basename(input_path).replace(".jpg", "")
                depth_path = depth_dir / f"{img_id}_depth.png"
                normal_path = normal_dir / f"{img_id}_normal.png"
                input_full = DATASETS_ROOT / input_path

                if not depth_path.exists() and input_full.exists():
                    samples.append((str(input_full), str(depth_path), str(normal_path)))

    print(f"\nCOCO: {len(samples)} depth+normal targets to generate")
    if limit:
        samples = samples[:limit]
        print(f"  Limited to {limit}")

    t0 = time.time()
    for i, (inp, depth_out, normal_out) in enumerate(samples):
        try:
            # Generate depth
            depth_img = generate_depth(model, processor, inp, device)
            depth_img.save(depth_out)

            # Generate normal from depth
            normal_img = generate_normal_from_depth(depth_out)
            normal_img.save(normal_out)

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(samples) - i - 1) / rate if rate > 0 else 0
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  COCO [{i+1}/{len(samples)}] {rate:.1f} img/s ETA {eta:.0f}s")
        except Exception as e:
            print(f"  COCO ERROR [{i+1}] {inp}: {e}")

    print(f"  COCO done: {len(samples)} in {time.time()-t0:.0f}s")


def process_object365(model, processor, device, limit=None):
    """Generate Object365 depth + normal targets."""
    # Find Object365 depth JSONL
    jsonl_path = None
    for name in ["object365_total_unified_depth_edit_regenerated_sampled.jsonl", "object365_depth_sampled.jsonl"]:
        candidate = SNVC_ROOT / name
        if candidate.exists():
            jsonl_path = candidate
            break
    if not jsonl_path:
        for name in os.listdir(SNVC_ROOT):
            if "object365" in name.lower() and "depth" in name.lower() and name.endswith("_sampled.jsonl"):
                jsonl_path = SNVC_ROOT / name
                break
    if not jsonl_path:
        print("Object365 depth JSONL not found, skipping")
        return

    depth_dir = DATASETS_ROOT / "object365" / "depth"
    normal_dir = DATASETS_ROOT / "object365" / "normal"
    depth_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            imgs = d.get("image", [])
            if isinstance(imgs, str):
                imgs = [imgs]
            if len(imgs) >= 2:
                input_path = imgs[0]
                depth_target = imgs[1]
                # Fix path: replace * with _
                depth_target = depth_target.replace("*", "_")

                # Find input image
                input_full = None
                for prefix in ["Objects365/", "Objects365/extracted/", ""]:
                    candidate = DATASETS_ROOT / (prefix + input_path)
                    if candidate.exists():
                        input_full = str(candidate)
                        break

                if input_full:
                    depth_path = DATASETS_ROOT / depth_target
                    normal_path = str(depth_path).replace("/depth/", "/normal/").replace("_depth.png", "_normal.png")

                    if not depth_path.exists():
                        samples.append((input_full, str(depth_path), normal_path))

    print(f"\nObject365: {len(samples)} depth+normal targets to generate")
    if limit:
        samples = samples[:limit]
        print(f"  Limited to {limit}")

    t0 = time.time()
    for i, (inp, depth_out, normal_out) in enumerate(samples):
        try:
            os.makedirs(os.path.dirname(depth_out), exist_ok=True)
            os.makedirs(os.path.dirname(normal_out), exist_ok=True)

            depth_img = generate_depth(model, processor, inp, device)
            depth_img.save(depth_out)

            normal_img = generate_normal_from_depth(depth_out)
            normal_img.save(normal_out)

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(samples) - i - 1) / rate if rate > 0 else 0
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  O365 [{i+1}/{len(samples)}] {rate:.1f} img/s ETA {eta:.0f}s")
        except Exception as e:
            print(f"  O365 ERROR [{i+1}] {inp}: {e}")

    print(f"  Object365 done: {len(samples)} in {time.time()-t0:.0f}s")


def process_coco_normal(model, processor, device, limit=None):
    """Generate COCO normal targets for the normal JSONL."""
    # Find COCO normal JSONL
    jsonl_path = None
    for name in ["coco_normal_sampled.jsonl", "coco_total_unified_normal_edit_regenerated_sampled.jsonl"]:
        candidate = SNVC_ROOT / name
        if candidate.exists():
            jsonl_path = candidate
            break
    if not jsonl_path:
        for name in os.listdir(SNVC_ROOT):
            if "coco" in name.lower() and "normal" in name.lower() and name.endswith("_sampled.jsonl"):
                jsonl_path = SNVC_ROOT / name
                break
    if not jsonl_path:
        print("COCO normal JSONL not found, skipping")
        return

    normal_dir = DATASETS_ROOT / "coco2017" / "normal"
    normal_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            imgs = d.get("image", [])
            if isinstance(imgs, str):
                imgs = [imgs]
            if len(imgs) >= 2:
                input_path = imgs[0]
                normal_target = imgs[1]

                img_id = os.path.basename(input_path).replace(".jpg", "")
                normal_path = normal_dir / f"{img_id}_normal.png"
                input_full = DATASETS_ROOT / input_path

                if not normal_path.exists() and input_full.exists():
                    samples.append((str(input_full), str(normal_path)))

    print(f"\nCOCO Normal: {len(samples)} targets to generate")
    if limit:
        samples = samples[:limit]
        print(f"  Limited to {limit}")

    t0 = time.time()
    for i, (inp, normal_out) in enumerate(samples):
        try:
            # Generate depth first, then normal
            depth_img = generate_depth(model, processor, inp, device)
            # Save temp depth
            tmp_depth = "/tmp/_tmp_depth.png"
            depth_img.save(tmp_depth)

            normal_img = generate_normal_from_depth(tmp_depth)
            normal_img.save(normal_out)

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(samples) - i - 1) / rate if rate > 0 else 0
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  COCO-N [{i+1}/{len(samples)}] {rate:.1f} img/s ETA {eta:.0f}s")
        except Exception as e:
            print(f"  COCO-N ERROR [{i+1}] {inp}: {e}")

    print(f"  COCO Normal done: {len(samples)} in {time.time()-t0:.0f}s")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--limit", type=int, default=None, help="Max samples per dataset")
    parser.add_argument("--datasets", default="all", help="Comma-separated: coco,coco_normal,object365")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, processor = load_model(device)

    datasets = args.datasets.split(",") if args.datasets != "all" else ["coco", "coco_normal", "object365"]

    if "coco" in datasets:
        process_coco(model, processor, device, args.limit)
    if "coco_normal" in datasets:
        process_coco_normal(model, processor, device, args.limit)
    if "object365" in datasets:
        process_object365(model, processor, device, args.limit)

    print("\nAll done!")


if __name__ == "__main__":
    main()
