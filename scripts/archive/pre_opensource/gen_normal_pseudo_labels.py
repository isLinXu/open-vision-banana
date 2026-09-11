#!/usr/bin/env python3
"""Generate high-quality normal pseudo-labels from depth maps.

Uses depth-to-normal conversion for generating surface normal maps.
This is a fallback when Marigold/DSINE models are not available.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_image(path: str, size: int = 512) -> Image.Image:
    """Load and resize image."""
    img = Image.open(path).convert("RGB")
    # Keep aspect ratio, resize so shorter side = size
    w, h = img.size
    if w < h:
        new_w = size
        new_h = int(h * size / w)
    else:
        new_h = size
        new_w = int(w * size / h)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


def load_depth(path: str, size: int = 512) -> np.ndarray:
    """Load depth map and resize."""
    img = Image.open(path).convert("L")
    w, h = img.size
    if w < h:
        new_w = size
        new_h = int(h * size / w)
    else:
        new_h = size
        new_w = int(w * size / h)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def depth_to_normal(depth: np.ndarray) -> np.ndarray:
    """Convert depth map to surface normal map.
    
    Uses gradient-based method to compute normals from depth.
    Returns normal map in [-1, 1] range.
    """
    # Compute gradients
    dy, dx = np.gradient(depth)
    
    # Normalize depth to [0, 1]
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max - depth_min > 0:
        depth_norm = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = depth
    
    # Recompute gradients on normalized depth
    dy, dx = np.gradient(depth_norm)
    
    # Scale gradients (adjust for depth scale)
    scale = 5.0
    dx = dx * scale
    dy = dy * scale
    
    # Compute normals
    # Normal = [-dx, -dy, 1] normalized
    normal_x = -dx
    normal_y = -dy
    normal_z = np.ones_like(dx)
    
    # Normalize
    norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_x = normal_x / norm
    normal_y = normal_y / norm
    normal_z = normal_z / norm
    
    # Stack [H, W, 3]
    normal = np.stack([normal_x, normal_y, normal_z], axis=-1)
    
    return normal


def normal_to_rgb(normal: np.ndarray) -> np.ndarray:
    """Convert normal map from [-1, 1] to [0, 255] RGB."""
    normal = np.clip(normal, -1.0, 1.0)
    rgb = ((normal + 1.0) / 2.0 * 255.0).astype(np.uint8)
    return rgb


def find_depth_path(input_path: str) -> str:
    """Find corresponding depth map for an input image."""
    # This is a simplified version - in practice, you'd need to look up
    # the depth path from the manifest or a mapping file
    # For now, return None to indicate no depth available
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate normal pseudo-labels from depth")
    parser.add_argument("--input-manifest", required=True, type=Path, help="Input pairs JSONL")
    parser.add_argument("--depth-manifest", type=Path, help="Depth pairs JSONL (for depth-to-normal)")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for normal maps")
    parser.add_argument("--output-manifest", required=True, type=Path, help="Output manifest JSONL")
    parser.add_argument("--resolution", type=int, default=512, help="Processing resolution")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process (None=all)")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load input manifest
    print(f"Loading manifest from: {args.input_manifest}")
    pairs = []
    with open(args.input_manifest) as f:
        for line in f:
            pairs.append(json.loads(line))
    
    # Filter normal pairs
    normal_pairs = [p for p in pairs if p["task"] == "normal"]
    print(f"Total normal pairs: {len(normal_pairs)}")
    
    if args.max_samples:
        normal_pairs = normal_pairs[:args.max_samples]
        print(f"Processing first {args.max_samples} samples")
    
    # Load depth manifest if available
    depth_pairs = {}
    if args.depth_manifest and args.depth_manifest.exists():
        print(f"Loading depth manifest from: {args.depth_manifest}")
        with open(args.depth_manifest) as f:
            for line in f:
                p = json.loads(line)
                if p["task"] == "depth":
                    # Use input path as key
                    depth_pairs[p["input"]] = p["target"]
        print(f"Loaded {len(depth_pairs)} depth pairs")
    
    # Process pairs
    output_pairs = []
    success_count = 0
    fail_count = 0
    no_depth_count = 0
    
    for idx, pair in enumerate(tqdm(normal_pairs, desc="Generating normals")):
        input_path = pair["input"]
        
        if not os.path.exists(input_path):
            fail_count += 1
            continue
        
        try:
            # Load image
            img = load_image(input_path, args.resolution)
            
            # Try to find depth map
            depth_path = depth_pairs.get(input_path)
            
            if depth_path and os.path.exists(depth_path):
                # Use depth-to-normal
                depth = load_depth(depth_path, args.resolution)
                normal = depth_to_normal(depth)
            else:
                # Fallback: use gradient-based normal from RGB
                # This is lower quality but better than uniform purple
                img_arr = np.array(img, dtype=np.float32) / 255.0
                # Convert to grayscale
                gray = np.mean(img_arr, axis=2)
                normal = depth_to_normal(gray)
                no_depth_count += 1
            
            # Convert to RGB
            normal_rgb = normal_to_rgb(normal)
            
            # Save
            output_filename = Path(input_path).stem + "_normal_v2.png"
            output_path = args.output_dir / output_filename
            Image.fromarray(normal_rgb).save(output_path)
            
            # Create output pair
            output_pair = pair.copy()
            output_pair["target"] = str(output_path)
            output_pair["target_version"] = "v2_depth2normal"
            output_pairs.append(output_pair)
            
            success_count += 1
            
        except Exception as e:
            fail_count += 1
            if idx < 10:
                print(f"  Error processing {input_path}: {e}")
    
    # Save output manifest
    print(f"\nSaving output manifest to: {args.output_manifest}")
    with open(args.output_manifest, "w") as f:
        for pair in output_pairs:
            f.write(json.dumps(pair) + "\n")
    
    print(f"\nGeneration complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  No depth (used RGB gradient): {no_depth_count}")
    print(f"  Output: {args.output_manifest}")


if __name__ == "__main__":
    main()
