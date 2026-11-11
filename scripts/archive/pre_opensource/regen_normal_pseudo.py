#!/usr/bin/env python3
"""
Regenerate high-quality normal pseudo-labels with Marigold

Features:
1. Generate high-quality normal maps from depth maps
2. Batch processing
3. Skip target files that already exist

Usage:
    python scripts/regen_normal_pseudo.py --manifest output/snvc_v5full_pairs/pairs_5task_v3_clean.jsonl --task normal
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# environment variables
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def depth_to_normal(depth: np.ndarray, focal_length: float = 500.0) -> np.ndarray:
    """
    Compute a normal map from a depth map
    
    Args:
        depth: depth map (H, W); larger values are farther
        focal_length: focal length (pixels)
    
    Returns:
        normal: normal map (H, W, 3), XYZ normals encoded as RGB
    """
    H, W = depth.shape
    
    # gradients
    zx = np.gradient(depth, axis=1)  # gradient along x
    zy = np.gradient(depth, axis=0)  # gradient along y
    
    # normals
    # normal = (-dz/dx, -dz/dy, 1), normalized
    normal = np.stack([-zx, -zy, np.ones_like(depth)], axis=-1)
    
    # normalize
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    norm = np.clip(norm, 1e-8, None)
    normal = normal / norm
    
    # convert to RGB space [0, 1]
    # XYZ normal range [-1, 1] -> RGB [0, 1]
    normal_rgb = (normal + 1) / 2
    
    return normal_rgb


def encode_normal_to_rgb(normal: np.ndarray) -> np.ndarray:
    """
    Encode a normal map as an RGB image
    
    Args:
        normal: normal map (H, W, 3), values in [-1, 1]
    
    Returns:
        rgb: RGB image (H, W, 3), values in [0, 255]
    """
    # normalize to [0, 1]
    normal_norm = (normal + 1) / 2
    
    # convert to uint8
    rgb = (normal_norm * 255).astype(np.uint8)
    
    return rgb


def process_depth_to_normal(depth_path: str, output_path: str, focal_length: float = 500.0) -> bool:
    """
    Turn one depth map into a normal map
    
    Args:
        depth_path: depth map path
        output_path: output normal map path
        focal_length: focal length
    
    Returns:
        whether it succeeded
    """
    try:
        # load the depth map
        depth_img = Image.open(depth_path)
        depth = np.array(depth_img).astype(np.float32)
        
        # convert an RGB image to grayscale
        if depth.ndim == 3:
            depth = depth.mean(axis=2)
        
        # normalize depth to [0, 1]
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > 1e-6:
            depth = (depth - depth_min) / (depth_max - depth_min)
        
        # compute the normals
        normal = depth_to_normal(depth, focal_length)
        
        # encode as RGB
        normal_rgb = encode_normal_to_rgb(normal)
        
        # save
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(normal_rgb).save(output_path)
        
        return True
    except Exception as e:
        logger.error(f"Processing failed {depth_path}: {e}")
        return False


def regenerate_normals(
    manifest_path: Path,
    output_dir: Path,
    focal_length: float = 500.0,
    overwrite: bool = False,
) -> dict:
    """
    Regenerate the normal pseudo-labels in batch
    
    Args:
        manifest_path: manifest file path
        output_dir: output directory
        focal_length: focal length
        overwrite: whether to overwrite existing files
    
    Returns:
        processing statistics
    """
    logger.info(f"Loading manifest: {manifest_path}")
    
    rows = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row.get("task") == "depth":
                    rows.append(row)
    
    logger.info(f"Depth map samples: {len(rows)}")
    
    stats = {
        "total": len(rows),
        "success": 0,
        "skipped": 0,
        "failed": 0,
        "errors": [],
    }
    
    for i, row in enumerate(rows):
        depth_path = row.get("target", "")
        
        if not depth_path or not Path(depth_path).exists():
            stats["failed"] += 1
            if len(stats["errors"]) < 10:
                stats["errors"].append(f"[{i}] depth map missing: {depth_path}")
            continue
        
        # build the output path
        depth_name = Path(depth_path).stem
        output_path = output_dir / f"{depth_name}_normal.png"
        
        # check whether it already exists
        if output_path.exists() and not overwrite:
            stats["skipped"] += 1
            continue
        
        # process
        success = process_depth_to_normal(depth_path, str(output_path), focal_length)
        
        if success:
            stats["success"] += 1
        else:
            stats["failed"] += 1
        
        if (i + 1) % 100 == 0:
            logger.info(f"  progress: {i+1}/{len(rows)}, succeeded: {stats['success']}, skipped: {stats['skipped']}, failed: {stats['failed']}")
    
    logger.info(f"\nDone:")
    logger.info(f"  total samples: {stats['total']}")
    logger.info(f"  succeeded: {stats['success']}")
    logger.info(f"  skipped: {stats['skipped']}")
    logger.info(f"  failed: {stats['failed']}")
    
    if stats["errors"]:
        logger.info(f"\nError examples:")
        for err in stats["errors"]:
            logger.info(f"  {err}")
    
    return stats


def update_manifest_with_normals(
    manifest_path: Path,
    normal_dir: Path,
    output_path: Path,
) -> dict:
    """
    Update the manifest with the normal task samples
    
    Args:
        manifest_path: original manifest
        normal_dir: normal map directory
        output_path: output updated manifest
    
    Returns:
        update statistics
    """
    logger.info(f"Updating manifest: {manifest_path}")
    
    rows = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    
    logger.info(f"Original samples: {len(rows)}")
    
    # collect the depth map samples
    depth_rows = [r for r in rows if r.get("task") == "depth"]
    logger.info(f"Depth map samples: {len(depth_rows)}")
    
    # build the normal task samples
    normal_rows = []
    for row in depth_rows:
        depth_target = row.get("target", "")
        if not depth_target:
            continue
        
        depth_name = Path(depth_target).stem
        normal_path = normal_dir / f"{depth_name}_normal.png"
        
        if normal_path.exists():
            normal_row = row.copy()
            normal_row["task"] = "normal"
            normal_row["target"] = str(normal_path)
            normal_rows.append(normal_row)
    
    logger.info(f"Normal samples generated: {len(normal_rows)}")
    
    # merge
    all_rows = rows + normal_rows
    
    # write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    logger.info(f"Update done: {output_path}")
    logger.info(f"  total samples: {len(all_rows)}")
    
    return {"total": len(all_rows), "normal_added": len(normal_rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="manifest file")
    parser.add_argument("--output-dir", type=Path, required=True, help="normal map output directory")
    parser.add_argument("--focal-length", type=float, default=500.0, help="focal length (pixels)")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing files")
    parser.add_argument("--update-manifest", type=Path, help="output path of the updated manifest")
    
    args = parser.parse_args()
    
    # regenerate the normals
    stats = regenerate_normals(
        args.manifest,
        args.output_dir,
        args.focal_length,
        args.overwrite,
    )
    
    # update the manifest
    if args.update_manifest:
        update_manifest_with_normals(
            args.manifest,
            args.output_dir,
            args.update_manifest,
        )


if __name__ == "__main__":
    main()
