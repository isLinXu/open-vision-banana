#!/usr/bin/env python3
"""Automated dataset download & preparation script for Vision Banana.

Downloads and prepares benchmark datasets from public sources.
Supports: NYU Depth v2, DIODE, KITTI, ADE20K, COCO-Stuff, Sintel, FlyingThings3D

Usage:
  python scripts/download_datasets.py --dataset nyu --output-dir data/vb_v6_depth
  python scripts/download_datasets.py --dataset all --output-dir data
  python scripts/download_datasets.py --list

Requires: wget, unzip, tar (system tools)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dataset_download")


# ─── Dataset configurations ─────────────────────────────────────────────────

DATASET_REGISTRY = {
    "nyu": {
        "name": "NYU Depth v2",
        "task": "depth",
        "urls": [
            # Official NYU v2 (labeled, 1449 pairs) - requires manual download
            # Alternative: use preprocessed splits from various mirrors
            {
                "url": "https://raw.githubusercontent.com/law-of-cycles/nyu-depth-v2-train-val-split/main/nyu_depth_v2_labeled.mat",
                "filename": "nyu_depth_v2_labeled.mat",
                "optional": True,
            },
        ],
        "huggingface": "patriziapol/stratified-depth-dataset-nyu-depth-v2",  # HF mirror
        "expected_structure": {
            "images": "*.jpg",
            "depths": "*.png",
        },
        "size_mb": 1500,
        "description": "Indoor RGB-D, 1449 labeled pairs, most common depth benchmark",
    },
    "diode": {
        "name": "DIODE",
        "task": "depth",
        "urls": [
            {
                "url": "http://diode-dataset.s3.amazonaws.com/train_val.tar.gz",
                "filename": "diode_train_val.tar.gz",
                "optional": False,
            },
        ],
        "expected_structure": {
            "train/indoor": "*_rgb.png",
            "train/outdoor": "*_rgb.png",
            "val/indoor": "*_rgb.png",
            "val/outdoor": "*_rgb.png",
        },
        "size_mb": 2500,
        "description": "First real indoor+outdoor dense depth, 25K train / 3.3K val",
    },
    "kitti": {
        "name": "KITTI Depth",
        "task": "depth",
        "urls": [
            {
                "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip",
                "filename": "data_depth_annotated.zip",
                "optional": False,
            },
        ],
        "expected_structure": {
            "train": "*.png",
            "val": "*.png",
        },
        "size_mb": 1500,
        "description": "Autonomous driving sparse LiDAR depth, 23K train / 1K test",
    },
    "ade20k": {
        "name": "ADE20K",
        "task": "semantic",
        "urls": [
            {
                "url": "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
                "filename": "ADEChallengeData2016.zip",
                "optional": False,
            },
        ],
        "expected_structure": {
            "images/training": "*.jpg",
            "images/validation": "*.jpg",
            "annotations/training": "*.png",
            "annotations/validation": "*.png",
        },
        "size_mb": 700,
        "description": "Scene parsing standard, 20K training / 150 classes / indoor+outdoor",
    },
    "coco_stuff": {
        "name": "COCO-Stuff",
        "task": "semantic",
        "urls": [
            {
                "url": "http://images.cocodataset.org/zips/train2017.zip",
                "filename": "train2017.zip",
                "optional": False,
            },
            {
                "url": "http://images.cocodataset.org/zips/val2017.zip",
                "filename": "val2017.zip",
                "optional": True,
            },
            {
                "url": "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip",
                "filename": "stuffthingmaps_trainval2017.zip",
                "optional": True,
            },
        ],
        "expected_structure": {
            "images/train2017": "*.jpg",
            "annotations": "*.json",
        },
        "size_mb": 19000,
        "description": "COCO + 80 stuff classes, 118K images, thing+stuff unified",
    },
    "sintel": {
        "name": "Sintel Flow",
        "task": "flow",
        "urls": [
            {
                "url": "http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip",
                "filename": "MPI-Sintel-complete.zip",
                "optional": False,
            },
        ],
        "expected_structure": {
            "training/clean": "*.png",
            "training/final": "*.png",
            "training/flow": "*.flo",
        },
        "size_mb": 6000,
        "description": "Classic optical flow benchmark, clean/final versions, 1K training frames",
    },
    "flyingthings3d": {
        "name": "FlyingThings3D",
        "task": "flow",
        "urls": [
            {
                "url": "https://lmb.informatik.uni-freiburg.de/data/FlyingThings3D/FlyingThings3D_subset.zip",
                "filename": "FlyingThings3D_subset.zip",
                "optional": False,
            },
        ],
        "expected_structure": {
            "train/image_clean/left": "*.png",
            "train/flow/left": "*.flo",
        },
        "size_mb": 42000,
        "description": "Synthetic 3D optical flow, 25K stereo frames, good for pretraining",
    },
    "eth3d": {
        "name": "ETH3D",
        "task": "depth",
        "urls": [],
        "expected_structure": {
            "images": "*.png",
            "depths": "*.npy",
        },
        "size_mb": 500,
        "description": "High-precision laser scanned depth, 10 scenes, requires registration",
        "note": "ETH3D requires registration at https://www.eth3d.net/ — download manually",
    },
    "oasis": {
        "name": "OASIS Normal",
        "task": "normal",
        "urls": [],
        "expected_structure": {
            "images": "*.jpg",
            "normals": "*.npy",
        },
        "size_mb": 50000,
        "description": "1.2M indoor surface normals, requires registration at https://oasis.cs.princeton.edu/",
        "note": "OASIS requires registration — download manually",
    },
    "virtual_kitti": {
        "name": "Virtual KITTI 2",
        "task": "depth",
        "urls": [
            {
                "url": "https://europa.eu/
        ],
        "expected_structure": {
            "Scene01/frames/rgb": "*.jpg",
            "Scene01/frames/depth": "*.png",
        },
        "size_mb": 7000,
        "description": "Synthetic KITTI with weather/time, 50K frames, perfect GT",
    },
    "tum_rgbd": {
        "name": "TUM RGB-D",
        "task": "depth",
        "urls": [
            {
                "url": "https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz",
                "filename": "rgbd_dataset_freiburg1_desk.tgz",
                "optional": False,
            },
        ],
        "expected_structure": {
            "rgb": "*.png",
            "depth": "*.png",
        },
        "size_mb": 400,
        "description": "SLAM/depth standard, 39 sequences, camera trajectories",
    },
}


# ─── Download utilities ────────────────────────────────────────────────────


def check_tool(name: str) -> bool:
    """Check if a system tool is available."""
    return shutil.which(name) is not None


def download_file(url: str, output_path: Path, timeout: int = 300) -> bool:
    """Download a file using wget or curl."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if check_tool("wget"):
        cmd = ["wget", "-c", "-O", str(output_path), "--timeout", str(timeout), url]
    elif check_tool("curl"):
        cmd = ["curl", "-L", "-o", str(output_path), "--max-time", str(timeout), url]
    else:
        logger.error("Neither wget nor curl available. Please install one.")
        return False
    
    logger.info(f"Downloading {url} → {output_path}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 30)
        if result.returncode != 0:
            logger.error(f"Download failed: {result.stderr[:500]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"Download timeout after {timeout}s")
        return False
    except Exception as e:
        logger.error(f"Download error: {e}")
        return False


def extract_archive(archive_path: Path, output_dir: Path) -> bool:
    """Extract a zip or tar archive."""
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = archive_path.suffix.lower()
    
    if suffix == ".zip":
        if not check_tool("unzip"):
            logger.error("unzip not available")
            return False
        cmd = ["unzip", "-q", "-o", str(archive_path), "-d", str(output_dir)]
    elif suffix in (".tar", ".gz", ".tgz"):
        if not check_tool("tar"):
            logger.error("tar not available")
            return False
        cmd = ["tar", "-xzf", str(archive_path), "-C", str(output_dir)]
    else:
        logger.error(f"Unknown archive format: {suffix}")
        return False
    
    logger.info(f"Extracting {archive_path.name} → {output_dir}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error(f"Extract failed: {result.stderr[:500]}")
            return False
        return True
    except Exception as e:
        logger.error(f"Extract error: {e}")
        return False


# ─── Dataset preparation ─────────────────────────────────────────────────────


def prepare_nyu(output_dir: Path) -> bool:
    """Prepare NYU Depth v2 dataset."""
    logger.info("NYU Depth v2 requires manual download or HF dataset.")
    logger.info("  Option 1: Download from https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html")
    logger.info("  Option 2: Use HuggingFace: huggingface-cli download patriziapol/stratified-depth-dataset-nyu-depth-v2")
    
    # Try to use HF dataset if available
    try:
        from datasets import load_dataset
        logger.info("Attempting HuggingFace download...")
        ds = load_dataset("patriziapol/stratified-depth-dataset-nyu-depth-v2", split="train")
        logger.info(f"Loaded {len(ds)} samples from HF")
        
        img_dir = output_dir / "images"
        depth_dir = output_dir / "depths"
        img_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        
        for i, sample in enumerate(ds):
            if i >= 500:  # Limit for quick testing
                break
            if "image" in sample and "depth" in sample:
                sample["image"].save(img_dir / f"nyu_{i:04d}.jpg")
                # Depth as numpy
                import numpy as np
                depth = np.array(sample["depth"])
                np.save(depth_dir / f"nyu_{i:04d}.npy", depth)
        
        logger.info(f"Saved {min(500, len(ds))} samples to {output_dir}")
        return True
    except ImportError:
        logger.info("datasets library not installed. Install: pip install datasets")
    except Exception as e:
        logger.warning(f"HF download failed: {e}")
    
    return False


def prepare_diode(output_dir: Path, download_dir: Path) -> bool:
    """Prepare DIODE dataset."""
    archive = download_dir / "diode_train_val.tar.gz"
    if not archive.exists():
        logger.info("DIODE download not available. Please download manually:")
        logger.info("  http://diode-dataset.s3.amazonaws.com/train_val.tar.gz")
        return False
    
    if extract_archive(archive, output_dir):
        logger.info(f"DIODE extracted to {output_dir}")
        return True
    return False


def prepare_kitti(output_dir: Path, download_dir: Path) -> bool:
    """Prepare KITTI depth dataset."""
    archive = download_dir / "data_depth_annotated.zip"
    if not archive.exists():
        logger.info("KITTI download requires registration. Please download manually:")
        logger.info("  https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction")
        return False
    
    if extract_archive(archive, output_dir):
        logger.info(f"KITTI extracted to {output_dir}")
        return True
    return False


def prepare_ade20k(output_dir: Path, download_dir: Path) -> bool:
    """Prepare ADE20K dataset."""
    archive = download_dir / "ADEChallengeData2016.zip"
    if not archive.exists():
        if not download_file(
            "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
            archive,
        ):
            return False
    
    if extract_archive(archive, output_dir):
        # Move to expected structure
        extracted = output_dir / "ADEChallengeData2016"
        if extracted.exists():
            extracted.rename(output_dir / "ade20k")
        logger.info(f"ADE20K prepared at {output_dir}")
        return True
    return False


def prepare_coco_stuff(output_dir: Path, download_dir: Path) -> bool:
    """Prepare COCO-Stuff dataset."""
    logger.info("COCO-Stuff requires downloading COCO images + stuff annotations.")
    logger.info("  Images: http://images.cocodataset.org/zips/train2017.zip")
    logger.info("  Annotations: http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip")
    
    # Try HuggingFace as alternative
    try:
        from datasets import load_dataset
        logger.info("Attempting HF COCO download...")
        ds = load_dataset("detection-datasets/coco", split="train", streaming=True)
        logger.info("COCO streaming dataset available from HF")
        logger.info("To save locally, use: ds.save_to_disk()")
        return True
    except ImportError:
        logger.info("datasets library not available for HF download")
    except Exception as e:
        logger.warning(f"HF download failed: {e}")
    
    return False


def prepare_sintel(output_dir: Path, download_dir: Path) -> bool:
    """Prepare Sintel optical flow dataset."""
    archive = download_dir / "MPI-Sintel-complete.zip"
    if not archive.exists():
        if not download_file(
            "http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip",
            archive,
            timeout=600,
        ):
            return False
    
    if extract_archive(archive, output_dir):
        logger.info(f"Sintel extracted to {output_dir}")
        return True
    return False


def prepare_flyingthings3d(output_dir: Path, download_dir: Path) -> bool:
    """Prepare FlyingThings3D optical flow dataset."""
    logger.info("FlyingThings3D is very large (~42GB). Consider using subset.")
    archive = download_dir / "FlyingThings3D_subset.zip"
    if not archive.exists():
        logger.info("Download from: https://lmb.informatik.uni-freiburg.de/data/FlyingThings3D/")
        return False
    
    if extract_archive(archive, output_dir):
        logger.info(f"FlyingThings3D extracted to {output_dir}")
        return True
    return False


# ─── Main ───────────────────────────────────────────────────────────────────


def download_dataset(dataset_name: str, output_dir: Path, download_dir: Path) -> bool:
    """Download and prepare a single dataset."""
    if dataset_name not in DATASET_REGISTRY:
        logger.error(f"Unknown dataset: {dataset_name}")
        return False
    
    config = DATASET_REGISTRY[dataset_name]
    logger.info(f"=" * 60)
    logger.info(f"Dataset: {config['name']}")
    logger.info(f"Task: {config['task']}")
    logger.info(f"Description: {config['description']}")
    logger.info(f"Expected size: ~{config['size_mb']} MB")
    if "note" in config:
        logger.info(f"Note: {config['note']}")
    logger.info(f"=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # Download all URLs
    for url_info in config.get("urls", []):
        if url_info.get("optional", False):
            logger.info(f"Skipping optional: {url_info['filename']}")
            continue
        
        output_path = download_dir / url_info["filename"]
        if output_path.exists():
            logger.info(f"Already downloaded: {output_path}")
            continue
        
        if not download_file(url_info["url"], output_path):
            if not url_info.get("optional", False):
                logger.error(f"Failed to download required file: {url_info['filename']}")
                return False
    
    # Prepare dataset-specific structure
    if dataset_name == "nyu":
        return prepare_nyu(output_dir)
    elif dataset_name == "diode":
        return prepare_diode(output_dir, download_dir)
    elif dataset_name == "kitti":
        return prepare_kitti(output_dir, download_dir)
    elif dataset_name == "ade20k":
        return prepare_ade20k(output_dir, download_dir)
    elif dataset_name == "coco_stuff":
        return prepare_coco_stuff(output_dir, download_dir)
    elif dataset_name == "sintel":
        return prepare_sintel(output_dir, download_dir)
    elif dataset_name == "flyingthings3d":
        return prepare_flyingthings3d(output_dir, download_dir)
    else:
        logger.info(f"Dataset '{dataset_name}' requires manual preparation.")
        logger.info(f"Expected structure: {config.get('expected_structure', {})}")
        return True


def list_datasets():
    """Print all available datasets."""
    print("\nAvailable datasets for Vision Banana:")
    print("=" * 70)
    for key, config in DATASET_REGISTRY.items():
        print(f"\n  {key:20s} | {config['name']}")
        print(f"  {'':20s} | Task: {config['task']}")
        print(f"  {'':20s} | Size: ~{config['size_mb']} MB")
        print(f"  {'':20s} | {config['description']}")
        if "note" in config:
            print(f"  {'':20s} | ⚠️  {config['note']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Download benchmark datasets for Vision Banana")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name or 'all' (use --list to see options)")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory for prepared datasets")
    parser.add_argument("--download-dir", type=str, default="downloads",
                        help="Temporary directory for downloaded archives")
    parser.add_argument("--list", action="store_true",
                        help="List all available datasets and exit")
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
        return
    
    output_dir = Path(args.output_dir)
    download_dir = Path(args.download_dir)
    
    if args.dataset == "all":
        datasets = list(DATASET_REGISTRY.keys())
    else:
        datasets = [args.dataset]
    
    results = {}
    for ds_name in datasets:
        success = download_dataset(ds_name, output_dir / f"vb_v6_{ds_name}", download_dir)
        results[ds_name] = success
    
    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    for ds_name, success in results.items():
        status = "✅ Success" if success else "❌ Failed / Manual required"
        print(f"  {ds_name:20s} | {status}")
    
    # Save manifest
    manifest_path = output_dir / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "datasets": {k: {"success": v, "config": DATASET_REGISTRY.get(k, {})}
                        for k, v in results.items()},
        }, f, indent=2, default=str)
    print(f"\nManifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
