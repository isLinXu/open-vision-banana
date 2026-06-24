#!/usr/bin/env python3
"""Automated dataset downloader and preparation pipeline for Vision Banana.

Downloads vision task datasets from public sources and prepares them in the
standard format (images/ + targets/ layout) expected by the training pipeline.

Supported datasets:
  - hypersim-depth: Hypersim depth estimation (synthetic indoor scenes)
  - scannet-normal: ScanNet surface normals (real indoor scans)
  - cityscapes-seg: Cityscapes semantic segmentation (urban street scenes)
  - sa1b-instance: SA-1B instance segmentation (automated masks)
  - refcocog-referring: RefCOCOg referring expression segmentation

Usage:
    # List available datasets
    python scripts/download_data.py --list

    # Download a specific dataset
    python scripts/download_data.py --dataset hypersim-depth --output data/hypersim/

    # Download with custom split
    python scripts/download_data.py --dataset scannet-normal --output data/scannet/ --train-split 0.8

    # Download all supported datasets
    python scripts/download_data.py --all --output data/

    # Dry run (show what would be downloaded)
    python scripts/download_data.py --dataset hypersim-depth --output data/hypersim/ --dry-run
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class DatasetSpec:
    """Specification for a downloadable dataset."""
    name: str
    task: str
    source: str  # URL or description of source
    size_estimate_gb: float
    requires_login: bool
    prepare_fn: Callable[[Path, Path], int] | None = None
    license: str = "unknown"


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    # === Paper-equivalent synthetic data (large) ===
    "hypersim-depth": DatasetSpec(
        name="Hypersim Depth",
        task="depth",
        source="https://github.com/apple/ml-hypersim",
        size_estimate_gb=1.5,
        requires_login=False,
        license="MIT",
    ),
    # === Lightweight alternatives to Hypersim ===
    "vkitti2-depth": DatasetSpec(
        name="Virtual KITTI 2 Depth",
        task="depth",
        source="https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/",
        size_estimate_gb=50.0,
        requires_login=False,
        license="Academic",
    ),
    "blendedmvs-depth": DatasetSpec(
        name="BlendedMVS Depth",
        task="depth",
        source="https://github.com/YoYo000/MVSNet",
        size_estimate_gb=17.0,
        requires_login=False,
        license="Academic",
    ),
    "hrwsi-depth": DatasetSpec(
        name="HRWSI Depth",
        task="depth",
        source="https://github.com/aim-uofa/DepthAnything",
        size_estimate_gb=20.0,
        requires_login=False,
        license="Academic",
    ),
    "diode-depth": DatasetSpec(
        name="DIODE Depth",
        task="depth",
        source="https://diode-dataset.org/",
        size_estimate_gb=40.0,
        requires_login=False,
        license="Academic",
    ),
    "tartanair-depth": DatasetSpec(
        name="TartanAir Depth",
        task="depth",
        source="https://github.com/castacks/tartanair_tools",
        size_estimate_gb=30.0,
        requires_login=False,
        license="Academic",
    ),
    # === Existing datasets ===
    "scannet-normal": DatasetSpec(
        name="ScanNet Normal",
        task="normal",
        source="http://www.scan-net.org/",
        size_estimate_gb=12.0,
        requires_login=True,
        license="ScanNet Terms",
    ),
    "cityscapes-seg": DatasetSpec(
        name="Cityscapes Semantic",
        task="semantic",
        source="https://www.cityscapes-dataset.com/",
        size_estimate_gb=11.0,
        requires_login=True,
        license="Cityscapes Terms",
    ),
    "sa1b-instance": DatasetSpec(
        name="SA-1B Instance",
        task="instance",
        source="https://segment-anything.com/",
        size_estimate_gb=0.5,
        requires_login=False,
        license="Apache-2.0",
    ),
    "refcocog-referring": DatasetSpec(
        name="RefCOCOg Referring",
        task="referring",
        source="https://github.com/lichengunc/refer",
        size_estimate_gb=0.05,
        requires_login=False,
        license="MIT",
    ),
}


def _download_file(url: str, dest: Path, chunk_size: int = 8192) -> Path:
    """Download a file with progress logging."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s", url, dest)

    def _report(block_num: int, block_size: int, total_size: int):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded * 100 / total_size, 100)
            if block_num % 100 == 0:
                logger.info("  %.1f%% (%d / %d bytes)", pct, downloaded, total_size)

    urllib.request.urlretrieve(url, dest, reporthook=_report)
    logger.info("Download complete: %s", dest)
    return dest


def _extract_archive(archive: Path, dest: Path) -> None:
    """Extract zip or tar archive to destination."""
    logger.info("Extracting %s → %s", archive, dest)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest)
    elif archive.suffix in (".tar", ".gz", ".tgz"):
        import tarfile
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(dest)
    else:
        logger.warning("Unknown archive format: %s", archive.suffix)


def prepare_hypersim_depth(raw_dir: Path, output_dir: Path) -> int:
    """Prepare Hypersim depth data from raw download.

    Expects raw_dir to contain scene directories with *_color.png and *_depth.png pairs.
    """
    from prepare_hypersim import prepare_hypersim
    return prepare_hypersim(raw_dir, output_dir, resolution=None, overwrite=False)


def prepare_scannet_normal(raw_dir: Path, output_dir: Path) -> int:
    """Prepare ScanNet normal data."""
    logger.info("ScanNet normal preparation: scanning %s", raw_dir)
    # ScanNet layout: sceneXXXX_XX/color/*.jpg + sceneXXXX_XX/normal/*.png
    import numpy as np
    from PIL import Image
    from open_vision_banana.core.normal_mapper import NormalRGBMapper

    img_dir = output_dir / "images"
    tgt_dir = output_dir / "targets"
    img_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    mapper = NormalRGBMapper()
    written = 0
    scene_dirs = sorted(raw_dir.rglob("scene*"))
    for scene_dir in scene_dirs:
        color_dir = scene_dir / "color"
        normal_dir = scene_dir / "normal"
        if not color_dir.exists() or not normal_dir.exists():
            continue
        for color_path in sorted(color_dir.glob("*.jpg")):
            frame_idx = color_path.stem
            normal_path = normal_dir / f"{frame_idx}.png"
            if not normal_path.exists():
                continue
            out_stem = f"{scene_dir.name}_{frame_idx}"
            out_img = img_dir / f"{out_stem}.png"
            out_tgt = tgt_dir / f"{out_stem}.png"
            if out_img.exists() and out_tgt.exists():
                written += 1
                continue
            img = Image.open(color_path).convert("RGB")
            img.save(out_img)
            # Load normal map (assumed in [-1,1] range, encoded as 8-bit PNG)
            normal_arr = np.array(Image.open(normal_path).convert("RGB")).astype(np.float32) / 127.5 - 1.0
            normal_tensor = torch.from_numpy(normal_arr.transpose(2, 0, 1))
            normal_tensor = normal_tensor / normal_tensor.norm(dim=0, keepdim=True).clamp(min=1e-8)
            rgb = mapper.normal_to_rgb(normal_tensor.unsqueeze(0)).squeeze(0)
            arr = (rgb.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
            Image.fromarray(arr, mode="RGB").save(out_tgt)
            written += 1
    logger.info("Wrote %d ScanNet normal pairs to %s", written, output_dir)
    return written


def prepare_cityscapes_seg(raw_dir: Path, output_dir: Path) -> int:
    """Prepare Cityscapes semantic segmentation data."""
    logger.info("Cityscapes preparation: scanning %s", raw_dir)
    img_dir = output_dir / "images"
    tgt_dir = output_dir / "targets"
    img_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    # Cityscapes layout: leftImg8bit/train/city/xxx_leftImg8bit.png
    #                   gtFine/train/city/xxx_gtFine_color.png
    img_base = raw_dir / "leftImg8bit"
    gt_base = raw_dir / "gtFine"
    if not img_base.exists() or not gt_base.exists():
        logger.warning("Expected Cityscapes layout not found. Skipping.")
        return 0

    for split in ("train", "val", "test"):
        img_split = img_base / split
        gt_split = gt_base / split
        if not img_split.exists() or not gt_split.exists():
            continue
        for city_dir in sorted(img_split.glob("*")):
            if not city_dir.is_dir():
                continue
            city_name = city_dir.name
            gt_city = gt_split / city_name
            for img_path in sorted(city_dir.glob("*.png")):
                stem = img_path.stem.replace("_leftImg8bit", "")
                gt_path = gt_city / f"{stem}_gtFine_color.png"
                if not gt_path.exists():
                    continue
                out_img = img_dir / f"{split}_{city_name}_{stem}.png"
                out_tgt = tgt_dir / f"{split}_{city_name}_{stem}.png"
                if out_img.exists() and out_tgt.exists():
                    written += 1
                    continue
                from shutil import copy2
                copy2(img_path, out_img)
                copy2(gt_path, out_tgt)
                written += 1
    logger.info("Wrote %d Cityscapes pairs to %s", written, output_dir)
    return written


def prepare_refcocog_referring(raw_dir: Path, output_dir: Path) -> int:
    """Prepare RefCOCOg referring expression segmentation data."""
    logger.info("RefCOCOg preparation: scanning %s", raw_dir)
    # RefCOCOg requires images from COCO 2014 train/val split
    # refs(data) contains {ref_id, ann_id, sentence, bbox, mask}
    import json

    img_dir = output_dir / "images"
    tgt_dir = output_dir / "targets"
    img_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    refs_file = raw_dir / "refs(unc).p"
    if not refs_file.exists():
        logger.warning("Refs file not found: %s", refs_file)
        return 0

    # Requires pickle loading (refs data structure)
    try:
        import pickle
        with open(refs_file, "rb") as f:
            refs_data = pickle.load(f)
    except Exception as exc:
        logger.warning("Could not load refs data: %s", exc)
        return 0

    written = 0
    for ref in refs_data:
        # Each ref has: ref_id, annotation_id, category_id, sentences, etc.
        # Image source must be COCO 2014
        pass  # Stub — requires COCO images for full implementation
    logger.info("RefCOCOg preparation stub: %d refs found, COCO images required", len(refs_data))
    return written


# Register preparation functions
DATASET_REGISTRY["hypersim-depth"].prepare_fn = prepare_hypersim_depth
DATASET_REGISTRY["scannet-normal"].prepare_fn = prepare_scannet_normal
DATASET_REGISTRY["cityscapes-seg"].prepare_fn = prepare_cityscapes_seg
DATASET_REGISTRY["refcocog-referring"].prepare_fn = prepare_refcocog_referring


def download_dataset(name: str, output_dir: Path, dry_run: bool = False) -> int:
    """Download and prepare a single dataset."""
    spec = DATASET_REGISTRY.get(name)
    if spec is None:
        logger.error("Unknown dataset: %s. Run --list for available datasets.", name)
        return 0

    logger.info("=" * 60)
    logger.info("Dataset: %s (%s)", spec.name, spec.task)
    logger.info("Source: %s", spec.source)
    logger.info("Estimated size: %.1f GB", spec.size_estimate_gb)
    logger.info("License: %s", spec.license)
    if spec.requires_login:
        logger.info("⚠️  This dataset requires manual download and agreement to terms.")
        logger.info("   Please download from %s and place in %s/", spec.source, output_dir)
    logger.info("=" * 60)

    if dry_run:
        logger.info("[DRY RUN] Would download/prepare %s → %s", name, output_dir)
        return 0

    if spec.requires_login:
        logger.info("Manual download required. Please visit %s", spec.source)
        logger.info("After downloading, run: python scripts/prepare_%s.py", name.replace("-", "_"))
        return 0

    # For datasets with direct download links, implement here
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Run preparation if available
    if spec.prepare_fn is not None:
        task_dir = output_dir / spec.task
        task_dir.mkdir(parents=True, exist_ok=True)
        return spec.prepare_fn(raw_dir, task_dir)

    logger.info("Preparation function not yet implemented for %s.", name)
    return 0


def list_datasets() -> None:
    """Print all available datasets with their status."""
    print("Available datasets:")
    print("-" * 70)
    print(f"  {'Name':<25} {'Task':<12} {'Size':<8} {'Login':<8} {'License':<15}")
    print("-" * 70)
    for name, spec in sorted(DATASET_REGISTRY.items()):
        login = "Yes" if spec.requires_login else "No"
        size = f"{spec.size_estimate_gb:.1f}GB"
        print(f"  {name:<25} {spec.task:<12} {size:<8} {login:<8} {spec.license:<15}")
    print("-" * 70)
    print(f"\nUsage: python scripts/download_data.py --dataset <name> --output <dir>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vision Banana dataset downloader")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name to download")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument("--all", action="store_true", help="Download all supported datasets")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    parser.add_argument("--train-split", type=float, default=0.9, help="Train split ratio")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        list_datasets()
        return 0

    if not args.dataset and not args.all:
        logger.error("Specify --dataset <name>, --all, or --list")
        return 1

    datasets = list(DATASET_REGISTRY.keys()) if args.all else [args.dataset]
    total = 0

    for name in datasets:
        if name not in DATASET_REGISTRY:
            logger.error("Unknown dataset: %s", name)
            continue
        count = download_dataset(name, Path(args.output), dry_run=args.dry_run)
        total += count

    logger.info("Total samples prepared: %d", total)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    sys.exit(main())
