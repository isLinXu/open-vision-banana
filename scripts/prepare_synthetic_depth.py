#!/usr/bin/env python3
"""Preprocessor for pure synthetic depth data — converts raw downloads to training format.

Converts synthetic depth datasets from their native formats into the standard
Vision Banana training layout (images/ + targets/) with DepthRGBMapper encoding.

All inputs are **synthetic** data from simulation engines (Hypersim, VKITTI2,
BlendedMVS, TartanAir). Zero real-world depth data is used or expected.

Usage:
    # Prepare a specific downloaded dataset
    python scripts/prepare_synthetic_depth.py --dataset hypersim-depth --raw data/hypersim/raw --output data/hypersim/train/

    # Prepare all downloaded synthetic datasets
    python scripts/prepare_synthetic_depth.py --all --root data/synthetic_depth/ --output-dir data/synthetic_depth_prepared/

    # Prepare with custom resolution and validate
    python scripts/prepare_synthetic_depth.py --dataset vkitti2-depth --raw data/vkitti2/raw --output data/vkitti2/train/ --resolution 512 --validate

    # Dry run to see what would be processed
    python scripts/prepare_synthetic_depth.py --dataset tartanair-depth --raw data/tartanair/raw --output data/tartanair/train/ --dry-run

    # Force reprocessing (overwrite existing)
    python scripts/prepare_synthetic_depth.py --dataset hypersim-depth --raw data/hypersim/raw --output data/hypersim/train/ --overwrite

    # Limit samples for quick smoke testing
    python scripts/prepare_synthetic_depth.py --dataset blendedmvs-depth --raw data/blendedmvs/raw --output data/blendedmvs/train/ --max-samples 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.data.depth_geometry import ray_distance_to_planar_depth

logger = logging.getLogger(__name__)


def _hypersim_module():
    """The dedicated Hypersim preparer, importable both as a package and as a script."""
    try:
        from scripts import prepare_hypersim
    except ImportError:  # run as `python scripts/prepare_synthetic_depth.py`
        import prepare_hypersim  # type: ignore[no-redef]
    return prepare_hypersim

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ═══════════════════════════════════════════════════════════════════════════════
# Colormap Augmentation Registry (Paper Section 3.2)
# ═══════════════════════════════════════════════════════════════════════════════

from open_vision_banana.core.depth_mapper import get_available_colormaps


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset Preparation Registry
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PreprocessorSpec:
    """Specification for a synthetic depth preprocessor."""

    name: str
    display_name: str
    description: str
    prepare_fn: Callable[[Path, Path, argparse.Namespace], int]


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────


def _tensor_to_pil_rgb(tensor: torch.Tensor) -> Image.Image:
    """Convert a [3, H, W] tensor in [0, 1] to PIL RGB image."""
    arr = (tensor.clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def _load_depth_meters(depth_path: Path) -> torch.Tensor:
    """Load a depth file and return metric depth [1, 1, H, W] in meters.

    Handles:
      - PNG uint16 (millimeters → meters)
      - PNG uint8 (normalized, try to infer scale)
      - NPY float32 (meters, raw)
      - EXR (if OpenEXR available)
    """
    suffix = depth_path.suffix.lower()

    if suffix == ".npy":
        depth = np.load(depth_path).astype(np.float32)
        depth_t = torch.from_numpy(depth).float()
        # Ensure shape [1, 1, H, W]
        while depth_t.dim() < 4:
            depth_t = depth_t.unsqueeze(0)
        return depth_t

    if suffix == ".exr":
        try:
            import OpenEXR
            import Imath
            exr = OpenEXR.InputFile(str(depth_path))
            header = exr.header()
            dw = header["dataWindow"]
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            depth_str = exr.channel("Z", pt)
            depth = np.frombuffer(depth_str, dtype=np.float32).reshape(size[1], size[0])
            depth_t = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0)
            return depth_t
        except ImportError:
            logger.warning("OpenEXR not available, skipping %s", depth_path)
            return torch.empty(1, 1, 0, 0)

    # Default: image-based depth (PNG/JPG)
    depth_img = Image.open(depth_path)
    depth = np.array(depth_img).astype(np.float32)

    if depth.ndim == 3:
        # Use first channel if multi-channel
        depth = depth[..., 0]

    depth_t = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0)

    # Heuristic: if max > 10, likely in millimeters; if max > 1000, definitely mm
    if depth_t.max() > 1000.0:
        depth_t = depth_t / 1000.0
    elif depth_t.max() > 10.0 and depth_t.max() <= 255.0:
        # Could be 0-255 normalized or uint8 cm scale — assume meters for synthetic
        pass

    return depth_t.clamp(min=0.0)


def _resize_depth(depth: torch.Tensor, resolution: int) -> torch.Tensor:
    """Resize depth tensor [B, C, H, W] to target resolution using nearest neighbor."""
    if depth.shape[-2:] == (resolution, resolution):
        return depth
    return torch.nn.functional.interpolate(
        depth, size=(resolution, resolution), mode="nearest"
    )


def _resize_rgb(img: Image.Image, resolution: int) -> Image.Image:
    """Resize RGB image to target resolution."""
    if img.size == (resolution, resolution):
        return img
    return img.resize((resolution, resolution), Image.Resampling.LANCZOS)


def _find_color_depth_pairs(
    raw_dir: Path,
    color_patterns: list[str] | None = None,
    depth_patterns: list[str] | None = None,
) -> list[tuple[Path, Path]]:
    """Find color/depth file pairs under raw_dir using flexible naming conventions.

    Args:
        raw_dir: Root directory to scan.
        color_patterns: Substrings that identify color files (e.g., ["color", "rgb", "image_left"]).
        depth_patterns: Substrings that identify depth files (e.g., ["depth", "depth_left"]).

    Returns:
        List of (color_path, depth_path) tuples.
    """
    color_patterns = color_patterns or ["color", "rgb", "image", "img"]
    depth_patterns = depth_patterns or ["depth"]

    pairs: list[tuple[Path, Path]] = []
    color_paths: list[Path] = []

    for ext in IMG_EXTS:
        for p in sorted(raw_dir.rglob(f"*{ext}")):
            if not p.is_file():
                continue
            stem_lower = p.stem.lower()
            if any(pat in stem_lower for pat in color_patterns):
                color_paths.append(p)

    for color_path in color_paths:
        # Derive depth path by replacing color indicator with depth indicator
        depth_path = None
        for cp in color_patterns:
            for dp in depth_patterns:
                candidate = Path(str(color_path).replace(cp, dp))
                if candidate.exists() and candidate != color_path:
                    depth_path = candidate
                    break
            if depth_path:
                break

        if not depth_path:
            # Fallback: same directory, same stem with depth suffix
            for dp in depth_patterns:
                candidate = color_path.parent / (color_path.stem + f"_{dp}" + color_path.suffix)
                if candidate.exists():
                    depth_path = candidate
                    break

        if depth_path and depth_path.exists():
            pairs.append((color_path, depth_path))

    return pairs


def _write_manifest(output_dir: Path, pairs: list[tuple[Path, Path]], args: argparse.Namespace) -> None:
    """Write a JSON manifest documenting the preprocessing run."""
    manifest = {
        "dataset_type": "synthetic_depth",
        "zero_real_world": True,
        "num_samples": len(pairs),
        "resolution": args.resolution,
        "colormap": "cube_edge",
        "depth_range": {"min_meters": 0.0, "max_meters": "unbounded"},
        "power_transform": {"lam": -3.0, "c": 10.0 / 3.0},
        "source_pairs": [
            {"color": str(c), "depth": str(d)} for c, d in pairs[:10]
        ],  # Only first 10 for brevity
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written: %s", manifest_path)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-Specific Preparers
# ─────────────────────────────────────────────────────────────────────────────


def prepare_hypersim(raw_dir: Path, output_dir: Path, args: argparse.Namespace) -> int:
    """Prepare Hypersim synthetic depth data.

    Hypersim layout (after extraction):
        raw/
          <scene_name>/
            images/
              scene_cam_00_final_preview/
                frame.0000.color.jpg
                frame.0000.depth_meters.png
                ...
    """
    logger.info("Preparing Hypersim synthetic depth data…")

    # Official layout (scene_cam_XX_final_preview/*.color.jpg <-> *_geometry_hdf5/*.depth_meters.hdf5)
    # and flat PNG/NPY exports are both handled by the dedicated Hypersim scanner.
    pairs = _hypersim_module().find_hypersim_pairs(raw_dir)
    if not pairs:
        pairs = _find_color_depth_pairs(
            raw_dir,
            color_patterns=["color", "rgb"],
            depth_patterns=["depth", "depth_meters"],
        )

    if not pairs:
        logger.error("No Hypersim color/depth pairs found in %s", raw_dir)
        return 0

    return _process_pairs(pairs, output_dir, args, name_prefix="hs_")


def prepare_vkitti2(raw_dir: Path, output_dir: Path, args: argparse.Namespace) -> int:
    """Prepare Virtual KITTI 2 synthetic depth data.

    VKITTI2 layout (after extraction):
        raw/
          Scene01/
            frames/
              rgb/Camera_0/rgb_00001.jpg
              depth/Camera_0/depth_00001.png
    """
    logger.info("Preparing Virtual KITTI 2 synthetic depth data…")

    pairs = _find_color_depth_pairs(
        raw_dir,
        color_patterns=["rgb"],
        depth_patterns=["depth"],
    )

    if not pairs:
        logger.error("No VKITTI2 color/depth pairs found in %s", raw_dir)
        return 0

    return _process_pairs(pairs, output_dir, args, name_prefix="vk_")


def prepare_blendedmvs(raw_dir: Path, output_dir: Path, args: argparse.Namespace) -> int:
    """Prepare BlendedMVS synthetic depth data.

    BlendedMVS layout:
        raw/
          <scene_name>/
            blended_images/
              <scene_name>_0000.jpg
            depth_maps/
              <scene_name>_0000.png
    """
    logger.info("Preparing BlendedMVS synthetic depth data…")

    pairs: list[tuple[Path, Path]] = []
    for color_path in sorted(raw_dir.rglob("*blended_images*")):
        if not color_path.is_file() or color_path.suffix.lower() not in IMG_EXTS:
            continue
        depth_dir = color_path.parent.parent / "depth_maps"
        depth_path = depth_dir / (color_path.stem + ".png")
        if depth_path.exists():
            pairs.append((color_path, depth_path))

    if not pairs:
        # Fallback: generic search
        pairs = _find_color_depth_pairs(raw_dir)

    if not pairs:
        logger.error("No BlendedMVS color/depth pairs found in %s", raw_dir)
        return 0

    return _process_pairs(pairs, output_dir, args, name_prefix="bm_")


def prepare_tartanair(raw_dir: Path, output_dir: Path, args: argparse.Namespace) -> int:
    """Prepare TartanAir synthetic depth data.

    TartanAir layout:
        raw/
          <env_name>/
            <difficulty>/
              <trajectory>/
                image_left/
                  000000_left.png
                depth_left/
                  000000_left.npy
    """
    logger.info("Preparing TartanAir synthetic depth data…")

    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(raw_dir.rglob("*image_left*")):
        if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXTS:
            continue
        depth_path = Path(str(img_path).replace("image_left", "depth_left"))
        if depth_path.exists() and depth_path.suffix == ".npy":
            pairs.append((img_path, depth_path))

    if not pairs:
        logger.error("No TartanAir image/depth pairs found in %s", raw_dir)
        return 0

    return _process_pairs(pairs, output_dir, args, name_prefix="ta_")


# ─────────────────────────────────────────────────────────────────────────────
# Core Processing Pipeline
# ─────────────────────────────────────────────────────────────────────────────


def _process_pairs(
    pairs: list[tuple[Path, Path]],
    output_dir: Path,
    args: argparse.Namespace,
    name_prefix: str = "",
) -> int:
    """Process color/depth pairs into standard training format.

    Args:
        pairs: List of (color_path, depth_path) tuples.
        output_dir: Directory to write images/ and targets/ subdirs.
        args: Command-line arguments.
        name_prefix: Prefix for output filenames.

    Returns:
        Number of successfully processed pairs.
    """
    if not pairs:
        return 0

    max_samples = getattr(args, "max_samples", None)
    if max_samples is not None:
        pairs = pairs[:max_samples]

    img_dir = output_dir / "images"
    tgt_dir = output_dir / "targets"
    img_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    mapper = DepthRGBMapper()
    resolution = args.resolution
    overwrite = getattr(args, "overwrite", False)

    written = 0
    errors = 0
    start_time = time.time()

    for idx, (color_path, depth_path) in enumerate(pairs):
        stem = f"{name_prefix}{idx:07d}"
        out_img = img_dir / f"{stem}.png"
        out_tgt = tgt_dir / f"{stem}.png"

        if out_img.exists() and out_tgt.exists() and not overwrite:
            written += 1
            continue

        try:
            # Load color image
            color = Image.open(color_path).convert("RGB")
            color = _resize_rgb(color, resolution)

            # Load depth. Hypersim depth_meters is ray distance; the paper's task is
            # planar z-depth, so the Hypersim loader converts it (see depth_geometry).
            ray_mode = getattr(args, "ray_distance", "auto")
            focal_px = getattr(args, "focal_px", None)
            if depth_path.suffix.lower() in {".hdf5", ".h5"}:
                depth, _ = _hypersim_module()._load_depth_meters(
                    depth_path, ray_distance=ray_mode, focal_px=focal_px
                )
            else:
                depth = _load_depth_meters(depth_path)
                if depth.numel() == 0:
                    logger.warning("Empty depth loaded from %s, skipping", depth_path)
                    errors += 1
                    continue
                if ray_mode == "always" or (
                    ray_mode == "auto" and name_prefix == "hs_" and "depth_meters" in depth_path.stem.lower()
                ):
                    depth = ray_distance_to_planar_depth(depth, focal_px)

            depth = _resize_depth(depth, resolution)

            # Encode depth to RGB via DepthRGBMapper
            target_rgb = mapper.depth_to_rgb(depth).squeeze(0)
            _tensor_to_pil_rgb(target_rgb).save(out_tgt)
            color.save(out_img)
            written += 1

        except Exception as e:
            logger.error("Failed to process pair (%s, %s): %s", color_path, depth_path, e)
            errors += 1
            continue

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            logger.info(
                "Processed %d / %d pairs (%.1f samples/sec, %d errors)",
                idx + 1, len(pairs), rate, errors,
            )

    elapsed = time.time() - start_time
    logger.info(
        "Wrote %d / %d pairs to %s (%.1f sec, %d errors)",
        written, len(pairs), output_dir.resolve(), elapsed, errors,
    )

    # Write manifest
    _write_manifest(output_dir, pairs, args)

    # Validate if requested
    if getattr(args, "validate", False):
        try:
            from open_vision_banana.data.validation import validate_vision_dataset
            report = validate_vision_dataset(output_dir)
            print(report.summary())
            if not report.ok:
                logger.warning("Dataset validation failed for %s", output_dir)
        except Exception as e:
            logger.warning("Validation skipped: %s", e)

    return written


# ═══════════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════════


PREPROCESSOR_REGISTRY: dict[str, PreprocessorSpec] = {
    "hypersim-depth": PreprocessorSpec(
        name="hypersim-depth",
        display_name="Hypersim Depth",
        description="Apple's synthetic indoor scenes from UE4",
        prepare_fn=prepare_hypersim,
    ),
    "vkitti2-depth": PreprocessorSpec(
        name="vkitti2-depth",
        display_name="Virtual KITTI 2 Depth",
        description="NAVER Labs synthetic driving scenes",
        prepare_fn=prepare_vkitti2,
    ),
    "blendedmvs-depth": PreprocessorSpec(
        name="blendedmvs-depth",
        display_name="BlendedMVS Depth",
        description="UoA synthetic multi-view stereo scenes",
        prepare_fn=prepare_blendedmvs,
    ),
    "tartanair-depth": PreprocessorSpec(
        name="tartanair-depth",
        display_name="TartanAir Depth",
        description="CMU synthetic autonomous navigation",
        prepare_fn=prepare_tartanair,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess pure synthetic depth data for Vision Banana training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare Hypersim (already downloaded)
  python scripts/prepare_synthetic_depth.py --dataset hypersim-depth \\
      --raw data/hypersim/raw --output data/hypersim/train/ --resolution 512

  # Prepare all synthetic datasets in a directory
  python scripts/prepare_synthetic_depth.py --all \\
      --root data/synthetic_depth/ --output-dir data/prepared/

  # Validate output and limit samples
  python scripts/prepare_synthetic_depth.py --dataset tartanair-depth \\
      --raw data/tartanair/raw --output data/tartanair/train/ --validate --max-samples 1000

  # Force reprocessing with overwrite
  python scripts/prepare_synthetic_depth.py --dataset vkitti2-depth \\
      --raw data/vkitti2/raw --output data/vkitti2/train/ --overwrite
""",
    )
    parser.add_argument("--list", action="store_true", help="List available preprocessors")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name to prepare")
    parser.add_argument("--raw", type=str, default=None, help="Raw (extracted) dataset directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory (images/ + targets/)")
    parser.add_argument("--all", action="store_true", help="Prepare all datasets under --root")
    parser.add_argument("--root", type=str, default="data/synthetic_depth", help="Root directory containing multiple datasets")
    parser.add_argument("--output-dir", type=str, default="data/prepared", help="Output directory for --all mode")
    parser.add_argument("--resolution", type=int, default=512, help="Target square resolution (default: 512)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without writing")
    parser.add_argument("--validate", action="store_true", help="Run validate_vision_dataset() after preparation")
    parser.add_argument(
        "--ray-distance", choices=("auto", "always", "never"), default="auto",
        help="Convert Euclidean ray distance to planar z-depth ('auto': Hypersim depth_meters only)",
    )
    parser.add_argument("--focal-px", type=float, default=None, help="Focal length in px for the conversion")
    parser.add_argument("--colormaps", type=str, default=None, help="Comma-separated colormaps for multi-colormap training")
    return parser.parse_args()


def list_preprocessors() -> None:
    """Print all available preprocessors."""
    print("\n" + "=" * 70)
    print("  Synthetic Depth Preprocessors")
    print("=" * 70)
    print(f"  {'Name':<22} {'Display Name':<25} Description")
    print("-" * 70)
    for name, spec in sorted(PREPROCESSOR_REGISTRY.items()):
        print(f"  {name:<22} {spec.display_name:<25} {spec.description}")
    print("-" * 70)
    print(f"\n  Total: {len(PREPROCESSOR_REGISTRY)} preprocessors")
    print("\n  All datasets are 100% synthetic (zero real-world data)")
    print("=" * 70 + "\n")


def prepare_single(name: str, raw_dir: Path, output_dir: Path, args: argparse.Namespace) -> int:
    """Prepare a single dataset."""
    spec = PREPROCESSOR_REGISTRY.get(name)
    if spec is None:
        logger.error("Unknown preprocessor: %s. Run --list for available preprocessors.", name)
        return 0

    if not raw_dir.exists():
        logger.error("Raw directory not found: %s", raw_dir)
        return 0

    if args.dry_run:
        logger.info("[DRY RUN] Would prepare %s → %s", raw_dir, output_dir)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    return spec.prepare_fn(raw_dir, output_dir, args)


def main() -> int:
    args = parse_args()

    if args.list:
        list_preprocessors()
        return 0

    if args.all:
        # Auto-discover datasets under root
        root = Path(args.root)
        if not root.exists():
            logger.error("Root directory not found: %s", root)
            return 1

        total = 0
        for name in PREPROCESSOR_REGISTRY:
            dataset_root = root / name
            raw_candidates = [dataset_root / "raw", dataset_root]
            raw_dir = None
            for cand in raw_candidates:
                if cand.exists():
                    raw_dir = cand
                    break
            if raw_dir is None:
                logger.warning("No raw data found for %s, skipping", name)
                continue

            output_dir = Path(args.output_dir) / name
            count = prepare_single(name, raw_dir, output_dir, args)
            total += count

        logger.info("Total samples prepared across all datasets: %d", total)
        return 0

    if not args.dataset or not args.raw or not args.output:
        logger.error("Specify --dataset, --raw, and --output (or use --all)")
        return 1

    count = prepare_single(args.dataset, Path(args.raw), Path(args.output), args)
    return 0 if count > 0 else 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    sys.exit(main())
