#!/usr/bin/env python3
"""Prepare BlendedMVS depth data for Vision Banana instruction tuning.

BlendedMVS is a large-scale synthetic dataset for generalized multi-view stereo
networks, containing **entirely synthetic depth data from Blender simulation
engines** — zero real-world depth data, fully compliant with the paper
requirement of using simulation-only depth for training.

Dataset source: https://github.com/YoYo000/BlendedMVS
License: Academic (free for research use)
Estimated size: ~17 GB raw, ~34 GB after preparation

Converts BlendedMVS-style paired color/depth files into the flat training layout
expected by ``VisionTaskDataset`` and ``scripts/validate_data.py``:

    output/
      images/     ← RGB input images (*.png)
      targets/    ← depth encoded as RGB via DepthRGBMapper

BlendedMVS source layout (scanned recursively per scene PID):
    <PID>/
      blended_images/
        00000000.jpg              ← rendered RGB image
        00000000_masked.jpg       ← masked version (ignored)
        00000001.jpg
        ...
      rendered_depth_maps/
        00000000.pfm              ← ground-truth depth map (float32, meters)
        00000001.pfm
        ...
      cams/                       ← camera parameters (ignored for depth task)

Depth values are stored as float32 PFM files in metric meters. Some scenes
may contain empty or invalid depth maps when the textured mesh model is
incomplete; these are automatically filtered out during preparation.

Usage:
    # ── Step 1: Download ──
    wget -P data/blendedmvs/raw/ \
        https://github.com/YoYo000/BlendedMVS/releases/download/v1.0.0/BlendedMVS.zip
    unzip data/blendedmvs/raw/BlendedMVS.zip -d data/blendedmvs/raw/

    # ── Step 2: Prepare ──
    python scripts/prepare_blendedmvs.py \
        --source data/blendedmvs/raw/ \
        --output data/blendedmvs_depth/

    # With limited samples + resize + validation
    python scripts/prepare_blendedmvs.py \
        --source data/blendedmvs/raw/ \
        --output data/blendedmvs_depth/ \
        --max-samples 5000 --resolution 512 --validate

    # Or use the unified downloader
    python scripts/download_data.py --dataset blendedmvs-depth --output data/
"""

from __future__ import annotations

import argparse
import logging
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.data.validation import validate_vision_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ──────────────────────────────────────────────────────────────
# PFM I/O helpers (Portable Float Map format)
# ──────────────────────────────────────────────────────────────


def _read_pfm(path: Path) -> tuple[np.ndarray, float]:
    """Read a PFM depth file and return (array, scale).

    PFM format (grayscale, little-endian):
        Pf
        <width> <height>
        <scale>   (negative = little-endian)
        <raw float32 data>
    """
    with open(path, "rb") as f:
        header = f.readline().strip()
        if header not in (b"Pf", b"PF"):
            raise ValueError(f"Invalid PFM header in {path}: {header}")
        is_color = header == b"PF"

        dims = f.readline().strip()
        while dims.startswith(b"#") or dims == b"":
            dims = f.readline().strip()
        width, height = map(int, dims.split())

        scale_line = f.readline().strip()
        scale = float(scale_line)
        endian = "<" if scale < 0 else ">"

        dtype = f"{endian}f4"
        num_channels = 3 if is_color else 1
        num_floats = width * height * num_channels
        raw = f.read()
        expected_bytes = num_floats * 4
        if len(raw) < expected_bytes:
            raise ValueError(
                f"PFM file truncated: {path} expected {expected_bytes} bytes, got {len(raw)}"
            )

        arr = np.frombuffer(raw, dtype=dtype)
        if len(arr) != num_floats:
            raise ValueError(
                f"PFM data size mismatch: expected {num_floats}, got {len(arr)}"
            )

        if is_color:
            arr = arr.reshape((height, width, 3))
            # Use first channel for depth (all should be identical in BlendedMVS)
            arr = arr[..., 0]
        else:
            arr = arr.reshape((height, width))

        # Flip vertically (PFM stores rows bottom-to-top)
        arr = np.flipud(arr).copy()
        return arr, scale


def _load_depth_meters(pfm_path: Path) -> torch.Tensor | None:
    """Load a BlendedMVS PFM depth map and return metric depth tensor [1, 1, H, W].

    Returns None for invalid/empty depth maps (as noted in the BlendedMVS
    documentation: "rendered depth map and blended images might be empty as the
textured mesh model is not necessarily to be complete").
    """
    try:
        arr, _ = _read_pfm(pfm_path)
    except (ValueError, OSError, struct.error) as exc:
        logger.warning("Failed to read PFM %s: %s", pfm_path, exc)
        return None

    # Detect and reject empty/invalid depth maps
    if arr.size == 0 or not np.isfinite(arr).any():
        logger.warning("Empty or non-finite depth map: %s", pfm_path)
        return None

    # BlendedMVS depth is in metric meters (float32, from simulation)
    # Clamp negatives to zero and extreme outliers
    depth = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    depth = torch.clamp(depth, min=0.0)

    # Reject effectively empty maps (all zeros or all same value)
    valid_mask = depth > 0
    if valid_mask.sum() < 10:
        logger.warning("Near-empty depth map (<%d valid pixels): %s", 10, pfm_path)
        return None

    return depth


# ──────────────────────────────────────────────────────────────
# Pair discovery
# ──────────────────────────────────────────────────────────────


def find_blendedmvs_pairs(source: Path) -> list[tuple[Path, Path]]:
    """Find all valid color/depth pairs under *source*.

    Scans for scene directories (PID folders) containing:
        - blended_images/00000000.jpg   (non-masked images only)
        - rendered_depth_maps/00000000.pfm

    Skips *_masked.jpg images and filters out empty/invalid PFM files.
    """
    pairs: list[tuple[Path, Path]] = []
    scanned = 0

    for pid_dir in sorted(source.rglob("*")):
        if not pid_dir.is_dir():
            continue

        img_dir = pid_dir / "blended_images"
        depth_dir = pid_dir / "rendered_depth_maps"
        if not img_dir.exists() or not depth_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("*.jpg")):
            # Skip masked versions — we want the clean blended render
            if "_masked" in img_path.stem:
                continue

            depth_name = img_path.stem + ".pfm"
            depth_path = depth_dir / depth_name
            if not depth_path.exists():
                continue

            pairs.append((img_path, depth_path))
            scanned += 1

    logger.info(
        "Scanned %d scenes, found %d color/depth pairs (before filtering)",
        scanned,
        len(pairs),
    )
    return pairs


# ──────────────────────────────────────────────────────────────
# Tensor ↔ PIL helpers
# ──────────────────────────────────────────────────────────────


def _tensor_to_pil_rgb(tensor: torch.Tensor) -> Image.Image:
    arr = (tensor.clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


# ──────────────────────────────────────────────────────────────
# Main preparation pipeline
# ──────────────────────────────────────────────────────────────


def prepare_blendedmvs(
    source: Path,
    output: Path,
    *,
    max_samples: int | None = None,
    resolution: int | None = None,
    overwrite: bool = False,
    skip_invalid: bool = True,
) -> int:
    """Write ``images/`` + ``targets/`` under *output* from BlendedMVS raw data.

    Args:
        source: Root directory containing BlendedMVS scene PID folders.
        output: Output directory (will contain images/ and targets/).
        max_samples: Limit number of pairs to process (useful for quick tests).
        resolution: If given, resize images and depth to this square resolution.
        overwrite: Overwrite existing output files.
        skip_invalid: Skip pairs with unreadable or empty depth maps.

    Returns:
        Number of successfully written pairs.
    """
    pairs = find_blendedmvs_pairs(source)
    if not pairs:
        logger.error("No BlendedMVS color/depth pairs found under %s", source)
        logger.error(
            "Expected structure: <PID>/blended_images/*.jpg + <PID>/rendered_depth_maps/*.pfm"
        )
        return 0

    if max_samples is not None:
        pairs = pairs[:max_samples]
        logger.info("Limited to first %d samples", max_samples)

    img_dir = output / "images"
    tgt_dir = output / "targets"
    img_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    mapper = DepthRGBMapper()
    written = 0
    skipped = 0
    errors = 0

    for idx, (color_path, depth_path) in enumerate(pairs):
        # Build a deterministic stem: scene index + original filename
        pid_name = depth_path.parent.parent.name
        stem = f"{idx:06d}_{pid_name}_{color_path.stem}"
        out_img = img_dir / f"{stem}.png"
        out_tgt = tgt_dir / f"{stem}.png"

        if out_img.exists() and out_tgt.exists() and not overwrite:
            written += 1
            continue

        # Load color image
        try:
            color = Image.open(color_path).convert("RGB")
        except OSError as exc:
            logger.warning("Failed to read image %s: %s", color_path, exc)
            errors += 1
            continue

        if resolution is not None:
            color = color.resize((resolution, resolution), Image.Resampling.LANCZOS)

        # Load depth PFM
        depth = _load_depth_meters(depth_path)
        if depth is None:
            if skip_invalid:
                skipped += 1
                continue
            else:
                # Create a black target as fallback
                depth = torch.zeros(1, 1, color.size[1], color.size[0])

        # Ensure depth matches image size before resize
        if resolution is not None:
            depth = torch.nn.functional.interpolate(
                depth, size=(resolution, resolution), mode="nearest"
            )

        # Encode depth as RGB target
        try:
            target_rgb = mapper.depth_to_rgb(depth).squeeze(0)
            _tensor_to_pil_rgb(target_rgb).save(out_tgt)
            color.save(out_img)
            written += 1
        except Exception as exc:
            logger.error("Failed to encode depth for %s: %s", depth_path, exc)
            errors += 1
            continue

        if (idx + 1) % 100 == 0 or idx == len(pairs) - 1:
            logger.info(
                "Processed %d / %d pairs (written=%d, skipped=%d, errors=%d)",
                idx + 1,
                len(pairs),
                written,
                skipped,
                errors,
            )

    logger.info(
        "Finished: wrote %d pairs to %s (skipped=%d, errors=%d)",
        written,
        output.resolve(),
        skipped,
        errors,
    )
    return written


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare BlendedMVS synthetic depth data for Vision Banana training"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="BlendedMVS root directory (contains PID folders with blended_images/ and rendered_depth_maps/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory (will contain images/ and targets/)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of pairs to process (useful for quick tests)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Resize images and depth to this square resolution",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--no-skip-invalid",
        action="store_true",
        help="Do NOT skip invalid/empty depth maps (use black fallback instead)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validate_vision_dataset() after preparation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)

    if not source.exists():
        logger.error("Source directory not found: %s", source)
        logger.error(
            "Download BlendedMVS from:\n"
            "  wget https://github.com/YoYo000/BlendedMVS/releases/download/v1.0.0/BlendedMVS.zip\n"
            "  unzip BlendedMVS.zip -d %s",
            source,
        )
        return 1

    count = prepare_blendedmvs(
        source,
        output,
        max_samples=args.max_samples,
        resolution=args.resolution,
        overwrite=args.overwrite,
        skip_invalid=not args.no_skip_invalid,
    )
    if count == 0:
        logger.error("No valid pairs were written. Check source directory structure.")
        return 1

    if args.validate:
        report = validate_vision_dataset(output)
        print(report.summary())
        if not report.ok:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
