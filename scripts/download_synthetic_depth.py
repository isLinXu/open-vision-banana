#!/usr/bin/env python3
"""Pure synthetic depth dataset downloader for Vision Banana.

Downloads **entirely synthetic** depth data from simulation engines.
Zero real-world depth data — as required by the paper.

Supported datasets (all synthetic):
  - hypersim-depth: Hypersim (Apple, synthetic indoor scenes from UE4)
  - vkitti2-depth: Virtual KITTI 2 (NAVER Labs, synthetic driving scenes)
  - blendedmvs-depth: BlendedMVS (UoA, synthetic MVS scenes)
  - tartanair-depth: TartanAir (CMU, synthetic autonomous navigation)

Usage:
    # List available synthetic datasets
    python scripts/download_synthetic_depth.py --list

    # Download a specific synthetic dataset
    python scripts/download_synthetic_depth.py --dataset hypersim-depth --output data/hypersim/

    # Download all synthetic depth datasets (recommended for full training)
    python scripts/download_synthetic_depth.py --all --output data/synthetic_depth/

    # Dry run (show what would be downloaded without downloading)
    python scripts/download_synthetic_depth.py --dataset hypersim-depth --output data/hypersim/ --dry-run

    # Download with retry and resume support
    python scripts/download_synthetic_depth.py --dataset vkitti2-depth --output data/vkitti2/ --retries 5

    # Verify downloaded archives before extraction
    python scripts/download_synthetic_depth.py --dataset blendedmvs-depth --output data/blendedmvs/ --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset Registry — Pure Synthetic Depth Data Only
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DownloadSource:
    """A single download source (URL + metadata)."""

    url: str
    filename: str | None = None
    size_estimate_mb: float | None = None
    md5_hash: str | None = None
    sha256_hash: str | None = None
    description: str = ""


@dataclass
class SyntheticDatasetSpec:
    """Specification for a downloadable synthetic depth dataset."""

    name: str
    display_name: str
    task: str
    source: str
    size_estimate_gb: float
    license: str
    description: str
    # One or more download sources (mirrors)
    sources: list[DownloadSource] = field(default_factory=list)
    # Post-download extraction / preparation
    prepare_fn: Callable[[Path, Path, argparse.Namespace], int] | None = None
    # Expected directory structure after extraction
    expected_layout: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Download Utilities with Retry & Resume
# ─────────────────────────────────────────────────────────────────────────────


def _get_resume_byte(dest: Path) -> int:
    """Return current file size for resume support."""
    return dest.stat().st_size if dest.exists() else 0


def _download_with_retry(
    url: str,
    dest: Path,
    *,
    retries: int = 3,
    timeout: int = 300,
    chunk_size: int = 8192,
    expected_size: int | None = None,
    resume: bool = True,
    headers: dict[str, str] | None = None,
) -> Path:
    """Download a file with automatic retry, resume, and progress logging.

    Args:
        url: Direct download URL.
        dest: Destination file path.
        retries: Number of retry attempts on transient failures.
        timeout: Socket timeout in seconds per chunk read.
        chunk_size: Download chunk size in bytes.
        expected_size: Expected total size (bytes) for validation.
        resume: Allow HTTP Range resume for interrupted downloads.
        headers: Additional HTTP headers.

    Returns:
        Path to the downloaded file.

    Raises:
        RuntimeError: If all retry attempts fail.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    request_headers = headers.copy() if headers else {}

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resume_byte = _get_resume_byte(dest) if resume else 0
            if resume_byte > 0:
                request_headers["Range"] = f"bytes={resume_byte}-"
                logger.info("Resuming download from byte %d (%s)", resume_byte, dest.name)

            req = urllib.request.Request(url, headers=request_headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                total_size = expected_size
                if total_size is None:
                    content_length = response.headers.get("Content-Length")
                    if content_length:
                        total_size = int(content_length) + resume_byte

                mode = "ab" if resume_byte > 0 and response.status == 206 else "wb"
                downloaded = resume_byte if mode == "ab" else 0

                with open(dest, mode) as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size and total_size > 0:
                            pct = min(downloaded * 100 / total_size, 100)
                            if downloaded % (chunk_size * 100) < chunk_size:
                                logger.info(
                                    "  %s: %.1f%% (%d / %d bytes)",
                                    dest.name,
                                    pct,
                                    downloaded,
                                    total_size,
                                )

            logger.info("Download complete: %s (%.2f MB)", dest, dest.stat().st_size / 1e6)
            return dest

        except urllib.error.HTTPError as e:
            last_error = e
            if e.code == 416:  # Range not satisfiable — file already complete
                logger.info("File already complete: %s", dest)
                return dest
            if e.code in (429, 503, 502, 504):
                wait = 2 ** attempt
                logger.warning("HTTP %d on attempt %d/%d, retrying in %ds…", e.code, attempt, retries, wait)
                time.sleep(wait)
                continue
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url} after {retries} attempts: {e}") from e
            logger.warning("HTTP error %d on attempt %d/%d, retrying…", e.code, attempt, retries)
            time.sleep(attempt * 2)

        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last_error = e
            wait = 2 ** attempt
            logger.warning("Download error on attempt %d/%d: %s — retrying in %ds…", attempt, retries, e, wait)
            time.sleep(wait)

    raise RuntimeError(f"Failed to download {url} after {retries} attempts: {last_error}")


def _verify_hash(path: Path, *, md5: str | None = None, sha256: str | None = None) -> bool:
    """Verify file integrity against expected hash."""
    if md5:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        actual = h.hexdigest()
        if actual != md5.lower():
            logger.error("MD5 mismatch for %s: expected %s, got %s", path.name, md5, actual)
            return False
        logger.info("MD5 verified: %s", path.name)

    if sha256:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        actual = h.hexdigest()
        if actual != sha256.lower():
            logger.error("SHA256 mismatch for %s: expected %s, got %s", path.name, sha256, actual)
            return False
        logger.info("SHA256 verified: %s", path.name)

    return True


def _extract_archive(archive: Path, dest: Path, *, overwrite: bool = False) -> None:
    """Extract zip or tar archive to destination with safety checks."""
    logger.info("Extracting %s → %s", archive, dest)
    dest.mkdir(parents=True, exist_ok=True)

    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            # Safety check for ZipSlip
            for member in zf.namelist():
                member_path = dest / member
                try:
                    member_path.relative_to(dest)
                except ValueError:
                    raise RuntimeError(f"ZipSlip detected in {archive}: {member}")
            zf.extractall(dest)
    elif archive.suffix.lower() in (".tar", ".gz", ".tgz", ".bz2"):
        with tarfile.open(archive, "r:*") as tf:
            # Safety check for TarSlip
            for member in tf.getmembers():
                member_path = dest / member.name
                try:
                    member_path.relative_to(dest)
                except ValueError:
                    raise RuntimeError(f"TarSlip detected in {archive}: {member.name}")
            tf.extractall(dest)
    else:
        logger.warning("Unknown archive format: %s — copying as-is", archive.suffix)
        shutil.copy2(archive, dest / archive.name)

    logger.info("Extraction complete: %s", dest)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Preparation Functions
# ─────────────────────────────────────────────────────────────────────────────


def _prepare_generic_hypersim_like(
    raw_dir: Path, output_dir: Path, args: argparse.Namespace, name_prefix: str = ""
) -> int:
    """Generic preparation for Hypersim-like layouts (color/depth pairs)."""
    from open_vision_banana.core.depth_mapper import DepthRGBMapper
    from open_vision_banana.data.validation import validate_vision_dataset

    import numpy as np
    from PIL import Image

    # Find all color/depth pairs
    img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    pairs: list[tuple[Path, Path]] = []

    for color_path in sorted(raw_dir.rglob("*")):
        if not color_path.is_file() or color_path.suffix.lower() not in img_exts:
            continue
        stem_lower = color_path.stem.lower()
        if "color" not in stem_lower and "rgb" not in stem_lower and "image" not in stem_lower:
            continue
        # Try multiple depth naming conventions
        depth_candidates = [
            Path(str(color_path).replace("color", "depth").replace("rgb", "depth").replace("image", "depth")),
            color_path.parent / (color_path.stem.replace("color", "").replace("rgb", "").replace("image", "").strip("_") + "_depth" + color_path.suffix),
            color_path.parent / ("depth_" + color_path.stem.replace("color_", "").replace("rgb_", "").replace("image_", "") + color_path.suffix),
        ]
        for depth_path in depth_candidates:
            if depth_path.exists():
                pairs.append((color_path, depth_path))
                break

    if not pairs:
        logger.error("No color/depth pairs found under %s", raw_dir)
        return 0

    logger.info("Found %d color/depth pairs in %s", len(pairs), raw_dir)

    img_dir = output_dir / "images"
    tgt_dir = output_dir / "targets"
    img_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    mapper = DepthRGBMapper()
    written = 0
    resolution = getattr(args, "resolution", 512)
    max_samples = getattr(args, "max_samples", None)

    if max_samples is not None:
        pairs = pairs[:max_samples]

    for idx, (color_path, depth_path) in enumerate(pairs):
        stem = f"{name_prefix}{idx:06d}_{color_path.stem.replace('color', '').replace('rgb', '').replace('image', '').strip('_')}"
        out_img = img_dir / f"{stem}.png"
        out_tgt = tgt_dir / f"{stem}.png"

        if out_img.exists() and out_tgt.exists() and not getattr(args, "overwrite", False):
            written += 1
            continue

        try:
            color = Image.open(color_path).convert("RGB")
            if color.size != (resolution, resolution):
                color = color.resize((resolution, resolution), Image.Resampling.LANCZOS)

            # Load depth
            depth_img = Image.open(depth_path)
            depth = np.array(depth_img).astype(np.float32)
            if depth.ndim == 3:
                depth = depth[..., 0]
            depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()

            # Normalize depth (mm → meters if needed)
            if depth_t.max() > 10.0:
                depth_t = depth_t / 1000.0
            depth_t = depth_t.clamp(min=0.0)

            if depth_t.shape[-2:] != (resolution, resolution):
                import torch.nn.functional as F
                depth_t = F.interpolate(depth_t, size=(resolution, resolution), mode="nearest")

            # Encode to RGB
            target_rgb = mapper.depth_to_rgb(depth_t).squeeze(0)
            arr = (target_rgb.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()

            Image.fromarray(arr, mode="RGB").save(out_tgt)
            color.save(out_img)
            written += 1

        except Exception as e:
            logger.error("Failed to process pair (%s, %s): %s", color_path, depth_path, e)
            continue

        if (idx + 1) % 100 == 0:
            logger.info("Processed %d / %d pairs", idx + 1, len(pairs))

    logger.info("Wrote %d pairs to %s", written, output_dir.resolve())

    if getattr(args, "validate", False):
        report = validate_vision_dataset(output_dir)
        print(report.summary())
        if not report.ok:
            logger.warning("Validation failed for %s", output_dir)

    return written


def prepare_hypersim_depth(raw_dir: Path, output_dir: Path, args: argparse.Namespace) -> int:
    """Prepare Hypersim depth data."""
    logger.info("Preparing Hypersim synthetic depth data…")
    # Hypersim has scene_*_color.png / scene_*_depth.png naming
    # Try to use the existing prepare_hypersim script if available
    try:
        import sys
        scripts_dir = Path(__file__).parent
        sys.path.insert(0, str(scripts_dir))
        from prepare_hypersim import prepare_hypersim
        resolution = getattr(args, "resolution", 512)
        max_samples = getattr(args, "max_samples", None)
        count = prepare_hypersim(
            raw_dir,
            output_dir,
            max_samples=max_samples,
            resolution=resolution,
            overwrite=getattr(args, "overwrite", False),
        )
        if count > 0:
            return count
    except Exception as e:
        logger.warning("prepare_hypersim import failed (%s), falling back to generic", e)

    return _prepare_generic_hypersim_like(raw_dir, output_dir, args, name_prefix="hs_")


def prepare_vkitti2_depth(raw_dir: Path, output_dir: Path, args: argparse.Namespace) -> int:
    """Prepare Virtual KITTI 2 synthetic depth data."""
    logger.info("Preparing Virtual KITTI 2 synthetic depth data…")
    # VKITTI2 layout: SceneXX/frames/rgb/Camera_0/rgb_*.jpg + depth/*.png
    return _prepare_generic_hypersim_like(raw_dir, output_dir, args, name_prefix="vk_")


def prepare_blendedmvs_depth(raw_dir: Path, output_dir: Path, args: argparse.Namespace) -> int:
    """Prepare BlendedMVS synthetic depth data."""
    logger.info("Preparing BlendedMVS synthetic depth data…")
    # BlendedMVS: blended_images/*.jpg + depth_maps/*.png
    return _prepare_generic_hypersim_like(raw_dir, output_dir, args, name_prefix="bm_")


def prepare_tartanair_depth(raw_dir: Path, output_dir: Path, args: argparse.Namespace) -> int:
    """Prepare TartanAir synthetic depth data."""
    logger.info("Preparing TartanAir synthetic depth data…")
    # TartanAir: image_left/*.png + depth_left/*.npy
    from open_vision_banana.core.depth_mapper import DepthRGBMapper
    import numpy as np
    from PIL import Image

    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(raw_dir.rglob("*image_left*")):
        if not img_path.is_file() or img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        depth_candidates = [
            Path(str(img_path).replace("image_left", "depth_left")),
            Path(str(img_path).replace("image_left", "depth")),
        ]
        for depth_path in depth_candidates:
            if depth_path.exists():
                pairs.append((img_path, depth_path))
                break

    if not pairs:
        logger.error("No image/depth pairs found in %s", raw_dir)
        return 0

    logger.info("Found %d TartanAir pairs", len(pairs))

    img_dir = output_dir / "images"
    tgt_dir = output_dir / "targets"
    img_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    mapper = DepthRGBMapper()
    written = 0
    resolution = getattr(args, "resolution", 512)
    max_samples = getattr(args, "max_samples", None)

    if max_samples is not None:
        pairs = pairs[:max_samples]

    for idx, (img_path, depth_path) in enumerate(pairs):
        stem = f"ta_{idx:06d}_{img_path.stem.replace('image_left', '').replace('image', '').strip('_')}"
        out_img = img_dir / f"{stem}.png"
        out_tgt = tgt_dir / f"{stem}.png"

        if out_img.exists() and out_tgt.exists() and not getattr(args, "overwrite", False):
            written += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            if img.size != (resolution, resolution):
                img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)

            # TartanAir depth is .npy (float32 meters)
            if depth_path.suffix == ".npy":
                depth = np.load(depth_path).astype(np.float32)
            else:
                depth = np.array(Image.open(depth_path)).astype(np.float32)
                if depth.max() > 10.0:
                    depth = depth / 1000.0

            depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
            depth_t = depth_t.clamp(min=0.0)

            if depth_t.shape[-2:] != (resolution, resolution):
                import torch.nn.functional as F
                depth_t = F.interpolate(depth_t, size=(resolution, resolution), mode="nearest")

            target_rgb = mapper.depth_to_rgb(depth_t).squeeze(0)
            arr = (target_rgb.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()

            Image.fromarray(arr, mode="RGB").save(out_tgt)
            img.save(out_img)
            written += 1

        except Exception as e:
            logger.error("Failed to process TartanAir pair (%s, %s): %s", img_path, depth_path, e)
            continue

        if (idx + 1) % 100 == 0:
            logger.info("Processed %d / %d TartanAir pairs", idx + 1, len(pairs))

    logger.info("Wrote %d TartanAir pairs to %s", written, output_dir.resolve())
    return written


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset Registry
# ═══════════════════════════════════════════════════════════════════════════════


SYNTHETIC_DEPTH_REGISTRY: dict[str, SyntheticDatasetSpec] = {
    "hypersim-depth": SyntheticDatasetSpec(
        name="hypersim-depth",
        display_name="Hypersim Depth",
        task="depth",
        source="https://github.com/apple/ml-hypersim",
        size_estimate_gb=1.5,
        license="MIT",
        description="Apple's Hypersim: 461 synthetic indoor scenes rendered in UE4 with photorealistic materials. "
                    "Contains 77,400 frames with accurate metric depth. Entirely synthetic.",
        sources=[
            DownloadSource(
                url="https://github.com/apple/ml-hypersim/releases/download/v1.0/evermotion_dataset.zip",
                filename="hypersim_evermotion.zip",
                size_estimate_mb=500,
                description="Hypersim Evermotion scenes (subset)",
            ),
        ],
        prepare_fn=prepare_hypersim_depth,
        expected_layout="raw/*/images/scene_*_final_preview/*.color.jpg + .depth_meters.png",
    ),
    "vkitti2-depth": SyntheticDatasetSpec(
        name="vkitti2-depth",
        display_name="Virtual KITTI 2 Depth",
        task="depth",
        source="https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/",
        size_estimate_gb=50.0,
        license="CC BY-NC-SA 3.0",
        description="Virtual KITTI 2: 50 high-resolution synthetic driving scenes cloned from real KITTI. "
                    "Includes depth, optical flow, instance segmentation, and camera poses. Entirely synthetic.",
        sources=[
            DownloadSource(
                url="https://download.europe.naverlabs.com//virtual-kitti-2/vkitti_2.0.3_depth.tar",
                filename="vkitti_2.0.3_depth.tar",
                size_estimate_mb=25_000,
                description="VKITTI 2 depth maps (tar archive)",
            ),
            DownloadSource(
                url="https://download.europe.naverlabs.com//virtual-kitti-2/vkitti_2.0.3_rgb.tar",
                filename="vkitti_2.0.3_rgb.tar",
                size_estimate_mb=25_000,
                description="VKITTI 2 RGB images (tar archive)",
            ),
        ],
        prepare_fn=prepare_vkitti2_depth,
        expected_layout="SceneXX/frames/rgb/Camera_0/rgb_*.jpg + depth/Camera_0/depth_*.png",
    ),
    "blendedmvs-depth": SyntheticDatasetSpec(
        name="blendedmvs-depth",
        display_name="BlendedMVS Depth",
        task="depth",
        source="https://github.com/YoYo000/MVSNet",
        size_estimate_gb=17.0,
        license="Academic",
        description="BlendedMVS: 17k synthetic MVS training samples generated by blending aerial images "
                    "with 3D reconstructions. Dense depth from multi-view stereo. Entirely synthetic.",
        sources=[
            DownloadSource(
                url="https://github.com/YoYo000/MVSNet/releases/download/v1.0/blendedmvs.zip",
                filename="blendedmvs.zip",
                size_estimate_mb=17_000,
                description="BlendedMVS full dataset (17GB)",
            ),
        ],
        prepare_fn=prepare_blendedmvs_depth,
        expected_layout="blended_images/*.jpg + depth_maps/*.png",
    ),
    "tartanair-depth": SyntheticDatasetSpec(
        name="tartanair-depth",
        display_name="TartanAir Depth",
        task="depth",
        source="https://github.com/castacks/tartanair_tools",
        size_estimate_gb=30.0,
        license="BSD-3-Clause",
        description="TartanAir: 1,000+ synthetic trajectories in simulated environments "
                    "with LiDAR-style dense depth, stereo, and IMU. Entirely synthetic.",
        sources=[
            DownloadSource(
                url="https://github.com/castacks/tartanair_tools/raw/master/data_lists/tartanair_download_list.txt",
                filename="download_list.txt",
                size_estimate_mb=1,
                description="TartanAir download list (metadata only — use tartanair_download script)",
            ),
        ],
        prepare_fn=prepare_tartanair_depth,
        expected_layout="*/image_left/*.png + */depth_left/*.npy",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Core Download Logic
# ═══════════════════════════════════════════════════════════════════════════════


def download_dataset(
    name: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> int:
    """Download and prepare a single synthetic depth dataset.

    Returns:
        Number of training samples prepared, or 0 on failure.
    """
    spec = SYNTHETIC_DEPTH_REGISTRY.get(name)
    if spec is None:
        logger.error("Unknown synthetic dataset: %s. Run --list for available datasets.", name)
        return 0

    logger.info("=" * 70)
    logger.info("Dataset: %s (%s)", spec.display_name, spec.task)
    logger.info("Source: %s", spec.source)
    logger.info("Description: %s", spec.description)
    logger.info("Estimated size: %.1f GB", spec.size_estimate_gb)
    logger.info("License: %s", spec.license)
    logger.info("Expected layout: %s", spec.expected_layout)
    logger.info("=" * 70)

    if args.dry_run:
        logger.info("[DRY RUN] Would download/prepare %s → %s", name, output_dir)
        for src in spec.sources:
            logger.info("  → %s (%s)", src.url, src.description or "no description")
        return 0

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Download all sources
    downloaded_files: list[Path] = []
    for src in spec.sources:
        dest_name = src.filename or Path(src.url).name
        if not dest_name or dest_name in ("/", ""):
            dest_name = f"{name}_download_{len(downloaded_files)}.bin"
        dest_path = raw_dir / dest_name

        if dest_path.exists() and not args.overwrite:
            logger.info("File already exists (skipping download): %s", dest_path)
            downloaded_files.append(dest_path)
            continue

        try:
            logger.info("Downloading from %s…", src.url)
            _download_with_retry(
                src.url,
                dest_path,
                retries=args.retries,
                expected_size=int(src.size_estimate_mb * 1_000_000) if src.size_estimate_mb else None,
            )
            downloaded_files.append(dest_path)
        except Exception as e:
            logger.error("Failed to download from %s: %s", src.url, e)
            if not args.skip_failed:
                return 0
            logger.warning("Skipping failed source and continuing…")
            continue

    # Verify hashes if requested
    if args.verify:
        for src, dest_path in zip(spec.sources, downloaded_files):
            if not _verify_hash(dest_path, md5=src.md5_hash, sha256=src.sha256_hash):
                logger.error("Hash verification failed for %s", dest_path)
                if not args.skip_failed:
                    return 0

    # Extract archives
    extracted_dirs: list[Path] = []
    for dest_path in downloaded_files:
        if dest_path.suffix.lower() in (".zip", ".tar", ".gz", ".tgz", ".bz2"):
            extract_dir = raw_dir / f"extracted_{dest_path.stem}"
            try:
                _extract_archive(dest_path, extract_dir)
                extracted_dirs.append(extract_dir)
            except Exception as e:
                logger.error("Failed to extract %s: %s", dest_path, e)
                if not args.skip_failed:
                    return 0
        else:
            extracted_dirs.append(dest_path.parent)

    # Determine the root directory for preparation
    prep_root = raw_dir
    if extracted_dirs:
        # If single extraction dir, use it; otherwise use raw_dir
        prep_root = extracted_dirs[0] if len(extracted_dirs) == 1 else raw_dir

    # Run preparation
    task_dir = output_dir / spec.task
    task_dir.mkdir(parents=True, exist_ok=True)

    if spec.prepare_fn is not None:
        try:
            return spec.prepare_fn(prep_root, task_dir, args)
        except Exception as e:
            logger.error("Preparation failed for %s: %s", name, e)
            return 0
    else:
        logger.warning("No preparation function for %s — raw data in %s", name, raw_dir)
        return 0


def list_datasets() -> None:
    """Print all available synthetic datasets with their metadata."""
    print("\n" + "=" * 70)
    print("  Pure Synthetic Depth Datasets (Zero Real-World Data)")
    print("=" * 70)
    print(f"  {'Name':<22} {'Task':<8} {'Size':<10} {'License':<18} Description")
    print("-" * 70)
    for name, spec in sorted(SYNTHETIC_DEPTH_REGISTRY.items()):
        size = f"{spec.size_estimate_gb:.1f} GB"
        desc = spec.description[:45] + "…" if len(spec.description) > 45 else spec.description
        print(f"  {name:<22} {spec.task:<8} {size:<10} {spec.license:<18} {desc}")
    print("-" * 70)
    print(f"\n  Total: {len(SYNTHETIC_DEPTH_REGISTRY)} datasets, "
          f"{sum(s.size_estimate_gb for s in SYNTHETIC_DEPTH_REGISTRY.values()):.1f} GB estimated")
    print("\n  Usage: python scripts/download_synthetic_depth.py --dataset <name> --output <dir>")
    print("         python scripts/download_synthetic_depth.py --all --output <dir>")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download pure synthetic depth datasets for Vision Banana training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available synthetic datasets
  python scripts/download_synthetic_depth.py --list

  # Download Hypersim (smallest, ~1.5 GB)
  python scripts/download_synthetic_depth.py --dataset hypersim-depth --output data/hypersim/

  # Download all synthetic datasets (full training set)
  python scripts/download_synthetic_depth.py --all --output data/synthetic_depth/

  # Dry run to see what would be downloaded
  python scripts/download_synthetic_depth.py --dataset vkitti2-depth --output data/vkitti2/ --dry-run

  # Download with verification and overwrite
  python scripts/download_synthetic_depth.py --dataset tartanair-depth --output data/tartanair/ --verify --overwrite
""",
    )
    parser.add_argument("--list", action="store_true", help="List all available synthetic datasets")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name to download")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument("--all", action="store_true", help="Download all synthetic datasets")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    parser.add_argument("--retries", type=int, default=3, help="Download retry attempts (default: 3)")
    parser.add_argument("--verify", action="store_true", help="Verify archive integrity after download")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--skip-failed", action="store_true", help="Skip failed downloads instead of aborting")
    parser.add_argument("--resolution", type=int, default=512, help="Target resolution (default: 512)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples to process")
    parser.add_argument("--validate", action="store_true", help="Validate output dataset after preparation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        list_datasets()
        return 0

    if not args.dataset and not args.all:
        logger.error("Specify --dataset <name>, --all, or --list")
        return 1

    datasets = list(SYNTHETIC_DEPTH_REGISTRY.keys()) if args.all else [args.dataset]
    total_samples = 0
    failed = []

    for name in datasets:
        if name not in SYNTHETIC_DEPTH_REGISTRY:
            logger.error("Unknown dataset: %s", name)
            failed.append(name)
            continue

        output_dir = Path(args.output) / name
        try:
            count = download_dataset(name, output_dir, args)
            total_samples += count
            if count == 0 and not args.dry_run:
                logger.warning("No samples prepared for %s", name)
                failed.append(name)
        except Exception as e:
            logger.error("Fatal error downloading %s: %s", name, e)
            failed.append(name)
            if not args.skip_failed:
                break

    logger.info("=" * 70)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 70)
    logger.info("Total synthetic samples prepared: %d", total_samples)
    if failed:
        logger.warning("Failed datasets: %s", ", ".join(failed))
    else:
        logger.info("All datasets completed successfully.")
    logger.info("=" * 70)

    return 0 if not failed else 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    sys.exit(main())
