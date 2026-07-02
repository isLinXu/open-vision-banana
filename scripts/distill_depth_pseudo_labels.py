#!/usr/bin/env python3
"""Distill pseudo depth labels using Depth-Anything V2 + cube_edge colormap.

This script generates a synthetic depth dataset by:
1. Loading source RGB images from the project's data directories.
2. Running Depth-Anything V2 Small (MPS-optimized) to predict relative depth.
3. Encoding the depth map via the Vision Banana cube-edge colormap (power transform
   + piecewise-linear RGB mapping).
4. Saving aligned {images, targets} pairs suitable for training scripts.

Usage:
    python scripts/distill_depth_pseudo_labels.py \
        --source-list data/.source_images_list.txt \
        --output-dir data/vb_depth_pseudo_5k \
        --augment-flip \
        --target-count 5000

The script is idempotent — re-running skips existing outputs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from transformers import pipeline

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_vision_banana.core.depth_mapper import DepthRGBMapper

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("depth_distill")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEPTH_MODEL_ID = "LiheYoung/depth-anything-small-hf"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_IMAGE_SIZE = 1024  # Resize very large images to keep MPS memory reasonable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def discover_source_images(
    source_list: Path | None = None,
    extra_dirs: list[Path] | None = None,
) -> list[Path]:
    """Collect source image paths from a text list and/or extra directories."""
    paths: set[Path] = set()

    if source_list and source_list.exists():
        with open(source_list, "r") as f:
            for line in f:
                p = Path(line.strip())
                if p.exists() and p.suffix.lower() in SUPPORTED_EXTS:
                    paths.add(p.resolve())
        logger.info("Loaded %d image paths from %s", len(paths), source_list)

    if extra_dirs:
        for d in extra_dirs:
            if d.exists() and d.is_dir():
                for p in d.rglob("*"):
                    if p.suffix.lower() in SUPPORTED_EXTS:
                        paths.add(p.resolve())
        logger.info("After scanning extra dirs: %d unique images", len(paths))

    return sorted(paths)


def resize_if_needed(img: Image.Image, max_size: int = MAX_IMAGE_SIZE) -> Image.Image:
    """Resize image so the longer edge <= max_size, preserving aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def augment_image(img: Image.Image, mode: str) -> Image.Image:
    """Apply a lightweight augmentation to increase dataset diversity."""
    if mode == "flip_h":
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if mode == "flip_v":
        return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if mode == "brightness_up":
        return ImageEnhance.Brightness(img).enhance(1.15)
    if mode == "brightness_down":
        return ImageEnhance.Brightness(img).enhance(0.85)
    if mode == "contrast_up":
        return ImageEnhance.Contrast(img).enhance(1.15)
    if mode == "contrast_down":
        return ImageEnhance.Contrast(img).enhance(0.85)
    return img


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> torch [1, 3, H, W] float32 in [0, 1]."""
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """torch [1, 3, H, W] float32 -> PIL RGB."""
    arr = (
        tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255
    ).astype(np.uint8)
    return Image.fromarray(arr)


def predict_depth(
    depth_pipe,
    image: Image.Image,
    device: torch.device,
) -> torch.Tensor:
    """Run Depth-Anything V2 and return a [1, 1, H, W] depth tensor."""
    # Pipeline returns a dict with "predicted_depth" (tensor) and "depth" (PIL)
    result = depth_pipe(image)
    # result["depth"] is PIL Image
    depth_pil = result["depth"]
    depth_arr = np.array(depth_pil).astype(np.float32)

    # Normalize to [0, 1] — Depth-Anything gives relative depth
    d_min, d_max = depth_arr.min(), depth_arr.max()
    if d_max > d_min:
        depth_arr = (depth_arr - d_min) / (d_max - d_min)
    else:
        depth_arr = np.zeros_like(depth_arr)

    depth_tensor = torch.from_numpy(depth_arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    return depth_tensor.to(device)


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------
def generate_dataset(
    source_images: list[Path],
    output_dir: Path,
    depth_pipe,
    device: torch.device,
    mapper: DepthRGBMapper,
    augment_modes: list[str] | None = None,
    target_count: int = 5000,
) -> dict:
    """Generate pseudo depth labels and return a statistics dict."""
    images_dir = output_dir / "images"
    targets_dir = output_dir / "targets"
    images_dir.mkdir(parents=True, exist_ok=True)
    targets_dir.mkdir(parents=True, exist_ok=True)

    # Determine how many augmentations per source we need
    n_sources = len(source_images)
    if n_sources == 0:
        raise RuntimeError("No source images found. Cannot generate dataset.")

    augment_modes = augment_modes or []
    variants_per_source = 1 + len(augment_modes)
    max_possible = n_sources * variants_per_source
    actual_count = min(target_count, max_possible)
    logger.info(
        "Sources: %d | Augment modes: %d | Variants/src: %d | Max possible: %d | Target: %d | Actual: %d",
        n_sources, len(augment_modes), variants_per_source, max_possible, target_count, actual_count,
    )

    stats = {
        "total_sources": n_sources,
        "augment_modes": augment_modes,
        "target_count": target_count,
        "actual_count": 0,
        "skipped": 0,
        "failed": 0,
        "output_dir": str(output_dir.resolve()),
    }

    generated = 0
    for src_idx, src_path in enumerate(source_images, start=1):
        if generated >= actual_count:
            break

        # Base image
        modes_to_process = ["base"] + augment_modes
        for mode in modes_to_process:
            if generated >= actual_count:
                break

            out_stem = f"{src_path.stem}_{mode}"
            out_img_path = images_dir / f"{out_stem}.png"
            out_tgt_path = targets_dir / f"{out_stem}.png"

            if out_img_path.exists() and out_tgt_path.exists():
                logger.debug("Skipping existing %s", out_stem)
                stats["skipped"] += 1
                generated += 1
                continue

            try:
                # Load and preprocess
                img = Image.open(src_path).convert("RGB")
                img = resize_if_needed(img, MAX_IMAGE_SIZE)
                if mode != "base":
                    img = augment_image(img, mode)

                # Predict depth
                depth_tensor = predict_depth(depth_pipe, img, device)

                # Encode depth -> cube_edge RGB
                rgb_tensor = mapper.depth_to_rgb(depth_tensor)  # [1, 3, H, W]

                # Convert back to PIL (match original image size)
                rgb_pil = tensor_to_pil(rgb_tensor)
                if rgb_pil.size != img.size:
                    rgb_pil = rgb_pil.resize(img.size, Image.Resampling.LANCZOS)

                # Save pair
                img.save(out_img_path, "PNG")
                rgb_pil.save(out_tgt_path, "PNG")

                generated += 1
                stats["actual_count"] = generated

                if generated % 50 == 0 or generated == actual_count:
                    logger.info(
                        "Progress: %d/%d (src %d/%d) | Last: %s",
                        generated, actual_count, src_idx, n_sources, out_stem,
                    )

            except Exception as exc:
                logger.warning("Failed to process %s mode=%s: %s", src_path.name, mode, exc)
                stats["failed"] += 1

    logger.info("Generation complete. Total: %d, Skipped: %d, Failed: %d", generated, stats["skipped"], stats["failed"])
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill pseudo depth labels using Depth-Anything V2")
    parser.add_argument(
        "--source-list",
        type=Path,
        default=PROJECT_ROOT / "data" / ".source_images_list.txt",
        help="Text file with one image path per line",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "vb_depth_pseudo_5k",
        help="Where to write {images,targets}/",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=5000,
        help="Desired number of output pairs (may be limited by source images)",
    )
    parser.add_argument(
        "--augment-flip",
        action="store_true",
        help="Add horizontal-flip augmentations",
    )
    parser.add_argument(
        "--augment-brightness",
        action="store_true",
        help="Add brightness +/- augmentations",
    )
    parser.add_argument(
        "--augment-contrast",
        action="store_true",
        help="Add contrast +/- augmentations",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable all augmentations (base images only)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cpu"],
        help="Torch device for inference",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for depth pipeline (keep 1 for MPS safety)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Device selection
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # Build augment mode list
    augment_modes: list[str] = []
    if not args.no_augment:
        if args.augment_flip:
            augment_modes.append("flip_h")
        if args.augment_brightness:
            augment_modes.extend(["brightness_up", "brightness_down"])
        if args.augment_contrast:
            augment_modes.extend(["contrast_up", "contrast_down"])
        # Default: if no specific augment flags given, use mild flip augmentation
        if not augment_modes and not any([args.augment_flip, args.augment_brightness, args.augment_contrast]):
            augment_modes = ["flip_h"]
            logger.info("No augment flags specified; defaulting to horizontal flip.")

    # Discover source images
    source_images = discover_source_images(args.source_list)
    if not source_images:
        logger.error("No source images found. Please check --source-list or data directories.")
        return 1

    # Load Depth-Anything V2 pipeline
    logger.info("Loading Depth-Anything V2 model: %s", DEPTH_MODEL_ID)
    depth_pipe = pipeline(
        "depth-estimation",
        model=DEPTH_MODEL_ID,
        device=str(device) if device.type != "mps" else device,
    )
    logger.info("Model loaded successfully.")

    # Initialize cube_edge mapper
    mapper = DepthRGBMapper(colormap="cube_edge", lam=-3.0, c=10.0 / 3.0)
    logger.info("DepthRGBMapper initialized with colormap=%s", mapper.colormap)

    # Generate dataset
    stats = generate_dataset(
        source_images=source_images,
        output_dir=args.output_dir,
        depth_pipe=depth_pipe,
        device=device,
        mapper=mapper,
        augment_modes=augment_modes,
        target_count=args.target_count,
    )

    # Save metadata
    import json
    meta_path = args.output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Metadata saved to %s", meta_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
