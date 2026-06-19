#!/usr/bin/env python3
"""Prepare Hypersim depth data for Vision Banana instruction tuning.

Converts Hypersim-style paired color/depth files into the flat training layout
expected by ``VisionTaskDataset`` and ``scripts/validate_data.py``:

    output/
      images/     ← RGB input images
      targets/    ← depth encoded as RGB (DepthRGBMapper)

Hypersim source layout (scanned recursively):
    .../scene_*_color.png  ↔  .../scene_*_depth.png

Usage:
    python scripts/prepare_hypersim.py --source /data/hypersim --output data/hypersim_depth/
    python scripts/prepare_hypersim.py --source /data/hypersim --output data/hypersim_depth/ --max-samples 500 --validate
"""

from __future__ import annotations

import argparse
import logging
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


def find_hypersim_pairs(source: Path) -> list[tuple[Path, Path]]:
    """Find color/depth file pairs under *source*."""
    pairs: list[tuple[Path, Path]] = []
    for color_path in sorted(source.rglob("*")):
        if not color_path.is_file():
            continue
        if color_path.suffix.lower() not in IMG_EXTS:
            continue
        if "color" not in color_path.stem.lower() and "rgb" not in color_path.stem.lower():
            continue
        depth_path = Path(str(color_path).replace("color", "depth").replace("rgb", "depth"))
        if depth_path.exists():
            pairs.append((color_path, depth_path))
    return pairs


def _load_depth_meters(depth_path: Path) -> torch.Tensor:
    depth_img = Image.open(depth_path)
    depth = torch.from_numpy(np.array(depth_img)).float()
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.unsqueeze(0).unsqueeze(0)
    if depth.max() > 1.0:
        depth = depth / 1000.0
    return depth.clamp(min=0.0)


def _tensor_to_pil_rgb(tensor: torch.Tensor) -> Image.Image:
    arr = (tensor.clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def prepare_hypersim(
    source: Path,
    output: Path,
    *,
    max_samples: int | None = None,
    resolution: int | None = None,
    overwrite: bool = False,
) -> int:
    """Write ``images/`` + ``targets/`` under *output*."""
    pairs = find_hypersim_pairs(source)
    if not pairs:
        logger.error("No color/depth pairs found under %s", source)
        return 0

    if max_samples is not None:
        pairs = pairs[:max_samples]

    img_dir = output / "images"
    tgt_dir = output / "targets"
    img_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    mapper = DepthRGBMapper()
    written = 0

    for idx, (color_path, depth_path) in enumerate(pairs):
        stem = f"{idx:06d}_{color_path.stem.replace('color', '').replace('rgb', '').strip('_')}"
        out_img = img_dir / f"{stem}.png"
        out_tgt = tgt_dir / f"{stem}.png"

        if out_img.exists() and out_tgt.exists() and not overwrite:
            written += 1
            continue

        color = Image.open(color_path).convert("RGB")
        if resolution is not None:
            color = color.resize((resolution, resolution), Image.LANCZOS)

        depth = _load_depth_meters(depth_path)
        if resolution is not None:
            depth = torch.nn.functional.interpolate(
                depth, size=(resolution, resolution), mode="nearest"
            )

        target_rgb = mapper.depth_to_rgb(depth).squeeze(0)
        _tensor_to_pil_rgb(target_rgb).save(out_tgt)
        color.save(out_img)
        written += 1

        if (idx + 1) % 100 == 0:
            logger.info("Processed %d / %d pairs", idx + 1, len(pairs))

    logger.info("Wrote %d pairs to %s", written, output.resolve())
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Hypersim depth data for Vision Banana training"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Hypersim root directory (contains *color* / *depth* pairs)",
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
        help="Limit number of pairs to process",
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
        return 1

    count = prepare_hypersim(
        source,
        output,
        max_samples=args.max_samples,
        resolution=args.resolution,
        overwrite=args.overwrite,
    )
    if count == 0:
        return 1

    if args.validate:
        report = validate_vision_dataset(output)
        print(report.summary())
        if not report.ok:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
