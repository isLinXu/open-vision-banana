#!/usr/bin/env python3
"""Prepare TartanAir synthetic depth data for Vision Banana instruction tuning.

Converts TartanAir-style paired RGB/depth files into the flat training layout
expected by ``VisionTaskDataset`` and ``scripts/validate_data.py``:

    output/
      images/     ← RGB input images
      targets/    ← depth encoded as RGB (DepthRGBMapper)

TartanAir source layout (scanned recursively):
    ENV_NAME/Easy/P000/image_left/000000_left.png
    ENV_NAME/Easy/P000/depth_left/000000_left_depth.npy

Depth values are metric (meters) stored as float32 NumPy arrays.
This is entirely synthetic data from Unreal Engine 4 simulation — zero
real-world depth data is used, satisfying the paper requirement.

Usage:
    python scripts/prepare_tartanair.py \
        --source data/tartanair/ \
        --output data/tartanair_depth/

    python scripts/prepare_tartanair.py \
        --source data/tartanair/ \
        --output data/tartanair_depth/ \
        --max-samples 1000 \
        --resolution 512 \
        --validate
"""

from __future__ import annotations

import argparse
import json
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

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEPTH_EXTS = {".npy"}


class TartanAirPairFinder:
    """Scan TartanAir directory tree for RGB/depth pairs.

    Expected layout after unzipping:
        source/
          ENV_NAME/
            Easy/
              P000/
                image_left/      ← 000000_left.png, 000001_left.png, ...
                depth_left/      ← 000000_left_depth.npy, 000001_left_depth.npy, ...
                ...
              P001/
                ...
            Hard/
              ...
    """

    def __init__(self, source: Path) -> None:
        self.source = source

    def find_pairs(self) -> list[tuple[Path, Path]]:
        """Find all (rgb_path, depth_path) pairs under source."""
        pairs: list[tuple[Path, Path]] = []

        # TartanAir stores RGB in image_left/ and depth in depth_left/
        for img_dir in sorted(self.source.rglob("image_left")):
            if not img_dir.is_dir():
                continue

            # Find corresponding depth directory
            depth_dir = img_dir.parent / "depth_left"
            if not depth_dir.is_dir():
                continue

            for img_path in sorted(img_dir.glob("*.png")):
                # Match depth file: 000000_left.png → 000000_left_depth.npy
                depth_name = img_path.stem + "_depth.npy"
                depth_path = depth_dir / depth_name
                if depth_path.exists():
                    pairs.append((img_path, depth_path))

        logger.info(
            "Found %d RGB/depth pairs under %s", len(pairs), self.source.resolve()
        )
        return pairs


def _load_depth_meters(depth_path: Path) -> torch.Tensor:
    """Load TartanAir depth .npy file and return metric depth tensor.

    TartanAir depth is stored as float32 NumPy arrays in meters.
    Returns shape [1, 1, H, W].
    """
    depth_arr = np.load(depth_path).astype(np.float32)

    # Ensure 2D shape
    if depth_arr.ndim > 2:
        # Some variants may have extra dims; squeeze singletons
        depth_arr = depth_arr.squeeze()

    depth = torch.from_numpy(depth_arr).float().unsqueeze(0).unsqueeze(0)

    # TartanAir depth is already in meters, but may contain inf/NaN
    depth = torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth))
    depth = depth.clamp(min=0.0)

    return depth


def _tensor_to_pil_rgb(tensor: torch.Tensor) -> Image.Image:
    """Convert a [3, H, W] float tensor in [0, 1] to a PIL RGB image."""
    arr = (tensor.clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def prepare_tartanair(
    source: Path,
    output: Path,
    *,
    max_samples: int | None = None,
    resolution: int | None = None,
    overwrite: bool = False,
) -> int:
    """Write ``images/`` + ``targets/`` under *output* from TartanAir source data.

    Args:
        source: Root directory containing extracted TartanAir data.
        output: Output directory (will contain images/ and targets/).
        max_samples: Limit number of pairs to process (for quick validation).
        resolution: Resize images and depth to this square resolution.
        overwrite: Overwrite existing output files.

    Returns:
        Number of successfully written pairs.
    """
    finder = TartanAirPairFinder(source)
    pairs = finder.find_pairs()
    if not pairs:
        logger.error("No RGB/depth pairs found under %s", source.resolve())
        return 0

    if max_samples is not None:
        pairs = pairs[:max_samples]
        logger.info("Limited to %d pairs for processing", max_samples)

    img_dir = output / "images"
    tgt_dir = output / "targets"
    metric_dir = output / "metric_depth"
    img_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    mapper = DepthRGBMapper()
    written = 0

    for idx, (color_path, depth_path) in enumerate(pairs):
        # Build a unique stem: env_diff_traj_frame
        # e.g., abandonedfactory_Easy_P000_000000
        rel_parts = color_path.relative_to(source).parts
        env_name = rel_parts[0] if len(rel_parts) > 0 else "unknown"
        diff = rel_parts[1] if len(rel_parts) > 1 else "unknown"
        traj = rel_parts[2] if len(rel_parts) > 2 else "unknown"
        frame = color_path.stem  # e.g., 000000_left

        stem = f"{idx:06d}_{env_name}_{diff}_{traj}_{frame}"
        out_img = img_dir / f"{stem}.png"
        out_tgt = tgt_dir / f"{stem}.png"
        out_metric = metric_dir / f"{stem}.npy"

        if out_img.exists() and out_tgt.exists() and out_metric.exists() and not overwrite:
            written += 1
            continue

        try:
            # Load RGB
            color = Image.open(color_path).convert("RGB")
            if resolution is not None:
                color = color.resize((resolution, resolution), Image.Resampling.LANCZOS)

            # Load depth (metric, meters)
            depth = _load_depth_meters(depth_path)
            if resolution is not None:
                depth = torch.nn.functional.interpolate(
                    depth, size=(resolution, resolution), mode="nearest"
                )

            # Encode depth to RGB via DepthRGBMapper
            target_rgb = mapper.depth_to_rgb(depth).squeeze(0)
            _tensor_to_pil_rgb(target_rgb).save(out_tgt)
            color.save(out_img)
            # Preserve actual metric evidence for fail-closed training admission;
            # never infer validity from the encoded RGB target.
            np.save(out_metric, depth.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32))
            written += 1

        except Exception as exc:
            logger.error("Failed to process pair %s / %s: %s", color_path, depth_path, exc)
            continue

        if (idx + 1) % 100 == 0:
            logger.info("Processed %d / %d pairs", idx + 1, len(pairs))

    rows = write_depth_pairs_jsonl(output)
    logger.info("Wrote %d pairs to %s (%d pairs.jsonl rows)", written, output.resolve(), rows)
    return written


def write_depth_pairs_jsonl(output: Path) -> int:
    """Rewrite ``pairs.jsonl`` (manifest-relative paths) from every images/targets pair.

    Same contract as ``prepare_hypersim.write_metric_depth_pairs_jsonl``: the unified
    Trainer only consumes ``pair_manifest`` files, so a bare images/ + targets/ layout
    is not loadable without this.
    """
    records = []
    for image_path in sorted((output / "images").glob("*.png")):
        target_path = output / "targets" / image_path.name
        if not target_path.is_file():
            continue
        records.append(
            {
                "task": "depth",
                "input": image_path.relative_to(output).as_posix(),
                "target": target_path.relative_to(output).as_posix(),
                "metric_depth": f"metric_depth/{image_path.stem}.npy",
                "split": "train",
                "target_kind": "tartanair_metric_rgb",
                "prompt": "Generate a metric depth visualization of this image.",
            }
        )
    (output / "pairs.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + ("\n" if records else ""),
        encoding="utf-8",
    )
    return len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare TartanAir synthetic depth data for Vision Banana training"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="TartanAir root directory (contains extracted ENV_NAME/Easy/P000/image_left etc.)",
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

    count = prepare_tartanair(
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
