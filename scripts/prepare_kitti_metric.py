#!/usr/bin/env python3
"""Prepare KITTI metric-depth pairs for Vision Banana (Eq.1 cube-edge RGB).

Pairs official raw ``image_02/data`` RGB with the depth-completion annotated
maps. The 13 Eigen/depth-completion val drives are excluded by default so
training cannot leak into a later KITTI val eval.

    python scripts/prepare_kitti_metric.py \
      --depth-root /data/kitti_depth_eval/extracted \
      --rgb-root /data/KITTI/raw_official \
      --output data/kitti_metric_512 \
      --resolution 512 --validate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.data.manifest import fingerprint_records

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Official KITTI depth-completion val drives. Never use these for training.
KITTI_VAL_DRIVES = {
    "2011_09_26_drive_0002_sync",
    "2011_09_26_drive_0005_sync",
    "2011_09_26_drive_0013_sync",
    "2011_09_26_drive_0020_sync",
    "2011_09_26_drive_0023_sync",
    "2011_09_26_drive_0036_sync",
    "2011_09_26_drive_0079_sync",
    "2011_09_26_drive_0095_sync",
    "2011_09_26_drive_0113_sync",
    "2011_09_28_drive_0037_sync",
    "2011_09_29_drive_0026_sync",
    "2011_09_30_drive_0016_sync",
    "2011_10_03_drive_0047_sync",
}


def find_rgb_frame(rgb_root: Path, drive: str, frame_name: str) -> Path | None:
    """Locate ``image_02/data/<frame>`` under either official or flattened raw extract."""
    date = "_".join(drive.split("_")[:3])
    candidates = [
        rgb_root / date / drive / "image_02" / "data" / frame_name,
        rgb_root / drive / "image_02" / "data" / frame_name,
        rgb_root / "train" / drive / "image_02" / "data" / frame_name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def find_kitti_metric_pairs(
    depth_root: Path,
    rgb_root: Path,
    *,
    split: str = "train",
    exclude_drives: set[str] | None = None,
) -> list[tuple[Path, Path, str]]:
    """Return ``(rgb, depth_png, drive)`` for annotated train frames with RGB."""
    blocked = set(KITTI_VAL_DRIVES)
    if exclude_drives:
        blocked.update(exclude_drives)
    split_dir = depth_root / split
    if not split_dir.is_dir():
        split_dir = depth_root
    pairs: list[tuple[Path, Path, str]] = []
    for drive_dir in sorted(path for path in split_dir.glob("*_sync") if path.is_dir()):
        drive = drive_dir.name
        if drive in blocked:
            continue
        depth_dir = drive_dir / "proj_depth" / "groundtruth" / "image_02"
        if not depth_dir.is_dir():
            continue
        for depth_path in sorted(depth_dir.glob("*.png")):
            rgb_path = find_rgb_frame(rgb_root, drive, depth_path.name)
            if rgb_path is not None:
                pairs.append((rgb_path, depth_path, drive))
    return pairs


def load_kitti_depth_meters(path: Path) -> torch.Tensor:
    """Load KITTI annotated depth PNG as ``[1, 1, H, W]`` metres (value / 256)."""
    with Image.open(path) as handle:
        array = np.array(handle, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D KITTI depth PNG, got {array.shape} from {path}")
    depth = torch.from_numpy(array).unsqueeze(0).unsqueeze(0) / 256.0
    return depth


def _tensor_to_pil_rgb(tensor: torch.Tensor) -> Image.Image:
    arr = (tensor.clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def write_metric_depth_pairs_jsonl(output: Path) -> int:
    records: list[dict] = []
    for image_path in sorted((output / "images").glob("*.png")):
        target_path = output / "targets" / image_path.name
        if not target_path.is_file():
            continue
        records.append(
            {
                "task": "metric_depth",
                "input": image_path.relative_to(output).as_posix(),
                "target": target_path.relative_to(output).as_posix(),
                "split": "train",
                "target_kind": "kitti_metric_rgb",
                "prompt": "Generate a metric depth visualization of this image.",
            }
        )
    rows = [json.dumps(record, ensure_ascii=False) for record in records]
    (output / "pairs.jsonl").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(records)


def prepare_kitti_metric(
    depth_root: Path,
    rgb_root: Path,
    output: Path,
    *,
    resolution: int = 512,
    max_samples: int | None = None,
    overwrite: bool = False,
    split: str = "train",
    exclude_drives: set[str] | None = None,
) -> int:
    pairs = find_kitti_metric_pairs(
        depth_root, rgb_root, split=split, exclude_drives=exclude_drives
    )
    if max_samples is not None:
        pairs = pairs[:max_samples]
    if not pairs:
        logger.error("No KITTI RGB/depth train pairs under %s + %s", depth_root, rgb_root)
        return 0

    img_dir = output / "images"
    tgt_dir = output / "targets"
    metric_dir = output / "metric_depth"
    img_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    mapper = DepthRGBMapper()
    written = 0
    digest = hashlib.sha256()
    depth_min, depth_max, depth_sum = float("inf"), float("-inf"), 0.0
    valid_pixels = 0

    for idx, (rgb_path, depth_path, drive) in enumerate(pairs):
        stem = f"{idx:06d}_{drive}_{depth_path.stem}"
        out_img = img_dir / f"{stem}.png"
        out_tgt = tgt_dir / f"{stem}.png"
        out_metric = metric_dir / f"{stem}.npy"
        digest.update(f"{stem}\0{rgb_path}\0{depth_path}\n".encode())
        if out_img.exists() and out_tgt.exists() and out_metric.exists() and not overwrite:
            written += 1
            continue

        color = Image.open(rgb_path).convert("RGB")
        color = color.resize((resolution, resolution), Image.Resampling.LANCZOS)
        raw_depth = load_kitti_depth_meters(depth_path)
        raw_depth = torch.nn.functional.interpolate(
            raw_depth, size=(resolution, resolution), mode="nearest"
        )
        valid = torch.isfinite(raw_depth) & (raw_depth > 0)
        depth = torch.where(valid, raw_depth, torch.zeros_like(raw_depth))
        target_rgb = mapper.depth_to_rgb(depth)
        _tensor_to_pil_rgb(target_rgb.squeeze(0)).save(out_tgt)
        color.save(out_img)
        metric = raw_depth.squeeze().cpu().numpy().astype(np.float32, copy=False)
        metric[~valid.squeeze().cpu().numpy()] = np.nan
        np.save(out_metric, metric)
        n_valid = int(valid.sum().item())
        valid_pixels += n_valid
        if n_valid:
            vals = raw_depth[valid]
            depth_sum += float(vals.sum().item())
            depth_min = min(depth_min, float(vals.min().item()))
            depth_max = max(depth_max, float(vals.max().item()))
        written += 1
        if (idx + 1) % 50 == 0:
            logger.info("Processed %d / %d KITTI pairs", idx + 1, len(pairs))

    jsonl_count = write_metric_depth_pairs_jsonl(output)
    records = [
        json.loads(line)
        for line in (output / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    manifest = {
        "schema_version": 2,
        # Preserve only portable provenance; pairs.jsonl carries the actual
        # training paths and is relative to this output directory.
        "source_depth": depth_root.name,
        "source_rgb": rgb_root.name,
        "split": split,
        "excluded_val_drives": sorted(KITTI_VAL_DRIVES),
        "pairs_written": written,
        "pair_manifest": "pairs.jsonl",
        "dataset_fingerprint": fingerprint_records(records, output),
        "fingerprint_algorithm": "sha256:canonical-pairs-v1",
        "path_policy": "manifest-relative",
        "output_resolution": resolution,
        "mapper": {"colormap": mapper.colormap, "lam": mapper.lam, "c": mapper.c},
        "depth_meters": {
            "min": None if valid_pixels == 0 else depth_min,
            "mean": None if valid_pixels == 0 else depth_sum / valid_pixels,
            "max": None if valid_pixels == 0 else depth_max,
            "valid_pixels": valid_pixels,
        },
        "sample_mapping_sha256": digest.hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %d KITTI metric pairs (%d jsonl rows)", written, jsonl_count)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--split", default="train")
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    count = prepare_kitti_metric(
        args.depth_root,
        args.rgb_root,
        args.output,
        resolution=args.resolution,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
        split=args.split,
    )
    if count == 0:
        return 1
    if args.validate:
        from open_vision_banana.data.validation import validate_vision_dataset

        report = validate_vision_dataset(args.output, task="depth")
        print(report.summary())
        if not report.ok:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
