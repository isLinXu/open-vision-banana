#!/usr/bin/env python3
"""Prepare Hypersim depth data for Vision Banana instruction tuning.

Converts Hypersim-style paired color/depth files into a flat training layout
while retaining lossless metric targets for benchmark-style evaluation:

    output/
      images/     ← RGB input images
      targets/    ← depth encoded as RGB (DepthRGBMapper)
      metric_depth/ ← float32 metric depth maps (NaN marks invalid pixels)
      manifest.json ← mapper parameters and data-quality audit

Hypersim source layout (scanned recursively):
    .../frame.0000.color.jpg  ↔  .../frame.0000.depth_meters.hdf5

Usage:
    python scripts/prepare_hypersim.py --source /data/hypersim --output data/hypersim_depth/
    python scripts/prepare_hypersim.py --source /data/hypersim --output data/hypersim_depth/ --max-samples 500 --validate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.data.manifest import fingerprint_records
from open_vision_banana.data.validation import validate_vision_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEPTH_EXTS = (".hdf5", ".h5", ".npy", ".png", ".tif", ".tiff")
_COLOR_MARKER = re.compile(r"(?:color|rgb)", re.IGNORECASE)
_FINAL_PREVIEW_DIR = re.compile(r"^(scene_cam_\d+)_final_preview$", re.IGNORECASE)


def _official_geometry_depth_candidates(color_path: Path) -> list[Path]:
    """Map official Hypersim preview RGB onto geometry-pass metric depth.

    Layout::

        .../scene_cam_00_final_preview/frame.0000.color.jpg
        .../scene_cam_00_geometry_hdf5/frame.0000.depth_meters.hdf5
    """
    match = _FINAL_PREVIEW_DIR.match(color_path.parent.name)
    if not match:
        return []
    geometry_dir = color_path.parent.parent / f"{match.group(1)}_geometry_hdf5"
    stem = _COLOR_MARKER.sub("depth_meters", color_path.stem)
    return [geometry_dir / f"{stem}{suffix}" for suffix in DEPTH_EXTS]


def _depth_candidates(color_path: Path) -> list[Path]:
    """Return conventional Hypersim depth names for a color image."""
    candidates: list[Path] = []
    candidates.extend(_official_geometry_depth_candidates(color_path))
    for replacement in ("depth_meters", "depth"):
        stem = _COLOR_MARKER.sub(replacement, color_path.stem)
        candidates.extend(color_path.with_name(f"{stem}{suffix}") for suffix in DEPTH_EXTS)

        # Some exports keep modality in a parent directory instead of the stem.
        swapped_parts = [_COLOR_MARKER.sub(replacement, part) for part in color_path.parts[:-1]]
        swapped_parent = Path(*swapped_parts)
        if swapped_parent != color_path.parent:
            candidates.extend(
                swapped_parent / f"{color_path.stem}{suffix}" for suffix in DEPTH_EXTS
            )
    return candidates


def find_hypersim_pairs(
    source: Path,
    *,
    exclude_scenes: set[str] | None = None,
) -> list[tuple[Path, Path]]:
    """Find color/depth file pairs under *source* without assuming PNG depth."""
    excluded = {name for name in (exclude_scenes or set()) if name}
    pairs: list[tuple[Path, Path]] = []
    color_paths: list[Path] = []
    for suffix in IMG_EXTS:
        color_paths.extend(path for path in source.rglob(f"*{suffix}") if path.is_file())
    for color_path in sorted(color_paths):
        if excluded and any(part in excluded for part in color_path.parts):
            continue
        if "color" not in color_path.stem.lower() and "rgb" not in color_path.stem.lower():
            continue
        # Preview PNGs under *_geometry_preview are visualizations, not metric metres.
        if color_path.parent.name.lower().endswith("_geometry_preview"):
            continue
        depth_path = next((path for path in _depth_candidates(color_path) if path.is_file()), None)
        if depth_path is not None:
            pairs.append((color_path, depth_path))
    return pairs


def write_metric_depth_pairs_jsonl(output: Path) -> int:
    """Rewrite portable ``pairs.jsonl`` from every image/target pair."""
    records: list[dict[str, Any]] = []
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
                "target_kind": "hypersim_metric_rgb",
                "prompt": "Generate a metric depth visualization of this image.",
            }
        )
    rows = [json.dumps(record, ensure_ascii=False) for record in records]
    (output / "pairs.jsonl").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(records)


def _load_depth_meters(depth_path: Path) -> torch.Tensor:
    """Load a raw metric depth map as ``[1, 1, H, W]`` in metres.

    Official Hypersim depth is stored in HDF5 and already uses metres. PNG
    exports are conventionally millimetres, matching the previous converter.
    """
    suffix = depth_path.suffix.lower()
    if suffix in {".hdf5", ".h5"}:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Reading Hypersim HDF5 depth requires h5py. Install "
                "open-vision-banana[data] or h5py."
            ) from exc
        with h5py.File(depth_path, "r") as handle:
            dataset = handle.get("dataset")
            if isinstance(dataset, h5py.Dataset):
                array = np.array(dataset, dtype=np.float32, copy=True)
            else:
                datasets = []
                handle.visititems(
                    lambda _name, value: (
                        datasets.append(value) if isinstance(value, h5py.Dataset) else None
                    )
                )
                if len(datasets) != 1:
                    raise ValueError(
                        f"Expected one depth dataset in {depth_path}, found {len(datasets)}"
                    )
                array = np.array(datasets[0], dtype=np.float32, copy=True)
    elif suffix == ".npy":
        array = np.load(depth_path)
    else:
        with Image.open(depth_path) as depth_img:
            array = np.asarray(depth_img)
        if array.dtype.kind in {"u", "i"} and array.dtype.itemsize >= 2:
            array = array.astype(np.float32) / 1000.0

    array = np.array(array, dtype=np.float32, copy=True)
    depth = torch.from_numpy(array)
    if depth.ndim == 3:
        if depth.shape[-1] == 1:
            depth = depth[..., 0]
        elif depth.shape[0] == 1:
            depth = depth[0]
        else:
            raise ValueError(f"Expected one-channel depth map, got {tuple(depth.shape)}")
    if depth.ndim != 2:
        raise ValueError(f"Expected 2-D depth map, got {tuple(depth.shape)} from {depth_path}")
    depth = depth.unsqueeze(0).unsqueeze(0)
    return depth


def _tensor_to_pil_rgb(tensor: torch.Tensor) -> Image.Image:
    arr = (tensor.clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def _update_depth_audit(
    audit: dict[str, float | int],
    depth: torch.Tensor,
    target_rgb: torch.Tensor,
    mapper: DepthRGBMapper,
) -> None:
    """Accumulate finite metric-depth and RGB roundtrip statistics."""
    valid = torch.isfinite(depth) & (depth > 0)
    audit["valid_pixels"] += int(valid.sum().item())
    audit["invalid_pixels"] += int((~valid).sum().item())
    if not valid.any():
        return

    valid_depth = depth[valid]
    audit["depth_sum"] += float(valid_depth.sum().item())
    audit["depth_min"] = min(float(audit["depth_min"]), float(valid_depth.min().item()))
    audit["depth_max"] = max(float(audit["depth_max"]), float(valid_depth.max().item()))

    quantized_rgb = _tensor_to_pil_rgb(target_rgb.squeeze(0))
    quantized = torch.from_numpy(np.asarray(quantized_rgb, dtype=np.float32)).permute(2, 0, 1)
    quantized = (quantized / 255.0).unsqueeze(0)
    decoded = mapper.rgb_to_depth(quantized)
    errors = (decoded[valid] - valid_depth).abs()
    audit["roundtrip_abs_error_sum"] += float(errors.sum().item())
    audit["roundtrip_abs_error_max"] = max(
        float(audit["roundtrip_abs_error_max"]), float(errors.max().item())
    )


def _write_manifest(
    output: Path,
    *,
    source: Path,
    resolution: int | None,
    mapper: DepthRGBMapper,
    pair_count: int,
    written: int,
    source_formats: dict[str, int],
    audit: dict[str, float | int],
    sample_digest: Any,
) -> None:
    valid_pixels = int(audit["valid_pixels"])
    payload = {
        "schema_version": 2,
        # Source data may live outside the repository. Keep provenance
        # portable; the pair manifest is the authoritative training input.
        "source": source.name,
        "output_resolution": resolution,
        "source_pairs": pair_count,
        "pairs_written": written,
        "pair_manifest": "pairs.jsonl",
        "dataset_fingerprint": fingerprint_records(
            [
                json.loads(line)
                for line in (output / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ],
            output,
        ),
        "fingerprint_algorithm": "sha256:canonical-pairs-v1",
        "path_policy": "manifest-relative",
        "source_depth_formats": source_formats,
        "mapper": {"colormap": mapper.colormap, "lam": mapper.lam, "c": mapper.c},
        "metric_depth": {
            "directory": "metric_depth",
            "dtype": "float32",
            "invalid_value": "NaN",
        },
        "data_contract": {
            "sample_mapping_sha256": sample_digest.hexdigest(),
            "valid_pixels": valid_pixels,
            "invalid_pixels": int(audit["invalid_pixels"]),
            "depth_meters": {
                "min": None if valid_pixels == 0 else audit["depth_min"],
                "mean": None if valid_pixels == 0 else audit["depth_sum"] / valid_pixels,
                "max": None if valid_pixels == 0 else audit["depth_max"],
            },
            "encoded_roundtrip_abs_error": {
                "mean": None
                if valid_pixels == 0
                else audit["roundtrip_abs_error_sum"] / valid_pixels,
                "max": None if valid_pixels == 0 else audit["roundtrip_abs_error_max"],
            },
        },
    }
    pairs_path = output / "pairs.jsonl"
    if pairs_path.is_file():
        records = [json.loads(line) for line in pairs_path.read_text(encoding="utf-8").splitlines() if line]
        payload["dataset_fingerprint"] = fingerprint_records(records, output)
        payload["fingerprint_algorithm"] = "sha256:canonical-pairs-v1"
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def prepare_hypersim(
    source: Path,
    output: Path,
    *,
    max_samples: int | None = None,
    resolution: int | None = None,
    overwrite: bool = False,
    start_index: int = 0,
    exclude_scenes: set[str] | None = None,
) -> int:
    """Write RGB targets and auditable lossless metric targets under *output*."""
    pairs = find_hypersim_pairs(source, exclude_scenes=exclude_scenes)
    if not pairs:
        logger.error("No color/depth pairs found under %s", source)
        return 0

    if max_samples is not None:
        pairs = pairs[:max_samples]

    img_dir = output / "images"
    tgt_dir = output / "targets"
    metric_dir = output / "metric_depth"
    img_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    mapper = DepthRGBMapper()
    written = 0
    source_formats: dict[str, int] = {}
    sample_digest = hashlib.sha256()
    audit: dict[str, float | int] = {
        "valid_pixels": 0,
        "invalid_pixels": 0,
        "depth_sum": 0.0,
        "depth_min": float("inf"),
        "depth_max": float("-inf"),
        "roundtrip_abs_error_sum": 0.0,
        "roundtrip_abs_error_max": 0.0,
    }

    for offset, (color_path, depth_path) in enumerate(pairs):
        idx = start_index + offset
        stem = f"{idx:06d}_{color_path.stem.replace('color', '').replace('rgb', '').strip('_')}"
        out_img = img_dir / f"{stem}.png"
        out_tgt = tgt_dir / f"{stem}.png"
        out_metric = metric_dir / f"{stem}.npy"
        source_formats[depth_path.suffix.lower()] = (
            source_formats.get(depth_path.suffix.lower(), 0) + 1
        )
        sample_digest.update(
            f"{stem}\0{color_path.relative_to(source)}\0{depth_path.relative_to(source)}\n".encode()
        )

        if out_img.exists() and out_tgt.exists() and out_metric.exists() and not overwrite:
            stored_metric = torch.from_numpy(np.load(out_metric)).float().unsqueeze(0).unsqueeze(0)
            stored_depth = torch.where(
                torch.isfinite(stored_metric) & (stored_metric > 0),
                stored_metric,
                torch.zeros_like(stored_metric),
            )
            with Image.open(out_tgt) as target_image:
                stored_target = (
                    torch.from_numpy(np.asarray(target_image.convert("RGB"), dtype=np.float32))
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    / 255.0
                )
            _update_depth_audit(audit, stored_depth, stored_target, mapper)
            written += 1
            continue

        color = Image.open(color_path).convert("RGB")
        if resolution is not None:
            color = color.resize((resolution, resolution), Image.Resampling.LANCZOS)

        raw_depth = _load_depth_meters(depth_path)
        target_size = (
            (resolution, resolution) if resolution is not None else (color.height, color.width)
        )
        if tuple(raw_depth.shape[-2:]) != target_size:
            raw_depth = torch.nn.functional.interpolate(raw_depth, size=target_size, mode="nearest")

        valid = torch.isfinite(raw_depth) & (raw_depth > 0)
        depth = torch.where(valid, raw_depth, torch.zeros_like(raw_depth))
        target_rgb = mapper.depth_to_rgb(depth)
        _tensor_to_pil_rgb(target_rgb.squeeze(0)).save(out_tgt)
        color.save(out_img)
        metric_depth = raw_depth.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        metric_depth[~valid.squeeze(0).squeeze(0).cpu().numpy()] = np.nan
        np.save(out_metric, metric_depth)
        _update_depth_audit(audit, depth, target_rgb, mapper)
        written += 1

        if (offset + 1) % 100 == 0:
            logger.info("Processed %d / %d pairs", offset + 1, len(pairs))

    jsonl_count = write_metric_depth_pairs_jsonl(output)
    _write_manifest(
        output,
        source=source,
        resolution=resolution,
        mapper=mapper,
        pair_count=len(pairs),
        written=written,
        source_formats=source_formats,
        audit=audit,
        sample_digest=sample_digest,
    )
    logger.info("Wrote %d pairs to %s (%d jsonl rows)", written, output.resolve(), jsonl_count)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Hypersim depth data for Vision Banana training"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Hypersim root directory (contains *color* images and HDF5/PNG depth pairs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory (images/, targets/, metric_depth/, manifest.json)",
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
        "--start-index",
        type=int,
        default=0,
        help="Start output filename numbering at this index (append to an existing split)",
    )
    parser.add_argument(
        "--exclude-scenes",
        type=str,
        default="",
        help="Comma-separated Hypersim scene names to skip (e.g. ai_001_001,ai_001_002)",
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

    exclude_scenes = {name.strip() for name in args.exclude_scenes.split(",") if name.strip()}
    count = prepare_hypersim(
        source,
        output,
        max_samples=args.max_samples,
        resolution=args.resolution,
        overwrite=args.overwrite,
        start_index=args.start_index,
        exclude_scenes=exclude_scenes or None,
    )
    if count == 0:
        return 1

    if args.validate:
        report = validate_vision_dataset(output, task="depth")
        print(report.summary())
        if not report.ok:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
