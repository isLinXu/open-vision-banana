#!/usr/bin/env python3
"""Prepare Hypersim depth data for Vision Banana instruction tuning.

Converts Hypersim-style paired color/depth files into a flat training layout
while retaining lossless metric targets for benchmark-style evaluation:

    output/
      images/     ← RGB input images
      targets/    ← depth encoded as RGB (DepthRGBMapper)
      metric_depth/ ← float32 metric depth maps (NaN marks invalid pixels)
      normal_targets/   ← --tasks normal: camera-space normals as RGB (paper frame)
      semantic_targets/ ← --tasks semantic: open-vocabulary NYU40 colour maps (+ prompt per row)
      instance_targets/ ← --tasks instance: one black-background target per class present
      pairs.jsonl   ← one row per (frame, task); the unified Trainer's input
      manifest.json ← mapper parameters and data-quality audit

``--tasks depth,normal,semantic,instance`` (default: depth) reads the sibling
``*.normal_cam.hdf5`` / ``*.semantic.hdf5`` / ``*.semantic_instance.hdf5`` of each
frame, so one downloaded scene feeds four of the paper's five tasks.

Hypersim source layout (scanned recursively):
    .../frame.0000.color.jpg  ↔  .../frame.0000.depth_meters.hdf5

Official ``depth_meters.hdf5`` holds the Euclidean ray distance from the optical
centre, not the planar z-depth the paper's metric-depth task is defined on. By
default (``--ray-distance auto``) HDF5 ``depth_meters`` files are converted with
the Hypersim 60 degree-FOV focal rule before encoding; ``metric_depth/*.npy`` and
the manifest record planar depth. Pass ``--ray-distance never`` for exports that
are already planar, ``--focal-px`` to override the focal length.

Usage:
    python scripts/prepare_hypersim.py --source /data/hypersim --output data/hypersim_depth/
    python scripts/prepare_hypersim.py --source /data/hypersim --output data/hypersim_depth/ --max-samples 500 --validate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.data.depth_geometry import (
    hypersim_focal_px,
    ray_distance_to_planar_depth,
)
from open_vision_banana.data.hypersim import (
    instance_targets,
    normal_backfacing_fraction,
    normal_to_target,
    semantic_sample,
)
from open_vision_banana.data.instance_palette import PALETTE
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


HYPERSIM_TASKS = ("depth", "normal", "semantic", "instance")


def geometry_sibling(depth_path: Path, channel: str) -> Path | None:
    """``.../frame.0000.depth_meters.hdf5`` -> ``.../frame.0000.<channel>.hdf5`` if present."""
    if "depth_meters" not in depth_path.name:
        return None
    candidate = depth_path.with_name(depth_path.name.replace("depth_meters", channel))
    return candidate if candidate.is_file() else None


def load_hdf5_array(path: Path) -> np.ndarray:
    """The single dataset of a Hypersim HDF5 file as a NumPy array."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Reading Hypersim HDF5 requires h5py") from exc
    with h5py.File(path, "r") as handle:
        dataset = handle.get("dataset")
        if dataset is None:
            found: list = []
            handle.visititems(lambda _n, v: found.append(v) if isinstance(v, h5py.Dataset) else None)
            if len(found) != 1:
                raise ValueError(f"Expected one dataset in {path}, found {len(found)}")
            dataset = found[0]
        return np.array(dataset)


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


RAY_DISTANCE_MODES = ("auto", "always", "never")


def is_hypersim_ray_distance(depth_path: Path) -> bool:
    """Official ``*.depth_meters.hdf5`` files store ray distance, not z-depth."""
    return depth_path.suffix.lower() in {".hdf5", ".h5"} and "depth_meters" in depth_path.stem.lower()


def _load_depth_meters(
    depth_path: Path,
    *,
    ray_distance: str = "auto",
    focal_px: float | None = None,
) -> tuple[torch.Tensor, bool]:
    """Load a metric depth map as ``[1, 1, H, W]`` planar z-depth in metres.

    Returns ``(depth, converted)``. Official Hypersim HDF5 ``depth_meters`` is
    Euclidean ray distance and is converted to planar depth
    (``ray_distance="auto"``); ``"always"`` / ``"never"`` force either reading.
    PNG exports are conventionally millimetres, matching the previous converter.
    """
    if ray_distance not in RAY_DISTANCE_MODES:
        raise ValueError(f"ray_distance must be one of {RAY_DISTANCE_MODES}")
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
    convert = ray_distance == "always" or (
        ray_distance == "auto" and is_hypersim_ray_distance(depth_path)
    )
    if convert:
        depth = ray_distance_to_planar_depth(depth, focal_px)
    return depth, convert


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
    depth_geometry: dict[str, Any] | None = None,
    tasks_summary: dict[str, Any] | None = None,
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
            # Paper Section 3.2: depth is the distance from the camera plane (planar z).
            "convention": "planar_z_depth",
        },
        "depth_geometry": depth_geometry or {},
        "tasks": tasks_summary or {"tasks": ["depth"]},
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


def _export_extra_tasks(
    depth_path: Path,
    stem: str,
    out_img: Path,
    tasks: tuple[str, ...],
    dirs: dict[str, Path],
    counts: dict[str, int],
    skipped: dict[str, int],
    rng: random.Random,
    *,
    semantic_variants: int,
    normal_flip: str,
    overwrite: bool,
    on_low_facing,
) -> list[dict[str, Any]]:
    """Write normal / semantic / instance targets for one frame; return manifest rows."""
    records: list[dict[str, Any]] = []
    image_rel = Path("images") / out_img.name
    if "normal" in tasks:
        src = geometry_sibling(depth_path, "normal_cam")
        if src is None:
            skipped["normal"] += 1
        else:
            out = dirs["normal"] / f"{stem}.png"
            if overwrite or not out.exists():
                normal = load_hdf5_array(src)
                backfacing = normal_backfacing_fraction(normal)
                if backfacing > 0.1:
                    logger.warning("%s: %.0f%% of normals point away from the camera; check --normal-flip",
                                   src.name, 100 * backfacing)
                    on_low_facing()
                target, _valid = normal_to_target(normal, flip=normal_flip)
                Image.fromarray(target).save(out)
            records.append({
                "task": "normal", "input": image_rel.as_posix(),
                "target": (Path("normal_targets") / out.name).as_posix(), "split": "train",
                "target_kind": "hypersim_normal_cam_rgb", "prompt": PROMPTS["normal"],
            })
            counts["normal"] += 1
    needs_semantic = "semantic" in tasks or "instance" in tasks
    semantic = None
    if needs_semantic:
        sem_src = geometry_sibling(depth_path, "semantic")
        if sem_src is None:
            for t in ("semantic", "instance"):
                if t in tasks:
                    skipped[t] += 1
        else:
            semantic = load_hdf5_array(sem_src)
    if semantic is not None and "semantic" in tasks:
        for variant in range(semantic_variants):
            out = dirs["semantic"] / f"{stem}_v{variant}.png"
            sample = semantic_sample(semantic, rng)
            if overwrite or not out.exists():
                Image.fromarray(sample.target_rgb).save(out)
            records.append({
                "task": "semantic", "input": image_rel.as_posix(),
                "target": (Path("semantic_targets") / out.name).as_posix(), "split": "train",
                "source": "hypersim", "prompt": sample.prompt, "style": sample.style,
                "mapping": {k: list(v) for k, v in sample.mapping.items()},
            })
            counts["semantic"] += 1
    if semantic is not None and "instance" in tasks:
        inst_src = geometry_sibling(depth_path, "semantic_instance")
        if inst_src is None:
            skipped["instance"] += 1
        else:
            for item in instance_targets(semantic, load_hdf5_array(inst_src), rng, palette=PALETTE):
                slug = item.class_name.replace(" ", "_")
                out = dirs["instance"] / f"{stem}_{slug}.png"
                if overwrite or not out.exists():
                    Image.fromarray(item.target).save(out)
                records.append({
                    "task": "instance", "input": image_rel.as_posix(),
                    "target": (Path("instance_targets") / out.name).as_posix(), "split": "train",
                    "prompt": item.prompt, "category": item.class_name,
                    "num_instances": item.num_instances, "background_color": [0, 0, 0],
                    "source_dataset": "hypersim",
                })
                counts["instance"] += 1
    return records


PROMPTS = {"normal": "Generate a surface normal estimation visualization of this image."}


def prepare_hypersim(
    source: Path,
    output: Path,
    *,
    max_samples: int | None = None,
    resolution: int | None = None,
    overwrite: bool = False,
    start_index: int = 0,
    exclude_scenes: set[str] | None = None,
    ray_distance: str = "auto",
    focal_px: float | None = None,
    tasks: tuple[str, ...] | list[str] = ("depth",),
    semantic_variants: int = 1,
    normal_flip: str = "none",
    seed: int = 42,
) -> int:
    """Write RGB targets and auditable lossless metric targets under *output*."""
    tasks = tuple(tasks)
    unknown_tasks = sorted(set(tasks) - set(HYPERSIM_TASKS))
    if unknown_tasks:
        raise ValueError(f"Unknown Hypersim tasks {unknown_tasks}; choose from {HYPERSIM_TASKS}")
    extra_dirs = {
        "normal": output / "normal_targets",
        "semantic": output / "semantic_targets",
        "instance": output / "instance_targets",
    }
    for task, directory in extra_dirs.items():
        if task in tasks:
            directory.mkdir(parents=True, exist_ok=True)
    extra_records: list[dict[str, Any]] = []
    extra_counts: dict[str, int] = {t: 0 for t in HYPERSIM_TASKS if t != "depth"}
    skipped_missing_channel: dict[str, int] = {t: 0 for t in extra_counts}
    low_facing = [0]
    rng = random.Random(seed)
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
    converted_pairs = 0
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

        extra_records.extend(
            _export_extra_tasks(
                depth_path, stem, out_img, tasks, extra_dirs, extra_counts, skipped_missing_channel,
                rng, semantic_variants=semantic_variants, normal_flip=normal_flip, overwrite=overwrite,
                on_low_facing=lambda: low_facing.__setitem__(0, low_facing[0] + 1),
            )
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

        raw_depth, converted = _load_depth_meters(
            depth_path, ray_distance=ray_distance, focal_px=focal_px
        )
        converted_pairs += int(converted)
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
    if extra_records:
        with (output / "pairs.jsonl").open("a", encoding="utf-8") as handle:
            for record in extra_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        jsonl_count += len(extra_records)
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
        tasks_summary={
            "tasks": list(tasks),
            "rows_per_task": {"depth": written, **{t: c for t, c in extra_counts.items() if t in tasks}},
            "skipped_missing_channel": {t: c for t, c in skipped_missing_channel.items() if t in tasks},
            "semantic_variants": semantic_variants,
            "normal_flip": normal_flip,
            "normal_backfacing_frames": low_facing[0],
        },
        depth_geometry={
            "ray_distance_mode": ray_distance,
            "ray_distance_converted_pairs": converted_pairs,
            "focal_px": focal_px,
            "focal_rule": (
                "explicit" if focal_px is not None
                else f"hypersim_{hypersim_focal_px.__defaults__[0]:.0f}deg_hfov_from_width"
            ),
        },
    )
    if converted_pairs:
        logger.info("Converted %d Hypersim ray-distance maps to planar z-depth", converted_pairs)
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
        "--ray-distance",
        choices=RAY_DISTANCE_MODES,
        default="auto",
        help=(
            "Treat source depth as Euclidean ray distance and convert to planar z-depth. "
            "'auto' converts official Hypersim *.depth_meters.hdf5 only."
        ),
    )
    parser.add_argument(
        "--focal-px",
        type=float,
        default=None,
        help="Focal length in pixels for the conversion (default: Hypersim 60 deg HFOV rule)",
    )
    parser.add_argument(
        "--tasks",
        default="depth",
        help="Comma-separated subset of depth,normal,semantic,instance (default: depth)",
    )
    parser.add_argument("--semantic-variants", type=int, default=1, help="Colour mappings sampled per frame")
    parser.add_argument(
        "--normal-flip", choices=("none", "z", "yz"), default="none",
        help="Axis flip for normal_cam exports not in the paper's +x right / +y up / +z out frame",
    )
    parser.add_argument("--seed", type=int, default=42)
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
        ray_distance=args.ray_distance,
        focal_px=args.focal_px,
        tasks=tuple(t.strip() for t in args.tasks.split(",") if t.strip()),
        semantic_variants=args.semantic_variants,
        normal_flip=args.normal_flip,
        seed=args.seed,
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
