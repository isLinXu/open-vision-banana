"""Deterministic, scene-disjoint Hypersim depth supplement preparation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.data.depth_geometry import ray_distance_to_planar_depth

UPSTREAM_REVISION = "3463c5c4a75f3cbfc65ed31cfd6e87204b3a2254"
METADATA_SHA256 = "47b7cce12f4659ffa31cf05e8fbf2aee3d69e1929d07b5ea25506603b69d5ce6"
ARCHIVE_ROOT = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes"
METADATA_PATH = "evermotion_dataset/analysis/metadata_images_split_scene_v1.csv"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_id(row: dict) -> str:
    if not re.fullmatch(r"ai_\d{3}_\d{3}", row["scene_name"]) or not re.fullmatch(
        r"cam_\d{2}", row["camera_name"]
    ):
        raise ValueError("Invalid Hypersim scene/camera identity")
    if not 0 <= int(row["frame_id"]) <= 9999:
        raise ValueError("Invalid Hypersim frame number")
    return f"{row['scene_name']}_{row['camera_name']}_frame.{int(row['frame_id']):04d}"


def select_candidates(
    metadata: str, *, seed: int = 42, excluded_max_volume: int = 10
) -> list[dict]:
    """Round-robin scenes, with hash-ranked even frames inside each scene.

    The original E1 download log names 56 scenes in volumes 1--10. Excluding
    those entire volumes is deliberately conservative. Only public, official
    scene-split training images are eligible. A stride of two excludes adjacent
    frame numbers; it does not claim statistical/perceptual independence.
    """
    scenes: dict[str, list[dict]] = defaultdict(list)
    seen = set()
    for row in csv.DictReader(io.StringIO(metadata)):
        identity = frame_id(row)
        if row["included_in_public_release"] != "True" or row["split_partition_name"] != "train":
            continue
        if int(row["scene_name"].split("_")[1]) <= excluded_max_volume or int(row["frame_id"]) % 2:
            continue
        if identity in seen:
            raise ValueError(f"Duplicate source identity: {identity}")
        seen.add(identity)
        scenes[row["scene_name"]].append(
            {
                "scene_name": row["scene_name"],
                "camera_name": row["camera_name"],
                "frame_id": int(row["frame_id"]),
                "source_split": "train",
            }
        )

    def rank(value: str) -> str:
        return sha256(f"{seed}:{value}".encode())

    names = sorted(scenes, key=rank)
    for rows in scenes.values():
        rows.sort(key=lambda r: rank(frame_id(r)))
    result: list[dict] = []
    for offset in range(max((len(rows) for rows in scenes.values()), default=0)):
        result.extend(scenes[name][offset] for name in names if offset < len(scenes[name]))
    return result


def member_names(row: dict) -> tuple[str, str]:
    prefix = f"{row['scene_name']}/images/scene_{row['camera_name']}"
    frame = f"frame.{int(row['frame_id']):04d}"
    return (
        f"{prefix}_final_preview/{frame}.color.jpg",
        f"{prefix}_geometry_hdf5/{frame}.depth_meters.hdf5",
    )


def decode_quantized_rgb(rgb: np.ndarray, mapper: DepthRGBMapper) -> np.ndarray:
    """Decode every pixel, evaluating the existing decoder once per distinct RGB.

    An 8-bit cube-edge image uses at most 1,786 colours. The decoder is a
    pointwise function, so an exact palette lookup avoids repeating its segment
    projections for every native-resolution pixel without sampling the image.
    """
    packed = rgb.astype(np.uint32)
    codes = (packed[..., 0] << 16) | (packed[..., 1] << 8) | packed[..., 2]
    unique, inverse = np.unique(codes, return_inverse=True)
    palette = np.stack([unique >> 16, (unique >> 8) & 255, unique & 255], axis=0)
    tensor = torch.from_numpy(palette.astype(np.float32) / 255)[None, :, None, :]
    decoded = mapper.rgb_to_depth(tensor).numpy().reshape(-1)
    return decoded[inverse].reshape(rgb.shape[:2])


def prepare_frame(row: dict, rgb_bytes: bytes, depth_bytes: bytes, output: Path) -> dict:
    """Keep the native JPEG and float32 planar depth; encode with the existing mapper."""
    import h5py

    identity = frame_id(row)
    with Image.open(io.BytesIO(rgb_bytes)) as image:
        rgb = image.convert("RGB")
        rgb.load()
    with h5py.File(io.BytesIO(depth_bytes), "r") as handle:
        distance = np.asarray(handle["dataset"], dtype=np.float32)
    if distance.ndim != 2 or distance.shape != (rgb.height, rgb.width):
        raise ValueError(f"Native RGB/depth size mismatch: {identity}")
    depth = ray_distance_to_planar_depth(torch.from_numpy(distance)).numpy()
    valid = np.isfinite(depth) & (depth > 0)
    if valid.mean() < 0.5:
        raise ValueError(f"Fewer than 50% valid depth pixels: {identity}")
    depth[~valid] = np.nan
    mapper = DepthRGBMapper()
    safe = torch.from_numpy(np.where(valid, depth, 0.0))[None, None]
    target = mapper.depth_to_rgb(safe)[0].permute(1, 2, 0).numpy()
    target_u8 = (target.clip(0, 1) * 255).astype(np.uint8)
    # Decode the complete quantized target, not a few favourable sample pixels.
    decoded = decode_quantized_rgb(target_u8, mapper)
    error = np.abs(decoded[valid] - depth[valid])
    image_path = output / "images" / f"{identity}.jpg"
    target_path = output / "targets" / f"{identity}.png"
    metric_path = output / "metric_depth" / f"{identity}.npy"
    for path in (image_path, target_path, metric_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(rgb_bytes)
    Image.fromarray(target_u8).save(target_path)
    np.save(metric_path, depth.astype(np.float32, copy=False), allow_pickle=False)
    pixels = f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()
    manifest_row = {
        "task": "depth",
        "split": "train",
        "sample_id": identity,
        "input": image_path.relative_to(output).as_posix(),
        "target": target_path.relative_to(output).as_posix(),
        "metric_depth": metric_path.relative_to(output).as_posix(),
        "target_kind": "hypersim_metric_rgb",
        "prompt": "Generate a metric depth visualization of this image.",
        "source_dataset": "hypersim",
        "source_revision": UPSTREAM_REVISION,
        "source_scene": row["scene_name"],
        "source_camera": row["camera_name"],
        "source_frame": int(row["frame_id"]),
        "source_split": "train",
        "depth_convention": "planar_z_depth",
        "depth_units": "metres",
    }
    return {
        "row": manifest_row,
        "source": row,
        "rgb_pixels_sha256": sha256(pixels),
        "source_rgb_sha256": sha256(rgb_bytes),
        "source_depth_sha256": sha256(depth_bytes),
        "assets": [
            {
                "path": p.relative_to(output).as_posix(),
                "sha256": file_sha256(p),
                "bytes": p.stat().st_size,
            }
            for p in (image_path, target_path, metric_path)
        ],
        "width": rgb.width,
        "height": rgb.height,
        "valid_pixels": int(valid.sum()),
        "invalid_pixels": int((~valid).sum()),
        "depth_sum": float(depth[valid].astype(np.float64).sum()),
        "depth_min": float(depth[valid].min()),
        "depth_max": float(depth[valid].max()),
        "roundtrip_error_sum": float(error.astype(np.float64).sum()),
        "roundtrip_error_max": float(error.max()),
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
