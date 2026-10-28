#!/usr/bin/env python3
"""Validate native depth assets and reuse the existing portable pair-bundle transport."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from PIL import Image

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.data.hypersim_supplement import (
    decode_quantized_rgb,
    file_sha256,
    sha256,
    write_json,
)
from open_vision_banana.data.manifest import read_jsonl
from open_vision_banana.res_bundle import package_dataset, safe_path


def validate_native(root: Path, workers: int = 8) -> dict:
    torch.set_num_threads(1)
    if (root / "BUILD_INCOMPLETE").exists():
        raise ValueError("Cannot publish an incomplete dataset")
    report = json.loads((root / "manifest.json").read_text())
    if file_sha256(root / "pairs.jsonl") != report["manifest_sha256"]:
        raise ValueError("Prepared manifest changed")
    if file_sha256(root / "provenance.jsonl") != report["provenance_sha256"]:
        raise ValueError("Prepared provenance changed")
    records = read_jsonl(root / "provenance.jsonl")
    rows = read_jsonl(root / "pairs.jsonl")
    if [r["row"] for r in records] != rows or len(rows) != report["pairs_written"]:
        raise ValueError("Manifest and provenance rows differ")

    def check(record):
        row = record["row"]
        if row["task"] != "depth" or row["split"] != "train":
            raise ValueError("Expected train-only depth rows")
        if {a["path"] for a in record["assets"]} != {
            row[k] for k in ("input", "target", "metric_depth")
        }:
            raise ValueError("Incomplete asset inventory")
        for asset in record["assets"]:
            path = safe_path(root, asset["path"])
            if path.stat().st_size != asset["bytes"] or file_sha256(path) != asset["sha256"]:
                raise ValueError(f"Changed asset: {asset['path']}")
        with Image.open(safe_path(root, row["input"])) as image:
            rgb = image.convert("RGB")
            digest = sha256(f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes())
            shape = (rgb.height, rgb.width)
        if digest != record["rgb_pixels_sha256"]:
            raise ValueError("Decoded source pixels changed")
        with Image.open(safe_path(root, row["target"])) as image:
            if image.mode != "RGB":
                raise ValueError("Depth target is not RGB")
            target = np.array(image)
        depth = np.load(safe_path(root, row["metric_depth"]), allow_pickle=False)
        if depth.dtype != np.float32 or depth.shape != shape or target.shape != (*shape, 3):
            raise ValueError("Native RGB/target/metric-depth dimensions or dtype differ")
        if np.isinf(depth).any() or (np.isfinite(depth) & (depth <= 0)).any():
            raise ValueError("Invalid depths must be NaN, not infinite or non-positive")
        valid = np.isfinite(depth)
        if valid.mean() < 0.5:
            raise ValueError("Insufficient valid depth")
        mapper = DepthRGBMapper()
        encoded = mapper.depth_to_rgb(torch.from_numpy(np.nan_to_num(depth))[None, None])
        expected = (encoded[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        if not np.array_equal(target, expected):
            raise ValueError("PNG does not encode the stored native metric depth")
        error = np.abs(decode_quantized_rgb(target, mapper)[valid] - depth[valid])
        if error.mean() > 0.05:
            raise ValueError("Frame exceeds the 5 cm quality threshold")
        maximum = report.get("quality_policy", {}).get("maximum_frame_encoded_max_abs_error_m")
        if maximum is not None and error.max() > maximum:
            raise ValueError("Frame exceeds the maximum pixel-error threshold")
        return digest, int(valid.sum()), float(error.astype(np.float64).sum())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        checked = list(pool.map(check, records))
    if len({v[0] for v in checked}) != len(rows):
        raise ValueError("Duplicate decoded input images")
    error_mean = sum(v[2] for v in checked) / sum(v[1] for v in checked)
    if error_mean >= 0.01:
        raise ValueError("Release fails the existing 1 cm roundtrip gate")
    inventory = sorted((a for r in records for a in r["assets"]), key=lambda a: a["path"])
    if len({a["path"] for a in inventory}) != len(inventory):
        raise ValueError("Repeated asset paths")
    (root / "assets.jsonl").write_text(
        "".join(json.dumps(a, sort_keys=True) + "\n" for a in inventory),
        encoding="utf-8",
        newline="\n",
    )
    names = [
        "pairs.jsonl",
        "manifest.json",
        "provenance.jsonl",
        "rejected.jsonl",
        "assets.jsonl",
        "selection_spec.json",
        "quality_policy.json",
    ]
    files = {
        name: {"bytes": (root / name).stat().st_size, "sha256": file_sha256(root / name)}
        for name in names
    }
    fingerprint = sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode())
    build = dict(
        status="complete",
        files=files,
        dataset_fingerprint=fingerprint,
        counts={"train": len(rows)},
        encoding="metric_power_cube_edge_v1",
    )
    validation = dict(
        status="passed",
        validator="hypersim_native_depth_v1",
        counts=build["counts"],
        dataset_fingerprint=fingerprint,
        assets_decoded=len(inventory),
        encoded_roundtrip_mae_m=error_mean,
        unique_images=len(checked),
    )
    write_json(root / "build_report.json", build)
    write_json(root / "validation_report.json", validation)
    return validation


def package(root: Path, output: Path, *, workers: int = 8, shard_mb: int = 512) -> dict:
    validation = validate_native(root, workers)
    index_path = output / "bundle.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if index["dataset_fingerprint"] != validation["dataset_fingerprint"]:
            raise ValueError("Existing bundle belongs to different validated data")
    else:
        index = package_dataset(root, output, shard_mb * 1024**2)

    def compress(item):
        source = safe_path(output, item["path"])
        if source.stat().st_size != item["size"] or file_sha256(source) != item["sha256"]:
            raise ValueError("Existing bundle payload changed")
        if item["kind"] != "tar" or item["path"].endswith(".gz"):
            return item
        target = source.with_suffix(source.suffix + ".gz")
        temporary = target.with_suffix(".gz.partial")
        with source.open("rb") as inp, temporary.open("wb") as out:  # noqa: SIM117 - close gzip before its underlying stream
            with gzip.GzipFile(
                filename="", fileobj=out, mode="wb", compresslevel=1, mtime=0
            ) as zipped:
                shutil.copyfileobj(inp, zipped, 1024 * 1024)
        temporary.replace(target)
        return dict(
            item,
            path=target.relative_to(output).as_posix(),
            size=target.stat().st_size,
            sha256=file_sha256(target),
        )

    old_entries = index["files"]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        index["files"] = list(pool.map(compress, old_entries))
    write_json(index_path, index)
    # Only delete redundant tar files created by this packager, after the new index is complete.
    for old, new in zip(old_entries, index["files"], strict=True):
        if old["path"] != new["path"]:
            safe_path(output, old["path"]).unlink()
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-size-mb", type=int, default=512)
    args = parser.parse_args()
    result = package(
        args.data_dir, args.output_dir, workers=args.workers, shard_mb=args.shard_size_mb
    )
    print(json.dumps({"assets": result["asset_files"], "payloads": len(result["files"])}))
