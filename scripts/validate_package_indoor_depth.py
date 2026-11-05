#!/usr/bin/env python3
"""Independently verify every native asset, then build an existing-format OVB bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.data.hypersim_supplement import (
    decode_quantized_rgb,
    file_sha256,
    write_json,
)
from open_vision_banana.res_bundle import package_dataset, safe_path


def check(record, root):
    row = record["row"]
    for asset in record["assets"]:
        path = safe_path(root, asset["path"])
        if path.stat().st_size != asset["bytes"] or file_sha256(path) != asset["sha256"]:
            raise ValueError("Changed native asset")
    rgb = np.array(Image.open(safe_path(root, row["input"])).convert("RGB"))
    target = np.array(Image.open(safe_path(root, row["target"])))
    native = np.array(Image.open(safe_path(root, row["metric_depth"])))
    mask_raw = np.array(Image.open(safe_path(root, row["valid_mask"])))
    if (
        native.dtype != np.uint16
        or native.shape != rgb.shape[:2]
        or target.shape != rgb.shape
        or mask_raw.shape != native.shape
    ):
        raise ValueError("Native size/dtype mismatch")
    if not np.isin(mask_raw, [0, 255]).all():
        raise ValueError("Nonbinary observation mask")
    mask = mask_raw > 0
    if (
        mask.mean() < 0.98
        or np.any(native[~mask] != 0)
        or np.any((native[mask] < 250) | (native[mask] > 20000))
    ):
        raise ValueError("Invalid observed measurements / less than 98%")
    if abs(float(mask.mean()) - record["statistics"]["valid_fraction"]) > 1e-12:
        raise ValueError("Observation statistics differ")
    if row["metric_depth_scale"] != 1000 or row["depth_convention"] != "planar_z_depth":
        raise ValueError("Metric convention differs")
    if row["source_dataset"] == "arkitscenes":
        if row["source_scene"] == "NA" or row["timestamp_error_seconds"] > 0.008:
            raise ValueError("Unidentified scene / unsynchronized frames")
        if (row["source_split"] == "Training") != (row["split"] == "train"):
            raise ValueError("Changed official split")
    mapper = DepthRGBMapper()
    depth = native.astype(np.float32) / 1000
    encoded = mapper.depth_to_rgb(torch.from_numpy(depth)[None, None])
    expected = (encoded[0].permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
    if not np.array_equal(expected, target):
        raise ValueError("RGB target does not encode native millimetres")
    error = np.abs(decode_quantized_rgb(target, mapper)[mask] - depth[mask])
    if error.max() > 0.05:
        raise ValueError("Encoding error above policy")
    digest = hashlib.sha256(f"{rgb.shape[1]}x{rgb.shape[0]}:".encode() + rgb.tobytes()).hexdigest()
    if digest != record["rgb_pixels_sha256"]:
        raise ValueError("Image fingerprint changed")
    return digest, int(mask.sum()), float(error.astype(np.float64).sum()), float(error.max())


def validate(root, workers):
    if (root / "BUILD_INCOMPLETE").exists():
        raise ValueError("Incomplete dataset")
    build = json.loads((root / "build_report.json").read_text(encoding="utf-8"))
    for name, meta in build["files"].items():
        path = safe_path(root, name)
        if file_sha256(path) != meta["sha256"] or path.stat().st_size != meta["bytes"]:
            raise ValueError("Metadata changed")
    pairs = [json.loads(line) for line in (root / "pairs.jsonl").open(encoding="utf-8")]
    records = [json.loads(line) for line in (root / "source_audit.jsonl").open(encoding="utf-8")]
    if pairs != [r["row"] for r in records] or len({r["sample_id"] for r in pairs}) != len(pairs):
        raise ValueError("Manifest/source audit differ")
    counts = Counter(f"{r['source_dataset']}/{r['split']}" for r in pairs)
    if dict(counts) != build["counts"]:
        raise ValueError("Counts differ")
    scenes = {
        split: {(r["source_dataset"], r["source_scene"]) for r in pairs if r["split"] == split}
        for split in ("train", "val")
    }
    if scenes["train"] & scenes["val"]:
        raise ValueError("Scene leakage")
    hashes = set()
    valid = 0
    error_sum = 0.0
    maximum = 0.0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (digest, n, e, m) in enumerate(pool.map(lambda r: check(r, root), records), 1):
            if digest in hashes:
                raise ValueError("Duplicate decoded images")
            hashes.add(digest)
            valid += n
            error_sum += e
            maximum = max(maximum, m)
            if i % 1000 == 0:
                print(json.dumps(dict(validated=i, total=len(records))), flush=True)
    if error_sum / valid >= 0.01:
        raise ValueError("Metric encoding MAE above 1cm")
    result = dict(
        status="passed",
        counts=dict(counts),
        dataset_fingerprint=build["dataset_fingerprint"],
        unique_images=len(hashes),
        asset_files=len(records) * 4,
        observed_pixels=valid,
        encoding_mae_m=error_sum / valid,
        encoding_max_error_m=maximum,
        minimum_observed_fraction=0.98,
        actual_minimum_observed_fraction=min(r["statistics"]["valid_fraction"] for r in records),
        mean_observed_fraction=float(np.mean([r["statistics"]["valid_fraction"] for r in records])),
        scene_splits_disjoint=True,
    )
    write_json(root / "validation_report.json", result)
    return result


def package(root, out, shard_mb):
    """Append provenance-mask TARs to the existing bundle without changing shared code."""
    index = package_dataset(root, out, shard_mb * 1024**2)
    assets = [
        json.loads(line) for line in (root / "valid_mask_assets.jsonl").open(encoding="utf-8")
    ]
    batch = []
    used = 0

    def emit(items):
        path = out / "assets" / f"observed-masks-{len(index['files']):05d}.tar"
        with tarfile.open(path, "w") as archive:
            for item in items:
                source = safe_path(root, item["path"])
                if file_sha256(source) != item["sha256"]:
                    raise ValueError("Mask changed after validation")
                info = tarfile.TarInfo(item["path"])
                info.size = source.stat().st_size
                info.mode = 0o644
                info.mtime = 0
                with source.open("rb") as f:
                    archive.addfile(info, f)
        index["files"].append(
            dict(
                path=path.relative_to(out).as_posix(),
                kind="tar",
                members=len(items),
                size=path.stat().st_size,
                sha256=file_sha256(path),
            )
        )
        index["asset_files"] += len(items)

    for asset in assets:
        if batch and used + asset["bytes"] + 1024 > shard_mb * 1024**2:
            emit(batch)
            batch = []
            used = 0
        batch.append(asset)
        used += asset["bytes"] + 1024
    if batch:
        emit(batch)
    write_json(out / "bundle.json", index)
    return index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data_dir", type=Path)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--bundle-dir", type=Path)
    p.add_argument("--shard-size-mb", type=int, default=256)
    a = p.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(validate(a.data_dir, a.workers)), flush=True)
    if a.bundle_dir:
        package(a.data_dir, a.bundle_dir, a.shard_size_mb)


if __name__ == "__main__":
    main()
