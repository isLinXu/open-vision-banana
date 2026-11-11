"""Auditable native DIODE indoor-train conversion.

Clean CLI gates, resizing, interpolation, model training or publishing.
Each output directory is an immutable build; --resume verifies existing bytes.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

VERSION = "diode-clean-native-train-v1"
PROMPT = "Generate a metric depth visualization of this image."
ROLES = ("input", "target", "metric_depth", "valid_mask")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value):
    return (
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode()


def immutable_write(path, data, resume=False):
    """Atomic publish; never repair/overwrite an existing mismatching artifact."""
    path = Path(path)
    if path.exists():
        if not resume or path.read_bytes() != data:
            raise ValueError(f"Existing output differs or resume not requested: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".building-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def array_bytes(array, png=False):
    stream = io.BytesIO()
    if png:
        Image.fromarray(array).save(stream, format="PNG")
    else:
        np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def prepare_arrays(rgb, depth, source_mask):
    """Keep every finite positive native valid depth, including distant pixels."""
    rgb, depth, source_mask = map(np.asarray, (rgb, depth, source_mask))
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Expected native HxWx3 uint8 RGB")
    if depth.ndim != 2 or depth.shape != rgb.shape[:2] or source_mask.shape != depth.shape:
        raise ValueError("Unaligned native RGB/depth/mask geometry")
    if not depth.size or depth.dtype.kind not in "fiu" or source_mask.dtype.kind not in "bifu":
        raise ValueError("Empty or nonnumeric depth/mask")
    if not np.isfinite(source_mask).all() or not np.isin(source_mask, [0, 1]).all():
        raise ValueError("Source validity mask must be finite binary 0/1")
    valid = (source_mask > 0) & np.isfinite(depth) & (depth > 0)
    metric = np.where(valid, depth, 0).astype(np.float32)
    if not np.isfinite(metric).all() or np.any(metric[valid] <= 0):
        raise ValueError("Depth is not representable as finite positive float32 metres")
    values = metric[valid]
    stats = dict(
        valid_pixels=int(valid.sum()),
        total_pixels=int(valid.size),
        valid_fraction=float(valid.mean()),
        missing_fraction=float((~valid).mean()),
        source_nonfinite_pixels=int((~np.isfinite(depth)).sum()),
        source_mask_positive_pixels=int((source_mask > 0).sum()),
        depth_min_m=float(values.min()) if values.size else None,
        depth_max_m=float(values.max()) if values.size else None,
        depth_p5_p50_p95_m=np.quantile(values, [0.05, 0.5, 0.95]).tolist() if values.size else None,
        rgb_mean=float(rgb.mean()),
        rgb_std=float(rgb.std()),
    )
    return metric, valid, stats


class EncodingQualityError(ValueError):
    """Explicit encoding gate rejection, carrying complete native quality statistics."""

    def __init__(self, stats, threshold):
        self.stats = dict(stats)
        self.threshold = threshold
        super().__init__(
            f"Encoding maximum error {stats['encoding_max_error_m']} m exceeds {threshold} m"
        )


def convert_pair(
    root, identity, rgb, depth, source_mask, provenance, resume=False, *, max_encoding_error_m=None
):
    import torch

    from open_vision_banana.core.depth_mapper import DepthRGBMapper
    from open_vision_banana.data.hypersim_supplement import decode_quantized_rgb

    if not re.fullmatch(r"[A-Za-z0-9_-]+", identity):
        raise ValueError("Unsafe sample ID")
    if max_encoding_error_m is not None and (
        not np.isfinite(max_encoding_error_m) or max_encoding_error_m < 0
    ):
        raise ValueError("Maximum encoding error must be finite and nonnegative")
    metric, valid, stats = prepare_arrays(rgb, depth, source_mask)
    mapper = DepthRGBMapper()
    with torch.no_grad():
        encoded = mapper.depth_to_rgb(torch.from_numpy(metric.copy())[None, None])
        target = (encoded[0].permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        target[~valid] = 0
        decoded = decode_quantized_rgb(target, mapper)
    error = np.abs(decoded[valid].astype(np.float64) - metric[valid])
    if not np.isfinite(error).all():
        raise ValueError("Nonfinite RGB encoding roundtrip")
    stats.update(
        encoding_mae_m=float(error.mean()) if error.size else None,
        encoding_max_error_m=float(error.max()) if error.size else None,
        encoding_abs_error_sum_m=float(error.sum()),
    )
    # Enforce the optional gate before encoding any output file or creating directories.
    if (
        max_encoding_error_m is not None
        and stats["encoding_max_error_m"] is not None
        and stats["encoding_max_error_m"] > max_encoding_error_m
    ):
        raise EncodingQualityError(stats, max_encoding_error_m)
    paths = dict(
        input=f"images/{identity}.png",
        target=f"targets/{identity}.png",
        metric_depth=f"metric_depth/{identity}.npy",
        valid_mask=f"valid_masks/{identity}.png",
    )
    payloads = (
        array_bytes(rgb, True),
        array_bytes(target, True),
        array_bytes(metric),
        array_bytes(valid.astype(np.uint8) * 255, True),
    )
    assets = {}
    for role, data in zip(ROLES, payloads, strict=True):
        immutable_write(Path(root) / paths[role], data, resume)
        assets[role] = dict(path=paths[role], sha256=sha(data), bytes=len(data))
    row = dict(
        provenance,
        sample_id=identity,
        task="depth",
        split="train",
        prompt=PROMPT,
        target_kind="metric_depth",
        encoding="cube_edge_metric_power",
        encoding_lambda=-3,
        encoding_c=10 / 3,
        metric_depth_scale=1,
        depth_units="metres",
        metric_depth_dtype="float32",
        native_width=rgb.shape[1],
        native_height=rgb.shape[0],
        policy=VERSION,
        loss_mask_applied=False,
        **paths,
        **stats,
    )
    record = dict(row=row, assets=assets)
    immutable_write(Path(root) / "records" / f"{identity}.json", json_bytes(record), resume)
    return record


def metadata_paths(tree, prefix=()):
    """Preserve scene/scan hierarchy; duplicate leaf IDs are rejected separately."""
    if isinstance(tree, list):
        for value in tree:
            if not isinstance(value, str) or PurePosixPath(value).name != value:
                raise ValueError("Invalid metadata sample ID")
            yield (*prefix, value)
    elif isinstance(tree, dict):
        for key in sorted(tree):
            if not isinstance(key, str) or key in (".", "..") or "/" in key or "\\" in key:
                raise ValueError("Invalid metadata path")
            yield from metadata_paths(tree[key], (*prefix, key))
    else:
        raise ValueError("Invalid DIODE metadata tree")


def diode_ids(metadata):
    paths = list(metadata_paths(metadata["train"]["indoors"]))
    allowed = {parts[-1]: parts for parts in paths}
    heldout = [parts[-1] for parts in metadata_paths(metadata["val"])]
    if (
        len(allowed) != len(paths)
        or len(set(heldout)) != len(heldout)
        or set(allowed) & set(heldout)
    ):
        raise ValueError("Duplicate or overlapping official DIODE IDs")
    return allowed, set(heldout)


def native_plane(array):
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError("Expected HxW or HxWx1 native depth/mask")
    return array


class DiodeSource:
    """Read native triplets without extraction; reject unsafe/duplicate archive paths."""

    def __init__(self, path):
        self.path = Path(path)
        self.tar = None
        if self.path.is_dir():
            self.files = {}
            for item in sorted(self.path.rglob("*")):
                if item.is_symlink():
                    raise ValueError("Symlink in DIODE source")
                if item.is_file():
                    self.files[item.relative_to(self.path).as_posix()] = item
            self.archive_sha = None
        else:
            # Official train.tar is seekable; compressed archives must be unpacked first.
            self.tar = tarfile.open(self.path, "r:")  # noqa: SIM115 - owned by this source; close() releases it
            self.files = {}
            for member in self.tar:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                    raise ValueError("Unsafe DIODE archive member")
                if member.issym() or member.islnk():
                    raise ValueError("Link in DIODE archive")
                if member.isfile():
                    name = path.as_posix()
                    if name in self.files:
                        raise ValueError("Duplicate DIODE archive member")
                    self.files[name] = member
            self.archive_sha = file_sha(self.path)

    def read(self, name):
        if self.tar:
            with self.tar.extractfile(self.files[name]) as handle:
                return handle.read()
        return self.files[name].read_bytes()

    def close(self):
        if self.tar:
            self.tar.close()

    def jobs(self, allowed, heldout):
        jobs = {}
        for name in sorted(self.files):
            path = PurePosixPath(name)
            if path.suffix != ".png" or "indoors" not in path.parts:
                continue
            # Allow an optional enclosing directory, but require official train/indoors.
            parts = path.parts
            at = parts.index("indoors")
            if at == 0 or parts[at - 1] != "train":
                if "val" in parts or "test" in parts:
                    continue
                raise ValueError(f"Missing official train path: {name}")
            stem = path.stem
            if stem in heldout or stem not in allowed:
                raise ValueError(f"Non-train DIODE ID in train tree: {stem}")
            expected = (*allowed[stem][:-1], stem + ".png")
            if parts[at + 1 :] != expected:
                raise ValueError(f"DIODE metadata hierarchy mismatch: {name}")
            if stem in jobs:
                raise ValueError(f"Duplicate DIODE sample ID: {stem}")
            triplet = (
                name,
                str(path.with_name(stem + "_depth.npy")),
                str(path.with_name(stem + "_depth_mask.npy")),
            )
            if any(item not in self.files for item in triplet):
                raise ValueError(f"Incomplete native DIODE triplet: {name}")
            jobs[stem] = triplet
        return jobs

    def samples(self, jobs, metadata_sha):
        for stem in sorted(jobs):
            names = jobs[stem]
            data = [self.read(name) for name in names]
            with Image.open(io.BytesIO(data[0])) as image:
                if image.mode != "RGB":
                    raise ValueError("DIODE input must be native RGB")
                rgb = np.asarray(image).copy()
            depth, mask = [
                native_plane(np.load(io.BytesIO(blob), allow_pickle=False)) for blob in data[1:]
            ]
            if rgb.shape != (768, 1024, 3):
                raise ValueError("Unexpected official DIODE native geometry")
            provenance = dict(
                source_dataset="diode_indoor",
                source_id=stem,
                source_split="official_train",
                depth_provenance="source_valid_measurements",
                source_rgb=names[0],
                source_depth=names[1],
                source_mask=names[2],
                source_sha256=dict(zip(("rgb", "depth", "mask"), map(sha, data), strict=True)),
                official_metadata_sha256=metadata_sha,
                depth_convention="source_native_no_ray_correction",
            )
            if self.archive_sha:
                provenance["source_archive_sha256"] = self.archive_sha
            yield "diode_" + stem, rgb, depth, mask, provenance


def excluded_ids(paths):
    """Read new and PR104 rows; scope IDs by dataset to avoid cross-source clashes."""
    found = set()
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sample = row.get("sample_id", "")
            source = row.get("source_dataset", "")
            if sample.startswith("diode_"):
                found.add(sample)
            if source.startswith("diode"):
                raw = row.get("source_id") or PurePosixPath(row.get("source_rgb", "")).stem
                if raw:
                    found.add("diode_" + raw)
    return found


def finalize(root, records, report, resume=False):
    records = sorted(records, key=lambda item: item["row"]["sample_id"])
    ids = [item["row"]["sample_id"] for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate output sample IDs")
    assets = []
    for record in records:
        for role in ROLES:
            asset = record["assets"][role]
            path = Path(root) / asset["path"]
            if path.stat().st_size != asset["bytes"] or file_sha(path) != asset["sha256"]:
                raise ValueError(f"Asset integrity failure: {path}")
            assets.append(dict(sample_id=record["row"]["sample_id"], role=role, **asset))
    rows = [record["row"] for record in records]
    total = sum(row["valid_pixels"] for row in rows)
    report.update(
        output_pairs=len(rows),
        output_assets=len(assets),
        encoding_pixel_weighted_mae_m=sum(row["encoding_abs_error_sum_m"] for row in rows) / total
        if total
        else None,
        encoding_max_error_m=max((row["encoding_max_error_m"] or 0 for row in rows), default=None),
        missing_pixel_fraction=1 - total / sum(row["total_pixels"] for row in rows)
        if rows
        else None,
        minimum_valid_fraction=min((row["valid_fraction"] for row in rows), default=None),
        no_quality_filter=report.get("min_valid_fraction", 0) == 0
        and report.get("max_encoding_error_m") is None,
        loss_mask_applied=False,
        limitations=[
            "Invalid depth targets are black; unchanged trainer loss is unmasked.",
            "RGB targets are quantized; float32 metric NPY retains source metric values.",
            "Without exclusion manifests, replace overlapping existing DIODE rows rather than append.",
        ],
    )
    pairs = b"".join(map(json_bytes, rows))
    inventory = b"".join(map(json_bytes, assets))
    report.update(pairs_sha256=sha(pairs), assets_sha256=sha(inventory))
    immutable_write(Path(root) / "pairs.jsonl", pairs, resume)
    immutable_write(Path(root) / "assets.jsonl", inventory, resume)
    immutable_write(Path(root) / "report.json", json_bytes(report), resume)
    immutable_write(
        Path(root) / "BUILD_COMPLETE.json",
        json_bytes(dict(report_sha256=sha(json_bytes(report)))),
        resume,
    )
    return report
