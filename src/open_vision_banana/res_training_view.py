"""Derive a model-selection manifest that cannot expose any official test image.

The official expression-level evaluation manifests remain unchanged. This
module needs only their records and previously validated asset hashes; it does
not decode images or import the model/data preparation dependencies.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import BinaryIO

VIEW_SCHEMA = "res_training_view_v1"
VIEW_NAME = "training.jsonl"


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path):
    with path.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{number}: expected a JSON object")
            yield row, line


def _derive(root: Path, destination: BinaryIO | None = None) -> dict:
    assets_path, pairs_path = root / "assets.jsonl", root / "pairs.jsonl"
    asset_sha = _file_digest(assets_path)
    pairs_sha = _file_digest(pairs_path)
    image_hashes: dict[str, str] = {}
    for asset, _ in _rows(assets_path):
        if asset.get("kind") != "image":
            continue
        name, digest = asset.get("path"), asset.get("sha256")
        if (
            not isinstance(name, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("Training view requires valid image paths and SHA256 asset entries")
        if name in image_hashes:
            raise ValueError(f"Duplicate image asset in training view: {name}")
        image_hashes[name] = digest

    def image_identity(row: dict) -> tuple[int, str]:
        image_id, name = row.get("image_id"), row.get("input")
        if not isinstance(image_id, int) or isinstance(image_id, bool) or image_id < 0:
            raise ValueError("Training view requires a nonnegative COCO image_id")
        if name not in image_hashes:
            raise ValueError(f"Training view references an unindexed image: {name}")
        if row.get("split") not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported training-view source split: {row.get('split')!r}")
        return image_id, image_hashes[name]

    test_ids: set[int] = set()
    test_hashes: set[str] = set()
    for row, _ in _rows(pairs_path):
        image_id, digest = image_identity(row)
        if row["split"] == "test":
            test_ids.add(image_id)
            test_hashes.add(digest)

    counts: Counter[str] = Counter({"train": 0, "val": 0})
    source_counts: Counter[str] = Counter()
    excluded_by_source: Counter[str] = Counter()
    excluded = excluded_id = excluded_hash = total_bytes = 0
    digest = hashlib.sha256()
    for row, line in _rows(pairs_path):
        image_id, image_hash = image_identity(row)
        split = row["split"]
        if split == "test":
            continue
        id_overlap, hash_overlap = image_id in test_ids, image_hash in test_hashes
        source = f"{row['source_dataset']}/{row['source_protocol']}/{row['source_split']}"
        if split == "train" and (id_overlap or hash_overlap):
            raise ValueError(
                "Training source already overlaps official test images; rebuild its global exclusion"
            )
        if split == "val" and (id_overlap or hash_overlap):
            excluded += 1
            excluded_id += int(id_overlap)
            excluded_hash += int(hash_overlap)
            excluded_by_source[source] += 1
            continue
        # Preserve the exact source serialization, including whitespace and line endings.
        if destination is not None:
            destination.write(line)
        digest.update(line)
        total_bytes += len(line)
        counts[split] += 1
        source_counts[source] += 1
    if _file_digest(assets_path) != asset_sha or _file_digest(pairs_path) != pairs_sha:
        raise ValueError("Source manifests changed while deriving the training view")
    return {
        "schema_version": VIEW_SCHEMA,
        "manifest": VIEW_NAME,
        "policy": "all train; val excludes all official test image IDs and exact image-file SHA256s",
        "sha256": digest.hexdigest(),
        "bytes": total_bytes,
        "counts": dict(sorted(counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "excluded_val": excluded,
        "excluded_val_by_source": dict(sorted(excluded_by_source.items())),
        "excluded_val_by_image_id": excluded_id,
        "excluded_val_by_exact_content": excluded_hash,
        "test_image_ids": len(test_ids),
        "test_content_sha256s": len(test_hashes),
        "input_manifests": {"assets.jsonl": asset_sha, "pairs.jsonl": pairs_sha},
    }


def write_training_view(root: str | Path) -> dict:
    """Atomically write all train and test-disjoint validation rows, returning metadata."""
    root = Path(root)
    temporary = root / f"{VIEW_NAME}.tmp"
    with temporary.open("xb") as stream:
        try:
            report = _derive(root, stream)
        except BaseException:
            stream.close()
            temporary.unlink()
            raise
    temporary.replace(root / VIEW_NAME)
    return report


def validate_training_view(root: str | Path, expected_report: dict) -> dict:
    """Re-derive exact row bytes/counts and verify both saved metadata and output SHA256."""
    root = Path(root)
    actual = _derive(root)
    if actual != expected_report:
        raise ValueError("Training view metadata does not match its current source manifests")
    path = root / VIEW_NAME
    if path.stat().st_size != actual["bytes"] or _file_digest(path) != actual["sha256"]:
        raise ValueError("training.jsonl differs from the exact expected rows/order/bytes")
    return actual
