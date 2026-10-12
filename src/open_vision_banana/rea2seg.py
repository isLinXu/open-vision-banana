"""Offline Rea²Seg adapter for the existing expression-conditioned pair loader.

No training or evaluation dispatch lives here. Optional readers are imported
only when used, and Hub access is cache-only.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .referring_contract import BINARY_REFERRING_ENCODING, build_binary_referring_prompt

DATASET = "snowball521/Rea2Seg-16K"
CATEGORIES = ("gqa-seg", "lisa-plus", "spatial", "reasonseg")
FIELDS = frozenset(("image", "question", "cot_answer", "mask", "category"))


def parquet_rows(paths: Iterable[Path]) -> Iterator[tuple[dict, dict, Path]]:
    """Yield rows, stable shard/row provenance, and the local image-path root."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Local Rea2Seg conversion requires pip install pyarrow") from exc
    # Keep snapshot symlink names: their targets in the Hub cache are blob hashes.
    paths = sorted(Path(path).absolute() for path in paths)
    if not paths or len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Provide distinct local Parquet files")
    names: set[str] = set()
    for path in paths:
        if path.name.casefold() in names:
            raise ValueError(f"Conflicting source shard name: {path.name}")
        names.add(path.name.casefold())
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        shard_hash = digest.hexdigest()
        parquet = pq.ParquetFile(path)
        if not FIELDS.issubset(parquet.schema_arrow.names):
            raise ValueError(f"{path.name}: required fields are {sorted(FIELDS)}")
        row_index = 0
        for batch in parquet.iter_batches(batch_size=1):
            for row in batch.to_pylist():
                yield (
                    row,
                    {
                        "source_shard": path.name,
                        "source_shard_sha256": shard_hash,
                        "source_row": row_index,
                    },
                    path.parent,
                )
                row_index += 1


def cached_hf_shards(dataset: str, revision: str) -> tuple[list[Path], str]:
    """Resolve an explicitly requested, already downloaded Hub snapshot."""
    if not revision:
        raise ValueError("HF input requires --revision (prefer an immutable commit SHA)")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("HF cache input requires pip install huggingface_hub") from exc
    snapshot = Path(
        snapshot_download(
            repo_id=dataset, repo_type="dataset", revision=revision, local_files_only=True
        )
    )
    paths = sorted((snapshot / "data").glob("train-*.parquet"))
    if not paths:
        raise ValueError(
            "Cached snapshot has no data/train-*.parquet; download it explicitly first"
        )
    # Official shards declare their expected count; do not quietly use a partial cache.
    expected = [f"train-{i:05d}-of-{len(paths):05d}.parquet" for i in range(len(paths))]
    if [path.name for path in paths] != expected:
        raise ValueError("Incomplete or unexpected cached train shard set")
    return paths, snapshot.name


def _image(value: Any, root: Path):
    from PIL import Image

    if isinstance(value, Image.Image):
        value.load()
        return value.convert("RGB")
    if not isinstance(value, dict):
        raise ValueError("image must be a PIL image or a Parquet {bytes, path} object")
    payload = value.get("bytes")
    if payload is not None:
        if not isinstance(payload, bytes):
            raise ValueError("image.bytes must contain encoded image bytes")
        source = io.BytesIO(payload)
    else:
        name = value.get("path")
        if not isinstance(name, str) or not name:
            raise ValueError("image requires bytes or a local path")
        source = _local_asset(root, name)
    with Image.open(source) as image:
        image.load()
        return image.convert("RGB")


def _local_asset(root: Path, name: Any) -> Path:
    if not isinstance(name, str) or not name or "://" in name:
        raise ValueError("Expected a local asset path")
    path = Path(name)
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        raise FileNotFoundError(f"Missing asset: {path}")
    return path


def _pixels(image) -> str:
    digest = hashlib.sha256(f"RGB:{image.width}:{image.height}:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected an object")
            except ValueError as exc:
                raise ValueError(f"{path.name}:{number}: {exc}") from exc
            yield number, row


def build_rea2seg(
    rows: Iterable[tuple[dict, dict, Path]],
    output_dir: str | Path,
    *,
    source: str = DATASET,
    revision: str | None = None,
    categories: Iterable[str] = CATEGORIES,
    max_samples: int | None = None,
    source_split: str = "train",
    merge: tuple[Path, Path] | None = None,
    exclude: Iterable[tuple[Path, Path]] = (),
) -> dict:
    """Export native RGB PNG pairs; fail closed on duplicates or held-out overlap.

    ``merge`` is one existing train/val manifest and its data root. ``exclude``
    contains known evaluation manifests and their roots; train rows in those
    manifests are ignored. No split is created, sampled or reassigned.
    """
    import numpy as np
    from PIL import Image

    from .data.res_contract import resolve_referring_prompt
    from .referring_masks import decode_coco_segmentation

    selected = set(categories)
    if not selected or selected - set(CATEGORIES):
        raise ValueError(f"categories must be a nonempty subset of {CATEGORIES}")
    if max_samples is not None and (
        not isinstance(max_samples, int) or isinstance(max_samples, bool) or max_samples <= 0
    ):
        raise ValueError("max_samples must be positive")
    if source_split != "train":
        raise ValueError("Rea2Seg conversion accepts only the original train split")
    root = Path(output_dir)
    # Exclusive creation prevents overwriting any previous dataset or partial build.
    root.mkdir(parents=True, exist_ok=False)
    marker = root / "BUILD_INCOMPLETE"
    marker.write_text("Do not train: conversion has not completed.\n", encoding="utf-8")
    (root / "images").mkdir()
    (root / "targets").mkdir()
    image_cache: dict[Path, str] = {}

    def image_hash(path: Path) -> str:
        path = path.resolve()
        if path not in image_cache:
            with Image.open(path) as image:
                image_cache[path] = _pixels(image.convert("RGB"))
        return image_cache[path]

    heldout: dict[str, set[str]] = {}
    exclusion_rows = 0
    for manifest, data_root in exclude:
        for number, row in _jsonl(Path(manifest)):
            split = row.get("split")
            if split == "train":
                continue
            if split not in ("val", "test", "testA", "testB"):
                raise ValueError(f"{manifest.name}:{number}: unknown exclusion split {split!r}")
            digest = image_hash(_local_asset(Path(data_root), row.get("input")))
            heldout.setdefault(digest, set()).add("val" if split == "val" else "test")
            exclusion_rows += 1

    images: dict[str, str] = {}
    targets: dict[str, str] = {}
    splits: dict[str, str] = {}
    identities: dict[tuple, str] = {}
    sample_ids: set[str] = set()
    base_train_images: set[str] = set()
    counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    scanned = filtered = added = base_rows = 0

    def check(row: dict, digest: str) -> None:
        split = row.get("split")
        if split not in ("train", "val"):
            raise ValueError("Merged training manifest must contain only explicit train/val splits")
        if row.get("source_split", split) != split:
            raise ValueError(
                "source_split conflicts with training split; evaluation cannot be relabeled"
            )
        if digest in heldout and (split == "train" or "test" in heldout[digest]):
            raise ValueError(f"Held-out image overlap for sample {row.get('sample_id')!r}")
        if digest in splits and splits[digest] != split:
            raise ValueError(f"Cross-split image overlap for sample {row.get('sample_id')!r}")
        splits[digest] = split
        task = row.get("task")
        if not isinstance(task, str) or not task:
            raise ValueError("Manifest record requires task")
        prompt = resolve_referring_prompt(row) if task == "referring" else row.get("prompt", "")
        identity = (digest, task, prompt)
        if identity in identities:
            raise ValueError(
                f"Duplicate image/query record: {row.get('sample_id')!r}; "
                f"first={identities[identity]!r} (including ReasonSeg duplicates)"
            )
        sample_id = row.get("sample_id")
        if sample_id is not None:
            if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
                raise ValueError(f"Invalid or duplicate sample_id: {sample_id!r}")
            sample_ids.add(sample_id)
        identities[identity] = sample_id or "existing manifest row"

    def save_image(image, digest: str) -> str:
        if digest not in images:
            name = f"images/{digest}.png"
            with (root / name).open("xb") as stream:
                image.save(stream, format="PNG")
            images[digest] = name
        return images[digest]

    temporary = root / "training.jsonl.tmp"
    with temporary.open("x", encoding="utf-8", newline="\n") as output:

        def emit(row: dict) -> None:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            counts[row["split"]] += 1

        if merge is not None:
            manifest, data_root = map(Path, merge)
            if any((p / "BUILD_INCOMPLETE").exists() for p in (manifest.parent, data_root)):
                raise ValueError("Cannot merge an incomplete dataset")
            for number, original in _jsonl(manifest):
                try:
                    row = dict(original)
                    input_path = _local_asset(data_root, row.get("input"))
                    target_path = _local_asset(data_root, row.get("target"))
                    digest = image_hash(input_path)
                    check(row, digest)
                    with Image.open(input_path) as image, Image.open(target_path) as target:
                        if image.size != target.size:
                            raise ValueError("Existing image/target dimensions differ")
                        for name, actual in (("width", image.width), ("height", image.height)):
                            if name in row and row[name] != actual:
                                raise ValueError(f"Existing {name} differs from image")
                        row["input"] = save_image(image.convert("RGB"), digest)
                        if row.get("encoding") == BINARY_REFERRING_ENCODING:
                            pixels = np.asarray(target.convert("RGB"))
                            if not np.all((pixels == 0) | (pixels == 255)) or not np.all(
                                pixels == pixels[..., :1]
                            ):
                                raise ValueError("Existing binary target is not white/black")
                    # Preserve other tasks' target bytes and suffix (e.g. depth PNGs).
                    target_hash = hashlib.sha256(target_path.read_bytes()).hexdigest()
                    target_name = f"targets/{target_hash}{target_path.suffix.lower()}"
                    if target_name not in targets:
                        with target_path.open("rb") as src, (root / target_name).open("xb") as dst:
                            shutil.copyfileobj(src, dst)
                        targets[target_name] = target_name
                    row["target"] = target_name
                    emit(row)
                    base_rows += 1
                    if row["split"] == "train":
                        base_train_images.add(digest)
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    raise ValueError(f"{manifest.name}:{number}: {exc}") from exc

        for raw, provenance, image_root in rows:
            label = f"{provenance.get('source_shard')}:{provenance.get('source_row')}"
            try:
                scanned += 1
                if not FIELDS.issubset(raw):
                    raise ValueError(f"Required fields are {sorted(FIELDS)}")
                category = raw["category"]
                if category not in CATEGORIES:
                    raise ValueError(f"Unknown category: {category!r}")
                if any(raw.get(key, source_split) != "train" for key in ("split", "source_split")):
                    raise ValueError("Evaluation rows cannot enter Rea2Seg training")
                if category not in selected:
                    filtered += 1
                    continue
                question = raw["question"]
                prompt = build_binary_referring_prompt(question)
                if not isinstance(raw["cot_answer"], str):
                    raise ValueError("cot_answer must be a string")
                image = _image(raw["image"], Path(image_root))
                for name, actual in (("width", image.width), ("height", image.height)):
                    if name in raw and raw[name] != actual:
                        raise ValueError(f"Source {name} differs from image")
                mask = json.loads(raw["mask"]) if isinstance(raw["mask"], str) else raw["mask"]
                if not isinstance(mask, dict) or not isinstance(mask.get("rle"), dict):
                    raise ValueError("mask must contain a COCO rle object")
                binary = decode_coco_segmentation(mask["rle"], image.height, image.width)
                digest = _pixels(image)
                if category == "reasonseg" and digest in base_train_images:
                    raise ValueError("ReasonSeg image already present in existing training data")
                source_id = raw.get("id", raw.get("sample_id"))
                identity = [source, revision, provenance, source_id]
                sample_id = (
                    "rea2seg_"
                    + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                )
                row = {
                    "sample_id": sample_id,
                    "source_sample_id": source_id,
                    "source_dataset": source,
                    "source_revision": revision,
                    "source_split": source_split,
                    **provenance,
                    "category": category,
                    "task": "referring",
                    "split": "train",
                    "expression": question,
                    "prompt": prompt,
                    "cot_answer": raw["cot_answer"],
                    "encoding": BINARY_REFERRING_ENCODING,
                    "width": image.width,
                    "height": image.height,
                    "image_sha256": digest,
                }
                check(row, digest)
                row["input"] = save_image(image, digest)
                target = Image.fromarray(binary.astype(np.uint8) * 255).convert("RGB")
                target_name = f"targets/mask_{_pixels(target)}.png"
                if target_name not in targets:
                    with (root / target_name).open("xb") as stream:
                        target.save(stream, format="PNG")
                    targets[target_name] = target_name
                row["target"] = target_name
                emit(row)
                category_counts[category] += 1
                added += 1
                if max_samples is not None and added >= max_samples:
                    break
            except (ValueError, OSError, KeyError, TypeError) as exc:
                raise ValueError(f"Rea2Seg {label}: {exc}") from exc
    if not added:
        raise ValueError("No Rea2Seg samples selected")
    report = {
        "schema_version": "rea2seg_training_v1",
        "source_dataset": source,
        "source_revision": revision,
        "source_split": source_split,
        "categories": sorted(selected),
        "max_samples": max_samples,
        "scanned_rows": scanned,
        "category_filtered_rows": filtered,
        "rea2seg_rows": added,
        "existing_rows": base_rows,
        "counts": dict(counts),
        "rea2seg_category_counts": dict(category_counts),
        "unique_images": len(images),
        "unique_targets": len(targets),
        "exclusion_rows": exclusion_rows,
        "exclusion_images": len(heldout),
        "image_identity": "sha256 of RGB dimensions and decoded pixels",
        "duplicates": "error; no records silently removed",
        "manifest": "training.jsonl",
    }
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.rename(root / "training.jsonl")
    marker.unlink()
    return report
