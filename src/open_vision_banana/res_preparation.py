"""Offline binary RES preparation shared by source adapters; no runtime loader."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .referring_contract import BINARY_REFERRING_ENCODING, build_binary_referring_prompt

FIELDS = frozenset(("image", "question", "cot_answer", "mask", "category"))


def _rejection_code(message: str) -> str:
    for text, code in (
        ("Held-out", "heldout_image_overlap"),
        ("Cross-split", "cross_split_image_overlap"),
        ("Duplicate image/query", "duplicate_image_query"),
        ("duplicate sample_id", "duplicate_sample_id"),
        ("empty foreground", "empty_mask"),
        ("native image dimensions", "mask_dimension_mismatch"),
        ("Foreground fraction", "foreground_fraction"),
        ("ReasonSeg image already", "reasonseg_duplicate_image"),
    ):
        if text in message:
            return code
    return "invalid_source_record"


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


def build_res_pairs(
    rows: Iterable[tuple[dict, dict, Path]],
    output_dir: str | Path,
    *,
    source: str,
    namespace: str,
    source_label: str,
    source_protocol: str,
    available_categories: Iterable[str],
    revision: str | None = None,
    categories: Iterable[str],
    max_samples: int | None = None,
    source_split: str = "train",
    merge: tuple[Path, Path] | None = None,
    exclude: Iterable[tuple[Path, Path]] = (),
    reject_invalid: bool = False,
    exclude_image_ids: Iterable[int] = (),
    min_mask_fraction: float = 0.0,
    max_mask_fraction: float = 1.0,
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
    available = set(available_categories)
    blocked_ids = set(exclude_image_ids)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in blocked_ids
    ):
        raise ValueError("Excluded COCO image ids must be nonnegative integers")
    if not 0 <= min_mask_fraction < max_mask_fraction <= 1:
        raise ValueError("Mask area bounds must satisfy 0 <= min < max <= 1")
    if not selected or selected - available:
        raise ValueError(f"categories must be a nonempty subset of {sorted(available)}")
    if max_samples is not None and (
        not isinstance(max_samples, int) or isinstance(max_samples, bool) or max_samples <= 0
    ):
        raise ValueError("max_samples must be positive")
    if source_split != "train":
        raise ValueError(f"{source_label} conversion accepts only the original train split")
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
    exclusion_sources = []
    for manifest, data_root in exclude:
        manifest = Path(manifest)
        exclusion_sources.append(
            {"manifest": manifest.name, "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
        )
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
    rejected: Counter[str] = Counter()

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
    with (
        temporary.open("x", encoding="utf-8", newline="\n") as output,
        (root / "rejected.jsonl").open("x", encoding="utf-8") as rejection_log,
    ):

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
                if category not in available:
                    raise ValueError(f"Unknown category: {category!r}")
                if any(raw.get(key, source_split) != "train" for key in ("split", "source_split")):
                    raise ValueError(f"Evaluation rows cannot enter {source_label} training")
                if category not in selected:
                    filtered += 1
                    continue
                if raw.get("image_id") in blocked_ids:
                    raise ValueError("Held-out COCO image id overlap")
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
                if not min_mask_fraction <= float(binary.mean()) <= max_mask_fraction:
                    raise ValueError("Foreground fraction outside selected quality bounds")
                digest = _pixels(image)
                if category == "reasonseg" and digest in base_train_images:
                    raise ValueError("ReasonSeg image already present in existing training data")
                source_id = raw.get("id", raw.get("sample_id"))
                identity = [source, revision, provenance, source_id]
                sample_id = (
                    f"{namespace}_"
                    + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                )
                row = {
                    "sample_id": sample_id,
                    "source_sample_id": source_id,
                    "source_dataset": source,
                    "source_revision": revision,
                    "source_protocol": source_protocol,
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
                for key in ("image_id", "source_image", "question_id", "mask_transform", "answer"):
                    if key in raw:
                        row[key] = raw[key]
                check(row, digest)
                row["input"] = save_image(image, digest)
                target = Image.fromarray(binary.astype(np.uint8) * 255).convert("RGB")
                row["target_sha256"] = _pixels(target)
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
                if not reject_invalid or isinstance(exc, FileNotFoundError):
                    raise ValueError(f"{source_label} {label}: {exc}") from exc
                reason = str(exc)
                code = _rejection_code(reason)
                rejected[code] += 1
                rejection_log.write(
                    json.dumps({**provenance, "code": code, "reason": reason}, sort_keys=True)
                    + "\n"
                )
    if not added:
        raise ValueError(f"No {source_label} samples selected")
    report = {
        "schema_version": f"{namespace}_training_v1",
        "source_dataset": source,
        "source_revision": revision,
        "source_split": source_split,
        "categories": sorted(selected),
        "max_samples": max_samples,
        "scanned_rows": scanned,
        "category_filtered_rows": filtered,
        f"{namespace}_rows": added,
        "existing_rows": base_rows,
        "counts": dict(counts),
        f"{namespace}_category_counts": dict(category_counts),
        "unique_images": len(images),
        "unique_targets": len(targets),
        "exclusion_rows": exclusion_rows,
        "exclusion_images": len(heldout),
        "image_identity": "sha256 of RGB dimensions and decoded pixels",
        "duplicates": "recorded in rejected.jsonl"
        if reject_invalid
        else "error; no records silently removed",
        "rejected_rows": sum(rejected.values()),
        "rejected_by_reason": dict(rejected),
        "reject_invalid": reject_invalid,
        "excluded_coco_image_ids": len(blocked_ids),
        "excluded_coco_image_ids_sha256": hashlib.sha256(
            json.dumps(sorted(blocked_ids)).encode()
        ).hexdigest(),
        "exclusion_sources": exclusion_sources,
        "manifest_sha256": hashlib.sha256(temporary.read_bytes()).hexdigest(),
        "manifest": "training.jsonl",
    }
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.rename(root / "training.jsonl")
    marker.unlink()
    return report


def validate_prepared_pairs(data_dir: str | Path) -> dict:
    """Decode source-adapter assets and prepare metadata for the existing RES bundler.

    This validates the generic binary pair contract, not the RefCOCO-specific
    official-expression accounting performed by res_manifest.validate_manifest.
    """
    import numpy as np
    from PIL import Image

    from .data.res_contract import resolve_referring_prompt
    from .res_bundle import safe_path, sha256_file

    root = Path(data_dir).resolve()
    if (root / "BUILD_INCOMPLETE").exists():
        raise ValueError("Cannot validate an incomplete dataset")
    manifest = root / "training.jsonl"
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    if sha256_file(manifest) != report["manifest_sha256"]:
        raise ValueError("Prepared manifest changed since conversion")
    inventory = {}
    images = {}
    targets = {}
    seen_ids = set()
    splits: dict[str, set[str]] = {}
    counts: Counter[str] = Counter()
    for _, row in _jsonl(manifest):
        if row.get("task") != "referring" or row.get("encoding") != BINARY_REFERRING_ENCODING:
            raise ValueError("Publication requires versioned binary referring rows")
        resolve_referring_prompt(row)
        for key in ("sample_id", "source_dataset", "source_protocol", "source_split"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing provenance field {key}")
        if row["sample_id"] in seen_ids:
            raise ValueError("Duplicate sample_id")
        seen_ids.add(row["sample_id"])
        if row.get("split") not in {"train", "val"} or row["split"] != row["source_split"]:
            raise ValueError("Invalid train/val split provenance")
        for key in ("input", "target"):
            name = row[key]
            if name in inventory:
                continue
            path = safe_path(root, name)
            with Image.open(path) as image:
                image.load()
                if image.format != "PNG" or image.mode != "RGB":
                    raise ValueError("Prepared assets must be lossless RGB PNGs")
                if key == "input":
                    images[name] = (image.size, _pixels(image))
                else:
                    pixels = np.asarray(image)
                    if (
                        not np.isin(pixels, [0, 255]).all()
                        or not (pixels == pixels[..., :1]).all()
                        or not pixels.any()
                    ):
                        raise ValueError("Target must be a nonempty native white/black mask")
                    targets[name] = (image.size, _pixels(image))
                inventory[name] = {
                    "path": name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
        size, digest = images[row["input"]]
        target_size, target_digest = targets[row["target"]]
        if size != target_size or size != (row["width"], row["height"]):
            raise ValueError("Native asset/metadata dimensions differ")
        if row.get("image_sha256") != digest:
            raise ValueError("Image pixels changed since conversion")
        if row.get("target_sha256") != target_digest:
            raise ValueError("Target pixels changed since conversion")
        splits.setdefault(digest, set()).add(row["split"])
        if len(splits[digest]) > 1:
            raise ValueError("Cross-split image overlap")
        counts[row["split"]] += 1
    if not seen_ids or dict(counts) != report["counts"]:
        raise ValueError("Conversion and validation counts differ")
    (root / "pairs.jsonl").write_bytes(manifest.read_bytes())
    (root / "assets.jsonl").write_text(
        "".join(json.dumps(inventory[name], sort_keys=True) + "\n" for name in sorted(inventory)),
        encoding="utf-8",
    )
    names = ("training.jsonl", "pairs.jsonl", "assets.jsonl", "report.json", "rejected.jsonl")
    files = {
        name: {"bytes": (root / name).stat().st_size, "sha256": sha256_file(root / name)}
        for name in names
    }
    fingerprint = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    build = {
        "status": "complete",
        "files": files,
        "dataset_fingerprint": fingerprint,
        "counts": dict(counts),
    }
    validation = {
        "status": "passed",
        "validator": "generic_binary_res_pairs_v1",
        "dataset_fingerprint": fingerprint,
        "counts": dict(counts),
        "unique_images": len(images),
        "unique_targets": len(targets),
        "assets_decoded": len(inventory),
    }
    (root / "build_report.json").write_text(json.dumps(build, indent=2) + "\n", encoding="utf-8")
    (root / "validation_report.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    return validation
