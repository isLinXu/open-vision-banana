"""Reconcile SeqTR/C3VG mixed RES annotations with the official source release.

Mixed training rows omit the original dataset/ref/sentence identities and use
REFER's processed ``sent`` text. They must not be appended as new human data
merely because that text differs from the canonical release's verbatim ``raw``.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .referring_masks import decode_coco_segmentation
from .res_manifest import (
    SOURCE_SPECS,
    _json,
    _read_refs,
    _stream_items,
    portable_path,
    sha256_file,
)

MIXED_SPLITS = (
    "train",
    "val_refcoco_unc",
    "testA_refcoco_unc",
    "testB_refcoco_unc",
    "val_refcocoplus_unc",
    "testA_refcocoplus_unc",
    "testB_refcocoplus_unc",
    "val_refcocog_umd",
    "test_refcocog_umd",
)


def _geometry(bbox: Any, mask: Any) -> str:
    return hashlib.sha256(_json({"bbox": bbox, "segmentation": mask}).encode()).hexdigest()


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Mixed RES {field} must be a positive integer")
    return value


def _references(root: Path) -> tuple[Counter, set[int], list[dict]]:
    expected: Counter = Counter()
    held_out: set[int] = set()
    fingerprints = []
    for source, protocol in SOURCE_SPECS:
        refs_path = root / source / f"refs({protocol}).p"
        instances = root / source / "instances.json"
        for path in (refs_path, instances):
            fingerprints.append(
                {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
            )
        refs = _read_refs(refs_path)
        wanted = {ref["ann_id"] for ref in refs}
        images = {
            row["id"]: (row["width"], row["height"])
            for row in _stream_items(instances, "images.item")
        }
        # MixedSeg stores contiguous zero-based categories, not COCO category IDs.
        category_ids = sorted(row["id"] for row in _stream_items(instances, "categories.item"))
        category_index = {value: index for index, value in enumerate(category_ids)}
        annotations = {
            row["id"]: (
                row["image_id"],
                _geometry(row["bbox"], row["segmentation"]),
                category_index[row["category_id"]],
            )
            for row in _stream_items(instances, "annotations.item")
            if row["id"] in wanted
        }
        if len(annotations) != len(wanted):
            raise ValueError(f"Official {source} references have missing annotations")
        for ref in refs:
            image_id, geometry, category = annotations[ref["ann_id"]]
            if ref["image_id"] != image_id:
                raise ValueError("Official reference/annotation image IDs differ")
            split = ref["split"]
            name = (
                "train" if split == "train" else f"{split}_{source.replace('+', 'plus')}_{protocol}"
            )
            if name not in MIXED_SPLITS:
                raise ValueError(f"Unsupported official split: {name}")
            if split != "train":
                held_out.add(image_id)
            width, height = images[image_id]
            for sentence in ref["sentences"]:
                text = sentence.get("sent")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Official processed expression must be nonempty")
                expected[(name, image_id, geometry, width, height, category, text)] += 1
    return expected, held_out, fingerprints


def _release(root: Path, sources: list[dict]) -> tuple[dict, dict, dict]:
    if (root / "BUILD_INCOMPLETE").exists():
        raise ValueError("Cannot reconcile against an incomplete RES release")
    build = json.loads((root / "build_report.json").read_text(encoding="utf-8"))
    validation = json.loads((root / "validation_report.json").read_text(encoding="utf-8"))
    if (
        build.get("status") != "complete"
        or validation.get("status") != "passed"
        or build.get("dataset_fingerprint") != validation.get("dataset_fingerprint")
    ):
        raise ValueError("A successfully validated canonical RES release is required")
    if {(s["path"], s["sha256"]) for s in sources} != {
        (s["path"], s["sha256"]) for s in build["source_files"]
    }:
        raise ValueError("Official source hashes differ from the canonical RES release")
    fingerprint = hashlib.sha256(_json(build["files"]).encode()).hexdigest()
    if fingerprint != build["dataset_fingerprint"]:
        raise ValueError("Canonical release metadata fingerprint mismatch")
    for name in ("pairs.jsonl", "assets.jsonl"):
        if sha256_file(root / name) != build["files"][name]["sha256"]:
            raise ValueError(f"Canonical release metadata changed: {name}")
    with (root / "assets.jsonl").open(encoding="utf-8") as handle:
        assets = {row["path"]: row for row in map(json.loads, handle)}
    masks: dict[tuple[int, str], tuple[str, int, int]] = {}
    with (root / "pairs.jsonl").open(encoding="utf-8") as handle:
        for row in map(json.loads, handle):
            key = (row["image_id"], row["geometry_sha256"])
            target = (row["target"], row["width"], row["height"])
            if key in masks and masks[key] != target:
                raise ValueError("Canonical manifest has conflicting mask metadata")
            masks[key] = target
    return build, assets, masks


def audit_mixed_res(
    annotations_json: str | Path,
    annotations_root: str | Path,
    image_root: str | Path,
    dataset_root: str | Path,
) -> dict:
    """Verify every mixed expression and native mask without changing any split.

    This deliberately accepts only a complete equivalent mixed RefCOCO-family
    source. Unknown expressions, missing/duplicate official records and altered
    masks fail reconciliation. New datasets require their own provenance/split
    review; this auditor does not silently label them equivalent or append them.
    """
    try:
        import ijson
    except ImportError as exc:
        raise ImportError("Mixed RES auditing requires open-vision-banana[data]") from exc
    annotations_json, annotations_root, image_root, dataset_root = map(
        Path, (annotations_json, annotations_root, image_root, dataset_root)
    )
    source_hash = sha256_file(annotations_json)
    expected, held_out, sources = _references(annotations_root)
    official_expressions = sum(expected.values())
    build, assets, released_masks = _release(dataset_root, sources)
    seen_splits: set[str] = set()
    seen_images: dict[int, tuple[int, int]] = {}
    seen_masks: set[tuple[int, str]] = set()
    matched_masks: set[tuple[int, str]] = set()
    mask_kinds: Counter = Counter()
    counts = {}
    held_out_train = 0
    with annotations_json.open("rb") as handle:
        for split, rows in ijson.kvitems(handle, "", use_float=True):
            if split not in MIXED_SPLITS or split in seen_splits:
                raise ValueError(f"Unknown or duplicate Mixed RES split: {split}")
            seen_splits.add(split)
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"Mixed RES split {split} must contain records")
            expressions = 0
            for index, row in enumerate(rows):
                context = f"{split}[{index}]"
                if not isinstance(row, dict):
                    raise ValueError(f"Mixed RES {context} must be an object")
                image_id = _positive_int(row.get("image_id"), "image_id")
                width = _positive_int(row.get("width"), "width")
                height = _positive_int(row.get("height"), "height")
                category = row.get("category_id")
                if not isinstance(category, int) or isinstance(category, bool) or category < 0:
                    raise ValueError(f"Mixed RES {context}: invalid contiguous category_id")
                texts = row.get("expressions")
                if not isinstance(texts, list) or not texts:
                    raise ValueError(f"Mixed RES {context}: expressions must be nonempty")
                geometry = _geometry(row.get("bbox"), row.get("mask"))
                for text in texts:
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError(f"Mixed RES {context}: expression must be nonempty")
                    key = (split, image_id, geometry, width, height, category, text)
                    if expected[key] <= 0:
                        raise ValueError(
                            f"Mixed RES {context}: record differs from official sources"
                        )
                    expected[key] -= 1
                    expressions += 1
                if split == "train" and image_id in held_out:
                    held_out_train += len(texts)
                image_name = f"COCO_train2014_{image_id:012d}.jpg"
                image_path = image_root / image_name
                if image_id not in seen_images:
                    with Image.open(image_path) as image:
                        image.load()
                        if image.size != (width, height):
                            raise ValueError(f"Mixed RES {context}: native image dimensions differ")
                    asset_name = f"images/coco2014/{image_name}"
                    entry = assets.get(asset_name)
                    if entry is None or sha256_file(image_path) != entry["sha256"]:
                        raise ValueError(
                            f"Mixed RES {context}: image differs from released COCO asset"
                        )
                    seen_images[image_id] = (width, height)
                elif seen_images[image_id] != (width, height):
                    raise ValueError(f"Mixed RES {context}: conflicting image dimensions")
                mask_key = (image_id, geometry)
                if mask_key in seen_masks:
                    continue
                mask = decode_coco_segmentation(row.get("mask"), height, width)
                seen_masks.add(mask_key)
                mask_kinds["polygon" if isinstance(row["mask"], list) else "rle"] += 1
                if mask_key in released_masks:
                    target, target_width, target_height = released_masks[mask_key]
                    target_path = portable_path(dataset_root, target)
                    if (target_width, target_height) != (width, height):
                        raise ValueError("Released mask dimensions differ from Mixed RES")
                    if sha256_file(target_path) != assets[target]["sha256"]:
                        raise ValueError(f"Released target checksum changed: {target}")
                    with Image.open(target_path) as image:
                        rgb = np.asarray(image.convert("RGB"))
                    if not np.array_equal(rgb, np.repeat(mask[..., None], 3, axis=2) * 255):
                        raise ValueError(f"Released mask pixels differ from Mixed RES: {target}")
                    matched_masks.add(mask_key)
            counts[split] = {"records": len(rows), "expressions": expressions}
    if seen_splits != set(MIXED_SPLITS) or any(expected.values()):
        raise ValueError("Mixed RES source omits official splits or expressions")
    if matched_masks != set(released_masks):
        raise ValueError("Mixed RES source does not cover all canonical released masks")
    if sha256_file(annotations_json) != source_hash:
        raise ValueError("Mixed RES source changed during audit")
    return {
        "schema_version": "res_mixed_audit_v1",
        "status": "passed",
        "source_file": annotations_json.name,
        "source_sha256": source_hash,
        "source_bytes": annotations_json.stat().st_size,
        "official_source_files": sources,
        "canonical_dataset_fingerprint": build["dataset_fingerprint"],
        "counts": counts,
        "expressions_matched_to_official_sent": official_expressions,
        "novel_expressions": 0,
        "unique_images_decoded_and_sha256_matched": len(seen_images),
        "unique_native_masks_decoded": len(seen_masks),
        "mask_types": dict(mask_kinds),
        "released_masks_pixel_identical": len(matched_masks),
        "source_masks_outside_filtered_release": len(seen_masks - matched_masks),
        "train_expressions_on_official_held_out_images": held_out_train,
        "canonical_released_counts": build["counts"],
        "training_action": "reuse canonical training.jsonl; do not concatenate this source",
        "text_policy": "exact official sent matching; canonical manifests preserve raw text",
        "visual_review": "not performed by this automated auditor",
    }
