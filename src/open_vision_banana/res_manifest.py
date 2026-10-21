"""Build and audit portable RES data without importing the training stack.

Official RefCOCO pickles contain only Python primitives. They are read with a
restricted unpickler; COCO JSON is streamed into a temporary SQLite index so
the build does not require all three annotation collections in memory.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import sqlite3
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
from PIL import Image

from .referring_contract import BINARY_REFERRING_ENCODING, build_binary_referring_prompt
from .referring_masks import decode_coco_segmentation
from .res_training_view import validate_training_view, write_training_view

SOURCE_SPECS = (("refcoco", "unc"), ("refcoco+", "unc"), ("refcocog", "umd"))
SOURCE_SPLITS = frozenset(("train", "val", "test", "testA", "testB"))
SCHEMA_VERSION = "res_manifest_v1"


class PrimitiveUnpickler(pickle.Unpickler):
    """Disallow executable globals and persistent object references."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"Globals are forbidden in refs: {module}.{name}")

    def persistent_load(self, pid: Any) -> Any:
        raise pickle.UnpicklingError("Persistent IDs are forbidden in refs")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def portable_path(root: Path, value: Any) -> Path:
    """Resolve a POSIX relative asset path, rejecting traversal and symlink escape."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"Expected a portable relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or PureWindowsPath(value).is_absolute() or ".." in path.parts:
        raise ValueError(f"Path escapes dataset root: {value!r}")
    if path.as_posix() != value or value in (".", ""):
        raise ValueError(f"Path is not canonical: {value!r}")
    if any(
        part.endswith((".", " "))
        or PureWindowsPath(part).is_reserved()
        or any(character in '<>"|?*' or ord(character) < 32 for character in part)
        for part in path.parts
    ):
        raise ValueError(f"Path is not portable to Windows: {value!r}")
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes dataset root: {value!r}")
    return resolved


def _integer(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _read_refs(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as stream:
        refs = PrimitiveUnpickler(stream).load()
    if not isinstance(refs, list) or not refs:
        raise ValueError(f"{path.name} must contain a nonempty list of refs")
    return refs


def _stream_items(path: Path, prefix: str):
    try:
        import ijson
    except ImportError as exc:
        raise ImportError("RES building requires ijson; install open-vision-banana[data]") from exc
    with path.open("rb") as stream:
        yield from ijson.items(stream, prefix, use_float=True)


def _sample_id(source: str, protocol: str, ref_id: int, sent_id: int) -> str:
    return f"{source}_{protocol}_ref{ref_id}_sent{sent_id}"


def write_image_attribution(annotations_root: str | Path, output_dir: str | Path) -> int:
    """Preserve available COCO image provenance for every exported image.

    Only source-provided values are recorded. Missing license or URL metadata
    remains null; complementary source records may fill null fields, but
    conflicting supplied values fail. Source JSON is streamed, so annotations
    and segmentation payloads need not be retained in memory.
    """
    annotations_root, output_dir = Path(annotations_root), Path(output_dir)
    exported: dict[str, dict[str, Any]] = {}
    with (output_dir / "assets.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            asset = json.loads(line)
            if asset.get("kind") != "image":
                continue
            portable_path(output_dir, asset["path"])
            name = PurePosixPath(asset["path"]).name
            if name in exported:
                raise ValueError(f"Ambiguous exported image basename for attribution: {name}")
            exported[name] = asset
    records: dict[str, dict[str, Any]] = {}
    image_paths: dict[int, str] = {}
    for source, _ in SOURCE_SPECS:
        instances_path = annotations_root / source / "instances.json"
        licenses: dict[int, dict[str, Any]] = {}
        for entry in _stream_items(instances_path, "licenses.item"):
            license_id = _integer(entry.get("id"), "license id")
            info = {"license_name": entry.get("name"), "license_url": entry.get("url")}
            previous = licenses.get(license_id)
            if previous is not None and previous != info:
                raise ValueError(f"Conflicting COCO license id={license_id} in {source}")
            licenses[license_id] = info
        seen_source: set[str] = set()
        for entry in _stream_items(instances_path, "images.item"):
            name = entry.get("file_name")
            if name not in exported:
                continue
            if name in seen_source:
                raise ValueError(f"Duplicate source image attribution for {name} in {source}")
            seen_source.add(name)
            asset = exported[name]
            image_id = _integer(entry.get("id"), "image id")
            previous_path = image_paths.get(image_id)
            if previous_path is not None and previous_path != asset["path"]:
                raise ValueError(f"Conflicting exported paths for COCO image_id={image_id}")
            image_paths[image_id] = asset["path"]
            for dimension in ("width", "height"):
                if entry.get(dimension) is not None and entry[dimension] != asset.get(dimension):
                    raise ValueError(
                        f"Image attribution {dimension} differs from exported asset: {name}"
                    )
            license_id = entry.get("license")
            info = licenses.get(license_id, {})
            record = {
                "image_id": image_id,
                "input": asset["path"],
                "license": license_id,
                "license_name": info.get("license_name"),
                "license_url": info.get("license_url"),
                "coco_url": entry.get("coco_url"),
                "flickr_url": entry.get("flickr_url"),
                "file_name": name,
                "width": entry.get("width"),
                "height": entry.get("height"),
            }
            previous = records.get(name)
            if previous is not None:
                for key, value in record.items():
                    if value is None:
                        continue
                    if previous[key] is not None and previous[key] != value:
                        raise ValueError(
                            f"Conflicting image attribution {key} for {name} in {source}"
                        )
                    previous[key] = value
            else:
                records[name] = record
    missing = set(exported) - set(records)
    if missing:
        raise ValueError(
            f"Missing COCO attribution for {len(missing)} exported images: {sorted(missing)[:3]}"
        )
    temporary = output_dir / "image_attribution.jsonl.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in sorted(records.values(), key=lambda item: item["input"]):
            stream.write(_json(record) + "\n")
    temporary.replace(output_dir / "image_attribution.jsonl")
    return len(records)


def _index_sources(connection: sqlite3.Connection, root: Path) -> dict[str, Any]:
    connection.executescript("""
        CREATE TABLE samples (
            id INTEGER PRIMARY KEY, source TEXT, protocol TEXT, source_split TEXT,
            image_id INTEGER, ann_id INTEGER, ref_id INTEGER, sent_id INTEGER, expression TEXT,
            UNIQUE(source, protocol, ref_id, sent_id));
        CREATE TABLE annotations (
            source TEXT, ann_id INTEGER, image_id INTEGER, geometry TEXT, geometry_hash TEXT,
            PRIMARY KEY(source, ann_id));
        CREATE TABLE images (image_id INTEGER PRIMARY KEY, file_name TEXT, width INTEGER, height INTEGER);
        CREATE TABLE selected_train (
            identity TEXT PRIMARY KEY, sample_id TEXT NOT NULL);
        CREATE INDEX sample_source ON samples(source);
    """)
    report: dict[str, Any] = {"sources": {}, "source_files": []}
    for source, protocol in SOURCE_SPECS:
        refs_path = root / source / f"refs({protocol}).p"
        instances_path = root / source / "instances.json"
        if not refs_path.is_file() or not instances_path.is_file():
            raise FileNotFoundError(f"Required official files missing for {source}/{protocol}")
        for path in (refs_path, instances_path):
            report["source_files"].append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        refs = _read_refs(refs_path)
        counts: Counter[str] = Counter()
        ref_counts: Counter[str] = Counter()
        seen_refs: set[int] = set()
        for ref in sorted(refs, key=lambda item: item["ref_id"]):
            ref_id = _integer(ref.get("ref_id"), "ref_id")
            if ref_id in seen_refs:
                raise ValueError(f"Duplicate ref_id {ref_id} in {source}")
            seen_refs.add(ref_id)
            image_id = _integer(ref.get("image_id"), "image_id")
            ann_id = _integer(ref.get("ann_id"), "ann_id")
            split = ref.get("split")
            if split not in SOURCE_SPLITS:
                raise ValueError(f"Unknown official split {split!r} in {source}")
            sentences = ref.get("sentences")
            if not isinstance(sentences, list) or not sentences:
                raise ValueError(f"No sentences for {source} ref {ref_id}")
            ref_counts[split] += 1
            for sentence in sorted(sentences, key=lambda item: item["sent_id"]):
                sent_id = _integer(sentence.get("sent_id"), "sent_id")
                # Keep the human text verbatim; case folding would change exact deduplication.
                expression = sentence.get("raw", sentence.get("sent"))
                if not isinstance(expression, str) or not expression.strip():
                    raise ValueError(f"Empty expression in {source} ref {ref_id}")
                connection.execute(
                    "INSERT INTO samples VALUES (NULL,?,?,?,?,?,?,?,?)",
                    (
                        source,
                        protocol,
                        split,
                        image_id,
                        ann_id,
                        ref_id,
                        sent_id,
                        expression,
                    ),
                )
                counts[split] += 1
        del refs
        wanted_images = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT image_id FROM samples WHERE source=?", (source,)
            )
        }
        wanted_anns = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT ann_id FROM samples WHERE source=?", (source,)
            )
        }
        found_images: set[int] = set()
        for entry in _stream_items(instances_path, "images.item"):
            image_id = entry.get("id")
            if image_id not in wanted_images:
                continue
            if image_id in found_images:
                raise ValueError(f"Duplicate COCO image ID {image_id} in {source}")
            found_images.add(image_id)
            width = _integer(entry.get("width"), "width")
            height = _integer(entry.get("height"), "height")
            if min(width, height) < 1:
                raise ValueError(f"Invalid dimensions for image {image_id}")
            name = entry.get("file_name")
            portable_path(root, name)
            if PurePosixPath(name).name != name:
                raise ValueError("COCO source image file_name must be a basename")
            previous = connection.execute(
                "SELECT file_name,width,height FROM images WHERE image_id=?", (image_id,)
            ).fetchone()
            if previous and previous != (name, width, height):
                raise ValueError(f"Conflicting source metadata for image {image_id}")
            connection.execute(
                "INSERT OR IGNORE INTO images VALUES (?,?,?,?)", (image_id, name, width, height)
            )
        if found_images != wanted_images:
            raise ValueError(
                f"Missing COCO image metadata in {source}: {len(wanted_images - found_images)}"
            )
        found_anns: set[int] = set()
        for entry in _stream_items(instances_path, "annotations.item"):
            ann_id = entry.get("id")
            if ann_id not in wanted_anns:
                continue
            if ann_id in found_anns:
                raise ValueError(f"Duplicate COCO annotation ID {ann_id} in {source}")
            found_anns.add(ann_id)
            image_id = _integer(entry.get("image_id"), "annotation image_id")
            geometry = _json({"bbox": entry.get("bbox"), "segmentation": entry.get("segmentation")})
            geometry_hash = hashlib.sha256(geometry.encode("utf-8")).hexdigest()
            connection.execute(
                "INSERT INTO annotations VALUES (?,?,?,?,?)",
                (source, ann_id, image_id, geometry, geometry_hash),
            )
        if found_anns != wanted_anns:
            raise ValueError(
                f"Missing COCO annotations in {source}: {len(wanted_anns - found_anns)}"
            )
        mismatches = connection.execute(
            """SELECT COUNT(*) FROM samples s JOIN annotations a
            ON s.source=a.source AND s.ann_id=a.ann_id
            WHERE s.source=? AND s.image_id!=a.image_id""",
            (source,),
        ).fetchone()[0]
        if mismatches:
            raise ValueError(f"{source}: {mismatches} refs have annotation/image ID mismatches")
        report["sources"][f"{source}/{protocol}"] = {
            "refs_by_split": dict(sorted(ref_counts.items())),
            "expressions_by_split": dict(sorted(counts.items())),
            "unique_images": len(wanted_images),
            "unique_annotations": len(wanted_anns),
        }
        connection.commit()
    return report


def build_res_manifest(
    annotations_root: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    *,
    max_train_samples: int | None = None,
    image_mode: str = "copy",
) -> dict[str, Any]:
    """Build all official evaluation rows and a globally disjoint train pool.

    A pilot limit applies only after the complete cross-dataset evaluation
    blacklist and train deduplication. It never changes official eval counts.
    The output directory must be empty, preventing a partial build from being
    mistaken for a previous validated dataset.
    """
    if image_mode != "copy":
        raise ValueError("Only image_mode='copy' is portable and supported")
    if max_train_samples is not None and max_train_samples < 0:
        raise ValueError("max_train_samples must be nonnegative")
    annotations_root, image_root, output_dir = map(Path, (annotations_root, image_root, output_dir))
    if not image_root.is_dir():
        raise FileNotFoundError("COCO image root does not exist")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory must be empty; use a new directory for each build")
    output_dir.mkdir(parents=True, exist_ok=True)
    incomplete = output_dir / "BUILD_INCOMPLETE"
    incomplete.write_text("Build has not completed. Do not train or publish.\n", encoding="utf-8")
    database = output_dir / ".build.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA cache_size=-16384")
    connection.execute("PRAGMA temp_store=FILE")
    handles: dict[str, Any] = {}
    cursor = None
    try:
        report = _index_sources(connection, annotations_root)
        blocked = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT image_id FROM samples WHERE source_split!='train'"
            )
        }
        _write_json(output_dir / "evaluation_image_ids.json", sorted(blocked))
        counts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        excluded: Counter[str] = Counter()
        duplicates: Counter[str] = Counter()
        pilot_omitted = 0
        image_paths: dict[int, str] = {}
        target_paths: dict[tuple[int, str], str] = {}
        train_images: set[int] = set()
        eval_images: set[int] = set()

        def stream(name: str):
            if name not in handles:
                handles[name] = (output_dir / name).open("w", encoding="utf-8", newline="\n")
            return handles[name]

        inventory = stream("assets.jsonl")
        duplicate_log = stream("train_duplicates.jsonl")
        stream("pairs.jsonl")
        stream("train.jsonl")
        stream("val.jsonl")
        stream("test.jsonl")
        cursor = connection.execute("""SELECT s.source,s.protocol,s.source_split,s.image_id,
            s.ann_id,s.ref_id,s.sent_id,s.expression,a.geometry,a.geometry_hash,
            i.file_name,i.width,i.height FROM samples s
            JOIN annotations a ON s.source=a.source AND s.ann_id=a.ann_id
            JOIN images i ON s.image_id=i.image_id ORDER BY s.id""")
        for (
            source,
            protocol,
            source_split,
            image_id,
            ann_id,
            ref_id,
            sent_id,
            expression,
            geometry_json,
            geometry_hash,
            file_name,
            width,
            height,
        ) in cursor:
            source_key = f"{source}/{protocol}"
            split = source_split if source_split in ("train", "val") else "test"
            sample_id = _sample_id(source, protocol, ref_id, sent_id)
            if split == "train":
                if image_id in blocked:
                    excluded[source_key] += 1
                    continue
                identity = hashlib.sha256(
                    _json([image_id, geometry_hash, expression]).encode("utf-8")
                ).hexdigest()
                primary = connection.execute(
                    "SELECT sample_id FROM selected_train WHERE identity=?", (identity,)
                ).fetchone()
                if primary:
                    duplicates[source_key] += 1
                    duplicate_log.write(
                        _json(
                            {
                                "primary_sample_id": primary[0],
                                "duplicate_sample_id": sample_id,
                                "source_dataset": source,
                                "source_protocol": protocol,
                                "image_id": image_id,
                                "ann_id": ann_id,
                                "ref_id": ref_id,
                                "sent_id": sent_id,
                                "identity_sha256": identity,
                            }
                        )
                        + "\n"
                    )
                    continue
                connection.execute("INSERT INTO selected_train VALUES (?,?)", (identity, sample_id))
                if max_train_samples is not None and counts["train"] >= max_train_samples:
                    pilot_omitted += 1
                    continue
                train_images.add(image_id)
            else:
                eval_images.add(image_id)
            if image_id not in image_paths:
                source_path = portable_path(image_root, file_name)
                with Image.open(source_path) as image:
                    image.load()
                    if image.size != (width, height):
                        raise ValueError(
                            f"COCO image {image_id} dimensions disagree with annotations"
                        )
                relative = f"images/coco2014/{file_name}"
                destination = portable_path(output_dir, relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, destination)
                inventory.write(
                    _json(
                        {
                            "path": relative,
                            "kind": "image",
                            "width": width,
                            "height": height,
                            "bytes": destination.stat().st_size,
                            "sha256": sha256_file(destination),
                        }
                    )
                    + "\n"
                )
                image_paths[image_id] = relative
            target_key = (image_id, geometry_hash)
            if target_key not in target_paths:
                geometry = json.loads(geometry_json)
                mask = decode_coco_segmentation(geometry["segmentation"], height, width)
                relative = f"targets/coco2014/{image_id:012d}_{geometry_hash}.png"
                destination = portable_path(output_dir, relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                rgb = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
                Image.fromarray(rgb, mode="RGB").save(destination)
                inventory.write(
                    _json(
                        {
                            "path": relative,
                            "kind": "mask",
                            "width": width,
                            "height": height,
                            "bytes": destination.stat().st_size,
                            "sha256": sha256_file(destination),
                        }
                    )
                    + "\n"
                )
                target_paths[target_key] = relative
            row = {
                "sample_id": sample_id,
                "task": "referring",
                "split": split,
                "input": image_paths[image_id],
                "target": target_paths[target_key],
                "expression": expression,
                "prompt": build_binary_referring_prompt(expression),
                "encoding": BINARY_REFERRING_ENCODING,
                "source_dataset": source,
                "source_protocol": protocol,
                "source_split": source_split,
                "image_id": image_id,
                "ann_id": ann_id,
                "ref_id": ref_id,
                "sent_id": sent_id,
                "supervision": "human_polygon",
                "geometry_sha256": geometry_hash,
                "width": width,
                "height": height,
            }
            serialized = _json(row) + "\n"
            stream("pairs.jsonl").write(serialized)
            stream(f"{split}.jsonl").write(serialized)
            stream(f"{source}_{protocol}_{source_split}.jsonl").write(serialized)
            counts[split] += 1
            source_counts[f"{source_key}/{source_split}"] += 1
        for handle in handles.values():
            handle.close()
        connection.commit()
        report.update(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "encoding": BINARY_REFERRING_ENCODING,
                "path_policy": "manifest-relative",
                "image_mode": image_mode,
                "expression_policy": "raw verbatim, falling back to sent only when raw is absent",
                "split_policy": "official unc/umd; all sources val/test image IDs excluded from every train",
                "deduplication": "train only: exact image_id + bbox/segmentation JSON + expression; eval multiplicity unchanged",
                "max_train_samples": max_train_samples,
                "counts": dict(sorted(counts.items())),
                "source_counts": dict(sorted(source_counts.items())),
                "excluded_train_by_source": dict(sorted(excluded.items())),
                "duplicate_train_by_source": dict(sorted(duplicates.items())),
                "pilot_omitted_unique_train": pilot_omitted,
                "evaluation_blacklist_images": len(blocked),
                "unique_images": len(image_paths),
                "unique_targets": len(target_paths),
                "unique_train_images": len(train_images),
                "unique_eval_images": len(eval_images),
                "train_eval_image_overlap": len(train_images & eval_images),
            }
        )
        write_image_attribution(annotations_root, output_dir)
        report["training_view"] = write_training_view(output_dir)
        report["files"] = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(output_dir.iterdir())
            if path.suffix in (".jsonl", ".json")
        }
        report["dataset_fingerprint"] = hashlib.sha256(
            _json(report["files"]).encode("utf-8")
        ).hexdigest()
        _write_json(output_dir / "build_report.json", report)
        incomplete.unlink()
        return report
    finally:
        for handle in handles.values():
            handle.close()
        if cursor is not None:
            cursor.close()
        connection.close()
        if database.exists():
            database.unlink()


def validate_res_manifest(dataset_root: str | Path, *, write_report: bool = True) -> dict[str, Any]:
    """Fully verify copied images, binary masks, paths, provenance, and partitions."""
    root = Path(dataset_root)
    if (root / "BUILD_INCOMPLETE").exists():
        raise ValueError("Dataset contains BUILD_INCOMPLETE; rebuild before validation")
    build_report = json.loads((root / "build_report.json").read_text(encoding="utf-8"))
    if (
        build_report.get("schema_version") != SCHEMA_VERSION
        or build_report.get("status") != "complete"
    ):
        raise ValueError("Missing or unsupported completed RES build report")
    assets: dict[str, dict[str, Any]] = {}
    for number, line in enumerate((root / "assets.jsonl").open(encoding="utf-8"), 1):
        item = json.loads(line)
        value = item["path"]
        path = portable_path(root, value)
        if value in assets:
            raise ValueError(f"Duplicate asset path at inventory row {number}")
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Asset integrity mismatch: {value}")
        with Image.open(path) as image:
            image.load()
            if image.size != (item["width"], item["height"]):
                raise ValueError(f"Asset dimensions mismatch: {value}")
            if item["kind"] == "mask":
                if image.mode != "RGB" or image.format != "PNG":
                    raise ValueError(f"Target must be a lossless RGB PNG: {value}")
                pixels = np.asarray(image)
                if not np.all((pixels == 0) | (pixels == 255)) or not np.all(
                    pixels == pixels[..., :1]
                ):
                    raise ValueError(f"Target is not binary white/black: {value}")
                if not np.any(pixels):
                    raise ValueError(f"Empty target mask: {value}")
            elif item["kind"] != "image":
                raise ValueError(f"Unknown asset kind: {item['kind']}")
        assets[value] = item
    blocked = set(json.loads((root / "evaluation_image_ids.json").read_text(encoding="utf-8")))
    counts: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    seen_samples: set[str] = set()
    seen_train: set[str] = set()
    seen_assets: set[str] = set()
    train_images: set[int] = set()
    eval_images: set[int] = set()
    train_content: set[str] = set()
    eval_content: set[str] = set()
    for number, line in enumerate((root / "pairs.jsonl").open(encoding="utf-8"), 1):
        row = json.loads(line)
        if row.get("task") != "referring" or row.get("encoding") != BINARY_REFERRING_ENCODING:
            raise ValueError(f"Unsupported RES contract at row {number}")
        expression = row.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError(f"Missing RES expression at row {number}")
        if row.get("prompt") != build_binary_referring_prompt(expression):
            raise ValueError(f"Expression/prompt mismatch at row {number}")
        source, protocol, source_split = (
            row["source_dataset"],
            row["source_protocol"],
            row["source_split"],
        )
        if (source, protocol) not in SOURCE_SPECS or source_split not in SOURCE_SPLITS:
            raise ValueError(f"Unsupported source/protocol/split at row {number}")
        expected_split = source_split if source_split in ("train", "val") else "test"
        if row.get("split") != expected_split or row.get("supervision") != "human_polygon":
            raise ValueError(f"Invalid provenance at row {number}")
        for key in ("image_id", "ann_id", "ref_id", "sent_id"):
            _integer(row.get(key), key)
        if row.get("sample_id") != _sample_id(source, protocol, row["ref_id"], row["sent_id"]):
            raise ValueError(f"Invalid sample_id at row {number}")
        if row["sample_id"] in seen_samples:
            raise ValueError(f"Duplicate sample ID at row {number}")
        seen_samples.add(row["sample_id"])
        for field, kind in (("input", "image"), ("target", "mask")):
            value = row[field]
            # Inventory paths were resolved and checked above. Membership in
            # that inventory also rejects unvalidated paths, without repeating
            # filesystem resolution for every expression sharing an asset.
            if not isinstance(value, str) or value not in assets or assets[value]["kind"] != kind:
                portable_path(root, value)  # Preserve precise errors for rejected paths.
                raise ValueError(f"Missing or incorrectly typed asset at row {number}: {value}")
            if (row["width"], row["height"]) != (assets[value]["width"], assets[value]["height"]):
                raise ValueError(f"Image/mask dimensions mismatch at row {number}")
            seen_assets.add(value)
        image_id = row["image_id"]
        content_sha = assets[row["input"]]["sha256"]
        if expected_split == "train":
            if image_id in blocked:
                raise ValueError(f"Train/evaluation leakage at row {number}")
            identity = hashlib.sha256(
                _json([image_id, row["geometry_sha256"], expression]).encode("utf-8")
            ).hexdigest()
            if identity in seen_train:
                raise ValueError(f"Duplicate training expression/geometry at row {number}")
            seen_train.add(identity)
            train_images.add(image_id)
            train_content.add(content_sha)
        else:
            if image_id not in blocked:
                raise ValueError(f"Evaluation row outside blacklist at row {number}")
            eval_images.add(image_id)
            eval_content.add(content_sha)
        counts[expected_split] += 1
        sources[f"{source}/{protocol}/{source_split}"] += 1
    if train_images & eval_images or train_content & eval_content:
        raise ValueError("Train/evaluation image identity or exact file-content leakage")
    if seen_assets != set(assets):
        raise ValueError("Asset inventory contains unreferenced files")
    if dict(counts) != build_report["counts"] or dict(sources) != build_report["source_counts"]:
        raise ValueError("Manifest counts disagree with build report")
    raw_train = sum(
        source["expressions_by_split"].get("train", 0)
        for source in build_report["sources"].values()
    )
    accounted_train = (
        counts["train"]
        + sum(build_report["excluded_train_by_source"].values())
        + sum(build_report["duplicate_train_by_source"].values())
        + build_report["pilot_omitted_unique_train"]
    )
    if raw_train != accounted_train:
        raise ValueError("Training source counts do not reconcile with exclusion/deduplication")
    for source_key, source_report in build_report["sources"].items():
        for split, expected in source_report["expressions_by_split"].items():
            if split != "train" and sources[f"{source_key}/{split}"] != expected:
                raise ValueError(
                    f"Official evaluation multiplicity changed for {source_key}/{split}"
                )
    for value, metadata in build_report["files"].items():
        path = portable_path(root, value)
        if path.stat().st_size != metadata["bytes"] or sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"Manifest/report integrity mismatch: {value}")
    training_view = validate_training_view(root, build_report["training_view"])
    fingerprint = hashlib.sha256(_json(build_report["files"]).encode("utf-8")).hexdigest()
    if fingerprint != build_report["dataset_fingerprint"]:
        raise ValueError("Dataset fingerprint mismatch")
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "counts": dict(sorted(counts.items())),
        "source_counts": dict(sorted(sources.items())),
        "verified_assets": len(assets),
        "unique_train_images": len(train_images),
        "unique_eval_images": len(eval_images),
        "train_eval_image_overlap": 0,
        "train_eval_exact_content_overlap": 0,
        "official_eval_multiplicity_preserved": True,
        "all_assets_fully_decoded": True,
        "dataset_fingerprint": fingerprint,
        "training_view": training_view,
        "limitations": [
            "Exact content and COCO identity checked; perceptual near-duplicate detection is not performed."
        ],
    }
    if write_report:
        _write_json(root / "validation_report.json", report)
    return report
