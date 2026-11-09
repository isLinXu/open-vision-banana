"""Native validity checks and immutable clean training manifest composition.

No RGB-color inference, input modifications, GPU work, or metadata-only trust.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
from PIL import Image

VERSION = "clean-depth-v1"
PATH_KEYS = (
    "input",
    "target",
    "metric_depth",
    "valid_mask",
    "image",
    "image_path",
    "target_path",
    "mask",
    "mask_path",
    "conditioning_image",
)
TRAIN_SPLITS = {"train", "training", "official_train", "raw_train_derived"}


def within_invalid_fraction(invalid_pixels, total_pixels, max_invalid_fraction=0.02):
    """Compare integers exactly; precisely 2% invalid is accepted by default."""
    limit = Fraction(str(max_invalid_fraction))
    if not 0 <= limit <= 1:
        raise ValueError("max_invalid_fraction must be between zero and one")
    if total_pixels <= 0 or not 0 <= invalid_pixels <= total_pixels:
        raise ValueError("Invalid pixel counts")
    return invalid_pixels * limit.denominator <= total_pixels * limit.numerator


def task_is_depth(row):
    return str(row.get("task", "")).lower() in {
        "depth",
        "metric_depth",
        "depth_estimation",
        "depth_edge",
    }


def _resolve(value, root):
    path = Path(value)
    return (path if path.is_absolute() else Path(root) / path).resolve()


def discover_metric_depth(row, root):
    """Recognize the legacy targets/X.png -> metric_depth/X.npy export layout."""
    if row.get("metric_depth"):
        return str(_resolve(row["metric_depth"], root))
    if row.get("target"):
        target = _resolve(row["target"], root)
        if target.parent.name == "targets":
            candidate = target.parent.parent / "metric_depth" / (target.stem + ".npy")
            if candidate.is_file():
                return str(candidate.resolve())
    return None


def _plane(path):
    try:
        if path.suffix.lower() == ".npy":
            data = np.load(path, allow_pickle=False)
        elif path.suffix.lower() == ".png":
            # Training modules allow truncated RGB images globally. Native
            # validity evidence must still pass PNG integrity verification.
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                data = np.array(image)
        else:
            raise ValueError("Expected NPY or scalar PNG")
        if data.ndim != 2 or not data.size or data.dtype.kind not in "bifu":
            raise ValueError("Expected nonempty numeric scalar HxW plane")
        return data
    except (OSError, ValueError, EOFError, SyntaxError) as exc:
        raise ValueError(f"Cannot read native validity source {path}: {exc}") from exc


def assess_depth_row(row, root, max_invalid_fraction=0.02):
    """Return (native bool mask, stats); fail on absent or unreadable sources.

    Intersect finite positive metric depth with finite positive native mask.
    RGB black is never used. All provided input/target images must be readable
    and have the same native geometry as validity evidence.
    """
    sources = {}
    metric = discover_metric_depth(row, root)
    if metric:
        sources["metric_depth"] = metric
    if row.get("valid_mask"):
        sources["valid_mask"] = str(_resolve(row["valid_mask"], root))
    if not sources:
        raise ValueError(
            "Depth sample requires metric_depth or valid_mask; RGB fallback is disabled"
        )
    valid = None
    for value in sources.values():
        plane = _plane(Path(value))
        current = np.isfinite(plane) & (plane > 0)
        if valid is not None and current.shape != valid.shape:
            raise ValueError("Native metric_depth and valid_mask shapes differ")
        valid = current if valid is None else valid & current
    for key in ("input", "target"):
        if row.get(key):
            path = _resolve(row[key], root)
            try:
                with Image.open(path) as image:
                    image.load()
                    if image.size != (valid.shape[1], valid.shape[0]):
                        raise ValueError(f"{key} image and native depth geometry differ: {path}")
            except OSError as exc:
                raise ValueError(f"Cannot read {key} image {path}: {exc}") from exc
    invalid = int(valid.size - np.count_nonzero(valid))
    return valid, dict(
        invalid_pixels=invalid,
        total_pixels=int(valid.size),
        invalid_fraction=invalid / valid.size,
        passes=within_invalid_fraction(invalid, int(valid.size), max_invalid_fraction),
        validity_sources=sources,
    )


LEGACY_DEPTH_EXEMPT = "legacy_baseline_exempt"
ADMISSION_RECORD_SUFFIX = ".depth-admission.json"


def is_legacy_depth_exempt(row):
    """True only for rows a pool composition explicitly admitted without native evidence."""
    return row.get("depth_admission") == LEGACY_DEPTH_EXEMPT


def validate_training_depth_row(row, root):
    """Read-only admission: reject dirty/unknown training depth, never filter it.

    Call before sample caps and before unavailable-pair skipping. No returned
    mask is applied to the original loader, so accepted sample processing stays
    exactly as it was, including resize, crops, interpolation and augmentation.
    Rows marked with the explicit legacy exemption skip only the native-depth
    assessment; the held-out split guard still applies to them.
    """
    source_split = str(row.get("source_split", "train")).strip().lower()
    if source_split not in TRAIN_SPLITS:
        raise ValueError(f"Depth admission rejects held-out source_split={source_split!r}")
    if is_legacy_depth_exempt(row):
        return
    _, stats = assess_depth_row(row, root)
    if not stats["passes"]:
        raise ValueError(
            f"Depth admission rejected {row.get('input')}: "
            f"{stats['invalid_pixels']}/{stats['total_pixels']} invalid native pixels "
            "exceeds 2%. Select an offline-cleaned manifest; no samples were filtered."
        )


def admission_record_path(manifest_path):
    path = Path(manifest_path)
    return path.with_name(path.name + ADMISSION_RECORD_SUFFIX)


def _row_error(item, root):
    line_no, row = item
    try:
        validate_training_depth_row(row, root)
    except ValueError as exc:
        return f"line {line_no}: {exc}"
    return None


def write_admission_record(manifest_path, root, workers=8, chunk_size=500):
    """Admit every train depth row once, in parallel, bound to the exact manifest bytes.

    Every rank of every training job otherwise decodes the whole depth pool again in
    its loader. A loader that finds a matching record skips that per-row decode; any
    other manifest is still checked row by row. Nothing is written if a row fails.
    """
    manifest_path, root = Path(manifest_path).resolve(), Path(root).resolve()
    digest = file_sha256(manifest_path)
    rows = []
    with manifest_path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if row.get("split") == "train" and task_is_depth(row):
                    rows.append((line_no, row))
    with ProcessPoolExecutor(max(1, workers)) as pool:
        check = partial(_row_error, root=root)
        errors = [e for e in pool.map(check, rows, chunksize=chunk_size) if e]
    if errors:
        raise ValueError(f"{len(errors)} depth rows rejected, no record written: {errors[:5]}")
    if file_sha256(manifest_path) != digest:
        raise ValueError("Manifest changed during admission, no record written")
    exempt = sum(is_legacy_depth_exempt(row) for _, row in rows)
    record = {
        "version": VERSION,
        "manifest": manifest_path.name,
        "manifest_sha256": digest,
        "root": str(root),
        "checked_rows": len(rows) - exempt,
        "legacy_exempt_rows": exempt,
    }
    admission_record_path(manifest_path).write_text(_json(record), encoding="utf-8")
    return record


@lru_cache(maxsize=8)
def _manifest_sha256(path, size, mtime_ns):
    # One hash per process and manifest state: a trainer builds one dataset per task.
    return file_sha256(path)


def has_admission_record(manifest_path, root):
    """True when a record binds exactly these manifest bytes, this root and this version."""
    try:
        record = json.loads(admission_record_path(manifest_path).read_text(encoding="utf-8"))
        stat = Path(manifest_path).stat()
    except (OSError, ValueError):
        return False
    return (
        isinstance(record, dict)
        and record.get("version") == VERSION
        and record.get("root") == str(Path(root).resolve())
        and record.get("manifest_sha256")
        == _manifest_sha256(str(Path(manifest_path).resolve()), stat.st_size, stat.st_mtime_ns)
    )


class DepthAdmission:
    """Training depth admission for one manifest, shared by the manifest loaders.

    A matching admission record replaces the per-row decode; the record lookup (a full
    manifest hash) happens only once a train depth row actually needs admission.
    """

    def __init__(self, manifest_path, root):
        self.manifest_path = manifest_path
        self.root = root
        self.legacy_exempt = 0
        self._recorded = None

    def check(self, row, line_no):
        if is_legacy_depth_exempt(row):
            self.legacy_exempt += 1
        if self._recorded is None:
            self._recorded = has_admission_record(self.manifest_path, self.root)
        if self._recorded:
            return
        try:
            validate_training_depth_row(row, self.root)
        except ValueError as exc:
            raise ValueError(f"{self.manifest_path}:{line_no}: {exc}") from exc

    def log_summary(self, logger):
        if self._recorded:
            record = admission_record_path(self.manifest_path)
            logger.info("%s: depth admission taken from %s", self.manifest_path, record)
        if self.legacy_exempt:
            logger.info(
                "%s: %d legacy baseline depth rows admitted without native depth",
                self.manifest_path,
                self.legacy_exempt,
            )


def require_depth_manifest(tasks, route):
    """Reject unsupported legacy depth input rather than silently bypass admission."""
    if any(task_is_depth({"task": task}) for task in tasks):
        raise ValueError(
            f"Depth admission cannot verify {route}. Use a clean --pair-manifest "
            "with native metric_depth or valid_mask evidence."
        )


def validate_training_dataset(dataset, _seen=None):
    """Read-only guard for known raw/V3/tar adapters at the legacy tuner entry.

    Constructors used for evaluation remain unchanged. Supported manifest
    loaders certify admission only after inspecting their full training pool.
    """
    seen = set() if _seen is None else _seen
    if id(dataset) in seen:
        return
    seen.add(id(dataset))
    children = []
    for name in ("datasets", "vision_datasets", "task_datasets"):
        collection = getattr(dataset, name, ()) or ()
        children.extend(collection.values() if isinstance(collection, Mapping) else collection)
    for name in ("dataset", "generation_dataset"):
        child = getattr(dataset, name, None)
        if child is not None:
            children.append(child)
    for child in children:
        validate_training_dataset(child, seen)
    if children:
        return
    tasks = {getattr(dataset, "task", None), getattr(dataset, "TASK_TYPE", None)}
    tasks.update(getattr(dataset, "tasks", ()) or ())
    for row in getattr(dataset, "samples", ()):
        if isinstance(row, dict):
            tasks.add(row.get("task", row.get("mapped_task")))
    if not getattr(dataset, "_depth_admission_checked", False):
        require_depth_manifest(tasks, type(dataset).__name__)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def _split(row):
    labels = [str(row[key]).lower() for key in ("split", "source_split") if row.get(key)]
    return "train" if not labels or all(s in TRAIN_SPLITS for s in labels) else "heldout"


def build_clean_manifest(
    manifests, output_dir, *, roots=None, max_invalid_fraction=0.02, expected_depth_rejections=None
):
    """Build a new completed directory, never overwrite or modify source data.

    Exact same-task duplicates are excluded. Conflicts and train/heldout overlap
    fail. Unlabelled legacy rows are assumed train and counted in the audit.
    A build is published via atomic rename from an exclusively reserved sibling
    staging directory, with checksums and COMPLETE.json. Existing outputs fail.
    """
    within_invalid_fraction(0, 1, max_invalid_fraction)
    manifests = [Path(p).resolve() for p in manifests]
    if not manifests:
        raise ValueError("At least one source manifest is required")
    roots = [Path(r).resolve() for r in roots] if roots else [p.parent for p in manifests]
    if len(roots) != len(manifests):
        raise ValueError("Provide one root per manifest")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Output must be a new directory: {output}")
    sources, records = [], []
    for manifest, root in zip(manifests, roots, strict=True):
        sources.append(dict(path=str(manifest), root=str(root), sha256=file_sha256(manifest)))
        with manifest.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected object: {manifest}:{line_no}")
                row = dict(row)
                for key in PATH_KEYS:
                    if row.get(key):
                        row[key] = str(_resolve(row[key], root))
                if task_is_depth(row):
                    metric = discover_metric_depth(row, root)
                    if metric:
                        row["metric_depth"] = metric
                records.append(
                    (
                        row,
                        root,
                        dict(
                            manifest=str(manifest),
                            line=line_no,
                            sample_id=row.get("sample_id", row.get("id")),
                            source_dataset=row.get("source_dataset"),
                        ),
                    )
                )
    seen, splits, kept, excluded, audits = {}, {}, [], [], []
    counts, asset_hashes = Counter(), {}
    for row, root, origin in records:
        split = _split(row)
        task = str(row.get("task", "")).lower()
        identities = []
        if row.get("source_dataset") and row.get("sample_id"):
            identities.append(("source", str(row["source_dataset"]).lower(), str(row["sample_id"])))
        if row.get("input"):
            identities.append(("input", os.path.normcase(row["input"])))
        if not identities:
            raise ValueError(f"No reliable source identity or input asset: {origin}")
        for identity in identities:
            if identity in splits and splits[identity] != split:
                raise ValueError(f"Train/heldout duplicate: {origin}")
            splits[identity] = split
        fingerprint = _json(row)
        duplicate = False
        for identity in identities:
            key = (*identity, task)
            if key in seen:
                if seen[key] != fingerprint:
                    raise ValueError(f"Conflicting duplicate: {origin}")
                duplicate = True
            seen[key] = fingerprint
        counts["input_rows"] += 1
        if duplicate or split != "train":
            reason = "duplicate" if duplicate else "non_train_split"
            excluded.append(dict(**origin, reason=reason, row=row))
            counts[reason] += 1
            continue
        if not row.get("split") and not row.get("source_split"):
            counts["assumed_train_rows"] += 1
        stats = None
        if task_is_depth(row):
            _, stats = assess_depth_row(row, root, max_invalid_fraction)
            counts["depth_rows_scanned"] += 1
        hashes = {}
        for key in PATH_KEYS:
            if row.get(key):
                path = row[key]
                if path not in asset_hashes:
                    asset_hashes[path] = file_sha256(path)
                hashes[key] = asset_hashes[path]
        audit = dict(
            **origin,
            asset_sha256=hashes,
            original_split_labels={
                key: row[key] for key in ("split", "source_split") if key in row
            },
        )
        if stats is not None:
            audit["quality"] = stats
        if stats is not None and not stats["passes"]:
            excluded.append(dict(**audit, reason="invalid_depth_fraction", row=row))
            counts["depth_rejected"] += 1
        else:
            kept.append(dict(row, split="train"))
            audits.append(audit)
            counts["kept_rows"] += 1
    actual = counts["depth_rejected"]
    if expected_depth_rejections is not None and actual != expected_depth_rejections:
        raise ValueError(
            f"Expected {expected_depth_rejections} depth rejections, measured {actual}"
        )
    for source in sources:
        if file_sha256(source["path"]) != source["sha256"]:
            raise ValueError(f"Source manifest changed while scanning: {source['path']}")
    payloads = {
        "pairs.jsonl": "".join(map(_json, kept)),
        "excluded.jsonl": "".join(map(_json, excluded)),
        "admitted_audit.jsonl": "".join(map(_json, audits)),
    }
    report = dict(
        version=VERSION,
        max_invalid_fraction=str(max_invalid_fraction),
        counts=dict(counts),
        sources=sources,
        missing_split_policy="assume train; explicit nontrain always excluded; admitted split normalized to train",
        rgb_fallback=False,
        assets_sha256=asset_hashes,
        output_sha256={
            name: hashlib.sha256(data.encode()).hexdigest() for name, data in payloads.items()
        },
    )
    payloads["report.json"] = _json(report)
    staging = output.with_name(output.name + ".building")
    staging.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        with (staging / name).open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    with (staging / "COMPLETE.json").open("x", encoding="utf-8") as handle:
        handle.write(_json({name: file_sha256(staging / name) for name in payloads}))
        handle.flush()
        os.fsync(handle.fileno())
    if output.exists():
        raise FileExistsError(f"Output appeared while building: {output}")
    staging.rename(output)
    return report
