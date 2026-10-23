"""Portable pair-manifest helpers used by dataset preparation and loading."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

_PATH_KEYS = frozenset({"input", "target", "image", "target_path", "image_path"})


def resolve_manifest_path(value: str | Path, manifest_path: str | Path) -> Path:
    """Resolve a manifest path relative to the manifest file when needed."""
    path = Path(value)
    if path.is_absolute():
        return path
    anchor = Path(manifest_path).resolve()
    base = anchor if anchor.is_dir() else anchor.parent
    return base / path


def relative_manifest_path(path: str | Path, root: str | Path) -> str:
    """``path`` relative to ``root`` (a builder's ``--data-root``) for a manifest row.

    Absolute or cwd-relative, inside ``root`` or outside it (``../`` steps). Symlinks are kept as
    the logical layout (no ``resolve()``), so a symlinked data tree does not write mount paths.
    Across drives, where no relative path exists, the absolute path is returned.
    """
    absolute = os.path.abspath(path)
    try:
        return Path(os.path.relpath(absolute, os.path.abspath(root))).as_posix()
    except ValueError:
        return Path(absolute).as_posix()


def _normalise_record(record: dict[str, Any], root: Path | None) -> dict[str, Any]:
    result = dict(record)
    if root is None:
        return result
    root = root.resolve()
    for key in _PATH_KEYS:
        value = result.get(key)
        if not isinstance(value, str):
            continue
        path = Path(value)
        if path.is_absolute():
            try:
                result[key] = path.resolve().relative_to(root).as_posix()
            except ValueError:
                result[key] = path.as_posix()
        else:
            result[key] = path.as_posix()
    return result


def fingerprint_records(records: list[dict[str, Any]], root: str | Path | None = None) -> str:
    """Return a deterministic SHA-256 over normalized manifest records."""
    base = Path(root) if root is not None else None
    canonical = [
        _normalise_record(record, base)
        for record in records
    ]
    canonical.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    payload = json.dumps(
        canonical,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL records in file order."""
    source = Path(path)
    records: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {source}:{line_no}")
            records.append(value)
    return records


__all__ = ["fingerprint_records", "read_jsonl", "relative_manifest_path", "resolve_manifest_path"]
