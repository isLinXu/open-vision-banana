#!/usr/bin/env python3
"""Audit v3 task declarations and paired files before multi-task training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SUPPORTED_TASKS = {
    "depth",
    "normal",
    "semantic",
    "generation",
    "edge",
}
TASK_DIRS = {
    "depth": "depth",
    "normal": "normal",
    "semantic": "segmentation",
    "generation": "generation",
    "edge": "structure",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
TEXT_EXTENSIONS = {".txt", ".json", ".jsonl"}


def manifest_checksum(payload: dict[str, Any]) -> str:
    """Hash canonical manifest JSON for reproducible audit metadata."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _names(payload: dict[str, Any], split: str, directory: str) -> list[str]:
    return [str(name) for name in payload.get(split, {}).get(directory, [])]


def audit_v3_dataset(
    data_dir: Path,
    *,
    tasks: list[str],
    require_nonempty: bool = True,
) -> dict[str, Any]:
    """Return task counts and fail closed on unsupported/incomplete data."""
    if not tasks:
        raise ValueError("at least one task is required")
    unsupported = sorted(set(tasks) - SUPPORTED_TASKS)
    if unsupported:
        raise ValueError(f"unsupported multi-task runner tasks: {unsupported}")
    manifest_path = data_dir / "train_val_split.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing split manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "train" not in payload or "val" not in payload:
        raise ValueError("split manifest must contain train and val mappings")
    report: dict[str, Any] = {
        "data_dir": str(data_dir.resolve()),
        "manifest": str(manifest_path.resolve()),
        "manifest_checksum": manifest_checksum(payload),
        "tasks": {},
    }
    for task in tasks:
        directory = TASK_DIRS[task]
        train_names = _names(payload, "train", directory)
        val_names = _names(payload, "val", directory)
        overlap = sorted(set(train_names) & set(val_names))
        task_dir = data_dir / directory
        image_names = {
            path.name
            for path in (task_dir / "images").iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        } if (task_dir / "images").is_dir() else set()
        target_extensions = TEXT_EXTENSIONS if task == "generation" else IMAGE_EXTENSIONS
        target_stems = {
            path.stem
            for path in (task_dir / "targets").iterdir()
            if path.is_file() and path.suffix.lower() in target_extensions
        } if (task_dir / "targets").is_dir() else set()
        missing_images = sorted(set(train_names + val_names) - image_names)
        missing_targets = sorted(
            name for name in train_names + val_names
            if Path(name).stem not in target_stems
        )
        if overlap:
            raise ValueError(f"{task}: train/val overlap: {overlap[:5]}")
        if require_nonempty and (not train_names or not val_names):
            raise ValueError(
                f"{task}: train/val declarations must both be non-empty "
                f"(train={len(train_names)} val={len(val_names)})"
            )
        if missing_images or missing_targets:
            raise ValueError(
                f"{task}: missing paired files images={missing_images[:5]} "
                f"targets={missing_targets[:5]}"
            )
        report["tasks"][task] = {
            "directory": directory,
            "train_count": len(train_names),
            "val_count": len(val_names),
            "train_names": train_names,
            "val_names": val_names,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(sorted(SUPPORTED_TASKS)))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    report = audit_v3_dataset(args.data_dir.expanduser().resolve(), tasks=tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": tasks, "manifest_checksum": report["manifest_checksum"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
