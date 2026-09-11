#!/usr/bin/env python3
"""Normalize prepared dataset manifests to portable, fingerprinted records."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.data.datasets import CityscapesSegDataset  # noqa: E402
from open_vision_banana.data.manifest import (  # noqa: E402
    fingerprint_records,
    resolve_manifest_path,
)
from open_vision_banana.inference.prompt import PromptBuilder  # noqa: E402

DEFAULT_DATASETS = (
    "hypersim_depth_512",
    "cityscapes_19class_512",
    "kitti_metric_512",
    "metric_mixed_replay_512",
    "sana_paper_gate_512",
    "flux_paper_gate_512",
)

_MANIFEST_PATH_KEYS = frozenset(
    {
        "output",
        "train_output",
        "paper_root",
        "generation",
        "depth_source",
        "segmentation_source",
        "generation_source",
        "source_depth",
        "source_rgb",
        "source",
        "pair_manifest",
        "baseline",
    }
)


def _portable_path(value: str, dataset_root: Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        if path == Path("."):
            return "."
        if (dataset_root / path).exists():
            return path.as_posix()
        project_candidate = PROJECT_ROOT / path
        if project_candidate.exists():
            return Path(os.path.relpath(project_candidate, dataset_root)).as_posix()
        return path.as_posix()
    if path.resolve() == dataset_root.resolve():
        return "."
    try:
        path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return path.name
    return Path(os.path.relpath(path, dataset_root)).as_posix()


def _normalize_record(record: dict[str, Any], pairs_path: Path) -> dict[str, Any]:
    result = dict(record)
    for key in ("input", "target", "image"):
        value = result.get(key)
        if not isinstance(value, str):
            continue
        resolved = resolve_manifest_path(value, pairs_path)
        if not resolved.is_file():
            raise FileNotFoundError(f"Missing manifest file for {key}: {resolved}")
        result[key] = Path(os.path.relpath(resolved, pairs_path.parent)).as_posix()
    if result.get("task") == "cityscapes":
        result["prompt"] = PromptBuilder.build_semantic_prompt(
            CityscapesSegDataset.CITYSCAPES_COLORS
        )
    return result


def _layout_records(dataset_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in ("depth", "segmentation"):
        image_dir = dataset_root / task / "images"
        target_dir = dataset_root / task / "targets"
        if not image_dir.is_dir():
            continue
        for image in sorted(path for path in image_dir.iterdir() if path.is_file()):
            target = target_dir / image.name
            if target.is_file():
                records.append(
                    {
                        "task": task,
                        "input": image.relative_to(dataset_root).as_posix(),
                        "target": target.relative_to(dataset_root).as_posix(),
                    }
                )
    prompts = dataset_root / "generation" / "prompts.jsonl"
    if prompts.is_file():
        for line in prompts.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            records.append(
                {
                    "task": "generation",
                    "image": f"generation/images/{row['image']}",
                    "prompt": row["prompt"],
                }
            )
    return records


def normalize_dataset(dataset_root: Path) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    pairs_path = dataset_root / "pairs.jsonl"
    if pairs_path.is_file():
        records = [
            _normalize_record(json.loads(line), pairs_path)
            for line in pairs_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        pairs_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
    else:
        records = _layout_records(dataset_root)
    if not records:
        raise ValueError(f"No manifest records found under {dataset_root}")

    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    for key in _MANIFEST_PATH_KEYS:
        value = manifest.get(key)
        if isinstance(value, str):
            manifest[key] = "." if key == "output" else _portable_path(value, dataset_root)
    manifest["schema_version"] = max(int(manifest.get("schema_version", 1)), 2)
    if pairs_path.is_file():
        manifest["pair_manifest"] = "pairs.jsonl"
    # Layout-based gates (depth/segmentation symlink trees plus generation
    # captions) do not have one pairs.jsonl, but must expose the same audited
    # record count in their manifest.
    manifest["pairs_written"] = len(records)
    manifest["dataset_fingerprint"] = fingerprint_records(records, dataset_root)
    manifest["fingerprint_algorithm"] = "sha256:canonical-pairs-v1"
    manifest["path_policy"] = "manifest-relative"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = args.datasets or [PROJECT_ROOT / "data" / name for name in DEFAULT_DATASETS]
    for root in roots:
        manifest = normalize_dataset(root)
        print(f"{root}: {manifest['dataset_fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
