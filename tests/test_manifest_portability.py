"""Regression tests for portable, fingerprinted dataset manifests."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from open_vision_banana.data.manifest import fingerprint_records, resolve_manifest_path
from scripts.archive.pre_opensource.normalize_dataset_manifests import normalize_dataset


def test_resolve_manifest_path_uses_manifest_directory(tmp_path: Path) -> None:
    manifest = tmp_path / "pairs.jsonl"
    assert resolve_manifest_path("images/sample.png", manifest) == tmp_path / "images/sample.png"
    assert resolve_manifest_path("images/sample.png", tmp_path) == tmp_path / "images/sample.png"


def test_fingerprint_is_independent_of_record_order(tmp_path: Path) -> None:
    records = [
        {"input": "images/a.png", "target": "targets/a.png", "task": "depth"},
        {"input": "images/b.png", "target": "targets/b.png", "task": "depth"},
    ]
    assert fingerprint_records(records, tmp_path) == fingerprint_records(records[::-1], tmp_path)


def test_normalize_dataset_rewrites_absolute_paths_and_adds_fingerprint(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    image = dataset / "images" / "sample.png"
    target = dataset / "targets" / "sample.png"
    image.parent.mkdir(parents=True)
    target.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(image)
    Image.new("RGB", (2, 2)).save(target)
    (dataset / "pairs.jsonl").write_text(
        json.dumps(
            {
                "task": "metric_depth",
                "input": str(image),
                "target": str(target),
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset / "manifest.json").write_text(
        json.dumps({"output": str(dataset), "pairs_written": 99}),
        encoding="utf-8",
    )

    manifest = normalize_dataset(dataset)
    row = json.loads((dataset / "pairs.jsonl").read_text(encoding="utf-8"))
    assert row["input"] == "images/sample.png"
    assert row["target"] == "targets/sample.png"
    assert manifest["output"] == "."
    assert manifest["pairs_written"] == 1
    assert len(manifest["dataset_fingerprint"]) == 64
    assert manifest["path_policy"] == "manifest-relative"
