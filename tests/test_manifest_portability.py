"""Regression tests for portable, fingerprinted dataset manifests."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from open_vision_banana.data.manifest import (
    fingerprint_records,
    relative_manifest_path,
    resolve_manifest_path,
)
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


def test_relative_manifest_path_round_trips_inside_and_outside_the_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = Path("prep") / "inst"
    (root / "targets").mkdir(parents=True)
    (Path("raw") / "coco").mkdir(parents=True)
    for path, expected in (
        (root / "targets" / "a.png", "targets/a.png"),
        (Path("raw") / "coco" / "b.jpg", "../../raw/coco/b.jpg"),
        ((tmp_path / "raw" / "coco" / "b.jpg"), "../../raw/coco/b.jpg"),
    ):
        path.touch()
        rel = relative_manifest_path(path, root)
        assert rel == expected
        assert resolve_manifest_path(rel, root).resolve() == path.resolve()


def test_relative_manifest_path_keeps_a_symlinked_tree_logical(tmp_path, monkeypatch):
    elsewhere = tmp_path / "mnt" / "coco"
    elsewhere.mkdir(parents=True)
    (elsewhere / "c.jpg").touch()
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "coco").symlink_to(elsewhere)
    (tmp_path / "prep" / "inst").mkdir(parents=True)
    rel = relative_manifest_path(tmp_path / "raw" / "coco" / "c.jpg", tmp_path / "prep" / "inst")
    assert rel == "../../raw/coco/c.jpg"  # not the physical ../../mnt/coco path
