"""Checkpoint validation must never observe another source's official test image."""

from __future__ import annotations

import hashlib
import json

import pytest

from open_vision_banana.res_training_view import validate_training_view, write_training_view


def fixture_manifests(root):
    root.mkdir()
    # Image 4 is a byte-identical alias of test image 3 despite a different COCO ID.
    hashes = {1: "a" * 64, 2: "b" * 64, 3: "c" * 64, 4: "c" * 64, 5: "d" * 64}
    assets = [
        {"path": f"images/{number}.jpg", "kind": "image", "sha256": digest}
        for number, digest in hashes.items()
    ]
    (root / "assets.jsonl").write_text("\n".join(json.dumps(item) for item in assets) + "\n")
    records = [
        (1, "train", "refcoco"),
        (2, "val", "refcoco"),
        (3, "val", "refcoco+"),
        (4, "val", "refcocog"),
        (5, "val", "refcocog"),
        (2, "test", "refcocog"),
        (3, "test", "refcoco"),
    ]
    lines = []
    for index, (number, split, source) in enumerate(records):
        row = {
            "image_id": number,
            "input": f"images/{number}.jpg",
            "split": split,
            "source_dataset": source,
            "source_protocol": "umd" if source == "refcocog" else "unc",
            "source_split": split,
            "expression": f"target {index}",
        }
        # Preserve intentionally noncanonical whitespace and CRLF in selected source rows.
        lines.append(("  " + json.dumps(row) + " \r\n").encode("utf-8"))
    (root / "pairs.jsonl").write_bytes(b"".join(lines))
    return lines


def test_global_test_id_and_exact_content_exclusion_preserves_rows(tmp_path):
    root = tmp_path / "data"
    lines = fixture_manifests(root)
    report = write_training_view(root)
    expected = lines[0] + lines[4]
    assert (root / "training.jsonl").read_bytes() == expected
    assert (root / "pairs.jsonl").read_bytes() == b"".join(lines)
    assert report["counts"] == {"train": 1, "val": 1}
    assert report["excluded_val"] == 3
    assert report["excluded_val_by_image_id"] == 2
    assert report["excluded_val_by_exact_content"] == 3
    assert report["sha256"] == hashlib.sha256(expected).hexdigest()
    assert validate_training_view(root, report) == report
    assert write_training_view(root) == report


def test_validation_rejects_changed_output_even_with_same_row_counts(tmp_path):
    root = tmp_path / "data"
    fixture_manifests(root)
    report = write_training_view(root)
    path = root / "training.jsonl"
    path.write_bytes(path.read_bytes().replace(b"target 0", b"target 9"))
    with pytest.raises(ValueError, match="exact expected"):
        validate_training_view(root, report)


def test_validation_rederives_source_view_and_rejects_stale_metadata(tmp_path):
    root = tmp_path / "data"
    fixture_manifests(root)
    report = write_training_view(root)
    path = root / "pairs.jsonl"
    path.write_bytes(path.read_bytes().replace(b"target 4", b"target 9"))
    with pytest.raises(ValueError, match="metadata"):
        validate_training_view(root, report)


def test_training_test_leak_is_rejected_without_partial_output(tmp_path):
    root = tmp_path / "data"
    fixture_manifests(root)
    path = root / "pairs.jsonl"
    path.write_bytes(path.read_bytes().replace(b'"split": "val"', b'"split": "train"', 1))
    with pytest.raises(ValueError, match="Training source already overlaps"):
        write_training_view(root)
    assert not (root / "training.jsonl").exists()
    assert not (root / "training.jsonl.tmp").exists()


def test_missing_image_hash_fails(tmp_path):
    root = tmp_path / "data"
    fixture_manifests(root)
    (root / "assets.jsonl").write_text("")
    with pytest.raises(ValueError, match="unindexed image"):
        write_training_view(root)
