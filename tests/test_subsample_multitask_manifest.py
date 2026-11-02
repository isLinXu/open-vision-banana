"""E2b samples only E1 vision-training rows and keeps its evaluation contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.subsample_multitask_manifest import VISION_TASKS, subsample_manifest


def _source(tmp_path: Path) -> tuple[Path, list[dict]]:
    root = tmp_path / "e1"
    root.mkdir()
    rows = []
    for task in sorted(VISION_TASKS | {"generation"}):
        for index in range(8):
            rows.append(
                {
                    "sample_id": f"{task}-train-{index}",
                    "task": task,
                    "split": "train",
                    "input": f"assets/{task}-{index}.png",
                    "target": f"assets/{task}-{index}-target.png",
                    "prompt": f"{task} prompt {index}",
                }
            )
        rows.append(
            {
                "sample_id": f"{task}-val",
                "task": task,
                "split": "val",
                "input": f"assets/{task}-val.png",
                "target": f"assets/{task}-val-target.png",
                "prompt": f"{task} validation",
            }
        )
    rows.append(
        {
            "sample_id": "referring-test",
            "task": "referring",
            "split": "test",
            "input": "assets/referring-test.png",
            "target": "assets/referring-test-target.png",
        }
    )
    for row in rows:
        if row["sample_id"] in {"generation-train-0", "depth-val"}:
            row["input"] = "assets/COCO_train2014_000000000123.jpg"
    source = root / "pairs.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    source.with_suffix(".audit.json").write_text(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "protocol": "strict",
                "zero_shot_claim_allowed": True,
                "evaluation_image_id_overlap_by_source": {"refcocog_umd": 0},
                "rejected_non_metric_depth_rows": 0,
            }
        ),
        encoding="utf-8",
    )
    return source, rows


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_only_vision_training_rows_are_quartered(tmp_path: Path) -> None:
    source, rows = _source(tmp_path)
    output = source.parent / "pairs_e2b.jsonl"

    audit = subsample_manifest(source, output)
    result = _rows(output)
    for task in VISION_TASKS:
        assert audit["task_split_counts_before"][f"{task}/train"] == 8
        assert audit["task_split_counts"][f"{task}/train"] == 2
    assert [row for row in result if row["task"] == "generation"] == [
        row for row in rows if row["task"] == "generation"
    ]
    assert [row for row in result if row["split"] != "train"] == [
        row for row in rows if row["split"] != "train"
    ]
    assert audit["source_manifest_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert audit["manifest_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert audit["train_val_coco_image_id_overlap"] == 1
    assert subsample_manifest(source, output)["manifest_sha256"] == audit["manifest_sha256"]


def test_rebased_assets_and_source_audit_are_checked(tmp_path: Path) -> None:
    source, _ = _source(tmp_path)
    output = tmp_path / "e2b" / "pairs.jsonl"
    subsample_manifest(source, output)
    original = next(row for row in _rows(source) if row["task"] == "generation")
    rebased = next(row for row in _rows(output) if row["sample_id"] == original["sample_id"])
    for key in ("input", "target"):
        assert (output.parent / rebased[key]).resolve() == (source.parent / original[key]).resolve()

    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="audit SHA256"):
        subsample_manifest(source, tmp_path / "other" / "pairs.jsonl")
