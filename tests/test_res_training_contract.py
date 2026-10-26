"""RES rows retain text conditioning without changing other task loading."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from open_vision_banana.data.res_contract import BINARY_RES_ENCODING, resolve_referring_prompt
from open_vision_banana.training.trainer import (
    ManifestVisionDataset,
    _collate_fn,
    uniform_task_prompts,
)


def test_expression_only_builds_prompt_and_legacy_prompt_survives():
    assert "left dog" in resolve_referring_prompt({"expression": "the left dog"}).casefold()
    assert resolve_referring_prompt({"prompt": "Segment the dog."}) == "Segment the dog."


@pytest.mark.parametrize("expression", ["A red car", "An orange cat", "The left dog"])
@pytest.mark.parametrize("encoding", [None, BINARY_RES_ENCODING])
def test_canonical_article_normalization_keeps_valid_source_expressions(expression, encoding):
    from open_vision_banana.referring_contract import build_binary_referring_prompt

    prompt = build_binary_referring_prompt(expression)
    row = {"expression": expression, "prompt": prompt, "encoding": encoding}
    assert resolve_referring_prompt(row) == prompt
    with pytest.raises(ValueError):
        resolve_referring_prompt({**row, "expression": "A different object"})


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"expression": " "},
        {"expression": 1},
        {"prompt": 7},
        {"expression": "cat", "prompt": "Segment the dog."},
        {"encoding": BINARY_RES_ENCODING, "prompt": "Segment the dog."},
        {"expression": "dog", "encoding": "unknown"},
        {"expression": "dog", "encoding": BINARY_RES_ENCODING, "prompt": "Segment dog in red."},
    ],
)
def test_invalid_res_conditioning_fails(row):
    with pytest.raises(ValueError):
        resolve_referring_prompt(row)


def test_same_image_multiple_expressions_and_targets_survive_resize(tmp_path):
    Image.new("RGB", (9, 7), "red").save(tmp_path / "image.png")
    rows = []
    for index, expression in enumerate(("left dog", "right dog")):
        mask = np.zeros((7, 9, 3), dtype=np.uint8)
        mask[1:6, (1 if index == 0 else 6) : (3 if index == 0 else 8)] = 255
        Image.fromarray(mask).save(tmp_path / f"mask{index}.png")
        rows.append(
            {
                "task": "referring",
                "split": "train",
                "input": "image.png",
                "target": f"mask{index}.png",
                "expression": expression,
                "sample_id": str(index),
                "encoding": BINARY_RES_ENCODING,
            }
        )
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ds = ManifestVisionDataset(path, 13, "train", {"referring"})
    assert len(ds) == 2
    left, right = ds[0], ds[1]
    assert left["prompt"] != right["prompt"]
    assert not left["target_rgb"].equal(right["target_rgb"])
    assert set(left["target_rgb"].unique().tolist()) == {0.0, 1.0}
    batch = _collate_fn([left, right])
    assert batch["expression"] == ["left dog", "right dog"]
    assert batch["image"].shape == (2, 3, 13, 13)
    assert uniform_task_prompts(ds) == {}


def test_single_res_expression_never_becomes_checkpoint_global_prompt():
    ds = SimpleNamespace(
        samples=[
            {"task": "referring", "prompt": "Segment this dog"},
            {"task": "depth", "prompt": "depth prompt"},
        ]
    )
    assert uniform_task_prompts(ds) == {"depth": "depth prompt"}


def test_training_refuses_incomplete_res_restore(tmp_path):
    (tmp_path / "BUILD_INCOMPLETE").write_text("incomplete", encoding="utf-8")
    (tmp_path / "pairs.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        ManifestVisionDataset(tmp_path / "pairs.jsonl", 8, "train", {"referring"})


@pytest.mark.parametrize("task", ["depth", "normal", "semantic", "instance"])
def test_non_referring_collate_preserves_historical_fields(task):
    sample = {
        "image": torch.zeros(3, 4, 4),
        "target_rgb": torch.ones(3, 4, 4),
        "task": task,
        "prompt": "existing task prompt",
        "expression": "unrelated metadata",
        "sample_id": "example",
    }
    batch = _collate_fn([sample])
    assert set(batch) == {"image", "target_rgb", "task", "prompt"}
    assert batch["task"] == [task]
    assert batch["prompt"] == [sample["prompt"]]
    assert torch.equal(batch["image"][0], sample["image"])
    assert torch.equal(batch["target_rgb"][0], sample["target_rgb"])
