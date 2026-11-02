"""Fig. 4 (a)/(b) multi-expression referring rows composed from a binary RES manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from open_vision_banana.data.openvocab_semantic import NAMED_COLORS
from open_vision_banana.inference.prompt import PromptBuilder
from open_vision_banana.referring_contract import build_binary_referring_prompt
from open_vision_banana.training.trainer import ManifestVisionDataset
from scripts.build_multi_referring_manifest import (
    TASK_NAME,
    build_multi_referring_manifest,
    expression_from_binary_prompt,
)


def _write_res(tmp_path: Path) -> Path:
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    Image.new("RGB", (64, 48), (90, 90, 90)).save(tmp_path / "images" / "a.jpg")
    Image.new("RGB", (64, 48), (90, 90, 90)).save(tmp_path / "images" / "b.jpg")

    def mask(name: str, x0: int, x1: int) -> None:
        arr = np.zeros((48, 64, 3), np.uint8)
        arr[:, x0:x1] = 255
        Image.fromarray(arr).save(tmp_path / "masks" / name)

    mask("a_cat.png", 0, 20)
    mask("a_dog.png", 24, 44)
    mask("a_ball.png", 40, 64)  # overlaps the dog on 40..44
    mask("b_only.png", 0, 10)
    rows = [
        {
            "task": "referring",
            "split": "train",
            "input": "images/a.jpg",
            "target": "masks/a_cat.png",
            "expression": "the stretching cat",
            "prompt": build_binary_referring_prompt("the stretching cat"),
            "encoding": "binary_white_black_v1",
            "sample_id": "a1",
        },
        {
            "task": "referring",
            "split": "train",
            "input": "images/a.jpg",
            "target": "masks/a_dog.png",
            "prompt": build_binary_referring_prompt("the sleeping dog"),
            "sample_id": "a2",
        },  # no expression field
        {
            "task": "referring",
            "split": "train",
            "input": "images/a.jpg",
            "target": "masks/a_ball.png",
            "expression": "a red ball",
            "prompt": build_binary_referring_prompt("a red ball"),
            "sample_id": "a3",
        },
        {
            "task": "referring",
            "split": "train",
            "input": "images/b.jpg",
            "target": "masks/b_only.png",
            "expression": "the lamp",
            "prompt": build_binary_referring_prompt("the lamp"),
            "sample_id": "b1",
        },
        {"task": "depth", "split": "train", "input": "images/a.jpg", "target": "masks/a_cat.png"},
    ]
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return manifest


def test_expression_round_trips_through_the_canonical_prompt():
    assert expression_from_binary_prompt(build_binary_referring_prompt("the red car")) == "red car"
    assert expression_from_binary_prompt("Paint it white.") is None


def test_multi_rows_have_fig4_prompts_and_colour_targets_in_sync(tmp_path: Path):
    manifest = _write_res(tmp_path)
    out = tmp_path / "multi.jsonl"
    audit = build_multi_referring_manifest(
        manifest,
        out,
        tmp_path / "multi_targets",
        data_root=tmp_path,
        max_expressions=3,
        rows_per_image=4,
        expression_from_prompt=True,
        seed=1,
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert audit["images_with_multiple_expressions"] == 1  # only image a
    assert audit["skipped_single_expression_images"] == 1  # image b
    assert audit["rows_written"] == 4 and all(r["task"] == TASK_NAME for r in rows)

    for row in rows:
        k = len(row["expressions"])
        assert 2 <= k <= 3 and len(set(row["colors"])) == k
        assert row["prompt"] == PromptBuilder.build_multi_referring_prompt(
            list(zip(row["expressions"], row["colors"], strict=True)), row["template_idx"]
        )
        assert row["prompt"].startswith("A segmentation map image.")
        target = np.asarray(Image.open(tmp_path / row["target"]).convert("RGB"))
        present = {tuple(c) for c in target.reshape(-1, 3)}
        assert (0, 0, 0) in present  # black background
        assert present - {(0, 0, 0)} == {NAMED_COLORS[c] for c in row["colors"]}
        # the expression order in the prompt is the paint order: a later colour wins overlaps
        if "a red ball" in row["expressions"] and "sleeping dog" in row["expressions"]:
            ball_idx = row["expressions"].index("a red ball")
            dog_idx = row["expressions"].index("sleeping dog")
            winner = row["colors"][max(ball_idx, dog_idx)]
            assert tuple(target[10, 42]) == NAMED_COLORS[winner]

    # rows load through the trainer dataset with nearest-neighbour resizing (no blended colours)
    ds = ManifestVisionDataset(out, 32, "train", {TASK_NAME}, data_root=tmp_path)
    sample = ds[0]
    cols = {
        tuple((sample["target_rgb"][:, y, x] * 255).round().int().tolist())
        for y in range(32)
        for x in range(32)
    }
    assert cols <= {(0, 0, 0)} | set(NAMED_COLORS.values())


def test_multi_prompt_rejects_degenerate_inputs():
    with pytest.raises(ValueError, match="at least two"):
        PromptBuilder.build_multi_referring_prompt([("the cat", "green")])
    with pytest.raises(ValueError, match="distinct colour"):
        PromptBuilder.build_multi_referring_prompt([("the cat", "green"), ("the dog", "green")])
    assert PromptBuilder.build_multi_referring_prompt(
        [("the man in pink t shirt", "white"), ("the other man", "green")], 0
    ) == (
        "A segmentation map image. The area that corresponds to the man in pink t shirt is "
        "rendered solid white; the other man is rendered in green."
    )
