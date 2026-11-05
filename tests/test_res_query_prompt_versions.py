"""Query text, legacy migration and ReasonSeg list regression coverage."""

import json

import pytest
from PIL import Image

from open_vision_banana.data.paper_eval_datasets import ReasonSegDataset, _reasonseg_query
from open_vision_banana.data.res_contract import resolve_referring_prompt
from open_vision_banana.inference.prompt import PromptBuilder
from open_vision_banana.referring_contract import build_binary_referring_prompt
from open_vision_banana.training.trainer import ManifestVisionDataset


@pytest.mark.parametrize(
    "query", ["What animal is crossing the road?", "Find the red car.", "the left dog"]
)
def test_training_migrates_old_text_to_inference_query(tmp_path, query):
    Image.new("RGB", (8, 8)).save(tmp_path / "input.png")
    Image.new("RGB", (8, 8), "white").save(tmp_path / "target.png")
    row = dict(
        task="referring",
        input="input.png",
        target="target.png",
        split="train",
        expression=query,
        encoding="binary_white_black_v1",
        prompt=build_binary_referring_prompt(query, "fig4_v1"),
    )
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(row) + "\n")
    dataset = ManifestVisionDataset(path, 8, "train", {"referring"})
    expected = PromptBuilder.build_referring_prompt(query)
    assert dataset[0]["prompt"] == expected
    assert f"\nQuery: {query}\n" in expected
    assert json.loads(path.read_text())["prompt"] == row["prompt"]
    legacy = ManifestVisionDataset(
        path, 8, "train", {"referring"}, referring_prompt_version="fig4_v1"
    )
    assert legacy[0]["prompt"] == row["prompt"]


@pytest.mark.parametrize("layout", ["sidecar", "manifest"])
def test_reasonseg_selects_first_query_without_changing_item_count(tmp_path, layout):
    Image.new("RGB", (8, 8)).save(tmp_path / "image.jpg")
    row = {"image": "image.jpg", "text": ["What is on the left?", "Find the left object."]}
    filename = "image.json" if layout == "sidecar" else "annotations.json"
    (tmp_path / filename).write_text(json.dumps(row if layout == "sidecar" else [row]))
    dataset = ReasonSegDataset(tmp_path, resolution=8)
    assert len(dataset) == 1
    assert dataset[0]["expression"] == row["text"][0]
    assert dataset[0]["prompt"] == build_binary_referring_prompt(row["text"][0])


def test_reasonseg_rewrite_precedence_and_bad_types():
    assert _reasonseg_query({"rewritten_text": ["rewritten"], "text": ["raw"]}) == "rewritten"
    assert _reasonseg_query({"text": "raw"}) == "raw"
    for value in ([], [""], [1], {"query": "bad"}, 12):
        with pytest.raises(ValueError):
            _reasonseg_query({"text": value})


def test_declared_prompt_version_must_match_and_unknown_versions_fail():
    row = {
        "expression": "the dog",
        "encoding": "binary_white_black_v1",
        "prompt_version": "fig4_v1",
        "prompt": build_binary_referring_prompt("the dog"),
    }
    with pytest.raises(ValueError, match="declared"):
        resolve_referring_prompt(row)
    with pytest.raises(ValueError, match="Unknown"):
        build_binary_referring_prompt("dog", "unknown")
