"""Query text, legacy migration and ReasonSeg list regression coverage."""

import json

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from open_vision_banana.data.paper_eval_datasets import ReasonSegDataset, _reasonseg_queries
from open_vision_banana.data.res_contract import resolve_referring_prompt
from open_vision_banana.evaluation.evaluator import VisionEvaluator
from open_vision_banana.evaluation.shards import merged_metrics
from open_vision_banana.inference.prompt import PromptBuilder
from open_vision_banana.referring_contract import build_binary_referring_prompt
from open_vision_banana.training.trainer import ManifestVisionDataset
from scripts.evaluate_paper_benchmarks import preflight_jobs


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
def test_reasonseg_expands_queries_and_retains_first_mode(tmp_path, layout):
    Image.new("RGB", (8, 8)).save(tmp_path / "image.jpg")
    row = {
        "image": "image.jpg",
        "text": ["What is on the left?", "Find the left object.", "Find the left object."],
    }
    filename = "image.json" if layout == "sidecar" else "annotations.json"
    (tmp_path / filename).write_text(json.dumps(row if layout == "sidecar" else [row]))
    dataset = ReasonSegDataset(tmp_path, resolution=8)
    assert len(dataset) == 3
    for index, expression in enumerate(row["text"]):
        assert dataset[index]["expression"] == expression
        assert dataset[index]["prompt"] == build_binary_referring_prompt(expression)
        assert dataset[index]["query_index"] == index
        assert dataset[index]["image_id"] == "image.jpg"
        assert torch.equal(dataset[index]["gt_mask"], dataset[0]["gt_mask"])
    first = ReasonSegDataset(tmp_path, resolution=8, query_mode="first")
    assert len(first) == 1
    assert first[0]["expression"] == row["text"][0]
    jobs = [("referring", "reasonseg", tmp_path, "reasonseg")]
    for mode, count in (("all", 3), ("first", 1)):
        result = preflight_jobs(jobs, split="val", resolution=8, reasonseg_query_mode=mode)
        assert result["ok"]
        assert result["jobs"][0]["samples"] == count
    with pytest.raises(ValueError, match="query_mode"):
        ReasonSegDataset(tmp_path, query_mode="unknown")


def test_reasonseg_rewrite_precedence_and_bad_types():
    assert _reasonseg_queries({"rewritten_text": ["rewrite1", "rewrite2"], "text": ["raw"]}) == [
        "rewrite1",
        "rewrite2",
    ]
    assert _reasonseg_queries({"text": " raw "}) == ["raw"]
    assert _reasonseg_queries({"query": [" query1 ", "query2"]}) == ["query1", "query2"]
    for value in ([], [""], [1], {"query": "bad"}, 12):
        with pytest.raises(ValueError):
            _reasonseg_queries({"text": value})


def test_reasonseg_all_queries_are_scored_equally_including_across_shards(tmp_path):
    for name, queries in (("a", ["Question one?", "Question two?"]), ("b", ["Question three?"])):
        Image.new("RGB", (8, 8)).save(tmp_path / f"{name}.jpg")
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                {
                    "text": queries,
                    "shapes": [{"points": [[1, 1], [6, 1], [6, 6], [1, 6]]}],
                }
            )
        )
    dataset = ReasonSegDataset(tmp_path, resolution=8)

    class QueryModel:
        def __init__(self):
            self.queries = []

        def eval(self):
            return self

        def predict(self, **kwargs):
            query = kwargs["expression"]
            self.queries.append(query)
            rgb = dataset[0]["target_rgb"].unsqueeze(0)
            return {"rgb_output": torch.zeros_like(rgb) if query == "Question three?" else rgb}

    model = QueryModel()
    evaluator = VisionEvaluator(model, device="cpu")
    metrics = evaluator.evaluate(DataLoader(dataset, batch_size=1), task="referring")
    assert model.queries == ["Question one?", "Question two?", "Question three?"]
    assert metrics["samples"] == 3
    assert metrics["gIoU"] == pytest.approx(2 / 3)  # image-weighted would be 1/2
    assert metrics["IoU"] == metrics["gIoU"]
    assert metrics["cIoU"] == pytest.approx(2 / 3)
    states = []
    for indices in ([0], [1, 2]):  # the first image's queries cross shard boundaries
        evaluator.evaluate(DataLoader(Subset(dataset, indices), batch_size=1), task="referring")
        states.append(json.loads(json.dumps(evaluator.last_accumulator_states)))
    assert merged_metrics(states) == pytest.approx(metrics)


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
