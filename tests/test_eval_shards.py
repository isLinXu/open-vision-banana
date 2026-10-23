"""Benchmark shards: contiguous seeded blocks whose accumulator sums merge into the unsharded result."""

from __future__ import annotations

import copy
import json
import sys

import pytest
import torch

from open_vision_banana.evaluation.benchmark import (
    DepthAccumulator,
    SemanticAccumulator,
    depth_report,
)
from open_vision_banana.evaluation.evaluator import VisionEvaluator
from open_vision_banana.evaluation.referring import ReferringAccumulator
from open_vision_banana.evaluation.shards import (
    accumulator_state,
    merge_accumulator_states,
    merged_metrics,
    shard_range,
)
from scripts.evaluate_paper_benchmarks import main as eval_main
from scripts.merge_eval_shards import main as merge_main
from scripts.merge_eval_shards import merge_shards


@pytest.mark.parametrize(
    ("total", "shards", "batch"), [(10, 3, 1), (4896, 8, 1), (7, 4, 2), (2, 5, 1), (0, 2, 1)]
)
def test_shard_ranges_tile_the_set_on_batch_boundaries(total, shards, batch):
    blocks = [shard_range(total, shards, index, batch) for index in range(shards)]
    assert [i for block in blocks for i in block] == list(range(total))
    assert all(block.start % batch == 0 for block in blocks)
    with pytest.raises(ValueError):
        shard_range(total, shards, shards)


def _halves(make, update, samples):
    whole, first, second = make(), make(), make()
    for index, sample in enumerate(samples):
        update(whole, *sample)
        update(first if index < len(samples) // 2 else second, *sample)
    states = [
        json.loads(json.dumps(accumulator_state(acc))) for acc in (first, second)
    ]  # through JSON
    return whole, merge_accumulator_states(states)


def test_depth_semantic_and_referring_merges_equal_one_pass():
    gen = torch.Generator().manual_seed(0)
    depth = [
        (
            torch.rand(1, 1, 6, 6, generator=gen) * 5 + 0.1,
            torch.rand(1, 1, 6, 6, generator=gen) * 5 + 0.1,
        )
        for _ in range(5)
    ]
    whole, merged = _halves(DepthAccumulator, lambda acc, p, g: acc.update(p, g), depth)
    assert merged.compute() == pytest.approx(whole.compute())

    labels = [
        (
            torch.randint(0, 4, (1, 6, 6), generator=gen),
            torch.randint(0, 4, (1, 6, 6), generator=gen),
        )
        for _ in range(5)
    ]
    whole, merged = _halves(
        lambda: SemanticAccumulator(4), lambda acc, p, g: acc.update(p, g), labels
    )
    assert torch.equal(merged.confusion, whole.confusion)
    assert merged.compute()["mIoU"] == pytest.approx(whole.compute()["mIoU"])

    masks = [
        (torch.rand(1, 6, 6, generator=gen) > 0.5, torch.rand(1, 6, 6, generator=gen) > 0.5)
        for _ in range(5)
    ]
    whole, merged = _halves(ReferringAccumulator, lambda acc, p, g: acc.update(p, g), masks)
    assert merged.compute() == pytest.approx(whole.compute())


def test_merge_refuses_mismatched_accumulators():
    with pytest.raises(ValueError, match="kind"):
        merge_accumulator_states(
            [accumulator_state(DepthAccumulator()), accumulator_state(ReferringAccumulator())]
        )
    with pytest.raises(ValueError, match="num_classes"):
        merge_accumulator_states(
            [accumulator_state(SemanticAccumulator(3)), accumulator_state(SemanticAccumulator(4))]
        )
    with pytest.raises(TypeError):
        accumulator_state(object())


class _QueueModel:
    """predict() returns the next queued RGB output."""

    def __init__(self, targets):
        self.targets = targets

    def eval(self):
        return self

    def predict(self, **kwargs):
        return {"rgb_output": self.targets.pop(0)}


def test_evaluator_exposes_states_that_reproduce_its_metrics():
    gen = torch.Generator().manual_seed(1)
    batches = [
        {
            "image": torch.rand(1, 3, 8, 8, generator=gen),
            "target_rgb": torch.rand(1, 3, 8, 8, generator=gen),
        }
        for _ in range(3)
    ]
    ev = VisionEvaluator(
        _QueueModel([torch.rand(1, 3, 8, 8, generator=gen) for _ in batches]), device="cpu"
    )
    metrics = ev.evaluate(batches, task="depth")
    assert set(ev.last_accumulator_states) == {
        "main",
        "relative",
        "skipped_samples",
        "depth_curve",
        "depth_units",
    }
    rebuilt = merged_metrics([json.loads(json.dumps(ev.last_accumulator_states))])
    assert set(rebuilt) == set(metrics)
    for key, value in metrics.items():
        assert rebuilt[key] == (value if isinstance(value, str) else pytest.approx(value)), key
    # an empty (or fully skipped) shard still reports its zero counts, so the merge can add them
    assert ev.evaluate(batches[:0], task="depth") == {}
    assert ev.last_accumulator_states["main"]["valid_pixels"] == 0


def test_non_finite_sums_survive_the_json_round_trip():
    from open_vision_banana.evaluation.results_io import json_safe

    acc = DepthAccumulator()
    acc.update(torch.full((1, 1, 2, 2), 2.0), torch.full((1, 1, 2, 2), 1.0))
    acc.sq_rel_sum = float("inf")
    state = json.loads(json.dumps(json_safe(accumulator_state(acc))))  # what the eval JSON holds
    assert state["sq_rel_sum"] == "inf"
    merged = merge_accumulator_states([state, accumulator_state(DepthAccumulator())])
    assert merged.sq_rel_sum == float("inf") and merged.valid_pixels == 4


def test_depth_report_matches_the_relative_reading():
    acc, rel = DepthAccumulator(), DepthAccumulator()
    acc.update(torch.full((1, 1, 2, 2), 2.0), torch.full((1, 1, 2, 2), 1.0))
    rel.update(torch.full((1, 1, 2, 2), 1.0), torch.full((1, 1, 2, 2), 1.0))
    report = depth_report(
        acc, rel, skipped_samples=1, depth_curve="metric_power", depth_units="metres"
    )
    assert report["AbsRel"] == pytest.approx(1.0) and report["relative_delta_1.25"] == 1.0
    assert report["skipped_samples"] == 1


def _shard_result(index, count, start, stop, total, state, samples):
    metadata = {"model_name": "qwen-image-2.1", "seed": 42}
    record = {
        "samples": samples,
        "decoded_metrics": {},
        "dataset_key": "referring",
        "shard": {
            "index": index,
            "count": count,
            "start": start,
            "stop": stop,
            "of_samples": total,
        },
        "accumulator_states": {"main": state},
    }
    return {
        "metadata": metadata,
        "preflight": {"ok": True},
        "benchmarks": {"referring/refcocog": record},
    }


def test_merge_shards_rebuilds_the_unsharded_result_and_checks_the_tiling(tmp_path, monkeypatch):
    accs = [ReferringAccumulator() for _ in range(3)]
    whole = ReferringAccumulator()
    gen = torch.Generator().manual_seed(2)
    for index in range(5):
        pred, gt = (
            torch.rand(1, 4, 4, generator=gen) > 0.5,
            torch.rand(1, 4, 4, generator=gen) > 0.5,
        )
        accs[min(index // 2, 2)].update(pred, gt)
        whole.update(pred, gt)
    blocks = [(0, 2), (2, 4), (4, 5)]
    shards = [
        _shard_result(i, 3, *blocks[i], 5, accumulator_state(accs[i]), blocks[i][1] - blocks[i][0])
        for i in range(3)
    ]
    merged = merge_shards(list(reversed(shards)))
    record = merged["benchmarks"]["referring/refcocog"]
    assert record["samples"] == 5 and record["decoded_metrics"] == pytest.approx(whole.compute())
    assert "shard" not in record and "accumulator_states" not in record
    assert record["merged_shards"] == {"count": 3, "blocks": blocks}

    with pytest.raises(ValueError, match="expected shards"):
        merge_shards(shards[:2])
    gap = copy.deepcopy(shards)
    gap[1]["benchmarks"]["referring/refcocog"]["shard"]["start"] = 3
    with pytest.raises(ValueError, match="tile"):
        merge_shards(gap)
    other = copy.deepcopy(shards)
    other[2]["metadata"]["seed"] = 43
    with pytest.raises(ValueError, match="metadata"):
        merge_shards(other)
    on_other_gpus = copy.deepcopy(shards)
    for i, shard in enumerate(on_other_gpus):
        shard["metadata"]["device"] = f"cuda:{i}"
    assert merge_shards(on_other_gpus)["benchmarks"]["referring/refcocog"]["samples"] == 5
    resized = copy.deepcopy(shards)
    resized[0]["benchmarks"]["referring/refcocog"]["shard"]["of_samples"] = 6
    with pytest.raises(ValueError, match="of_samples"):
        merge_shards(resized)
    stateless = copy.deepcopy(shards)
    stateless[1]["benchmarks"]["referring/refcocog"]["accumulator_states"] = None
    with pytest.raises(ValueError, match="accumulator states"):
        merge_shards(stateless)

    paths = []
    for i, shard in enumerate(shards):
        paths.append(tmp_path / f"shard{i}.json")
        paths[-1].write_text(json.dumps(shard), encoding="utf-8")
    out = tmp_path / "merged.json"
    argv = [
        "merge_eval_shards.py",
        *map(str, paths),
        "--output-json",
        str(out),
        "--output-csv",
        str(tmp_path / "m.csv"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert merge_main() == 0
    assert json.loads(out.read_text())["benchmarks"]["referring/refcocog"]["samples"] == 5
    assert "cIoU" in (tmp_path / "m.csv").read_text()


def test_eval_cli_refuses_to_shard_unshardable_jobs(tmp_path, monkeypatch):
    argv = [
        "evaluate_paper_benchmarks.py",
        "--benchmark",
        "normal",
        "--normal-root",
        str(tmp_path),
        "--num-shards",
        "2",
        "--preflight-only",
        "--output-json",
        str(tmp_path / "out.json"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="--num-shards supports"):
        eval_main()
    monkeypatch.setattr(sys, "argv", [*argv, "--shard-index", "2"])
    with pytest.raises(SystemExit, match="--shard-index"):
        eval_main()
