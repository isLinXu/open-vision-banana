"""Generation-preservation shards: per-image states whose merge equals the unsharded report."""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest
import torch

import open_vision_banana.evaluation.generation_preservation as gp
import scripts.evaluate_generation_preservation as gen_script
from open_vision_banana.evaluation.generation_preservation import (
    evaluate_generation_preservation,
    extract_rgb_features,
    generation_metrics,
    generation_shard_state,
)
from open_vision_banana.evaluation.shards import shard_range
from scripts.merge_eval_shards import main as merge_main
from scripts.merge_eval_shards import merge_generation_shards


class _FakeLpips(torch.nn.Module):
    """Per-image distance shaped like ``lpips.LPIPS`` output ([N, 1, 1, 1])."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a - b).abs().mean(dim=(1, 2, 3), keepdim=True)


def _assert_same_metrics(merged: dict, whole: dict) -> None:
    """Equal up to float rounding; NaN (or JSON null) matches NaN (or null)."""
    assert list(merged) == list(whole)
    for key, value in whole.items():
        if value is None or math.isnan(value):
            assert merged[key] is None or math.isnan(merged[key]), key
        else:
            assert merged[key] == pytest.approx(value, rel=1e-5, abs=1e-7), key


@pytest.fixture(params=[False, True], ids=["mse", "lpips"])
def with_lpips(request, monkeypatch):
    monkeypatch.setattr(gp, "_lpips_available", lambda: request.param)
    monkeypatch.setattr(gp, "_get_lpips_model", lambda device, net="alex": _FakeLpips())
    return request.param


@pytest.fixture
def no_lpips(monkeypatch):
    """Pixel-MSE path whether or not the ``lpips`` package is installed."""
    monkeypatch.setattr(gp, "_lpips_available", lambda: False)


@pytest.mark.parametrize("size", [16, 8], ids=["same-size", "resized"])
def test_block_states_give_the_metrics_of_the_whole_set(with_lpips, size):
    gen = torch.Generator().manual_seed(0)
    real = torch.rand(7, 3, 16, 16, generator=gen)
    fake = torch.rand(7, 3, size, size, generator=gen)
    whole = evaluate_generation_preservation(
        real_rgb=real,
        gen_rgb=fake,
        pred_rgb=fake,
        target_rgb=real,
        feature_extractor=extract_rgb_features,
    )
    blocks = [shard_range(7, 3, index) for index in range(3)]
    states = [
        generation_shard_state(real[b.start : b.stop], fake[b.start : b.stop], extract_rgb_features)
        for b in blocks
    ]
    assert states[0]["lpips"] is with_lpips
    _assert_same_metrics(generation_metrics(states), whole)


def test_metrics_skip_blocks_without_images_and_reject_inconsistent_states(no_lpips):
    for empty in ([], [None, None]):
        with pytest.raises(ValueError, match="no generated image"):
            generation_metrics(empty)
    state = generation_shard_state(
        torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8), extract_rgb_features
    )
    _assert_same_metrics(generation_metrics([None, state, None]), generation_metrics([state]))
    with pytest.raises(ValueError, match="LPIPS"):
        generation_metrics([state, {**state, "lpips": True}])


class _FakeDataset(torch.utils.data.Dataset):
    def __init__(self, root, size: int = 7, resolution: int = 16):
        self.root_dir = root
        gen = torch.Generator().manual_seed(3)
        self.images = torch.rand(size, 3, resolution, resolution, generator=gen)
        self.samples = [
            {"image_path": str(root / f"{i}.png"), "prompt": f"prompt {i}"} for i in range(size)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return {"prompt": self.samples[index]["prompt"], "target_rgb": self.images[index]}


class _FakeGenerator:
    fail_from_seed: int | None = None  # samples whose seed reaches this raise

    def __init__(self, **kwargs):
        pass

    def generate(self, *, prompt, image, num_inference_steps, height, width, seed):
        if self.fail_from_seed is not None and seed >= self.fail_from_seed:
            raise RuntimeError("CUDA out of memory")
        return torch.rand(1, 3, height, width, generator=torch.Generator().manual_seed(seed))


@pytest.fixture
def fake_runtime(monkeypatch, tmp_path):
    dataset = _FakeDataset(tmp_path)
    monkeypatch.setattr(gen_script, "build_generation_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(gen_script, "BaseGenerator", _FakeGenerator)
    return dataset


def _run_script(monkeypatch, tmp_path, out, *extra):
    argv = ["evaluate_generation_preservation.py", "--model-name", "fake/model"]
    argv += ["--dataset-root", str(tmp_path), "--output-json", str(out)]
    argv += ["--resolution", "16", "--device", "cpu", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    return gen_script.main()


def _run_merge(monkeypatch, out, *paths, extra=()):
    argv = ["merge_eval_shards.py", *map(str, paths), "--output-json", str(out), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    return merge_main()


def _run_shards(monkeypatch, tmp_path, count=3, *extra):
    paths = [tmp_path / f"gen.shard{index}.json" for index in range(count)]
    for index, path in enumerate(paths):
        shard_args = ("--num-shards", str(count), "--shard-index", str(index), *extra)
        assert _run_script(monkeypatch, tmp_path, path, *shard_args) == 0
    return paths


def _merged_equals_unsharded(monkeypatch, tmp_path, paths, *extra) -> dict:
    assert _run_script(monkeypatch, tmp_path, tmp_path / "gen.json", *extra) == 0
    whole = json.loads((tmp_path / "gen.json").read_text())
    assert _run_merge(monkeypatch, tmp_path / "merged.json", *reversed(paths)) == 0
    merged = json.loads((tmp_path / "merged.json").read_text())
    _assert_same_metrics(merged["metrics"], whole["metrics"])
    for key in ("requested_samples", "successful_samples", "failed_samples", "dataset_fingerprint"):
        assert merged["metadata"][key] == whole["metadata"][key], key
    return merged


def test_sharded_runs_merge_into_the_unsharded_report(
    monkeypatch, tmp_path, fake_runtime, with_lpips
):
    paths = _run_shards(monkeypatch, tmp_path)
    shard = json.loads(paths[1].read_text())
    assert shard["metadata"]["shard"] == {
        "index": 1,
        "count": 3,
        "start": 3,
        "stop": 6,
        "of_samples": 7,
    }
    assert shard["shard_state"] == "gen.shard1.state.pt" and "metrics" not in shard

    meta = _merged_equals_unsharded(monkeypatch, tmp_path, paths)["metadata"]
    assert meta["successful_samples"] == 7 and "shard" not in meta
    assert meta["merged_shards"] == {"count": 3, "blocks": [[0, 3], [3, 6], [6, 7]]}
    assert meta["failure_jsonl"] == str(tmp_path / "merged.failures.jsonl")


def test_empty_and_wholly_failed_shards_still_merge(monkeypatch, tmp_path, fake_runtime, no_lpips):
    # more shards than samples: shard 1 of 2 over one sample has an empty block
    paths = _run_shards(monkeypatch, tmp_path, 2, "--max-samples", "1")
    assert json.loads(paths[1].read_text())["metadata"]["requested_samples"] == 0
    _merged_equals_unsharded(monkeypatch, tmp_path, paths, "--max-samples", "1")

    # every sample of shard 1 (indices 4..6, seeds 46..48) fails: a proxy from the rest
    monkeypatch.setattr(_FakeGenerator, "fail_from_seed", 46)
    with pytest.raises(RuntimeError, match="3 generation samples failed"):
        _run_script(monkeypatch, tmp_path, tmp_path / "x.json")
    paths = _run_shards(monkeypatch, tmp_path, 2, "--allow-failures")
    meta = _merged_equals_unsharded(monkeypatch, tmp_path, paths, "--allow-failures")["metadata"]
    assert meta["failed_samples"] == 3 and meta["successful_samples"] == 4
    assert len(Path(meta["failure_jsonl"]).read_text().splitlines()) == 3


def test_merge_cli_rejects_wrong_inputs(monkeypatch, tmp_path, fake_runtime, no_lpips):
    paths = _run_shards(monkeypatch, tmp_path, count=2)
    out = tmp_path / "merged.json"
    with pytest.raises(SystemExit, match="output-csv"):
        _run_merge(monkeypatch, out, *paths, extra=("--output-csv", str(tmp_path / "m.csv")))
    benchmark = tmp_path / "bench.json"
    benchmark.write_text(json.dumps({"metadata": {}, "benchmarks": {}}))
    with pytest.raises(SystemExit, match="cannot merge"):
        _run_merge(monkeypatch, out, paths[0], benchmark)
    assert _run_script(monkeypatch, tmp_path, tmp_path / "gen.json") == 0
    with pytest.raises(SystemExit, match="not the output of a sharded evaluation"):
        _run_merge(monkeypatch, out, tmp_path / "gen.json")


def test_script_rejects_a_shard_index_out_of_range(monkeypatch, tmp_path, fake_runtime):
    with pytest.raises(SystemExit, match="shard-index"):
        _run_script(
            monkeypatch, tmp_path, tmp_path / "g.json", "--num-shards", "2", "--shard-index", "2"
        )


def test_merge_checks_run_metadata_and_saved_states(no_lpips):
    blocks = [(0, 2), (2, 5)]
    shards = []
    for index, (start, stop) in enumerate(blocks):
        record = {
            "metadata": {
                "evaluation_mode": "generation_preservation_proxy",
                "seed": 42,
                "device": f"cuda:{index}",
                "requested_samples": stop - start,
                "successful_samples": stop - start,
                "failed_samples": 0,
                "failure_jsonl": f"f{index}.jsonl",
                "elapsed_seconds": 1.5 + index,
                "shard": {
                    "index": index,
                    "count": 2,
                    "start": start,
                    "stop": stop,
                    "of_samples": 5,
                },
            },
            "shard_state": f"s{index}.state.pt",
        }
        images = torch.rand(stop - start, 3, 8, 8), torch.rand(stop - start, 3, 8, 8)
        shards.append((record, generation_shard_state(*images, extract_rgb_features)))
    merged = merge_generation_shards(shards)
    assert merged["metadata"]["elapsed_seconds"] == pytest.approx(2.5)  # wall time of the slowest
    assert "device" not in merged["metadata"] and "shard_state" not in merged

    with pytest.raises(ValueError, match="no generation shards"):
        merge_generation_shards([])
    other = copy.deepcopy(shards)
    other[1][0]["metadata"]["seed"] = 43
    with pytest.raises(ValueError, match="metadata"):
        merge_generation_shards(other)
    stale = [shards[0], (shards[1][0], shards[0][1])]
    with pytest.raises(ValueError, match="stale or foreign"):
        merge_generation_shards(stale)
