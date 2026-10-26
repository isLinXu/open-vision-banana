"""Generation-preservation shards: per-image states whose merge equals the unsharded report."""

from __future__ import annotations

import copy
import json
import math
import sys

import pytest
import torch

import open_vision_banana.evaluation.generation_preservation as gp
import scripts.evaluate_generation_preservation as gen_script
from open_vision_banana.evaluation.generation_preservation import (
    evaluate_generation_preservation,
    extract_rgb_features,
    generation_shard_state,
    merge_generation_shard_states,
)
from open_vision_banana.evaluation.shards import shard_range
from scripts.merge_eval_shards import main as merge_main
from scripts.merge_eval_shards import merge_generation_shards


class _FakeLpips(torch.nn.Module):
    """Per-image distance shaped like ``lpips.LPIPS`` output ([N, 1, 1, 1])."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a - b).abs().mean(dim=(1, 2, 3), keepdim=True)


def _assert_same_metrics(merged: dict, whole: dict) -> None:
    assert list(merged) == list(whole)
    for key, value in whole.items():
        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(merged[key]), key
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


def test_shard_states_merge_to_the_unsharded_metrics(with_lpips):
    gen = torch.Generator().manual_seed(0)
    real, fake = torch.rand(7, 3, 16, 16, generator=gen), torch.rand(7, 3, 16, 16, generator=gen)
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
    assert (states[0]["lpips"] is not None) == with_lpips
    _assert_same_metrics(merge_generation_shard_states(states), whole)


def test_shard_state_resizes_generated_images_like_the_unsharded_run(no_lpips):
    real, fake = torch.rand(2, 3, 8, 8), torch.rand(2, 3, 16, 16)
    state = generation_shard_state(real, fake, extract_rgb_features)
    resized = torch.nn.functional.interpolate(
        fake, size=(8, 8), mode="bilinear", align_corners=False
    )
    assert torch.allclose(state["mse"], (resized - real).pow(2).flatten(1).mean(dim=1))


def test_merge_rejects_empty_or_inconsistent_states(no_lpips):
    with pytest.raises(ValueError, match="no generation shard states"):
        merge_generation_shard_states([])
    state = generation_shard_state(
        torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8), extract_rgb_features
    )
    other = {**state, "lpips": torch.zeros(2)}
    with pytest.raises(ValueError, match="LPIPS"):
        merge_generation_shard_states([state, other])


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
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate(self, *, prompt, image, num_inference_steps, height, width, seed):
        return torch.rand(1, 3, height, width, generator=torch.Generator().manual_seed(seed))


def _run_script(monkeypatch, tmp_path, out, *extra):
    argv = [
        "evaluate_generation_preservation.py",
        "--model-name",
        "fake/model",
        "--dataset-root",
        str(tmp_path),
        "--output-json",
        str(out),
        "--resolution",
        "16",
        "--device",
        "cpu",
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return gen_script.main()


@pytest.fixture
def fake_runtime(monkeypatch, tmp_path):
    dataset = _FakeDataset(tmp_path)
    monkeypatch.setattr(gen_script, "build_generation_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(gen_script, "BaseGenerator", _FakeGenerator)
    return dataset


def test_sharded_script_runs_merge_into_the_unsharded_report(
    monkeypatch, tmp_path, fake_runtime, with_lpips
):
    whole_path = tmp_path / "gen.json"
    assert _run_script(monkeypatch, tmp_path, whole_path) == 0
    whole = json.loads(whole_path.read_text())

    shard_paths = []
    for index in range(3):
        shard_paths.append(tmp_path / f"gen.shard{index}.json")
        assert (
            _run_script(
                monkeypatch,
                tmp_path,
                shard_paths[-1],
                "--num-shards",
                "3",
                "--shard-index",
                str(index),
            )
            == 0
        )
    shard = json.loads(shard_paths[1].read_text())
    assert shard["metadata"]["shard"] == {
        "index": 1,
        "count": 3,
        "start": 3,
        "stop": 6,
        "of_samples": 7,
    }
    assert shard["shard_state"] == "gen.shard1.state.pt" and "metrics" not in shard

    merged_path = tmp_path / "merged.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_eval_shards.py",
            *map(str, reversed(shard_paths)),
            "--output-json",
            str(merged_path),
        ],
    )
    assert merge_main() == 0
    merged = json.loads(merged_path.read_text())
    for key, value in whole["metrics"].items():
        assert merged["metrics"][key] == (
            None if value is None else pytest.approx(value, rel=1e-5, abs=1e-7)
        ), key
    meta = merged["metadata"]
    assert (
        meta["requested_samples"] == meta["successful_samples"] == 7 and meta["failed_samples"] == 0
    )
    assert meta["merged_shards"] == {"count": 3, "blocks": [[0, 3], [3, 6], [6, 7]]}
    assert meta["dataset_fingerprint"] == whole["metadata"]["dataset_fingerprint"]
    assert "shard" not in meta and len(meta["failure_jsonl"]) == 3

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_eval_shards.py",
            *map(str, shard_paths),
            "--output-json",
            str(merged_path),
            "--output-csv",
            "x.csv",
        ],
    )
    with pytest.raises(SystemExit, match="output-csv"):
        merge_main()
    benchmark = tmp_path / "bench.json"
    benchmark.write_text(json.dumps({"metadata": {}, "benchmarks": {}}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_eval_shards.py",
            str(shard_paths[0]),
            str(benchmark),
            "--output-json",
            str(merged_path),
        ],
    )
    with pytest.raises(SystemExit, match="cannot merge"):
        merge_main()


def test_script_rejects_bad_or_empty_shards(monkeypatch, tmp_path, fake_runtime):
    out = tmp_path / "gen.json"
    with pytest.raises(SystemExit, match="shard-index"):
        _run_script(monkeypatch, tmp_path, out, "--num-shards", "2", "--shard-index", "2")
    with pytest.raises(SystemExit, match="empty"):
        _run_script(
            monkeypatch,
            tmp_path,
            out,
            "--max-samples",
            "1",
            "--num-shards",
            "2",
            "--shard-index",
            "1",
        )


def _generation_shard(index: int, blocks: list[tuple[int, int]], **metadata) -> dict:
    start, stop = blocks[index]
    return {
        "metadata": {
            "evaluation_mode": "generation_preservation_proxy",
            "model_name": "m",
            "seed": 42,
            "device": f"cuda:{index}",
            "requested_samples": stop - start,
            "successful_samples": stop - start,
            "failed_samples": 0,
            "failure_jsonl": f"f{index}.jsonl",
            "elapsed_seconds": 1.5,
            "shard": {
                "index": index,
                "count": len(blocks),
                "start": start,
                "stop": stop,
                "of_samples": blocks[-1][1],
            },
            **metadata,
        },
        "shard_state": f"s{index}.state.pt",
    }


def test_merge_generation_shards_checks_the_set_of_shards(no_lpips):
    blocks = [(0, 2), (2, 4), (4, 5)]
    states = [
        generation_shard_state(
            torch.rand(b - a, 3, 8, 8), torch.rand(b - a, 3, 8, 8), extract_rgb_features
        )
        for a, b in blocks
    ]
    shards = [_generation_shard(index, blocks) for index in range(3)]
    merged = merge_generation_shards(shards, states)
    assert merged["metadata"]["elapsed_seconds"] == pytest.approx(4.5)
    assert merged["metadata"]["merged_shards"] == {"count": 3, "blocks": blocks}
    assert "device" not in merged["metadata"] and "shard_state" not in merged

    with pytest.raises(ValueError, match="one saved state"):
        merge_generation_shards(shards, states[:2])
    with pytest.raises(ValueError, match="expected shards"):
        merge_generation_shards(shards[:2], states[:2])
    gap = copy.deepcopy(shards)
    gap[1]["metadata"]["shard"]["start"] = 3
    with pytest.raises(ValueError, match="tile"):
        merge_generation_shards(gap, states)
    other = copy.deepcopy(shards)
    other[2]["metadata"]["seed"] = 43
    with pytest.raises(ValueError, match="metadata"):
        merge_generation_shards(other, states)
    resized = copy.deepcopy(shards)
    resized[0]["metadata"]["shard"]["of_samples"] = 6
    with pytest.raises(ValueError, match="of_samples"):
        merge_generation_shards(resized, states)
