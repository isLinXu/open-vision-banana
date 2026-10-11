"""Benchmark entry points exercise public mapper APIs without loading a model."""

from __future__ import annotations

import json
import math

import torch

from scripts import benchmark, benchmark_pipeline


def test_single_iteration_statistics_are_finite():
    statistics, result = benchmark.benchmark_fn(lambda value: value + 1, 3, warmup=0, iterations=1)
    assert result == 4
    assert statistics["std_ms"] == 0.0
    assert all(math.isfinite(value) for value in statistics.values())


def test_mapper_cli_does_not_load_model_and_writes_all_operations(tmp_path, monkeypatch):
    from open_vision_banana.inference.pipeline import VisionBanana

    def forbidden_model_load(*args, **kwargs):
        raise AssertionError("Mapper-only must not load model weights")

    monkeypatch.setattr(VisionBanana, "from_pretrained", forbidden_model_load)
    destination = tmp_path / "new" / "mapper.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark_pipeline.py",
            "--mapper-only",
            "--device",
            "cpu",
            "--resolution",
            "8",
            "--num-samples",
            "1",
            "--warmup",
            "1",
            "--output",
            str(destination),
        ],
    )
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        assert benchmark_pipeline.main() == 0
    finally:
        torch.set_num_threads(previous_threads)
    results = json.loads(destination.read_text())
    assert {result["task"] for result in results} == {"depth", "normal", "segmentation"}
    for result in results:
        assert result["preset"] == "mapper_only"
        assert result["num_inference_steps"] == 0
        assert result["median_latency_ms"] > 0
        assert result["std_latency_ms"] == 0
        assert all(math.isfinite(value) for value in result.values() if isinstance(value, float))


def test_core_cli_writes_native_json_into_new_directory(tmp_path, monkeypatch):
    destination = tmp_path / "new" / "core.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark.py",
            "--device",
            "cpu",
            "--resolution",
            "8",
            "--iterations",
            "1",
            "--output",
            str(destination),
        ],
    )
    benchmark.main()
    results = json.loads(destination.read_text())
    assert set(results) == {
        "Depth encode",
        "Depth decode",
        "Normal encode",
        "Normal decode",
        "Instance decode",
        "Unified encode(depth)",
        "Unified decode(depth)",
    }
    assert all(value["median_ms"] > 0 for value in results.values())
