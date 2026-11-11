"""Depth seed ensemble: log-scale median over K seeds per image, scored next to the single-seed reading."""

from __future__ import annotations

import json
import sys

import pytest
import torch

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.evaluation.benchmark import DepthAccumulator, apply_depth_benchmark_protocol
from open_vision_banana.evaluation.evaluator import (
    DEPTH_ENSEMBLE_SEED_STRIDE,
    VisionEvaluator,
    decode_depth_rgb,
    seed_ensemble_depth,
)
from open_vision_banana.evaluation.shards import merged_metrics
from scripts.evaluate_paper_benchmarks import main as eval_main


def test_ensemble_takes_the_median_scale_and_the_median_structure():
    base = torch.rand(2, 1, 4, 5, generator=torch.Generator().manual_seed(0)) + 0.5
    # odd count: the middle member's scale; even count: the geometric mean of the two middle scales
    assert torch.allclose(
        seed_ensemble_depth([base * 2, base * 0.5, base * 8]), base * 2, rtol=1e-5
    )
    assert torch.allclose(seed_ensemble_depth([base, base * 4]), base * 2, rtol=1e-5)
    assert torch.allclose(seed_ensemble_depth([base]), base, rtol=1e-5)
    # one member with a different structure is outvoted pixelwise, whatever its scale
    odd = base.clone()
    odd[:, :, 0, 0] *= 50
    assert torch.allclose(seed_ensemble_depth([base, odd * 3, base]), base, rtol=1e-5)
    # scales are per image
    scaled = base * torch.tensor([1.0, 10.0]).view(2, 1, 1, 1)
    assert torch.allclose(seed_ensemble_depth([scaled, scaled]), scaled, rtol=1e-5)


class _SeededModel:
    """predict() returns an RGB drawn from its seed and records the seeds it was asked for."""

    def __init__(self, fail_seed: int | None = None):
        self.seeds: list[int] = []
        self.fail_seed = fail_seed

    def eval(self):
        return self

    def predict(self, *, seed, **kwargs):
        self.seeds.append(seed)
        if seed == self.fail_seed:
            raise ValueError("Generator produced NaN or Inf")
        return {"rgb_output": torch.rand(1, 3, 8, 8, generator=torch.Generator().manual_seed(seed))}


def _batches(count: int) -> list[dict[str, torch.Tensor]]:
    gen = torch.Generator().manual_seed(1)
    return [
        {
            "image": torch.rand(1, 3, 8, 8, generator=gen),
            "target_rgb": torch.rand(1, 3, 8, 8, generator=gen),
        }
        for _ in range(count)
    ]


def _rgb(seed: int) -> torch.Tensor:
    return torch.rand(1, 3, 8, 8, generator=torch.Generator().manual_seed(seed))


def test_evaluator_scores_the_ensemble_next_to_the_unchanged_single_seed_reading():
    batches = _batches(2)
    single = VisionEvaluator(_SeededModel(), device="cpu").evaluate(batches, task="depth", seed=42)
    model = _SeededModel()
    ev = VisionEvaluator(model, device="cpu")
    report = ev.evaluate(batches, task="depth", seed=42, depth_seed_ensemble=3)
    stride = DEPTH_ENSEMBLE_SEED_STRIDE
    assert model.seeds == [42, 42 + stride, 42 + 2 * stride, 43, 43 + stride, 43 + 2 * stride]

    ensemble = report.pop("seed_ensemble")
    assert report.pop("seed_ensemble_members") == 3
    assert report == single

    mapper = DepthRGBMapper(colormap="cube_edge")
    expected = DepthAccumulator()
    for index, batch in enumerate(batches):
        members = [decode_depth_rgb(_rgb(42 + index + k * stride), mapper) for k in range(3)]
        gt = mapper.rgb_to_depth(batch["target_rgb"])
        expected.update(*apply_depth_benchmark_protocol("", seed_ensemble_depth(members), gt, None))
    assert ensemble == pytest.approx(expected.compute())
    assert ensemble != pytest.approx(single)

    states = json.loads(json.dumps(ev.last_accumulator_states))
    assert states["seed_ensemble_members"] == 3
    rebuilt = merged_metrics([states])
    assert rebuilt["seed_ensemble"] == pytest.approx(ensemble)
    assert rebuilt["seed_ensemble_members"] == 3


def test_ensemble_at_native_resolution_is_upsampled_like_the_main_prediction():
    batch = _batches(1)[0]
    native_gt = torch.rand(1, 1, 12, 16, generator=torch.Generator().manual_seed(3)) * 4 + 0.5
    batch.update(
        gt_depth_native=native_gt, valid_mask_native=torch.ones(1, 12, 16, dtype=torch.bool)
    )
    report = VisionEvaluator(_SeededModel(), device="cpu").evaluate(
        [batch], task="depth", seed=7, native_resolution=True, depth_seed_ensemble=2
    )
    mapper = DepthRGBMapper(colormap="cube_edge")
    members = [decode_depth_rgb(_rgb(7 + k * DEPTH_ENSEMBLE_SEED_STRIDE), mapper) for k in range(2)]
    upsampled = torch.nn.functional.interpolate(
        seed_ensemble_depth(members), size=(12, 16), mode="bilinear", align_corners=False
    )
    expected = DepthAccumulator()
    expected.update(
        *apply_depth_benchmark_protocol("", upsampled, native_gt, batch["valid_mask_native"])
    )
    assert report["seed_ensemble"] == pytest.approx(expected.compute())


def test_a_non_finite_member_skips_the_sample_for_both_readings():
    model = _SeededModel(fail_seed=43 + DEPTH_ENSEMBLE_SEED_STRIDE)
    report = VisionEvaluator(model, device="cpu", strict_nonfinite=False).evaluate(
        _batches(2), task="depth", seed=42, depth_seed_ensemble=2
    )
    single = VisionEvaluator(_SeededModel(), device="cpu").evaluate(
        _batches(1), task="depth", seed=42
    )
    assert report["skipped_samples"] == 1
    assert (
        report["valid_pixels"] == single["valid_pixels"] == report["seed_ensemble"]["valid_pixels"]
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"task": "depth", "seed": 42, "depth_seed_ensemble": 0}, "at least 1"),
        ({"task": "normal", "seed": 42, "depth_seed_ensemble": 2}, "task='depth'"),
        ({"task": "depth", "seed": None, "depth_seed_ensemble": 2}, "a seed"),
    ],
)
def test_evaluator_rejects_an_ensemble_it_cannot_seed(kwargs, match):
    with pytest.raises(ValueError, match=match):
        VisionEvaluator(_SeededModel(), device="cpu").evaluate(_batches(1), **kwargs)


def test_merge_refuses_shards_that_disagree_on_the_ensemble():
    batches = _batches(2)
    states = []
    for index, members in enumerate((2, 1)):
        ev = VisionEvaluator(_SeededModel(), device="cpu")
        ev.evaluate(
            batches[index : index + 1], task="depth", seed=42 + index, depth_seed_ensemble=members
        )
        states.append(ev.last_accumulator_states)
    with pytest.raises(ValueError, match="seed ensemble"):
        merged_metrics(states)
    assert "seed_ensemble" not in merged_metrics(states[1:])


def test_eval_cli_rejects_an_empty_ensemble(tmp_path, monkeypatch, capsys):
    argv = [
        "evaluate_paper_benchmarks.py",
        "--depth-seed-ensemble",
        "0",
        "--output-json",
        str(tmp_path / "o.json"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        eval_main()
    assert "--depth-seed-ensemble must be at least 1" in capsys.readouterr().err
