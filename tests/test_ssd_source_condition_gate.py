"""Tests for the source-conditioned SSD-1B overfit gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from _paths import ARCHIVE_SCRIPTS
from PIL import Image
from torch import nn

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.data.depth_encoding import relative_depth_prompt

SCRIPT = ARCHIVE_SCRIPTS / "quick_overfit_ssd_source_condition.py"


def load_module():
    spec = importlib.util.spec_from_file_location("quick_overfit_ssd_source_condition", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expand_unet_input_preserves_original_forward():
    module = load_module()

    class FakeUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_in = nn.Conv2d(4, 6, 3, padding=1)

    torch.manual_seed(7)
    unet = FakeUNet()
    latent = torch.randn(2, 4, 8, 8)
    expected = unet.conv_in(latent)

    expanded = module.expand_unet_input(unet, condition_channels=4)
    actual = expanded(torch.cat((latent, torch.randn_like(latent)), dim=1))

    assert expanded.in_channels == 8
    assert torch.allclose(actual, expected)
    assert torch.count_nonzero(expanded.weight[:, 4:]) == 0


def test_conditioned_input_noises_target_and_appends_source():
    module = load_module()

    class FakeScheduler:
        timesteps = torch.tensor([3.0])
        sigmas = torch.tensor([0.0, 0.0])

        @staticmethod
        def add_noise(original, noise, timesteps):
            return original + noise

        @staticmethod
        def index_for_timestep(timestep, schedule):
            return 0

    target = torch.full((1, 4, 2, 2), 2.0)
    source = torch.full((1, 4, 2, 2), 7.0)
    noise = torch.full((1, 4, 2, 2), 3.0)
    result = module.build_conditioned_model_input(
        FakeScheduler(), target, source, noise, torch.tensor([3.0])
    )

    assert result.shape == (1, 8, 2, 2)
    assert torch.equal(result[:, :4], target + noise)
    assert torch.equal(result[:, 4:], source)


def test_repeat_final_timestep_only_oversamples_lowest_noise_step():
    module = load_module()
    timesteps = torch.tensor([751.0, 501.0, 251.0, 1.0])

    result = module.repeat_final_timestep(timesteps, repeats=4)

    assert result.tolist() == [751.0, 501.0, 251.0, 1.0, 1.0, 1.0, 1.0]


def test_resumed_training_schedule_uses_cumulative_step():
    module = load_module()

    sample_index, timestep_index = module.training_schedule_indices(
        600,
        sample_count=8,
        timestep_count=7,
    )

    assert sample_index == 0
    assert timestep_index == 5


def test_completed_steps_support_current_and_legacy_reports():
    module = load_module()

    assert module.infer_completed_steps({"training_steps": {"cumulative": 2400}}) == 2400
    assert module.infer_completed_steps({"config": {"max_steps": 600}}) == 600
    assert module.infer_completed_steps({"loss": {"history": [0.4, 0.3]}}) == 2
    assert module.infer_completed_steps(None) == 0


def test_sigma_for_timesteps_broadcasts_over_latents():
    module = load_module()

    class FakeScheduler:
        timesteps = torch.tensor([3.0, 1.0])
        sigmas = torch.tensor([4.0, 0.5, 0.0])

        @staticmethod
        def index_for_timestep(timestep, schedule):
            return int(torch.nonzero(schedule == timestep)[0])

    reference = torch.zeros(2, 4, 3, 3)
    sigma = module.sigma_for_timesteps(FakeScheduler(), torch.tensor([3.0, 1.0]), reference)

    assert sigma.shape == (2, 1, 1, 1)
    assert sigma[:, 0, 0, 0].tolist() == [4.0, 0.5]


def test_depth_audit_overrides_stale_per_sample_prompt(tmp_path: Path):
    module = load_module()
    image_dir = tmp_path / "depth" / "images"
    target_dir = tmp_path / "depth" / "targets"
    prompt_dir = tmp_path / "depth" / "prompts"
    image_dir.mkdir(parents=True)
    target_dir.mkdir()
    prompt_dir.mkdir()
    source = image_dir / "sample.jpg"
    target = target_dir / "sample.png"
    Image.new("RGB", (32, 32), "black").save(source)
    normalized = torch.linspace(0, 1, 32).view(1, 1, 1, 32).expand(1, 1, 32, 32)
    rgb = DepthRGBMapper("viridis").normalized_to_rgb(normalized)[0]
    array = (rgb.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    Image.fromarray(array).save(target)
    (prompt_dir / "sample.txt").write_text(
        "Use a power-transform RGB encoding.",
        encoding="utf-8",
    )

    audit, prompt = module.audit_depth_pairs(tmp_path, [(source, target)])

    assert audit.detected_colormap == "viridis"
    assert prompt == relative_depth_prompt("viridis")
    assert "power-transform" not in prompt


def test_split_clean_pairs_follow_manifest_without_overlap(tmp_path: Path):
    module = load_module()
    image_dir = tmp_path / "depth" / "images"
    target_dir = tmp_path / "depth" / "targets"
    image_dir.mkdir(parents=True)
    target_dir.mkdir()
    for name in ("train_a", "train_b", "val_a", "val_b"):
        Image.new("RGB", (8, 8), "black").save(image_dir / f"{name}.jpg")
        Image.new("RGB", (8, 8), "black").save(target_dir / f"{name}.png")
    (tmp_path / "train_val_split.json").write_text(
        json.dumps(
            {
                "train": {"depth": ["train_a.jpg", "train_b.jpg"]},
                "val": {"depth": ["val_a.jpg", "val_b.jpg"]},
            }
        ),
        encoding="utf-8",
    )

    train_pairs, eval_pairs = module.find_split_clean_depth_pairs(
        tmp_path,
        train_count=2,
        eval_count=2,
    )

    assert [source.stem for source, _ in train_pairs] == ["train_a", "train_b"]
    assert [source.stem for source, _ in eval_pairs] == ["val_a", "val_b"]

    manifest = json.loads((tmp_path / "train_val_split.json").read_text())
    manifest["val"]["depth"].append("train_a.jpg")
    (tmp_path / "train_val_split.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlap"):
        module.find_split_clean_depth_pairs(
            tmp_path,
            train_count=2,
            eval_count=2,
        )


def test_gate_metrics_use_audited_relative_depth_position():
    module = load_module()
    normalized = torch.linspace(0, 1, 32).view(1, 1, 1, 32).expand(1, 1, 32, 32)
    mapper = DepthRGBMapper("viridis")
    target_rgb = mapper.normalized_to_rgb(normalized)[0]
    reversed_rgb = mapper.normalized_to_rgb(1.0 - normalized)[0]

    def to_pil(rgb: torch.Tensor) -> Image.Image:
        array = (rgb.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
        return Image.fromarray(array)

    metrics = module.image_metrics(
        to_pil(reversed_rgb),
        to_pil(target_rgb),
        colormap="viridis",
    )

    assert metrics["relative_position_pearson_r"] < -0.999
    assert metrics["relative_position_spearman_r"] < -0.999


def test_metric_summaries_and_gate_are_split_scoped():
    module = load_module()
    samples = [
        {
            "after": {
                "rgb_mae": 0.10,
                "relative_position_pearson_r": 0.90,
            }
        },
        {
            "after": {
                "rgb_mae": 0.12,
                "relative_position_pearson_r": 0.80,
            }
        },
    ]

    summary = module.summarize_sample_metrics(samples, "after")

    assert summary == {
        "rgb_mae": pytest.approx(0.11),
        "relative_position_pearson_r": pytest.approx(0.85),
    }
    assert module.gate_passed(summary, max_mae=0.14, min_correlation=0.8)
    assert not module.gate_passed(summary, max_mae=0.10, min_correlation=0.8)


def test_gate_calibration_projects_prediction_to_audited_curve(tmp_path: Path):
    module = load_module()
    image_dir = tmp_path / "depth" / "images"
    target_dir = tmp_path / "depth" / "targets"
    image_dir.mkdir(parents=True)
    target_dir.mkdir()
    source = image_dir / "sample.png"
    target = target_dir / "sample.png"
    Image.new("RGB", (32, 32), "black").save(source)
    normalized = torch.linspace(0, 1, 32).view(1, 1, 1, 32).expand(1, 1, 32, 32)
    target_rgb = DepthRGBMapper("viridis").normalized_to_rgb(normalized)[0]
    target_array = (target_rgb.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    Image.fromarray(target_array).save(target)
    audit, _ = module.audit_depth_pairs(tmp_path, [(source, target)])
    off_curve = Image.new("RGB", (32, 32), (180, 140, 120))

    calibrated = module.calibrate_depth_prediction(
        off_curve,
        audit,
        mode="moments_and_project",
    )
    metrics = module.image_metrics(
        calibrated,
        Image.open(target),
        colormap="viridis",
    )

    assert metrics["prediction_curve_mae"] < 0.01
