"""Tests for the frozen SSD/VAE scalar-depth architecture probe."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from _paths import ARCHIVE_SCRIPTS
from torch import nn

from open_vision_banana.depth_probe_checkpoint import (
    DEPTH_PROBE_CONFIG_NAME,
    DEPTH_PROBE_WEIGHTS_NAME,
    load_depth_probe_artifact,
    save_depth_probe_artifact,
)
from open_vision_banana.inference.depth_probe import FrozenDepthEstimator
from open_vision_banana.training.depth_probe import (
    DepthFeatureContract,
    FrozenSDXLDepthFeatureExtractor,
    ScalarDepthProbe,
    depth_feature_contribution_gate_passed,
    depth_probe_gate_passed,
    evaluate_scalar_depth_predictions,
    scalar_depth_probe_loss,
)


def load_probe_script():
    script_path = ARCHIVE_SCRIPTS / "overfit_frozen_depth_probe.py"
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_feature_comparison_script():
    script_path = ARCHIVE_SCRIPTS / "compare_frozen_depth_features.py"
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_production_refit_script():
    # the script imports its sibling compare_frozen_depth_features by bare name
    sys.path.insert(0, str(ARCHIVE_SCRIPTS))
    script_path = ARCHIVE_SCRIPTS / "train_frozen_depth_estimator.py"
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_probe_visualization_script():
    sys.path.insert(0, str(ARCHIVE_SCRIPTS))  # bare-name sibling imports, as above
    script_path = ARCHIVE_SCRIPTS / "visualize_frozen_depth_probe.py"
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeLatentDistribution:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    def mode(self) -> torch.Tensor:
        return self.value


class FakeVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, 3, stride=2, padding=1)

    def encode(self, images: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(latent_dist=FakeLatentDistribution(self.encoder(images)))


class FakeDownBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            input_channels,
            output_channels,
            3,
            stride=2,
            padding=1,
        )

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor]]:
        output = F.silu(self.conv(hidden))
        return output, (output,)


class FakeMidBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.silu(self.conv(hidden))


class FakeUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(in_channels=4)
        self.down_blocks = nn.ModuleList([FakeDownBlock(4, 8), FakeDownBlock(8, 16)])
        self.mid_block = FakeMidBlock(16)
        self.output = nn.Conv2d(16, 4, 1)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor]:
        del timestep, kwargs
        hidden = sample
        for block in self.down_blocks:
            hidden = block(hidden)[0]
        hidden = self.mid_block(hidden)
        hidden = F.interpolate(
            hidden,
            size=sample.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return (self.output(hidden),)


class FakeGenerator:
    def __init__(self) -> None:
        self.pipe = SimpleNamespace(vae=FakeVAE(), unet=FakeUNet())
        self.model_config = SimpleNamespace(
            family="sdxl",
            vae_has_latent_dist=True,
            vae_scaling_factor=1.0,
            vae_shift_factor=0.0,
            needs_added_cond_kwargs=False,
        )

    @staticmethod
    def encode_prompts(prompts: list[str]) -> dict[str, torch.Tensor]:
        batch = len(prompts)
        return {
            "prompt_embeds": torch.zeros(batch, 2, 4),
            "pooled_prompt_embeds": torch.zeros(batch, 4),
        }


def test_frozen_extractor_returns_ordered_deterministic_features_and_removes_hooks():
    generator = FakeGenerator()
    extractor = FrozenSDXLDepthFeatureExtractor(generator, timestep=0)
    images = torch.randn(2, 3, 16, 16)

    contract, features = extractor.extract(images, prompt="relative depth")
    second_contract, second_features = extractor.extract(
        images,
        prompt="relative depth",
    )

    assert contract == DepthFeatureContract(
        names=("vae_latent", "down_0", "down_1", "mid", "unet_output"),
        channels=(4, 8, 16, 16, 4),
    )
    assert second_contract == contract
    assert [tuple(feature.shape) for feature in features] == [
        (2, 4, 8, 8),
        (2, 8, 4, 4),
        (2, 16, 2, 2),
        (2, 16, 2, 2),
        (2, 4, 8, 8),
    ]
    assert all(torch.equal(a, b) for a, b in zip(features, second_features, strict=True))
    assert all(not feature.requires_grad for feature in features)
    assert all(not parameter.requires_grad for parameter in generator.pipe.vae.parameters())
    assert all(not parameter.requires_grad for parameter in generator.pipe.unet.parameters())
    modules = [*generator.pipe.unet.down_blocks, generator.pipe.unet.mid_block]
    assert all(not module._forward_hooks for module in modules)


def test_scalar_depth_probe_fuses_multiscale_features_and_backpropagates():
    contract = DepthFeatureContract(
        names=("fine", "coarse"),
        channels=(4, 8),
    )
    head = ScalarDepthProbe(contract, width=16, upsample_stages=1)
    features = (
        torch.randn(2, 4, 8, 8),
        torch.randn(2, 8, 4, 4),
    )
    target = torch.rand(2, 1, 16, 16)

    prediction = head(features, output_size=(16, 16))
    loss = scalar_depth_probe_loss(prediction, target)
    loss.backward()

    assert prediction.shape == target.shape
    assert prediction.min() >= 0 and prediction.max() <= 1
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in head.parameters()
    )


def test_scalar_depth_probe_rejects_feature_contract_mismatch():
    contract = DepthFeatureContract(names=("fine",), channels=(4,))
    head = ScalarDepthProbe(contract, width=8, upsample_stages=0)

    with pytest.raises(ValueError, match="feature count"):
        head((), output_size=(8, 8))
    with pytest.raises(ValueError, match="channels"):
        head((torch.randn(1, 3, 8, 8),), output_size=(8, 8))


def test_scalar_depth_probe_mask_removes_projected_feature_information():
    torch.manual_seed(4)
    contract = DepthFeatureContract(
        names=("vae_latent", "ssd_feature"),
        channels=(4, 8),
    )
    head = ScalarDepthProbe(contract, width=8, upsample_stages=0).eval()
    vae = torch.randn(2, 4, 8, 8)
    first_ssd = torch.randn(2, 8, 4, 4)
    second_ssd = torch.randn(2, 8, 4, 4) * 100.0

    first = head(
        (vae, first_ssd),
        output_size=(8, 8),
        feature_mask=(True, False),
    )
    second = head(
        (vae, second_ssd),
        output_size=(8, 8),
        feature_mask=(True, False),
    )

    assert torch.equal(first, second)


def test_scalar_depth_probe_rejects_invalid_feature_mask():
    contract = DepthFeatureContract(names=("vae_latent",), channels=(4,))
    head = ScalarDepthProbe(contract, width=8, upsample_stages=0)
    features = (torch.randn(1, 4, 8, 8),)

    with pytest.raises(ValueError, match="feature_mask length"):
        head(features, output_size=(8, 8), feature_mask=())
    with pytest.raises(ValueError, match="boolean"):
        head(features, output_size=(8, 8), feature_mask=(1,))


def test_depth_probe_gate_requires_all_metrics():
    passing = {"mae": 0.04, "pearson": 0.96, "spearman": 0.97}

    assert depth_probe_gate_passed(passing)
    assert not depth_probe_gate_passed({**passing, "mae": 0.06})
    assert not depth_probe_gate_passed({**passing, "pearson": 0.94})
    assert not depth_probe_gate_passed({**passing, "spearman": 0.94})


def test_feature_contribution_gate_requires_both_correlations_and_mae_guardrail():
    vae = {"mae": 0.20, "pearson": 0.30, "spearman": 0.25}
    passing = {"mae": 0.19, "pearson": 0.36, "spearman": 0.31}

    assert depth_feature_contribution_gate_passed(vae, passing)
    assert not depth_feature_contribution_gate_passed(
        vae,
        {**passing, "pearson": 0.34},
    )
    assert not depth_feature_contribution_gate_passed(
        vae,
        {**passing, "spearman": 0.29},
    )
    assert not depth_feature_contribution_gate_passed(
        vae,
        {**passing, "mae": 0.22},
    )


def test_scalar_depth_metrics_are_exact_for_matching_predictions():
    target = torch.linspace(0, 1, 64).view(1, 1, 8, 8)

    metrics = evaluate_scalar_depth_predictions(target, target)

    assert metrics["mae"] == 0.0
    assert metrics["pearson"] > 0.999
    assert metrics["spearman"] > 0.999


def test_scalar_depth_probe_can_fit_a_direct_synthetic_feature():
    torch.manual_seed(7)
    target = torch.rand(2, 1, 8, 8) * 0.8 + 0.1
    contract = DepthFeatureContract(names=("direct",), channels=(1,))
    head = ScalarDepthProbe(contract, width=8, upsample_stages=0)
    optimizer = torch.optim.Adam(head.parameters(), lr=3e-2)

    with torch.no_grad():
        initial = F.l1_loss(head((target,), output_size=(8, 8)), target)
    for _ in range(120):
        prediction = head((target,), output_size=(8, 8))
        loss = F.l1_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final = F.l1_loss(head((target,), output_size=(8, 8)), target)

    assert final < initial * 0.25


def test_probe_cli_reads_declared_train_names_in_manifest_order(tmp_path: Path):
    module = load_probe_script()
    (tmp_path / "train_val_split.json").write_text(
        json.dumps(
            {
                "train": {"depth": ["second.png", "first.png", "third.png"]},
                "val": {"depth": ["held-out.png"]},
            }
        ),
        encoding="utf-8",
    )

    names = module.declared_depth_train_names(tmp_path, num_samples=2)

    assert names == ["second.png", "first.png"]


def test_probe_cli_writes_strict_json_report(tmp_path: Path):
    module = load_probe_script()
    output = tmp_path / "result.json"
    payload = {"gate_passed": True, "metrics": {"mae": 0.01}}

    module.write_json_report(output, payload)

    assert json.loads(output.read_text(encoding="utf-8")) == payload
    with pytest.raises(ValueError):
        module.write_json_report(output, {"loss": float("nan")})


def test_feature_comparison_cli_selects_disjoint_manifest_splits(tmp_path: Path):
    module = load_feature_comparison_script()
    manifest = {
        "train": {"depth": ["train-a.png", "train-b.png"]},
        "val": {"depth": ["val-a.png", "val-b.png"]},
    }
    (tmp_path / "train_val_split.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    train, val = module.declared_depth_split_names(
        tmp_path,
        train_samples=2,
        val_samples=1,
    )

    assert train == ["train-a.png", "train-b.png"]
    assert val == ["val-a.png"]
    manifest["val"]["depth"][0] = "train-a.png"
    (tmp_path / "train_val_split.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlap"):
        module.declared_depth_split_names(
            tmp_path,
            train_samples=2,
            val_samples=1,
        )


def test_feature_comparison_cli_reuses_schedule_and_state_checksums():
    module = load_feature_comparison_script()
    first = module.build_batch_schedule(
        num_samples=5,
        batch_size=3,
        steps=4,
        seed=9,
    )
    second = module.build_batch_schedule(
        num_samples=5,
        batch_size=3,
        steps=4,
        seed=9,
    )
    state = {"weight": torch.arange(6).view(2, 3).float()}

    assert torch.equal(first, second)
    assert first.shape == (4, 3)
    assert module.tensor_checksum(first) == module.tensor_checksum(second)
    assert module.state_dict_checksum(state) == module.state_dict_checksum(state)
    changed = {"weight": state["weight"].clone()}
    changed["weight"][0, 0] = -1
    assert module.state_dict_checksum(state) != module.state_dict_checksum(changed)


def test_stage8_split_is_deterministic_disjoint_and_checksumed(tmp_path: Path):
    module = load_feature_comparison_script()
    (tmp_path / "train_val_split.json").write_text(
        json.dumps(
            {
                "train": {"depth": [f"train-{i}.png" for i in range(8)]},
                "val": {"depth": [f"val-{i}.png" for i in range(3)]},
            }
        ),
        encoding="utf-8",
    )

    first = module.declared_depth_stage8_split_names(
        tmp_path,
        train_samples=4,
        dev_samples=2,
        val_samples=2,
        split_seed=17,
    )
    second = module.declared_depth_stage8_split_names(
        tmp_path,
        train_samples=4,
        dev_samples=2,
        val_samples=2,
        split_seed=17,
    )

    assert first == second
    fit, dev, official = first
    assert not set(fit) & set(dev)
    assert not set(fit) & set(official)
    assert not set(dev) & set(official)
    assert module.split_names_checksum(*first) == module.split_names_checksum(*second)


def test_stage8_dev_checkpoint_selection_ignores_official_validation():
    module = load_feature_comparison_script()
    history = [
        {
            "step": 10,
            "dev": {"mae": 0.30, "pearson": 0.60, "spearman": 0.60},
        },
        {
            "step": 20,
            "dev": {"mae": 0.20, "pearson": 0.70, "spearman": 0.65},
        },
    ]
    assert module.select_dev_checkpoint(history)["step"] == 20
    history[0]["official_val"] = {"mae": 0.0, "pearson": 1.0, "spearman": 1.0}
    history[1]["official_val"] = {"mae": 1.0, "pearson": -1.0, "spearman": -1.0}
    assert module.select_dev_checkpoint(history)["step"] == 20


def test_stage8_multiseed_gate_requires_majority_and_dev_val_agreement():
    module = load_feature_comparison_script()
    per_seed = [
        {
            "delta": {"mae": -0.02, "pearson": 0.10, "spearman": 0.08},
            "dev_delta": {"pearson": 0.05, "spearman": 0.06},
        },
        {
            "delta": {"mae": -0.01, "pearson": 0.08, "spearman": 0.07},
            "dev_delta": {"pearson": 0.02, "spearman": 0.04},
        },
        {
            "delta": {"mae": 0.00, "pearson": -0.01, "spearman": -0.02},
            "dev_delta": {"pearson": -0.01, "spearman": -0.02},
        },
    ]
    aggregate = {"mae": -0.01, "pearson": 0.0567, "spearman": 0.0433}
    assert not module.stage8_gate_passed(
        aggregate,
        per_seed,
        min_correlation_delta=0.05,
        max_mae_regression=0.01,
        required_positive_seeds=2,
    )
    aggregate["spearman"] = 0.06
    assert module.stage8_gate_passed(
        aggregate,
        per_seed,
        min_correlation_delta=0.05,
        max_mae_regression=0.01,
        required_positive_seeds=2,
    )


def test_stage8_aggregates_seed_metrics_with_population_std():
    module = load_feature_comparison_script()
    aggregate = module.aggregate_seed_metrics(
        [
            {"mae": 0.1, "rmse": 0.2, "pearson": 0.3, "spearman": 0.4},
            {"mae": 0.3, "rmse": 0.4, "pearson": 0.5, "spearman": 0.6},
        ]
    )

    assert aggregate["mae"]["mean"] == pytest.approx(0.2)
    assert aggregate["mae"]["std"] == pytest.approx(0.1)
    assert aggregate["pearson"]["mean"] == pytest.approx(0.4)
    assert aggregate["pearson"]["std"] == pytest.approx(0.1)


def test_stage8_parser_supports_full_dataset_split_defaults():
    module = load_feature_comparison_script()
    assert module._parse_seeds("42, 43,44") == [42, 43, 44]
    with pytest.raises(ValueError, match="unique"):
        module._parse_seeds("42,42")


def test_depth_probe_artifact_round_trip_and_checksum(tmp_path: Path):
    contract = DepthFeatureContract(names=("fine", "coarse"), channels=(4, 8))
    head = ScalarDepthProbe(contract, width=8, upsample_stages=1)
    artifact_dir = save_depth_probe_artifact(
        tmp_path / "probe",
        head_state_dict=head.state_dict(),
        feature_contract=contract.to_dict(),
        feature_mask=(True, True),
        model="segmind/SSD-1B",
        prompt="relative depth",
        colormap="viridis",
        resolution=128,
        timestep=0,
        width=8,
        upsample_stages=1,
        selected_step=25,
        metrics={"dev": {"pearson": 0.7}},
        splits={"official_val_names": ["held-out.png"]},
        training={"seed": 42},
    )

    artifact = load_depth_probe_artifact(artifact_dir)

    assert (artifact_dir / DEPTH_PROBE_CONFIG_NAME).is_file()
    assert (artifact_dir / DEPTH_PROBE_WEIGHTS_NAME).is_file()
    assert DepthFeatureContract.from_dict(artifact.feature_contract) == contract
    assert artifact.feature_mask == (True, True)
    assert artifact.selected_step == 25
    assert artifact.metrics["dev"]["pearson"] == 0.7
    assert not artifact.legacy
    assert set(artifact.head_state_dict) == set(head.state_dict())

    weights_path = artifact_dir / DEPTH_PROBE_WEIGHTS_NAME
    weights_path.write_bytes(weights_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_depth_probe_artifact(artifact_dir)


def test_depth_probe_artifact_loads_validated_legacy_checkpoint(tmp_path: Path):
    contract = DepthFeatureContract(names=("fine",), channels=(4,))
    head = ScalarDepthProbe(contract, width=8, upsample_stages=0)
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "feature_contract": contract.to_dict(),
            "feature_mask": (True,),
            "model": "segmind/SSD-1B",
            "prompt": "relative depth",
            "colormap": "viridis",
            "resolution": 128,
            "timestep": 0,
            "width": 8,
            "upsample_stages": 0,
            "selected_step": 10,
            "official_val_names": ["held-out.png"],
        },
        checkpoint,
    )

    artifact = load_depth_probe_artifact(checkpoint)

    assert artifact.legacy
    assert artifact.format_version == 0
    assert artifact.splits["official_val_names"] == ["held-out.png"]


def test_frozen_depth_estimator_restores_original_image_size(tmp_path: Path):
    generator = FakeGenerator()
    generator.model_name = "segmind/SSD-1B"
    extractor = FrozenSDXLDepthFeatureExtractor(generator)
    contract, _ = extractor.extract(torch.zeros(1, 3, 16, 16), prompt="relative depth")
    head = ScalarDepthProbe(contract, width=8, upsample_stages=1)
    artifact_dir = save_depth_probe_artifact(
        tmp_path / "probe",
        head_state_dict=head.state_dict(),
        feature_contract=contract.to_dict(),
        feature_mask=(True,) * len(contract.names),
        model=generator.model_name,
        prompt="relative depth",
        colormap="viridis",
        resolution=16,
        timestep=0,
        width=8,
        upsample_stages=1,
        selected_step=20,
    )
    estimator = FrozenDepthEstimator(generator, load_depth_probe_artifact(artifact_dir))

    result = estimator.predict(torch.zeros(2, 3, 20, 12))

    assert result["relative_depth"].shape == (2, 1, 20, 12)
    assert result["visualization"].shape == (2, 3, 20, 12)
    assert result["metadata"]["resolution"] == 16
    assert not result["metadata"]["legacy_checkpoint"]


def test_production_refit_scales_steps_by_sample_exposure():
    module = load_production_refit_script()

    steps = module.scale_refit_steps(
        selected_steps=250,
        selection_train_count=128,
        selection_batch_size=8,
        refit_train_count=2181,
        refit_batch_size=8,
    )

    assert steps == 4260
    with pytest.raises(ValueError, match="positive integers"):
        module.scale_refit_steps(
            selected_steps=0,
            selection_train_count=128,
            selection_batch_size=8,
            refit_train_count=2181,
            refit_batch_size=8,
        )


def test_visualizer_resolves_production_final_artifact(tmp_path: Path):
    module = load_probe_visualization_script()
    artifact = tmp_path / "seed43_vae_ssd_final"
    artifact.mkdir()

    assert module._resolve_checkpoint(tmp_path, 43) == artifact
