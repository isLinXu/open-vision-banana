"""Focused tests for production SSD/SDXL source conditioning."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from _paths import ARCHIVE_SCRIPTS
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from open_vision_banana.conditioning import (
    SourceConditioningConfig,
    build_source_conditioned_input,
    calibrate_source_conditioned_output,
    expand_unet_conv_in,
    read_source_conditioning_config,
    sample_source_conditioned_timesteps,
    scheduler_prediction_type,
    set_scheduler_prediction_type,
    validate_scheduler_prediction_type,
)
from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.data.depth_encoding import audit_depth_target_encoding
from open_vision_banana.evaluation.relative_depth import evaluate_relative_depth_rgb
from open_vision_banana.inference.generator import (
    DIFFUSERS_LORA_WEIGHT_NAME as GENERATOR_LORA_WEIGHT_NAME,
)
from open_vision_banana.inference.generator import (
    BaseGenerator,
)
from open_vision_banana.inference.pipeline import VisionBanana
from open_vision_banana.spatial_adapter import (
    SPATIAL_ADAPTER_STATE_NAME,
    T2I_ADAPTER_XL,
    SourceSpatialAdapter,
)
from open_vision_banana.task_prompts import SEMANTIC_SEGMENTATION_PROMPT
from open_vision_banana.training.checkpoint import (
    DIFFUSERS_LORA_WEIGHT_NAME,
    CheckpointManager,
)
from open_vision_banana.training.structure_loss import (
    decoded_x0_scalar_geometry_loss,
    decoded_x0_structure_loss,
    soft_colormap_position,
)
from open_vision_banana.training.tuner import (
    InstructionTuner,
    VisionTaskDataset,
    euler_v_prediction_target,
)


class FakeUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(4, 6, 3, padding=1)
        self.config = SimpleNamespace(in_channels=4)

    def register_to_config(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self.config, key, value)


class FakeScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.arange(4)
        self.sigmas = torch.tensor([0.0, 0.5, 1.0, 2.0, 0.0])
        self.config = SimpleNamespace(
            prediction_type="epsilon",
            num_train_timesteps=4,
        )
        self.noised_original = None

    def register_to_config(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self.config, key, value)

    def add_noise(self, original, noise, timesteps):
        self.noised_original = original.detach().clone()
        return original + noise

    def set_timesteps(self, steps, device=None):
        self.timesteps = torch.linspace(3, 0, steps, device=device)
        self.sigmas = torch.linspace(2, 0, steps + 1, device=device)

    @staticmethod
    def index_for_timestep(timestep, schedule):
        return int(torch.nonzero(schedule == timestep)[0])


class MockAdapter:
    def __init__(self) -> None:
        self.linear = nn.Linear(2, 2)

    def save_pretrained(self, path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.linear.state_dict(), Path(path) / "adapter_model.pt")


def test_expansion_preserves_four_channel_forward_and_is_idempotent():
    torch.manual_seed(9)
    unet = FakeUNet()
    target = torch.randn(2, 4, 8, 8)
    expected = unet.conv_in(target)

    expanded = expand_unet_conv_in(unet, condition_channels=4)
    actual = expanded(torch.cat((target, torch.randn_like(target)), dim=1))
    second = expand_unet_conv_in(unet, condition_channels=4)

    assert expanded is second
    assert expanded.in_channels == 8
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(expanded.weight[:, 4:]) == 0


def test_conditioned_input_noises_target_and_keeps_source_clean():
    scheduler = FakeScheduler()
    target = torch.full((1, 4, 2, 2), 2.0)
    source = torch.full((1, 4, 2, 2), 7.0)
    noise = torch.full_like(target, 3.0)

    model_input, noisy, sigma = build_source_conditioned_input(
        scheduler,
        target,
        source,
        noise,
        torch.tensor([1]),
    )

    assert torch.equal(scheduler.noised_original, target)
    assert torch.equal(noisy, target + noise)
    assert torch.equal(model_input[:, 4:], source)
    assert sigma.shape == (1, 1, 1, 1)


def test_source_timestep_sampling_uses_deployed_schedule():
    scheduler = FakeScheduler()
    torch.manual_seed(3)

    timesteps = sample_source_conditioned_timesteps(
        scheduler,
        batch_size=12,
        device=torch.device("cpu"),
        mode="inference",
        inference_steps=4,
    )

    assert set(timesteps.tolist()).issubset(set(scheduler.timesteps.tolist()))
    assert scheduler.timesteps.numel() == 4


def test_output_calibration_projects_to_declared_colormap():
    torch.manual_seed(4)
    output = torch.rand(2, 3, 8, 8)
    config = SourceConditioningConfig(
        target_colormap="viridis",
        target_channel_mean=(0.30, 0.57, 0.43),
        target_channel_std=(0.19, 0.25, 0.13),
        output_calibration="moments_and_project",
    )

    calibrated = calibrate_source_conditioned_output(output, config)
    mapper = DepthRGBMapper("viridis")
    projected = mapper.normalized_to_rgb(mapper.rgb_to_normalized(calibrated))

    assert torch.allclose(calibrated, projected, atol=1e-6)
    assert calibrated.min() >= 0 and calibrated.max() <= 1


def test_checkpoint_round_trip_restores_config_and_conv(tmp_path: Path):
    manager = CheckpointManager(str(tmp_path))
    adapter = MockAdapter()
    optimizer = torch.optim.SGD(adapter.linear.parameters(), lr=0.1)
    conv = nn.Conv2d(8, 4, 3, padding=1)
    config = SourceConditioningConfig(
        x0_loss_weight=2.0,
        scalar_geometry_loss_weight=3.0,
        scalar_geometry_gradient_weight=0.5,
        scalar_geometry_ordinal_weight=2.0,
        target_colormap="viridis",
    )
    expected_weight = conv.weight.detach().clone()

    checkpoint = manager.save(
        adapter,
        optimizer,
        step=3,
        epoch=0,
        loss=0.5,
        lr=0.1,
        conditioning_module=conv,
        conditioning_config=config,
    )
    with torch.no_grad():
        conv.weight.zero_()
    manager.load(
        adapter,
        conditioning_module=conv,
        conditioning_config=config,
        step=3,
    )

    assert read_source_conditioning_config(checkpoint) == config
    assert torch.equal(conv.weight, expected_weight)


def test_config_requires_colormap_for_scalar_geometry_loss():
    config = SourceConditioningConfig(scalar_geometry_loss_weight=1.0)

    with pytest.raises(ValueError, match="target_colormap"):
        config.validate()


def test_checkpoint_resume_allows_updated_loss_weights(tmp_path: Path):
    manager = CheckpointManager(str(tmp_path))
    adapter = MockAdapter()
    optimizer = torch.optim.SGD(adapter.linear.parameters(), lr=0.1)
    conv = nn.Conv2d(8, 4, 3, padding=1)
    manager.save(
        adapter,
        optimizer,
        step=1,
        epoch=0,
        loss=0.5,
        lr=0.1,
        conditioning_module=conv,
        conditioning_config=SourceConditioningConfig(x0_loss_weight=1.0),
    )

    manager.load(
        adapter,
        step=1,
        conditioning_module=conv,
        conditioning_config=SourceConditioningConfig(
            x0_loss_weight=10.0,
            pixel_loss_weight=1.0,
            structure_loss_weight=0.5,
            timestep_sampling="inference",
        ),
    )


def test_checkpoint_round_trip_restores_spatial_adapter(tmp_path: Path):
    manager = CheckpointManager(str(tmp_path))
    lora = MockAdapter()
    optimizer = torch.optim.SGD(lora.linear.parameters(), lr=0.1)
    conv = nn.Conv2d(8, 4, 3, padding=1)
    spatial_adapter = nn.Conv2d(3, 4, 1)
    config = SourceConditioningConfig(spatial_adapter_type=T2I_ADAPTER_XL)
    expected_weight = spatial_adapter.weight.detach().clone()

    checkpoint = manager.save(
        lora,
        optimizer,
        step=2,
        epoch=0,
        loss=0.4,
        lr=0.1,
        conditioning_module=conv,
        conditioning_config=config,
        spatial_adapter=spatial_adapter,
    )
    with torch.no_grad():
        spatial_adapter.weight.zero_()
    manager.load(
        lora,
        step=2,
        conditioning_module=conv,
        conditioning_config=config,
        spatial_adapter=spatial_adapter,
    )

    assert (checkpoint / SPATIAL_ADAPTER_STATE_NAME).is_file()
    assert torch.equal(spatial_adapter.weight, expected_weight)


def test_structure_loss_matches_values_and_gradients():
    prediction = torch.zeros(1, 3, 16, 16, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[:, :, 4:12, 4:12] = 1.0

    loss = decoded_x0_structure_loss(
        prediction,
        target,
        gradient_weight=2.0,
    )
    loss.backward()

    assert loss > 0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad) > 0


def test_soft_colormap_position_recovers_viridis_curve_and_gradients():
    expected = torch.linspace(0, 1, 25).view(1, 1, 5, 5)
    rgb = DepthRGBMapper("viridis").normalized_to_rgb(expected).detach().requires_grad_(True)

    actual = soft_colormap_position(
        rgb,
        colormap="viridis",
        temperature=1e-4,
    )
    actual.square().mean().backward()

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=2e-2)
    assert rgb.grad is not None
    assert torch.isfinite(rgb.grad).all()
    assert torch.count_nonzero(rgb.grad) > 0


def test_scalar_geometry_loss_prefers_correct_depth_order():
    normalized = torch.linspace(0, 1, 64).view(1, 1, 8, 8)
    mapper = DepthRGBMapper("viridis")
    target = mapper.normalized_to_rgb(normalized) * 2.0 - 1.0
    correct = target.detach().clone().requires_grad_(True)
    reversed_prediction = (
        mapper.normalized_to_rgb(1.0 - normalized).detach().mul(2.0).sub(1.0).requires_grad_(True)
    )

    correct_loss = decoded_x0_scalar_geometry_loss(
        correct,
        target,
        colormap="viridis",
    )
    reversed_loss = decoded_x0_scalar_geometry_loss(
        reversed_prediction,
        target,
        colormap="viridis",
    )
    reversed_loss.backward()

    assert correct_loss < reversed_loss
    assert reversed_prediction.grad is not None
    assert torch.isfinite(reversed_prediction.grad).all()
    assert torch.count_nonzero(reversed_prediction.grad) > 0


def test_scalar_geometry_loss_rejects_invalid_projection_temperature():
    image = torch.zeros(1, 3, 4, 4)

    with pytest.raises(ValueError, match="projection_temperature"):
        decoded_x0_scalar_geometry_loss(
            image,
            image,
            colormap="viridis",
            projection_temperature=0.0,
        )


def test_spatial_adapter_starts_as_zero_residual_and_receives_gradients():
    adapter = SourceSpatialAdapter(
        channels=(4, 8, 16, 16),
        downscale_factor=16,
        num_res_blocks=1,
    )
    source = torch.rand(1, 3, 128, 128)

    residuals = adapter(source)
    sum(residual.sum() for residual in residuals).backward()

    assert [tuple(residual.shape) for residual in residuals] == [
        (1, 4, 8, 8),
        (1, 8, 8, 8),
        (1, 16, 4, 4),
        (1, 16, 4, 4),
    ]
    assert all(torch.count_nonzero(residual) == 0 for residual in residuals)
    assert all(projection.weight.grad is not None for projection in adapter.zero_convs)


def test_tuner_source_path_noises_target_and_builds_eight_channels():
    tuner = InstructionTuner.__new__(InstructionTuner)
    tuner.peft_model = nn.Conv2d(1, 1, 1)
    tuner.base_model = SimpleNamespace(_t2i_pipe=SimpleNamespace(scheduler=FakeScheduler()))
    tuner.model_config = SimpleNamespace(family="sdxl")
    tuner.use_source_conditioning = True
    tuner.x0_loss_weight = 0.0
    tuner.pixel_loss_weight = 0.0
    tuner.source_timestep_sampling = "uniform"
    tuner.source_training_inference_steps = 4
    tuner.use_min_snr = False
    captured = {}

    def encode_latents(pixels):
        return torch.cat((pixels, pixels[:, :1]), dim=1)

    def encode_text(prompts, device):
        batch = len(prompts)
        return {
            "prompt_embeds": torch.zeros(batch, 1, 2, device=device),
            "pooled_prompt_embeds": torch.zeros(batch, 2, device=device),
            "attention_mask": None,
        }

    def forward_registry(model_input, *args, **kwargs):
        captured["model_input"] = model_input.detach().clone()
        return torch.zeros_like(model_input[:, :4])

    tuner._encode_latents = encode_latents
    tuner._encode_text = encode_text
    tuner._forward_registry = forward_registry
    batch = {
        "target_rgb": torch.full((1, 3, 8, 8), 0.75),
        "target_rgb_range": "0_1",
        "image": torch.full((1, 3, 8, 8), -0.75),
        "image_range": "-1_1",
        "prompt": ["depth"],
    }

    loss = tuner._training_step_sdxl(batch)
    scheduler = tuner.base_model._t2i_pipe.scheduler
    encoded_target = encode_latents(batch["target_rgb"] * 2.0 - 1.0)
    encoded_source = encode_latents(batch["image"])

    assert loss.ndim == 0
    assert torch.equal(scheduler.noised_original, encoded_target)
    assert captured["model_input"].shape[1] == 8
    assert torch.equal(captured["model_input"][:, 4:], encoded_source)
    assert tuner._last_step_metrics["train/raw_epsilon_mse"] >= 0
    assert tuner._last_step_metrics["train/raw_x0_mse"] >= 0


def test_tuner_rejects_paired_four_channel_sdxl_training():
    tuner = InstructionTuner.__new__(InstructionTuner)
    tuner.peft_model = nn.Conv2d(1, 1, 1)
    scheduler = FakeScheduler()
    tuner.base_model = SimpleNamespace(_t2i_pipe=SimpleNamespace(scheduler=scheduler))
    tuner.model_config = SimpleNamespace(family="sdxl")
    tuner.use_source_conditioning = False
    tuner.prediction_type = "epsilon"
    tuner.use_min_snr = False
    tuner._encode_latents = lambda pixels: torch.cat((pixels, pixels[:, :1]), dim=1)
    batch = {
        "target_rgb": torch.full((1, 3, 8, 8), 0.75),
        "target_rgb_range": "0_1",
        "image": torch.full((1, 3, 8, 8), -0.75),
        "image_range": "-1_1",
        "prompt": ["depth"],
    }

    with pytest.raises(RuntimeError, match="requires use_source_conditioning=True"):
        tuner._training_step_sdxl(batch)

    assert scheduler.noised_original is None


def test_source_conditioned_generation_uses_zero_source_channels():
    tuner = InstructionTuner.__new__(InstructionTuner)
    tuner.peft_model = nn.Conv2d(1, 1, 1)
    tuner.base_model = SimpleNamespace(_t2i_pipe=SimpleNamespace(scheduler=FakeScheduler()))
    tuner.model_config = SimpleNamespace(family="sdxl")
    tuner.use_source_conditioning = True
    tuner.prediction_type = "epsilon"
    tuner.x0_loss_weight = 0.0
    tuner.pixel_loss_weight = 0.0
    tuner.source_timestep_sampling = "uniform"
    tuner.source_training_inference_steps = 4
    tuner.use_min_snr = False
    captured = {}
    tuner._encode_latents = lambda pixels: torch.cat((pixels, pixels[:, :1]), dim=1)
    tuner._encode_text = lambda prompts, device: {
        "prompt_embeds": torch.zeros(len(prompts), 1, 2, device=device),
        "pooled_prompt_embeds": torch.zeros(len(prompts), 2, device=device),
        "attention_mask": None,
    }

    def forward_registry(model_input, *args, **kwargs):
        captured["model_input"] = model_input.detach().clone()
        return torch.zeros_like(model_input[:, :4])

    tuner._forward_registry = forward_registry
    loss = tuner._training_step_sdxl(
        {
            "target_rgb": torch.full((1, 3, 8, 8), 0.75),
            "target_rgb_range": "0_1",
            "image": None,
            "prompt": ["a photograph"],
        }
    )

    assert loss.ndim == 0
    assert captured["model_input"].shape[1] == 8
    assert torch.count_nonzero(captured["model_input"][:, 4:]) == 0


def test_tuner_pixel_loss_backpropagates_through_frozen_vae():
    tuner = InstructionTuner.__new__(InstructionTuner)
    tuner.peft_model = nn.Conv2d(8, 4, 1, bias=False)
    vae = nn.Conv2d(4, 3, 1, bias=False)
    vae.requires_grad_(False)
    scheduler = FakeScheduler()
    tuner.base_model = SimpleNamespace(_t2i_pipe=SimpleNamespace(scheduler=scheduler, vae=vae))
    tuner.model_config = SimpleNamespace(family="sdxl")
    tuner.use_source_conditioning = True
    tuner.x0_loss_weight = 0.0
    tuner.pixel_loss_weight = 1.0
    tuner.source_timestep_sampling = "uniform"
    tuner.source_training_inference_steps = 4
    tuner.use_min_snr = False

    tuner._encode_latents = lambda pixels: torch.cat(
        (pixels, pixels[:, :1]),
        dim=1,
    )
    tuner._encode_text = lambda prompts, device: {
        "prompt_embeds": torch.zeros(len(prompts), 1, 2, device=device),
        "pooled_prompt_embeds": torch.zeros(len(prompts), 2, device=device),
        "attention_mask": None,
    }
    tuner._forward_registry = lambda model_input, *args, **kwargs: tuner.peft_model(model_input)
    batch = {
        "target_rgb": torch.full((1, 3, 8, 8), 0.75),
        "target_rgb_range": "0_1",
        "image": torch.full((1, 3, 8, 8), -0.75),
        "image_range": "-1_1",
        "prompt": ["depth"],
    }

    with patch(
        "open_vision_banana.training.tuner.registry_vae_decode",
        side_effect=lambda live_vae, latent, config: live_vae(latent),
    ):
        loss = tuner._training_step_sdxl(batch)
    loss.backward()

    assert tuner.peft_model.weight.grad is not None
    assert torch.isfinite(tuner.peft_model.weight.grad).all()
    assert all(parameter.grad is None for parameter in vae.parameters())


def test_tuner_scalar_geometry_loss_backpropagates_through_frozen_vae():
    tuner = InstructionTuner.__new__(InstructionTuner)
    tuner.peft_model = nn.Conv2d(8, 4, 1, bias=False)
    vae = nn.Conv2d(4, 3, 1, bias=False)
    vae.requires_grad_(False)
    scheduler = FakeScheduler()
    tuner.base_model = SimpleNamespace(_t2i_pipe=SimpleNamespace(scheduler=scheduler, vae=vae))
    tuner.model_config = SimpleNamespace(family="sdxl")
    tuner.use_source_conditioning = True
    tuner.x0_loss_weight = 0.0
    tuner.pixel_loss_weight = 0.0
    tuner.structure_loss_weight = 0.0
    tuner.scalar_geometry_loss_weight = 1.0
    tuner.scalar_geometry_gradient_weight = 1.0
    tuner.scalar_geometry_ordinal_weight = 1.0
    tuner.source_target_colormap = "viridis"
    tuner.source_timestep_sampling = "uniform"
    tuner.source_training_inference_steps = 4
    tuner.use_min_snr = False

    tuner._encode_latents = lambda pixels: torch.cat(
        (pixels, pixels[:, :1]),
        dim=1,
    )
    tuner._encode_text = lambda prompts, device: {
        "prompt_embeds": torch.zeros(len(prompts), 1, 2, device=device),
        "pooled_prompt_embeds": torch.zeros(len(prompts), 2, device=device),
        "attention_mask": None,
    }
    tuner._forward_registry = lambda model_input, *args, **kwargs: tuner.peft_model(model_input)
    normalized = torch.linspace(0, 1, 64).view(1, 1, 8, 8)
    target = DepthRGBMapper("viridis").normalized_to_rgb(normalized)
    batch = {
        "target_rgb": target,
        "target_rgb_range": "0_1",
        "image": torch.full((1, 3, 8, 8), -0.75),
        "image_range": "-1_1",
        "prompt": ["depth"],
    }

    with patch(
        "open_vision_banana.training.tuner.registry_vae_decode",
        side_effect=lambda live_vae, latent, config: live_vae(latent),
    ):
        loss = tuner._training_step_sdxl(batch)
    loss.backward()

    assert tuner.peft_model.weight.grad is not None
    assert torch.isfinite(tuner.peft_model.weight.grad).all()
    assert torch.count_nonzero(tuner.peft_model.weight.grad) > 0
    assert all(parameter.grad is None for parameter in vae.parameters())


def test_generator_dispatches_configured_image_to_custom_path():
    with patch.object(BaseGenerator, "_load_model"):
        generator = BaseGenerator(model_name="dummy", dtype=torch.float32, device="cpu")
    generator.source_conditioning_config = SourceConditioningConfig()
    generator._generate_source_conditioned = MagicMock(return_value=torch.zeros(1, 3, 8, 8))

    output = generator.generate(
        "depth",
        image=Image.new("RGB", (8, 8)),
        num_inference_steps=4,
    )

    assert output.shape == (1, 3, 8, 8)
    generator._generate_source_conditioned.assert_called_once()


def test_generator_dispatches_source_checkpoint_t2i_with_zero_condition():
    with patch.object(BaseGenerator, "_load_model"):
        generator = BaseGenerator(model_name="dummy", dtype=torch.float32, device="cpu")
    generator.source_conditioning_config = SourceConditioningConfig()
    generator._generate_source_conditioned = MagicMock(return_value=torch.zeros(1, 3, 8, 8))

    output = generator.generate(
        "a photograph",
        image=None,
        height=16,
        width=24,
        num_inference_steps=4,
    )

    assert output.shape == (1, 3, 8, 8)
    kwargs = generator._generate_source_conditioned.call_args.kwargs
    assert kwargs["image"] is None
    assert kwargs["height"] == 16
    assert kwargs["width"] == 24
    assert kwargs["num_inference_steps"] == 4


def test_source_conditioned_t2i_appends_zero_latent_channels():
    captured = {}

    class TinyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))

        def forward(self, model_input, *args, **kwargs):
            captured["model_input"] = model_input.detach().clone()
            return (torch.zeros_like(model_input[:, :4]),)

    class TinyVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))

    class TinyScheduler:
        init_noise_sigma = 1.0

        def __init__(self):
            self.config = SimpleNamespace(prediction_type="epsilon")
            self.timesteps = torch.tensor([1.0])

        @classmethod
        def from_config(cls, config):
            return cls()

        def register_to_config(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self.config, key, value)

        def set_timesteps(self, steps, device):
            self.timesteps = torch.tensor([1.0], device=device)

        def scale_model_input(self, latents, timestep):
            return latents

        def step(self, predicted_noise, timestep, latents, return_dict=False):
            return (latents,)

    generator = BaseGenerator.__new__(BaseGenerator)
    generator.source_conditioning_config = SourceConditioningConfig()
    generator.source_spatial_adapter = None
    generator._t2i_pipe = SimpleNamespace(
        unet=TinyUNet(),
        vae=TinyVAE(),
        scheduler=TinyScheduler(),
    )
    generator.model_config = SimpleNamespace(vae_downsample_factor=8)
    generator.encode_prompts = lambda prompts: {
        "prompt_embeds": torch.zeros(len(prompts), 1, 2),
        "pooled_prompt_embeds": torch.zeros(len(prompts), 2),
    }

    with (
        patch(
            "open_vision_banana.model_registry.build_added_cond_kwargs",
            return_value={},
        ),
        patch(
            "open_vision_banana.model_registry.vae_decode",
            side_effect=lambda vae, latents, config: torch.zeros(
                latents.shape[0],
                3,
                latents.shape[-2] * 8,
                latents.shape[-1] * 8,
            ),
        ),
    ):
        output = generator._generate_source_conditioned(
            prompt="a photograph",
            image=None,
            num_inference_steps=1,
            guidance_scale=0.0,
            strength=0.85,
            height=16,
            width=24,
            seed=7,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
        )

    assert output.shape == (1, 3, 16, 24)
    assert captured["model_input"].shape == (1, 8, 2, 3)
    assert torch.count_nonzero(captured["model_input"][:, 4:]) == 0


def test_pipeline_uses_checkpoint_depth_colormap_and_prompt():
    config = SourceConditioningConfig(
        target_colormap="viridis",
        recommended_prompt="Generate relative viridis depth.",
    )
    generator = MagicMock()
    generator.source_conditioning_config = config
    generator.generate.return_value = torch.full((1, 3, 8, 8), 0.5)
    pipeline = VisionBanana(base_generator=generator)

    result = pipeline.predict(
        torch.zeros(1, 3, 8, 8),
        task="depth",
        seed=3,
    )

    assert pipeline.rgb_mapper.depth_mapper.colormap == "viridis"
    assert result["prompt"] == config.recommended_prompt
    assert generator.generate.call_args.kwargs["prompt"] == config.recommended_prompt


def test_prediction_type_uses_frozendict_safe_registration():
    scheduler = MagicMock()
    scheduler.config = {"prediction_type": "sample"}

    set_scheduler_prediction_type(scheduler, "epsilon")

    scheduler.register_to_config.assert_called_once_with(prediction_type="epsilon")


def test_prediction_type_contract_rejects_scheduler_mismatch():
    scheduler = SimpleNamespace(config={"prediction_type": "sample"})

    assert scheduler_prediction_type(scheduler) == "sample"
    with pytest.raises(RuntimeError, match="prediction_type mismatch"):
        validate_scheduler_prediction_type(scheduler, "epsilon")


def test_manual_lora_merge_supports_conv2d(tmp_path: Path):
    checkpoint = tmp_path / "adapter"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text(
        '{"r": 2, "lora_alpha": 2, "use_rslora": false, "target_modules": ["proj"]}',
        encoding="utf-8",
    )
    factor_a = torch.tensor([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]])
    factor_b = torch.tensor(
        [
            [[[1.0]], [[0.0]]],
            [[[0.0]], [[1.0]]],
            [[[1.0]], [[1.0]]],
        ]
    )
    save_file(
        {
            "base_model.model.proj.lora_A.weight": factor_a,
            "base_model.model.proj.lora_B.weight": factor_b,
        },
        checkpoint / "adapter_model.safetensors",
    )
    pipe = SimpleNamespace(unet=nn.Module())
    pipe.unet.proj = nn.Conv2d(2, 3, 1, bias=False)
    with torch.no_grad():
        pipe.unet.proj.weight.zero_()
    generator = BaseGenerator.__new__(BaseGenerator)

    merged = generator._load_and_merge_lora(pipe, str(checkpoint))

    expected = torch.einsum("or,rihw->oihw", factor_b[:, :, 0, 0], factor_a)
    assert merged == 1
    assert torch.equal(pipe.unet.proj.weight, expected)


def test_dataset_prefers_per_sample_prompt(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "targets").mkdir()
    (tmp_path / "prompts").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "images" / "x.png")
    Image.new("RGB", (8, 8)).save(tmp_path / "targets" / "x.png")
    (tmp_path / "prompts" / "x.txt").write_text(
        "Use the v3 depth encoding.",
        encoding="utf-8",
    )

    dataset = VisionTaskDataset(str(tmp_path), resolution=8, task="depth")

    assert dataset[0]["prompt"] == "Use the v3 depth encoding."


def test_euler_v_target_round_trips_scheduler_formula():
    noisy = torch.full((1, 2, 1, 1), 3.0)
    clean = torch.full_like(noisy, 1.5)
    sigma = torch.full((1, 1, 1, 1), 2.0)
    target = euler_v_prediction_target(noisy, clean, sigma)
    reconstructed = target * (-sigma / (sigma.square() + 1).sqrt()) + noisy / (sigma.square() + 1)
    assert torch.allclose(reconstructed, clean)


def test_semantic_dataset_uses_complete_cityscapes_legend(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "targets").mkdir()
    (tmp_path / "prompts").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "images" / "x.png")
    Image.new("RGB", (8, 8)).save(tmp_path / "targets" / "x.png")
    (tmp_path / "prompts" / "x.txt").write_text(
        "Generate segmentation. " + ", ".join(f"class_{i}=red" for i in range(100)),
        encoding="utf-8",
    )

    dataset = VisionTaskDataset(str(tmp_path), resolution=8, task="semantic")

    assert dataset[0]["prompt"] == SEMANTIC_SEGMENTATION_PROMPT
    assert '"road": <128, 64, 128>' in dataset[0]["prompt"]
    assert '"bicycle": <119, 11, 32>' in dataset[0]["prompt"]


def test_diffusers_lora_companion_uses_component_prefix_and_rslora_metadata(
    tmp_path: Path,
):
    from peft import LoraConfig, TaskType, get_peft_model

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4)

        def forward(self, x, **kwargs):
            return self.proj(x)

    model = get_peft_model(
        TinyModel(),
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["proj"],
            use_rslora=True,
            task_type=TaskType.FEATURE_EXTRACTION,
        ),
    )

    path = CheckpointManager._save_diffusers_lora_weights(
        model,
        tmp_path,
        component="unet",
    )
    state_dict = load_file(path)

    assert path.name == DIFFUSERS_LORA_WEIGHT_NAME
    assert path.name == GENERATOR_LORA_WEIGHT_NAME
    assert state_dict
    assert all(key.startswith("unet.") for key in state_dict)
    assert all("base_model.model" not in key for key in state_dict)
    assert any(key.endswith(".lora_A.weight") for key in state_dict)
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    assert metadata is not None
    assert '"unet.use_rslora": true' in metadata["lora_adapter_metadata"]


def test_depth_encoding_audit_detects_viridis(tmp_path: Path):
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    normalized = torch.linspace(0, 1, 32).view(1, 1, 1, 32).expand(1, 1, 32, 32)
    rgb = DepthRGBMapper("viridis").normalized_to_rgb(normalized)[0]
    array = (rgb.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    Image.fromarray(array).save(target_dir / "sample.png")

    audit = audit_depth_target_encoding(
        target_dir,
        candidates=["cube_edge", "viridis"],
        max_images=1,
        resolution=32,
    )

    assert audit.detected_colormap == "viridis"
    assert audit.is_curve_consistent


def test_depth_encoding_audit_uses_all_allowed_targets_for_channel_stats(
    tmp_path: Path,
):
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    Image.new("RGB", (8, 8), (0, 0, 0)).save(target_dir / "a.png")
    Image.new("RGB", (8, 8), (255, 255, 255)).save(target_dir / "b.png")
    Image.new("RGB", (8, 8), (255, 0, 0)).save(target_dir / "val.png")

    audit = audit_depth_target_encoding(
        target_dir,
        candidates=["viridis"],
        max_images=1,
        resolution=8,
        allowed_names={"a.png", "b.png"},
    )

    assert audit.sample_count == 1
    assert audit.statistics_sample_count == 2
    assert audit.channel_mean == (0.5, 0.5, 0.5)
    assert audit.channel_std == (0.5, 0.5, 0.5)


def test_relative_depth_metrics_use_colormap_position():
    normalized = torch.linspace(0, 1, 16).view(1, 1, 4, 4)
    target = DepthRGBMapper("viridis").normalized_to_rgb(normalized)

    metrics = evaluate_relative_depth_rgb(
        target,
        target,
        colormap="viridis",
    )

    assert metrics["rgb_mae"] == 0.0
    assert metrics["relative_position_mae"] == 0.0
    assert metrics["relative_position_pearson_r"] > 0.999
    assert metrics["relative_position_spearman_r"] > 0.999


def test_relative_depth_spearman_detects_reversed_order():
    normalized = torch.linspace(0, 1, 16).view(1, 1, 4, 4)
    mapper = DepthRGBMapper("viridis")
    target = mapper.normalized_to_rgb(normalized)
    prediction = mapper.normalized_to_rgb(1.0 - normalized)

    metrics = evaluate_relative_depth_rgb(
        prediction,
        target,
        colormap="viridis",
    )

    assert metrics["relative_position_spearman_r"] < -0.999


def test_v3_evaluator_separates_raw_and_calibrated_metrics():
    import importlib.util

    script_path = ARCHIVE_SCRIPTS / "evaluate_v3_relative_depth.py"
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(13)
    raw = torch.rand(1, 3, 8, 8)
    target = DepthRGBMapper("viridis").normalized_to_rgb(torch.rand(1, 1, 8, 8))
    config = SourceConditioningConfig(
        target_colormap="viridis",
        target_channel_mean=(0.33, 0.51, 0.47),
        target_channel_std=(0.29, 0.32, 0.27),
        output_calibration="moments_and_project",
    )

    metrics = module.evaluate_prediction_variants(
        raw,
        target,
        config=config,
        colormap="viridis",
    )

    assert "raw_rgb_mae" in metrics
    assert "calibrated_rgb_mae" in metrics
    assert metrics["raw_prediction_curve_mae"] > 0
    assert metrics["calibrated_prediction_curve_mae"] < 1e-5
