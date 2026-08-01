"""Source-image conditioning shared by SSD/SDXL training and inference.

The source-conditioned path follows the InstructPix2Pix convention: the UNet
receives the noisy target latent and a clean source-image latent concatenated
along the channel dimension.  It is deliberately opt-in so legacy 4-channel
LoRA checkpoints keep their existing img2img behaviour.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

CONDITIONING_CONFIG_NAME = "conditioning_config.json"
CONV_IN_STATE_NAME = "conv_in.pt"
SOURCE_LATENT_CONCAT = "source_latent_concat"

# Inference protocols a checkpoint can declare. Multi-step epsilon is the
# legacy iterative sampler; single-step x0 performs one UNet forward at a
# fixed deployed timestep and decodes the (implied) x0 directly.
MULTI_STEP_EPSILON = "multi_step_epsilon"
SINGLE_STEP_X0 = "single_step_x0"
INFERENCE_PROTOCOLS = (MULTI_STEP_EPSILON, SINGLE_STEP_X0)


@dataclass(frozen=True)
class SourceConditioningConfig:
    """Serializable contract for an expanded source-conditioned UNet."""

    mode: str = SOURCE_LATENT_CONCAT
    in_channels: int = 8
    condition_channels: int = 4
    prediction_type: str = "epsilon"
    x0_loss_weight: float = 0.0
    pixel_loss_weight: float = 0.0
    structure_loss_weight: float = 0.0
    structure_gradient_weight: float = 1.0
    scalar_geometry_loss_weight: float = 0.0
    scalar_geometry_gradient_weight: float = 1.0
    scalar_geometry_ordinal_weight: float = 1.0
    timestep_sampling: str = "uniform"
    training_num_inference_steps: int = 4
    inference_protocol: str = MULTI_STEP_EPSILON
    single_step_timestep_index: int | None = None
    target_colormap: str = "unknown"
    target_channel_mean: tuple[float, float, float] | None = None
    target_channel_std: tuple[float, float, float] | None = None
    output_calibration: str = "none"
    recommended_prompt: str = ""
    spatial_adapter_type: str = "none"
    spatial_adapter_conditioning_scale: float = 1.0
    spatial_adapter_downscale_factor: int = 16
    default_num_inference_steps: int = 4
    version: int = 1

    def validate(self) -> None:
        if self.mode != SOURCE_LATENT_CONCAT:
            raise ValueError(f"Unsupported conditioning mode: {self.mode!r}")
        if self.condition_channels <= 0:
            raise ValueError("condition_channels must be positive")
        if self.in_channels <= self.condition_channels:
            raise ValueError("in_channels must include target and condition channels")
        if self.prediction_type not in {"epsilon", "sample"}:
            raise ValueError(
                "Source-latent concatenation supports epsilon or sample (x0) prediction"
            )
        if self.inference_protocol not in INFERENCE_PROTOCOLS:
            raise ValueError(
                f"Unsupported inference protocol: {self.inference_protocol!r}"
            )
        if self.prediction_type == "sample" and self.inference_protocol != SINGLE_STEP_X0:
            raise ValueError(
                "sample (x0) prediction requires the single_step_x0 inference protocol"
            )
        if self.inference_protocol == SINGLE_STEP_X0:
            if self.single_step_timestep_index is None:
                raise ValueError(
                    "single_step_x0 requires single_step_timestep_index"
                )
            if not 0 <= self.single_step_timestep_index < self.training_num_inference_steps:
                raise ValueError(
                    "single_step_timestep_index must index the deployed schedule "
                    f"[0, {self.training_num_inference_steps})"
                )
        elif self.single_step_timestep_index is not None:
            raise ValueError(
                "single_step_timestep_index is only valid with single_step_x0"
            )
        if self.x0_loss_weight < 0:
            raise ValueError("x0_loss_weight must be non-negative")
        if self.pixel_loss_weight < 0:
            raise ValueError("pixel_loss_weight must be non-negative")
        if self.structure_loss_weight < 0:
            raise ValueError("structure_loss_weight must be non-negative")
        if self.structure_gradient_weight < 0:
            raise ValueError("structure_gradient_weight must be non-negative")
        if self.scalar_geometry_loss_weight < 0:
            raise ValueError("scalar_geometry_loss_weight must be non-negative")
        if self.scalar_geometry_gradient_weight < 0:
            raise ValueError("scalar_geometry_gradient_weight must be non-negative")
        if self.scalar_geometry_ordinal_weight < 0:
            raise ValueError("scalar_geometry_ordinal_weight must be non-negative")
        if self.scalar_geometry_loss_weight > 0 and self.target_colormap == "unknown":
            raise ValueError("scalar geometry loss requires target_colormap")
        if self.timestep_sampling not in {"uniform", "inference"}:
            raise ValueError(
                "timestep_sampling must be either 'uniform' or 'inference'"
            )
        if self.training_num_inference_steps <= 0:
            raise ValueError("training_num_inference_steps must be positive")
        if self.output_calibration not in {
            "none",
            "moments",
            "moments_and_project",
        }:
            raise ValueError(f"Unsupported output calibration: {self.output_calibration}")
        if self.output_calibration != "none":
            if self.target_channel_mean is None or self.target_channel_std is None:
                raise ValueError("Output calibration requires target channel statistics")
            if len(self.target_channel_mean) != 3 or len(self.target_channel_std) != 3:
                raise ValueError("Target channel statistics must contain three values")
            if any(value <= 0 for value in self.target_channel_std):
                raise ValueError("Target channel standard deviations must be positive")
        if self.output_calibration == "moments_and_project" and self.target_colormap == "unknown":
            raise ValueError("Colormap projection requires target_colormap")
        if self.spatial_adapter_type not in {"none", "t2i_adapter_xl"}:
            raise ValueError(f"Unsupported spatial adapter: {self.spatial_adapter_type}")
        if self.spatial_adapter_conditioning_scale < 0:
            raise ValueError("spatial_adapter_conditioning_scale must be non-negative")
        if self.spatial_adapter_downscale_factor <= 0:
            raise ValueError("spatial_adapter_downscale_factor must be positive")
        if self.default_num_inference_steps <= 0:
            raise ValueError("default_num_inference_steps must be positive")
        if self.version != 1:
            raise ValueError(f"Unsupported conditioning config version: {self.version}")

    @property
    def target_channels(self) -> int:
        return self.in_channels - self.condition_channels

    def architecture_signature(self) -> tuple[str, int, int, str, str, str, int, int]:
        """Fields that must match when resuming model/optimizer state."""
        return (
            self.mode,
            self.in_channels,
            self.condition_channels,
            self.prediction_type,
            self.inference_protocol,
            self.spatial_adapter_type,
            self.spatial_adapter_downscale_factor,
            self.version,
        )

    def is_architecture_compatible(self, other: SourceConditioningConfig) -> bool:
        return self.architecture_signature() == other.architecture_signature()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> SourceConditioningConfig:
        payload = dict(payload)
        for key in ("target_channel_mean", "target_channel_std"):
            if isinstance(payload.get(key), list):
                payload[key] = tuple(float(value) for value in payload[key])
        config = cls(**payload)
        config.validate()
        return config


def read_source_conditioning_config(
    checkpoint_dir: str | Path,
) -> SourceConditioningConfig | None:
    """Read a conditioning contract, returning ``None`` for legacy checkpoints."""
    path = Path(checkpoint_dir) / CONDITIONING_CONFIG_NAME
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid conditioning config payload in {path}")
    return SourceConditioningConfig.from_dict(payload)


def write_source_conditioning_config(
    checkpoint_dir: str | Path,
    config: SourceConditioningConfig,
) -> Path:
    """Write the conditioning contract beside the LoRA adapter."""
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CONDITIONING_CONFIG_NAME
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)
        handle.write("\n")
    return path


def expand_unet_conv_in(
    unet: nn.Module,
    *,
    condition_channels: int = 4,
) -> nn.Conv2d:
    """Expand ``conv_in`` and zero-init source channels without changing output."""
    old = getattr(unet, "conv_in", None)
    if not isinstance(old, nn.Conv2d):
        raise TypeError(f"Expected Conv2d conv_in, got {type(old).__name__}")
    if condition_channels <= 0:
        raise ValueError("condition_channels must be positive")

    configured_channels = getattr(getattr(unet, "config", None), "in_channels", None)
    # An already-expanded model is identified by config and physical layer
    # agreeing on at least two latent channel groups.
    if (
        configured_channels == old.in_channels
        and old.in_channels >= 2 * condition_channels
    ):
        return old

    expanded = nn.Conv2d(
        in_channels=old.in_channels + condition_channels,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    ).to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        expanded.weight.zero_()
        expanded.weight[:, : old.in_channels].copy_(old.weight)
        if old.bias is not None:
            expanded.bias.copy_(old.bias)
    unet.conv_in = expanded
    register = getattr(unet, "register_to_config", None)
    if callable(register):
        register(in_channels=expanded.in_channels)
    return expanded


def sigma_for_timesteps(
    scheduler,
    timesteps: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return Euler sigmas broadcast over a latent batch."""
    schedule = scheduler.timesteps.to(device=timesteps.device)
    indices = [
        scheduler.index_for_timestep(timestep, schedule)
        for timestep in timesteps
    ]
    sigma = scheduler.sigmas.to(
        device=reference.device,
        dtype=reference.dtype,
    )[indices]
    return sigma.view(reference.shape[0], *([1] * (reference.ndim - 1)))


def build_source_conditioned_input(
    scheduler,
    target_latent: torch.Tensor,
    source_latent: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Noise/scale only the target latent, then append the clean source latent."""
    if target_latent.shape != source_latent.shape or noise.shape != target_latent.shape:
        raise ValueError("target, source, and noise latents must have identical shapes")
    noisy_target = scheduler.add_noise(target_latent, noise, timesteps)
    sigma = sigma_for_timesteps(scheduler, timesteps, noisy_target)
    scaled_target = noisy_target / torch.sqrt(sigma.square() + 1.0)
    return torch.cat((scaled_target, source_latent), dim=1), noisy_target, sigma


def sample_source_conditioned_timesteps(
    scheduler,
    *,
    batch_size: int,
    device: torch.device,
    mode: str,
    inference_steps: int = 4,
) -> torch.Tensor:
    """Sample either the full training schedule or the deployed SSD schedule."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if mode == "uniform":
        config = getattr(scheduler, "config", scheduler)
        n_train = int(getattr(config, "num_train_timesteps", 1000))
        return torch.randint(
            0,
            n_train,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )
    if mode != "inference":
        raise ValueError(f"Unsupported source timestep sampling mode: {mode!r}")
    if inference_steps <= 0:
        raise ValueError("inference_steps must be positive")
    scheduler.set_timesteps(inference_steps, device=device)
    candidates = scheduler.timesteps.to(device=device)
    indices = torch.randint(0, candidates.numel(), (batch_size,), device=device)
    return candidates[indices]


def calibrate_source_conditioned_output(
    output: torch.Tensor,
    config: SourceConditioningConfig,
) -> torch.Tensor:
    """Match training-target colour moments and optionally project to its curve."""
    config.validate()
    if config.output_calibration == "none":
        return output
    if output.ndim != 4 or output.shape[1] != 3:
        raise ValueError("Expected output tensor shaped [B, 3, H, W]")
    mean = torch.tensor(
        config.target_channel_mean,
        device=output.device,
        dtype=output.dtype,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        config.target_channel_std,
        device=output.device,
        dtype=output.dtype,
    ).view(1, 3, 1, 1)
    output_mean = output.mean(dim=(2, 3), keepdim=True)
    output_std = output.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
    calibrated = ((output - output_mean) / output_std * std + mean).clamp(0, 1)
    if config.output_calibration == "moments_and_project":
        from .core.depth_mapper import DepthRGBMapper

        mapper = DepthRGBMapper(config.target_colormap)
        normalized = mapper.rgb_to_normalized(calibrated)
        calibrated = mapper.normalized_to_rgb(normalized)
    return calibrated


def set_scheduler_prediction_type(scheduler, prediction_type: str) -> None:
    """Update scheduler config through Diffusers' FrozenDict-safe API."""
    register = getattr(scheduler, "register_to_config", None)
    if callable(register):
        register(prediction_type=prediction_type)
    else:
        config = getattr(scheduler, "config", None)
        if isinstance(config, dict):
            config["prediction_type"] = prediction_type
        else:
            config.prediction_type = prediction_type


def scheduler_prediction_type(scheduler) -> str:
    """Read the scheduler output contract from dict- or object-like config."""
    config = getattr(scheduler, "config", None)
    if isinstance(config, dict):
        prediction_type = config.get("prediction_type")
    else:
        prediction_type = getattr(config, "prediction_type", None)
    if not isinstance(prediction_type, str) or not prediction_type:
        raise RuntimeError("Scheduler config does not declare prediction_type")
    return prediction_type


def validate_scheduler_prediction_type(scheduler, expected: str) -> str:
    """Fail fast when scheduler semantics diverge from the model contract."""
    actual = scheduler_prediction_type(scheduler)
    if actual != expected:
        raise RuntimeError(
            "Scheduler prediction_type mismatch: "
            f"model expects {expected!r}, scheduler uses {actual!r}"
        )
    return actual


__all__ = [
    "CONDITIONING_CONFIG_NAME",
    "CONV_IN_STATE_NAME",
    "INFERENCE_PROTOCOLS",
    "MULTI_STEP_EPSILON",
    "SINGLE_STEP_X0",
    "SOURCE_LATENT_CONCAT",
    "SourceConditioningConfig",
    "build_source_conditioned_input",
    "calibrate_source_conditioned_output",
    "expand_unet_conv_in",
    "read_source_conditioning_config",
    "sample_source_conditioned_timesteps",
    "scheduler_prediction_type",
    "set_scheduler_prediction_type",
    "sigma_for_timesteps",
    "validate_scheduler_prediction_type",
    "write_source_conditioning_config",
]
