"""Frozen SSD/VAE feature extraction and a lightweight scalar-depth probe."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..model_registry import build_added_cond_kwargs, vae_encode


@dataclass(frozen=True)
class DepthFeatureContract:
    """Ordered names and channels required by a depth-probe checkpoint."""

    names: tuple[str, ...]
    channels: tuple[int, ...]

    def validate(self) -> None:
        if not self.names:
            raise ValueError("feature contract must not be empty")
        if len(self.names) != len(self.channels):
            raise ValueError("feature names and channels must have equal length")
        if len(set(self.names)) != len(self.names):
            raise ValueError("feature names must be unique")
        if any(not name for name in self.names):
            raise ValueError("feature names must be non-empty")
        if any(channels <= 0 for channels in self.channels):
            raise ValueError("feature channels must be positive")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DepthFeatureContract:
        contract = cls(
            names=tuple(str(name) for name in payload["names"]),
            channels=tuple(int(channels) for channels in payload["channels"]),
        )
        contract.validate()
        return contract


def _primary_tensor(output: Any, *, name: str) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        candidate = output[0]
        if isinstance(candidate, torch.Tensor):
            return candidate
    raise TypeError(f"{name} produced no primary tensor")


class FrozenSDXLDepthFeatureExtractor:
    """Read deterministic VAE and SSD/SDXL UNet hierarchy features."""

    def __init__(self, generator: Any, *, timestep: int = 0) -> None:
        if timestep < 0:
            raise ValueError("timestep must be non-negative")
        model_config = generator.model_config
        if getattr(model_config, "family", None) != "sdxl":
            raise ValueError("Frozen depth probe currently requires an SDXL/SSD model")
        pipe = getattr(generator, "pipe", None) or getattr(generator, "_t2i_pipe", None)
        if pipe is None:
            raise AttributeError("generator has no loaded text-to-image pipeline")
        vae = getattr(pipe, "vae", None)
        unet = getattr(pipe, "unet", None)
        if vae is None or unet is None:
            raise AttributeError("Frozen depth probe requires pipeline VAE and UNet")
        if int(getattr(getattr(unet, "config", None), "in_channels", 0)) != 4:
            raise ValueError("Frozen depth probe requires the unmodified 4-channel SSD UNet")
        down_blocks = getattr(unet, "down_blocks", None)
        mid_block = getattr(unet, "mid_block", None)
        if down_blocks is None or len(down_blocks) == 0 or mid_block is None:
            raise AttributeError("UNet must expose down_blocks and mid_block")

        self.generator = generator
        self.model_config = model_config
        self.vae: nn.Module = vae
        self.unet: nn.Module = unet
        self.down_blocks: tuple[nn.Module, ...] = tuple(down_blocks)
        self.mid_block: nn.Module = mid_block
        self.timestep = timestep
        self.vae.eval().requires_grad_(False)
        self.unet.eval().requires_grad_(False)

    @torch.inference_mode()
    def extract(
        self,
        images: torch.Tensor,
        *,
        prompt: str,
    ) -> tuple[DepthFeatureContract, tuple[torch.Tensor, ...]]:
        """Extract an ordered frozen feature tuple for images in ``[-1, 1]``."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected images shaped [B, 3, H, W]")
        if not prompt:
            raise ValueError("prompt must be non-empty")

        vae_parameter = next(self.vae.parameters())
        vae_images = images.to(
            device=vae_parameter.device,
            dtype=vae_parameter.dtype,
        )
        latents = vae_encode(self.vae, vae_images, self.model_config)
        unet_parameter = next(self.unet.parameters())
        latents = latents.to(
            device=unet_parameter.device,
            dtype=unet_parameter.dtype,
        )
        batch_size = latents.shape[0]

        text = self.generator.encode_prompts([prompt] * batch_size)
        prompt_embeds = text["prompt_embeds"]
        pooled_embeds = text.get("pooled_prompt_embeds")
        if not isinstance(prompt_embeds, torch.Tensor):
            raise TypeError("Prompt encoder returned no prompt embedding tensor")
        prompt_embeds = prompt_embeds.to(
            device=unet_parameter.device,
            dtype=unet_parameter.dtype,
        )
        if isinstance(pooled_embeds, torch.Tensor):
            pooled_embeds = pooled_embeds.to(
                device=unet_parameter.device,
                dtype=unet_parameter.dtype,
            )
        added_cond_kwargs = build_added_cond_kwargs(
            config=self.model_config,
            pooled_embeds=pooled_embeds,
            resolution=(images.shape[-2], images.shape[-1]),
            device=unet_parameter.device,
            dtype=unet_parameter.dtype,
            batch_size=batch_size,
        )

        captures: dict[str, torch.Tensor] = {}
        handles: list[Any] = []

        def capture(name: str):
            def hook(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
                del module, args
                captures[name] = _primary_tensor(output, name=name)

            return hook

        try:
            for index, block in enumerate(self.down_blocks):
                name = f"down_{index}"
                handles.append(block.register_forward_hook(capture(name)))
            handles.append(self.mid_block.register_forward_hook(capture("mid")))
            timesteps = torch.full(
                (batch_size,),
                self.timestep,
                device=unet_parameter.device,
                dtype=torch.long,
            )
            unet_output = self.unet(
                latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )
            final = _primary_tensor(unet_output, name="unet_output")
        finally:
            for handle in handles:
                handle.remove()

        ordered_names = (
            "vae_latent",
            *(f"down_{index}" for index in range(len(self.down_blocks))),
            "mid",
            "unet_output",
        )
        missing = [name for name in ordered_names[1:-1] if name not in captures]
        if missing:
            raise RuntimeError(f"Missing frozen UNet features: {missing}")
        ordered_tensors = (
            latents,
            *(captures[name] for name in ordered_names[1:-1]),
            final,
        )
        features = tuple(tensor.detach().float() for tensor in ordered_tensors)
        if any(tensor.ndim != 4 for tensor in features):
            raise ValueError("All frozen depth features must be 4D tensors")
        if any(tensor.shape[0] != batch_size for tensor in features):
            raise ValueError("Frozen depth feature batch sizes do not match")
        contract = DepthFeatureContract(
            names=ordered_names,
            channels=tuple(int(tensor.shape[1]) for tensor in features),
        )
        contract.validate()
        return contract, features


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _conv_block(input_channels: int, output_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, padding=1),
        nn.GroupNorm(_group_count(output_channels), output_channels),
        nn.SiLU(),
    )


class ScalarDepthProbe(nn.Module):
    """FPN-style decoder for an ordered frozen feature contract."""

    def __init__(
        self,
        contract: DepthFeatureContract,
        *,
        width: int = 64,
        upsample_stages: int = 3,
    ) -> None:
        super().__init__()
        contract.validate()
        if width <= 0:
            raise ValueError("width must be positive")
        if upsample_stages < 0:
            raise ValueError("upsample_stages must be non-negative")
        self.contract = contract
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, width, 1),
                    nn.GroupNorm(_group_count(width), width),
                    nn.SiLU(),
                )
                for channels in contract.channels
            ]
        )
        self.fusion = nn.Sequential(
            _conv_block(width * len(contract.channels), width),
            _conv_block(width, width),
        )
        blocks = []
        current_width = width
        for _ in range(upsample_stages):
            next_width = max(8, current_width // 2)
            blocks.append(_conv_block(current_width, next_width))
            current_width = next_width
        self.upsample_blocks = nn.ModuleList(blocks)
        self.output = nn.Conv2d(current_width, 1, 1)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        *,
        output_size: tuple[int, int],
        feature_mask: Sequence[bool] | None = None,
    ) -> torch.Tensor:
        if len(features) != len(self.contract.names):
            raise ValueError(
                f"feature count {len(features)} does not match contract "
                f"{len(self.contract.names)}"
            )
        if len(output_size) != 2 or any(size <= 0 for size in output_size):
            raise ValueError("output_size must contain two positive values")
        if feature_mask is None:
            active_features = (True,) * len(features)
        else:
            if len(feature_mask) != len(features):
                raise ValueError("feature_mask length must match feature count")
            if any(type(active) is not bool for active in feature_mask):
                raise ValueError("feature_mask values must be boolean")
            active_features = tuple(feature_mask)
        batch_size = features[0].shape[0]
        for name, channels, feature in zip(
            self.contract.names,
            self.contract.channels,
            features,
            strict=True,
        ):
            if feature.ndim != 4:
                raise ValueError(f"feature {name} must be 4D")
            if feature.shape[0] != batch_size:
                raise ValueError(f"feature {name} has a mismatched batch size")
            if feature.shape[1] != channels:
                raise ValueError(
                    f"feature {name} channels {feature.shape[1]} do not match {channels}"
                )

        reference_size = max(
            (tuple(feature.shape[-2:]) for feature in features),
            key=lambda size: size[0] * size[1],
        )
        projected = []
        for projection, feature, active in zip(
            self.projections,
            features,
            active_features,
            strict=True,
        ):
            parameter = next(projection.parameters())
            level = projection(
                feature.to(device=parameter.device, dtype=parameter.dtype)
            )
            if not active:
                level = torch.zeros_like(level)
            if level.shape[-2:] != reference_size:
                level = F.interpolate(
                    level,
                    size=reference_size,
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(level)
        hidden = self.fusion(torch.cat(projected, dim=1))
        for block in self.upsample_blocks:
            hidden = F.interpolate(
                hidden,
                scale_factor=2.0,
                mode="bilinear",
                align_corners=False,
            )
            hidden = block(hidden)
        logits = self.output(hidden)
        if logits.shape[-2:] != output_size:
            logits = F.interpolate(
                logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return torch.sigmoid(logits)


def _spatial_gradient_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = prediction.new_zeros(())
    if prediction.shape[-1] > 1:
        result = result + F.l1_loss(
            prediction[..., :, 1:] - prediction[..., :, :-1],
            target[..., :, 1:] - target[..., :, :-1],
        )
    if prediction.shape[-2] > 1:
        result = result + F.l1_loss(
            prediction[..., 1:, :] - prediction[..., :-1, :],
            target[..., 1:, :] - target[..., :-1, :],
        )
    return result


def _pearson_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    pred_flat = prediction.flatten(1)
    target_flat = target.flatten(1)
    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    numerator = (pred_centered * target_centered).sum(dim=1)
    denominator = (
        pred_centered.square().sum(dim=1)
        * target_centered.square().sum(dim=1)
    ).clamp_min(1e-12).sqrt()
    return numerator / denominator


def scalar_depth_probe_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    gradient_weight: float = 0.5,
    pearson_weight: float = 0.1,
) -> torch.Tensor:
    """Optimize scalar value, spatial gradients, and linear correlation."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError("Expected scalar depth shaped [B, 1, H, W]")
    if gradient_weight < 0 or pearson_weight < 0:
        raise ValueError("loss weights must be non-negative")
    value_loss = F.smooth_l1_loss(prediction, target, beta=0.02)
    gradient_loss = _spatial_gradient_l1(prediction, target)
    correlation_loss = 1.0 - _pearson_per_sample(prediction, target).mean()
    result = value_loss + gradient_weight * gradient_loss + pearson_weight * correlation_loss
    if not torch.isfinite(result):
        raise FloatingPointError("scalar depth probe loss is non-finite")
    return result


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    values = values.flatten().to(dtype=torch.float64)
    order = torch.argsort(values)
    sorted_values = values[order]
    _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
    ends = counts.cumsum(0)
    starts = ends - counts
    midranks = (starts + ends - 1).to(dtype=torch.float64) / 2.0 + 1.0
    sorted_ranks = torch.repeat_interleave(midranks, counts)
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


@torch.no_grad()
def evaluate_scalar_depth_predictions(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Return mean per-image scalar error and order correlations."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError("Expected scalar depth shaped [B, 1, H, W]")
    prediction = prediction.detach().float().cpu().clamp(0, 1)
    target = target.detach().float().cpu().clamp(0, 1)
    maes = []
    rmses = []
    pearsons = []
    spearmans = []
    for pred_sample, target_sample in zip(prediction, target, strict=True):
        pred_flat = pred_sample.flatten()
        target_flat = target_sample.flatten()
        maes.append(float((pred_flat - target_flat).abs().mean()))
        rmses.append(float((pred_flat - target_flat).square().mean().sqrt()))
        pearson = 0.0
        if pred_flat.std() > 1e-8 and target_flat.std() > 1e-8:
            pearson = float(
                torch.corrcoef(torch.stack((pred_flat, target_flat)))[0, 1]
            )
        pred_ranks = _average_ranks(pred_flat)
        target_ranks = _average_ranks(target_flat)
        spearman = 0.0
        if pred_ranks.std() > 1e-8 and target_ranks.std() > 1e-8:
            spearman = float(
                torch.corrcoef(torch.stack((pred_ranks, target_ranks)))[0, 1]
            )
        pearsons.append(pearson)
        spearmans.append(spearman)
    return {
        "mae": sum(maes) / len(maes),
        "rmse": sum(rmses) / len(rmses),
        "pearson": sum(pearsons) / len(pearsons),
        "spearman": sum(spearmans) / len(spearmans),
    }


def depth_probe_gate_passed(
    metrics: dict[str, float],
    *,
    max_mae: float = 0.05,
    min_pearson: float = 0.95,
    min_spearman: float = 0.95,
) -> bool:
    """Require error and both correlation thresholds simultaneously."""
    if max_mae < 0 or not -1 <= min_pearson <= 1 or not -1 <= min_spearman <= 1:
        raise ValueError("invalid depth probe gate thresholds")
    required = ("mae", "pearson", "spearman")
    if any(key not in metrics for key in required):
        raise KeyError("depth probe metrics are incomplete")
    if any(not math.isfinite(float(metrics[key])) for key in required):
        return False
    return (
        metrics["mae"] <= max_mae
        and metrics["pearson"] >= min_pearson
        and metrics["spearman"] >= min_spearman
    )


def depth_feature_contribution_gate_passed(
    vae_only_metrics: dict[str, float],
    vae_ssd_metrics: dict[str, float],
    *,
    min_correlation_delta: float = 0.05,
    max_mae_regression: float = 0.01,
) -> bool:
    """Gate held-out SSD contribution beyond an otherwise matched VAE arm."""
    if min_correlation_delta < 0 or max_mae_regression < 0:
        raise ValueError("feature contribution thresholds must be non-negative")
    required = ("mae", "pearson", "spearman")
    if any(
        key not in metrics
        for metrics in (vae_only_metrics, vae_ssd_metrics)
        for key in required
    ):
        raise KeyError("feature contribution metrics are incomplete")
    if any(
        not math.isfinite(float(metrics[key]))
        for metrics in (vae_only_metrics, vae_ssd_metrics)
        for key in required
    ):
        return False
    return (
        vae_ssd_metrics["pearson"] - vae_only_metrics["pearson"]
        >= min_correlation_delta
        and vae_ssd_metrics["spearman"] - vae_only_metrics["spearman"]
        >= min_correlation_delta
        and vae_ssd_metrics["mae"]
        <= vae_only_metrics["mae"] + max_mae_regression
    )


# ──────────────────────────────────────────────────────────────
# V5 Loss Components: SSIM + Edge-Aware Loss
# ──────────────────────────────────────────────────────────────


def _gaussian_kernel(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a 2D Gaussian kernel for SSIM computation."""
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g.unsqueeze(0) * g.unsqueeze(1)
    return kernel.unsqueeze(0).unsqueeze(0)


def _ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Compute Structural Similarity Index (SSIM) per sample.

    Returns mean SSIM across the batch (higher = better).
    Input: [B, 1, H, W] tensors in [0, 1] range.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError("Expected scalar depth shaped [B, 1, H, W]")
    if prediction.shape[-1] < window_size or prediction.shape[-2] < window_size:
        # Fallback for small spatial sizes: reduce window
        window_size = min(prediction.shape[-1], prediction.shape[-2])
        if window_size % 2 == 0:
            window_size -= 1
        if window_size < 3:
            return torch.ones((), device=prediction.device, dtype=prediction.dtype)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    kernel = _gaussian_kernel(window_size, sigma, prediction.device, prediction.dtype)
    pad = window_size // 2

    mu_pred = F.conv2d(prediction, kernel, padding=pad)
    mu_target = F.conv2d(target, kernel, padding=pad)
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_cross = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(prediction * prediction, kernel, padding=pad) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, kernel, padding=pad) - mu_target_sq
    sigma_cross = F.conv2d(prediction * target, kernel, padding=pad) - mu_cross

    ssim_map = (
        (2 * mu_cross + c1) * (2 * sigma_cross + c2)
    ) / (
        (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    )
    return ssim_map.mean()


def _ssim_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - SSIM loss (lower = better)."""
    return 1.0 - _ssim(prediction, target)


def _sobel_edges(tensor: torch.Tensor) -> torch.Tensor:
    """Compute Sobel edge magnitude map for a [B, 1, H, W] tensor."""
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=tensor.dtype, device=tensor.device,
    ).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=tensor.dtype, device=tensor.device,
    ).unsqueeze(0).unsqueeze(0)
    gx = F.conv2d(tensor, sobel_x, padding=1)
    gy = F.conv2d(tensor, sobel_y, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


def _edge_aware_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Edge-weighted L1 loss that penalizes errors at object boundaries more heavily."""
    edge_weight = _sobel_edges(target)
    edge_weight = edge_weight / (edge_weight.mean() + 1e-8)
    edge_weight = edge_weight.clamp(0.5, 5.0)
    pixel_diff = (prediction - target).abs()
    return (pixel_diff * edge_weight).mean()


def v5_composite_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    gradient_weight: float = 0.5,
    pearson_weight: float = 0.3,
    ssim_weight: float = 0.2,
    edge_weight: float = 0.15,
) -> torch.Tensor:
    """V5 composite loss: SmoothL1 + gradient + Pearson + SSIM + edge-aware.

    The SSIM term captures structural similarity (addresses edge blurring).
    The edge-aware term uses Sobel edge maps to weight boundary errors higher.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError("Expected scalar depth shaped [B, 1, H, W]")
    for w_name, w_val in [
        ("gradient_weight", gradient_weight),
        ("pearson_weight", pearson_weight),
        ("ssim_weight", ssim_weight),
        ("edge_weight", edge_weight),
    ]:
        if w_val < 0:
            raise ValueError(f"{w_name} must be non-negative")

    value_loss = F.smooth_l1_loss(prediction, target, beta=0.02)
    gradient_loss = _spatial_gradient_l1(prediction, target)
    correlation_loss = 1.0 - _pearson_per_sample(prediction, target).mean()
    ssim_loss = _ssim_loss(prediction, target)
    edge_loss = _edge_aware_loss(prediction, target)

    result = (
        value_loss
        + gradient_weight * gradient_loss
        + pearson_weight * correlation_loss
        + ssim_weight * ssim_loss
        + edge_weight * edge_loss
    )
    if not torch.isfinite(result):
        raise FloatingPointError("V5 composite loss is non-finite")
    return result


# ──────────────────────────────────────────────────────────────
# V5 Probe Architecture: Skip-Connection Decoder
# ──────────────────────────────────────────────────────────────


class _SkipDecoderStage(nn.Module):
    """One decoder stage: upsample → reduce channels → skip-inject → refine."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1)
        has_skip = skip_channels is not None
        if has_skip:
            self.skip_adapter = nn.Conv2d(skip_channels, out_channels, 1)
        else:
            self.skip_adapter = None
        decoder_in = out_channels * 2 if has_skip else out_channels
        self.refine = nn.Sequential(
            nn.Conv2d(decoder_in, out_channels, 3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, hidden: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        hidden = F.interpolate(hidden, scale_factor=2.0, mode="bilinear", align_corners=False)
        hidden = self.reduce(hidden)
        if self.skip_adapter is not None and skip is not None:
            if skip.shape[-2:] != hidden.shape[-2:]:
                skip = F.interpolate(skip, size=hidden.shape[-2:], mode="bilinear", align_corners=False)
            skip = self.skip_adapter(skip)
            hidden = torch.cat([hidden, skip], dim=1)
        return self.refine(hidden)


class ScalarDepthProbeV5(nn.Module):
    """V5 FPN-style decoder with multi-scale skip connections from frozen features.

    Improvements over V4 ScalarDepthProbe:
    1. **Skip connections** — frozen features at different spatial scales are
       injected into the corresponding decoder stages, preserving fine-grained
       spatial detail (addresses edge blurring).
    2. **Progressive channel reduction** — wider early layers for capacity,
       narrower late layers for resolution.
    3. **Residual refinement** — each decoder stage adds a residual refinement
       rather than replacing the upsampled features.
    """

    def __init__(
        self,
        contract: DepthFeatureContract,
        *,
        width: int = 128,
        upsample_stages: int = 3,
    ) -> None:
        super().__init__()
        contract.validate()
        if width <= 0:
            raise ValueError("width must be positive")
        if upsample_stages < 0:
            raise ValueError("upsample_stages must be non-negative")
        self.contract = contract

        # Project each frozen feature to a consistent width
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, width, 1),
                    nn.GroupNorm(_group_count(width), width),
                    nn.SiLU(),
                )
                for channels in contract.channels
            ]
        )

        # Fusion of all projected features at reference resolution
        num_features = len(contract.channels)
        self.fusion = nn.Sequential(
            _conv_block(width * num_features, width),
            _conv_block(width, width),
        )

        # Build decoder stages with optional skip connections.
        # Assign skips from the deepest features (unet_output, mid, etc.)
        # to the earliest decoder stages.
        num_skips = min(upsample_stages, num_features)
        skip_proj_indices = list(range(num_features - 1, num_features - 1 - num_skips, -1))
        self._skip_proj_indices = tuple(skip_proj_indices)

        stages = []
        current_width = width
        for stage in range(upsample_stages):
            next_width = max(16, current_width // 2)
            # projected features all have `width` channels
            skip_ch = width if stage < len(skip_proj_indices) else None
            stages.append(_SkipDecoderStage(current_width, next_width, skip_ch))
            current_width = next_width
        self.decoder_stages = nn.ModuleList(stages)
        self.output = nn.Conv2d(current_width, 1, 1)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        *,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        if len(features) != len(self.contract.names):
            raise ValueError(
                f"feature count {len(features)} does not match contract "
                f"{len(self.contract.names)}"
            )

        # Project all features and store for skip connections
        reference_size = max(
            (tuple(f.shape[-2:]) for f in features),
            key=lambda s: s[0] * s[1],
        )
        projected_all = []
        for projection, feature in zip(self.projections, features, strict=True):
            param = next(projection.parameters())
            level = projection(feature.to(device=param.device, dtype=param.dtype))
            if level.shape[-2:] != reference_size:
                level = F.interpolate(level, size=reference_size, mode="bilinear", align_corners=False)
            projected_all.append(level)

        # Fuse all features
        hidden = self.fusion(torch.cat(projected_all, dim=1))

        # Progressive decoding with skip connections
        for stage_idx, stage_module in enumerate(self.decoder_stages):
            skip = None
            if stage_idx < len(self._skip_proj_indices):
                proj_idx = self._skip_proj_indices[stage_idx]
                skip = projected_all[proj_idx]
            hidden = stage_module(hidden, skip)

        logits = self.output(hidden)
        if logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(logits)


# ──────────────────────────────────────────────────────────────
# V6: Multi-Timestep Feature Extraction + LoRA-Adapted UNet
# ──────────────────────────────────────────────────────────────


class MultiTimestepFeatureExtractor:
    """Extract frozen UNet features at multiple timesteps for richer representation.

    Instead of a single forward pass at t=0, this extractor runs the UNet at
    multiple timesteps (e.g., t=0, 250, 500, 750) and concatenates the captured
    down/mid features along the channel dimension. This provides the probe
    decoder with temporal multi-scale information — early timesteps capture
    coarse structure, late timesteps capture fine details.

    The VAE latent is extracted once (timestep-independent) and prepended.
    """

    def __init__(
        self,
        generator: Any,
        *,
        timesteps: Sequence[int] = (0, 250, 500, 750),
    ) -> None:
        if not timesteps or any(t < 0 for t in timesteps):
            raise ValueError("timesteps must be non-empty and non-negative")
        model_config = generator.model_config
        if getattr(model_config, "family", None) != "sdxl":
            raise ValueError("Multi-timestep extractor requires an SDXL/SSD model")
        pipe = getattr(generator, "pipe", None) or getattr(generator, "_t2i_pipe", None)
        if pipe is None:
            raise AttributeError("generator has no loaded text-to-image pipeline")
        vae = getattr(pipe, "vae", None)
        unet = getattr(pipe, "unet", None)
        if vae is None or unet is None:
            raise AttributeError("Extractor requires pipeline VAE and UNet")
        if int(getattr(getattr(unet, "config", None), "in_channels", 0)) != 4:
            raise ValueError("Extractor requires the unmodified 4-channel SSD UNet")

        self.generator = generator
        self.model_config = model_config
        self.vae: nn.Module = vae
        self.unet: nn.Module = unet
        self.timesteps: tuple[int, ...] = tuple(timesteps)
        self.down_blocks: tuple[nn.Module, ...] = tuple(unet.down_blocks)
        self.mid_block: nn.Module = unet.mid_block
        self.vae.eval().requires_grad_(False)
        self.unet.eval().requires_grad_(False)

    @torch.inference_mode()
    def extract(
        self,
        images: torch.Tensor,
        *,
        prompt: str,
    ) -> tuple[DepthFeatureContract, tuple[torch.Tensor, ...]]:
        """Extract multi-timestep features for images in ``[-1, 1]``.

        Returns a contract with per-timestep features concatenated channel-wise:
        - vae_latent: [B, 4, H, W] (extracted once)
        - down_{i}_ts{t}: [B, C_i, h_i, w_i] for each timestep
        - mid_ts{t}: [B, 1280, h, w] for each timestep
        - unet_output_ts{t}: [B, 4, H, W] for each timestep
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected images shaped [B, 3, H, W]")
        if not prompt:
            raise ValueError("prompt must be non-empty")

        vae_parameter = next(self.vae.parameters())
        vae_images = images.to(
            device=vae_parameter.device, dtype=vae_parameter.dtype,
        )
        latents = vae_encode(self.vae, vae_images, self.model_config)
        unet_parameter = next(self.unet.parameters())
        latents = latents.to(
            device=unet_parameter.device, dtype=unet_parameter.dtype,
        )
        batch_size = latents.shape[0]

        text = self.generator.encode_prompts([prompt] * batch_size)
        prompt_embeds = text["prompt_embeds"].to(
            device=unet_parameter.device, dtype=unet_parameter.dtype,
        )
        pooled_embeds = text.get("pooled_prompt_embeds")
        if isinstance(pooled_embeds, torch.Tensor):
            pooled_embeds = pooled_embeds.to(
                device=unet_parameter.device, dtype=unet_parameter.dtype,
            )
        added_cond_kwargs = build_added_cond_kwargs(
            config=self.model_config,
            pooled_embeds=pooled_embeds,
            resolution=(images.shape[-2], images.shape[-1]),
            device=unet_parameter.device,
            dtype=unet_parameter.dtype,
            batch_size=batch_size,
        )

        # Collect per-level features across all timesteps
        # Structure: {level_name: [tensor_per_timestep]}
        multi_ts_features: dict[str, list[torch.Tensor]] = {
            f"down_{i}": [] for i in range(len(self.down_blocks))
        }
        multi_ts_features["mid"] = []
        multi_ts_features["unet_output"] = []

        for ts_val in self.timesteps:
            captures: dict[str, torch.Tensor] = {}
            handles: list[Any] = []

            def capture(name: str, captures: dict[str, torch.Tensor] = captures):
                def hook(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
                    del module, args
                    captures[name] = _primary_tensor(output, name=name)
                return hook

            try:
                for index, block in enumerate(self.down_blocks):
                    handles.append(block.register_forward_hook(capture(f"down_{index}")))
                handles.append(self.mid_block.register_forward_hook(capture("mid")))
                timesteps_tensor = torch.full(
                    (batch_size,), ts_val,
                    device=unet_parameter.device, dtype=torch.long,
                )
                unet_output = self.unet(
                    latents,
                    timesteps_tensor,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )
                final = _primary_tensor(unet_output, name="unet_output")
            finally:
                for handle in handles:
                    handle.remove()

            for level_name in multi_ts_features:
                multi_ts_features[level_name].append(
                    captures.get(level_name, final if level_name == "unet_output" else None)
                )

        # Build ordered feature tuple: vae_latent + concatenated per-level
        ordered_names = ["vae_latent"]
        ordered_tensors = [latents.detach().float()]
        ordered_channels = [int(latents.shape[1])]

        for level_name in [f"down_{i}" for i in range(len(self.down_blocks))] + ["mid", "unet_output"]:
            ts_tensors = multi_ts_features[level_name]
            # All timesteps should have the same spatial shape for a given level
            concatenated = torch.cat(
                [t.detach().float() for t in ts_tensors], dim=1,
            )
            ordered_names.append(level_name)  # e.g., "down_0" (multi-ts concatenated)
            ordered_tensors.append(concatenated)
            ordered_channels.append(int(concatenated.shape[1]))

        contract = DepthFeatureContract(
            names=tuple(ordered_names),
            channels=tuple(ordered_channels),
        )
        contract.validate()
        return contract, tuple(ordered_tensors)


class LoRAAdaptedFeatureExtractor:
    """Extract features from a UNet partially adapted via LoRA.

    Instead of fully freezing the UNet, this extractor applies LoRA adapters
    to selected attention/MLP layers. The LoRA weights are trainable during
    the feature extraction phase, allowing the UNet features to adapt to the
    depth estimation task while keeping the base weights frozen.

    Features are extracted at a single timestep (default t=0) but with
    LoRA-adapted weights, providing task-relevant features.
    """

    def __init__(
        self,
        generator: Any,
        *,
        timestep: int = 0,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_targets: Sequence[str] | None = None,
    ) -> None:
        model_config = generator.model_config
        if getattr(model_config, "family", None) != "sdxl":
            raise ValueError("LoRA extractor requires an SDXL/SSD model")
        pipe = getattr(generator, "pipe", None) or getattr(generator, "_t2i_pipe", None)
        if pipe is None:
            raise AttributeError("generator has no loaded text-to-image pipeline")
        vae = getattr(pipe, "vae", None)
        unet = getattr(pipe, "unet", None)
        if vae is None or unet is None:
            raise AttributeError("Extractor requires pipeline VAE and UNet")

        self.generator = generator
        self.model_config = model_config
        self.vae: nn.Module = vae
        self.unet: nn.Module = unet
        self.timestep = timestep
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_targets = tuple(lora_targets) if lora_targets else (
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
        )
        self.down_blocks: tuple[nn.Module, ...] = tuple(unet.down_blocks)
        self.mid_block: nn.Module = unet.mid_block

        # Freeze base weights
        self.vae.eval().requires_grad_(False)
        self.unet.eval().requires_grad_(False)

        # Inject LoRA adapters
        self._lora_params: list[nn.Parameter] = []
        self._inject_lora()

    def _inject_lora(self) -> None:
        """Inject LoRA adapters into targeted attention/MLP layers."""
        from peft import LoraConfig, get_peft_model

        target_modules = set(self.lora_targets)
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=list(target_modules),
            lora_dropout=0.0,
            bias="none",
        )
        self.unet = get_peft_model(self.unet, lora_config)
        # Collect trainable LoRA params
        for _name, param in self.unet.named_parameters():
            if param.requires_grad:
                self._lora_params.append(param)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self._lora_params)

    @torch.inference_mode()
    def extract(
        self,
        images: torch.Tensor,
        *,
        prompt: str,
    ) -> tuple[DepthFeatureContract, tuple[torch.Tensor, ...]]:
        """Extract LoRA-adapted UNet features (same format as FrozenSDXL)."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected images shaped [B, 3, H, W]")
        if not prompt:
            raise ValueError("prompt must be non-empty")

        vae_parameter = next(self.vae.parameters())
        vae_images = images.to(
            device=vae_parameter.device, dtype=vae_parameter.dtype,
        )
        latents = vae_encode(self.vae, vae_images, self.model_config)
        unet_parameter = next(self.unet.parameters())
        latents = latents.to(
            device=unet_parameter.device, dtype=unet_parameter.dtype,
        )
        batch_size = latents.shape[0]

        text = self.generator.encode_prompts([prompt] * batch_size)
        prompt_embeds = text["prompt_embeds"].to(
            device=unet_parameter.device, dtype=unet_parameter.dtype,
        )
        pooled_embeds = text.get("pooled_prompt_embeds")
        if isinstance(pooled_embeds, torch.Tensor):
            pooled_embeds = pooled_embeds.to(
                device=unet_parameter.device, dtype=unet_parameter.dtype,
            )
        added_cond_kwargs = build_added_cond_kwargs(
            config=self.model_config,
            pooled_embeds=pooled_embeds,
            resolution=(images.shape[-2], images.shape[-1]),
            device=unet_parameter.device,
            dtype=unet_parameter.dtype,
            batch_size=batch_size,
        )

        captures: dict[str, torch.Tensor] = {}
        handles: list[Any] = []

        def capture(name: str):
            def hook(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
                del module, args
                captures[name] = _primary_tensor(output, name=name)
            return hook

        try:
            for index, block in enumerate(self.down_blocks):
                handles.append(block.register_forward_hook(capture(f"down_{index}")))
            handles.append(self.mid_block.register_forward_hook(capture("mid")))
            timesteps = torch.full(
                (batch_size,), self.timestep,
                device=unet_parameter.device, dtype=torch.long,
            )
            unet_output = self.unet(
                latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )
            final = _primary_tensor(unet_output, name="unet_output")
        finally:
            for handle in handles:
                handle.remove()

        ordered_names = (
            "vae_latent",
            *(f"down_{index}" for index in range(len(self.down_blocks))),
            "mid",
            "unet_output",
        )
        ordered_tensors = (
            latents.detach().float(),
            *(captures[name] for name in ordered_names[1:-1]),
            final.detach().float(),
        )
        contract = DepthFeatureContract(
            names=ordered_names,
            channels=tuple(int(t.shape[1]) for t in ordered_tensors),
        )
        contract.validate()
        return contract, tuple(ordered_tensors)


class ScalarDepthProbeV6(nn.Module):
    """V6 decoder with multi-timestep fusion + progressive skip decoder.

    Key improvements over V5:
    1. **Multi-timestep feature fusion** — handles channel-concatenated features
       from multiple timesteps, providing richer temporal information.
    2. **Progressive multi-scale skip decoder** — assigns features to decoder
       stages by spatial resolution match (not just deepest-first).
    3. **Feature compression** — uses 1x1 conv to compress high-channel multi-ts
       features before injection, reducing parameter count.
    4. **Deep supervision** — auxiliary output heads at intermediate decoder
       stages for gradient flow during training.
    """

    def __init__(
        self,
        contract: DepthFeatureContract,
        *,
        width: int = 128,
        upsample_stages: int = 4,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        contract.validate()
        if width <= 0:
            raise ValueError("width must be positive")
        if upsample_stages < 0:
            raise ValueError("upsample_stages must be non-negative")
        self.contract = contract
        self.deep_supervision = deep_supervision

        # Compress each feature to `width` channels (handles multi-ts concat)
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, width, 1),
                nn.GroupNorm(_group_count(width), width),
                nn.SiLU(),
            )
            for channels in contract.channels
        ])

        num_features = len(contract.channels)
        self.fusion = nn.Sequential(
            _conv_block(width * num_features, width * 2),
            _conv_block(width * 2, width),
        )

        # Group features by spatial resolution for skip assignment
        # Features with different spatial sizes get assigned to matching decoder stages
        # Decoder starts at the smallest spatial size and upsamples progressively
        stages = []
        current_width = width
        for stage in range(upsample_stages):
            next_width = max(16, current_width // 2)
            # Assign skip from features, cycling through projected features
            skip_ch = width if stage < num_features else None
            stages.append(_SkipDecoderStage(current_width, next_width, skip_ch))
            current_width = next_width
        self.decoder_stages = nn.ModuleList(stages)

        # Deep supervision auxiliary heads
        if deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(max(16, width // (2 ** (s + 1))), 1, 1)
                for s in range(upsample_stages - 1)
            ])
        else:
            self.aux_heads = nn.ModuleList()

        self.output = nn.Conv2d(current_width, 1, 1)

        # Compute skip assignment: match features to stages by resolution
        # We'll use a simple heuristic: assign features in reverse order
        # (deepest features to earliest stages)
        num_skips = min(upsample_stages, num_features)
        self._skip_proj_indices = tuple(
            range(num_features - 1, num_features - 1 - num_skips, -1)
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        *,
        output_size: tuple[int, int],
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if len(features) != len(self.contract.names):
            raise ValueError(
                f"feature count {len(features)} does not match contract "
                f"{len(self.contract.names)}"
            )

        # Project all features
        reference_size = max(
            (tuple(f.shape[-2:]) for f in features),
            key=lambda s: s[0] * s[1],
        )
        projected_all = []
        for projection, feature in zip(self.projections, features, strict=True):
            param = next(projection.parameters())
            level = projection(feature.to(device=param.device, dtype=param.dtype))
            if level.shape[-2:] != reference_size:
                level = F.interpolate(
                    level, size=reference_size,
                    mode="bilinear", align_corners=False,
                )
            projected_all.append(level)

        # Fuse
        hidden = self.fusion(torch.cat(projected_all, dim=1))

        # Progressive decoding with skip connections
        aux_outputs: list[torch.Tensor] = []
        for stage_idx, stage_module in enumerate(self.decoder_stages):
            skip = None
            if stage_idx < len(self._skip_proj_indices):
                proj_idx = self._skip_proj_indices[stage_idx]
                skip = projected_all[proj_idx]
            hidden = stage_module(hidden, skip)

            # Deep supervision: auxiliary output at intermediate stages
            if self.deep_supervision and stage_idx < len(self.aux_heads):
                aux_logits = self.aux_heads[stage_idx](hidden)
                aux_out = torch.sigmoid(aux_logits)
                if aux_out.shape[-2:] != output_size:
                    aux_out = F.interpolate(
                        aux_out, size=output_size,
                        mode="bilinear", align_corners=False,
                    )
                aux_outputs.append(aux_out)

        logits = self.output(hidden)
        if logits.shape[-2:] != output_size:
            logits = F.interpolate(
                logits, size=output_size,
                mode="bilinear", align_corners=False,
            )
        main_output = torch.sigmoid(logits)

        if return_aux and aux_outputs:
            return main_output, aux_outputs
        return main_output


def v6_composite_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    aux_predictions: list[torch.Tensor] | None = None,
    *,
    gradient_weight: float = 0.5,
    pearson_weight: float = 0.3,
    ssim_weight: float = 0.2,
    edge_weight: float = 0.15,
    aux_weight: float = 0.3,
    cosine_schedule: bool = True,
    epoch: int = 0,
    max_epochs: int = 250,
) -> torch.Tensor:
    """V6 composite loss with deep supervision.

    Extends V5 with:
    - Deep supervision auxiliary losses (weighted, optionally cosine-scheduled)
    - Cosine schedule gradually reduces aux weight as training progresses,
      letting the main output dominate late in training.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError("Expected scalar depth shaped [B, 1, H, W]")

    # Base V5 loss
    value_loss = F.smooth_l1_loss(prediction, target, beta=0.02)
    gradient_loss = _spatial_gradient_l1(prediction, target)
    correlation_loss = 1.0 - _pearson_per_sample(prediction, target).mean()
    ssim_loss = _ssim_loss(prediction, target)
    edge_loss = _edge_aware_loss(prediction, target)

    result = (
        value_loss
        + gradient_weight * gradient_loss
        + pearson_weight * correlation_loss
        + ssim_weight * ssim_loss
        + edge_weight * edge_loss
    )

    # Deep supervision auxiliary losses
    if aux_predictions:
        if cosine_schedule and max_epochs > 0:
            progress = min(epoch / max_epochs, 1.0)
            effective_aux_weight = aux_weight * (0.5 * (1 + math.cos(math.pi * progress)))
        else:
            effective_aux_weight = aux_weight

        for aux_pred in aux_predictions:
            if aux_pred.shape != target.shape:
                aux_pred = F.interpolate(
                    aux_pred, size=target.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
            aux_value = F.smooth_l1_loss(aux_pred, target, beta=0.02)
            aux_grad = _spatial_gradient_l1(aux_pred, target)
            result = result + effective_aux_weight * (aux_value + 0.3 * aux_grad)

    if not torch.isfinite(result):
        raise FloatingPointError("V6 composite loss is non-finite")
    return result


__all__ = [
    "DepthFeatureContract",
    "FrozenSDXLDepthFeatureExtractor",
    "MultiTimestepFeatureExtractor",
    "LoRAAdaptedFeatureExtractor",
    "ScalarDepthProbe",
    "ScalarDepthProbeV5",
    "ScalarDepthProbeV6",
    "depth_feature_contribution_gate_passed",
    "depth_probe_gate_passed",
    "evaluate_scalar_depth_predictions",
    "scalar_depth_probe_loss",
    "v5_composite_loss",
    "v6_composite_loss",
]
