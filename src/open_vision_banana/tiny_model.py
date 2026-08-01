"""Tiny standalone vision model for Open-Vision-Banana.

A lightweight encoder-decoder that directly maps RGB images to vision-task
RGB outputs (depth, normal, edge, semantic, generation) without any
diffusion-model backbone dependency.  The architecture is a compact U-Net
with task-conditioned modulation, designed for < 50 M parameters and
on-device inference.

Key design decisions:
- Input:  [B, 3, H, W] in [-1, 1]  (matches existing V3FlatDataset)
- Output: [B, 3, H, W] in [0, 1]   (matches all RGB-encoded targets)
- Task selection via learned embedding added to the bottleneck
- Depthwise-separable convolutions to keep parameter count low
- GroupNorm + SiLU activation (standard for dense prediction)
- Multi-scale skip connections (U-Net style)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

# ── Task registry ────────────────────────────────────────────────

TINY_TASKS: tuple[str, ...] = (
    "depth",
    "normal",
    "edge",
    "semantic",
    "generation",
)
"""Canonical task names accepted by :class:`TinyVisionBanana`."""

DEFAULT_TASK_INDEX: dict[str, int] = {name: i for i, name in enumerate(TINY_TASKS)}


# ── Config ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class TinyVisionBananaConfig:
    """Configuration for :class:`TinyVisionBanana`."""

    in_channels: int = 3
    out_channels: int = 3
    base_width: int = 32
    """Starting feature width.  Doubled at each downsample stage."""
    num_stages: int = 4
    """Number of encoder downsample stages."""
    block_depth: int = 2
    """Conv blocks per stage."""
    task_embed_dim: int = 32
    """Dimension of the learned task embedding."""
    task_embed_init_std: float = 0.02
    use_residual: bool = True
    """Add a residual skip from input (scaled) to output."""
    output_activation: str = "sigmoid"
    """Activation applied to the final output ('sigmoid' or 'none')."""
    resolution: int = 256
    """Expected input resolution (informational, not enforced)."""

    def validate(self) -> None:
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("channel counts must be positive")
        if self.base_width < 8:
            raise ValueError("base_width must be >= 8")
        if not 2 <= self.num_stages <= 6:
            raise ValueError("num_stages must be in [2, 6]")
        if self.block_depth < 1:
            raise ValueError("block_depth must be >= 1")
        if self.task_embed_dim < 1:
            raise ValueError("task_embed_dim must be positive")
        if self.task_embed_init_std < 0:
            raise ValueError("task_embed_init_std must be non-negative")
        if self.output_activation not in ("sigmoid", "none"):
            raise ValueError(f"unsupported output_activation: {self.output_activation!r}")


# ── Building blocks ───────────────────────────────────────────────


def _group_count(channels: int, max_groups: int = 8) -> int:
    """Pick the largest group count that divides ``channels``."""
    for g in (max_groups, 4, 2):
        if channels % g == 0:
            return g
    return 1


class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable conv (depthwise + pointwise) for efficiency."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False,
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.norm = nn.GroupNorm(_group_count(out_ch), out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


class ConvBlock(nn.Module):
    """Stack of ``depth`` depthwise-separable conv blocks with residual."""

    def __init__(self, in_ch: int, out_ch: int, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(depth):
            layers.append(DepthwiseSeparableConv(in_ch if i == 0 else out_ch, out_ch))
        self.layers = nn.Sequential(*layers)
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        if self.residual is not None:
            out = out + self.residual(x)
        else:
            out = out + x
        return out


class Downsample(nn.Module):
    """Strided 3x3 conv for spatial downsampling."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Upsample(nn.Module):
    """Bilinear upsample + conv refinement."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.act(self.norm(self.conv(x)))


# ── Main model ───────────────────────────────────────────────────


class TinyVisionBanana(nn.Module):
    """Standalone lightweight vision-to-vision model.

    Architecture::

        Input [-1,1] RGB
            |
        [Stem] 3 -> base_width
            |
        [Encoder] base_width -> ... -> base_width * 2^num_stages
            |        (skip connections saved at each stage)
            v
        [Bottleneck] + task embedding modulation
            |
        [Decoder] ... -> base_width
            |        (skip connections fused)
            v
        [Output head] base_width -> 3
            |
        Output [0,1] RGB

    Parameters
    ----------
    config : TinyVisionBananaConfig
        Architecture configuration.
    tasks : sequence of str
        Task names to register embeddings for.  Defaults to
        :data:`TINY_TASKS`.
    """

    def __init__(
        self,
        config: TinyVisionBananaConfig | None = None,
        *,
        tasks: Sequence[str] = TINY_TASKS,
    ) -> None:
        super().__init__()
        self.config = config or TinyVisionBananaConfig()
        self.config.validate()
        self.task_names = tuple(tasks)
        self.task_index = {name: i for i, name in enumerate(self.task_names)}

        w = self.config.base_width
        num_stages = self.config.num_stages
        depth = self.config.block_depth

        # Stem: 3 -> base_width
        self.stem = ConvBlock(self.config.in_channels, w, depth=1)

        # Encoder: each stage = ConvBlock + Downsample
        # After ConvBlock: width doubles (skip saved here)
        # After Downsample: spatial halves, width stays
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.encoder_widths: list[int] = []  # output width of each encoder block (skip channels)
        current_width = w
        for stage in range(num_stages):
            out_width = current_width * 2
            self.encoder_blocks.append(ConvBlock(current_width, out_width, depth))
            self.downsamplers.append(Downsample(out_width))
            self.encoder_widths.append(out_width)
            current_width = out_width

        # Bottleneck
        bottleneck_width = current_width
        self.bottleneck = ConvBlock(current_width, current_width, depth)

        # Task embedding modulation at bottleneck
        self.task_embed = nn.Embedding(len(self.task_names), self.config.task_embed_dim)
        nn.init.normal_(self.task_embed.weight, std=self.config.task_embed_init_std)
        self.task_to_bottleneck = nn.Linear(
            self.config.task_embed_dim, bottleneck_width * 2,
        )

        # Decoder: reverse order through encoder skips
        # Each decoder stage: Upsample(current) -> cat(skip) -> ConvBlock(current+skip -> skip_width)
        self.upsamplers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for stage in range(num_stages):
            skip_width = self.encoder_widths[num_stages - 1 - stage]
            self.upsamplers.append(Upsample(current_width))
            self.decoder_blocks.append(
                ConvBlock(current_width + skip_width, skip_width, depth)
            )
            current_width = skip_width

        # Output head
        self.output_head = nn.Sequential(
            nn.Conv2d(current_width, current_width, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(current_width), current_width),
            nn.SiLU(),
            nn.Conv2d(current_width, self.config.out_channels, 1),
        )

        # Residual scaling (input -> output)
        if self.config.use_residual:
            self.residual_scale = nn.Parameter(torch.tensor(0.1))

    # ── Public API ──────────────────────────────────────────────

    def forward(
        self,
        image: torch.Tensor,
        task: str | torch.Tensor,
        *,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Run a forward pass.

        Parameters
        ----------
        image : Tensor
            Input image ``[B, 3, H, W]`` in ``[-1, 1]``.
        task : str or Tensor
            Task name (str) or task index tensor ``[B]`` (long).
        output_size : optional
            If provided, the output is interpolated to this ``(H, W)``.

        Returns
        -------
        Tensor
            Output ``[B, 3, H, W]`` in ``[0, 1]``.
        """
        if image.ndim != 4 or image.shape[1] != self.config.in_channels:
            raise ValueError(
                f"Expected image [{image.shape[0]}, {self.config.in_channels}, H, W], "
                f"got {list(image.shape)}"
            )

        # Resolve task indices
        if isinstance(task, str):
            if task not in self.task_index:
                raise ValueError(
                    f"Unknown task {task!r}. Registered: {self.task_names}"
                )
            task_idx = torch.full(
                (image.shape[0],), self.task_index[task],
                device=image.device, dtype=torch.long,
            )
        else:
            task_idx = task.to(device=image.device, dtype=torch.long)

        # Stem
        x = self.stem(image)

        # Encoder with skip connections
        skips: list[torch.Tensor] = []
        for enc_block, down in zip(self.encoder_blocks, self.downsamplers, strict=True):
            x = enc_block(x)
            skips.append(x)
            x = down(x)

        # Bottleneck with task modulation
        x = self.bottleneck(x)
        task_emb = self.task_embed(task_idx)
        modulation = self.task_to_bottleneck(task_emb)
        scale = modulation[:, : x.shape[1]].unsqueeze(-1).unsqueeze(-1)
        shift = modulation[:, x.shape[1]:].unsqueeze(-1).unsqueeze(-1)
        x = x * (1.0 + scale) + shift

        # Decoder with skip fusion
        for up, dec_block, skip in zip(
            self.upsamplers,
            self.decoder_blocks,
            reversed(skips),
            strict=True,
        ):
            x = up(x)
            # Ensure spatial dimensions match
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False,
                )
            x = torch.cat([x, skip], dim=1)
            x = dec_block(x)

        # Output head
        out = self.output_head(x)

        # Residual from input
        if self.config.use_residual:
            # Map input from [-1,1] to [0,1] space and add scaled residual
            input_01 = (image + 1.0) / 2.0
            out = out + self.residual_scale * input_01

        if self.config.output_activation == "sigmoid":
            out = torch.sigmoid(out)

        if output_size is not None and out.shape[-2:] != output_size:
            out = F.interpolate(out, size=output_size, mode="bilinear", align_corners=False)

        return out

    def num_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        """Save model weights and config."""
        import json
        from pathlib import Path

        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path_obj / "model.safetensors")
        config_dict = {
            "architecture": "tiny_vision_banana",
            "config": {
                "in_channels": self.config.in_channels,
                "out_channels": self.config.out_channels,
                "base_width": self.config.base_width,
                "num_stages": self.config.num_stages,
                "block_depth": self.config.block_depth,
                "task_embed_dim": self.config.task_embed_dim,
                "use_residual": self.config.use_residual,
                "output_activation": self.config.output_activation,
                "resolution": self.config.resolution,
            },
            "tasks": list(self.task_names),
            "num_parameters": self.num_parameters(),
        }
        (path_obj / "config.json").write_text(
            json.dumps(config_dict, indent=2), encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str, *, device: str = "cpu") -> TinyVisionBanana:
        """Load model from a saved directory."""
        import json
        from pathlib import Path

        path_obj = Path(path)
        with (path_obj / "config.json").open("r") as f:
            meta = json.load(f)
        cfg = TinyVisionBananaConfig(**meta["config"])
        model = cls(cfg, tasks=meta["tasks"])
        state = torch.load(path_obj / "model.safetensors", map_location=device)
        model.load_state_dict(state)
        model.to(device)
        return model


# ── Loss functions ────────────────────────────────────────────────


def tiny_vision_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    gradient_weight: float = 0.5,
    pearson_weight: float = 0.1,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """Multi-component loss for tiny vision model training.

    Combines:
    - Smooth L1 on RGB values
    - Spatial gradient matching (multi-scale)
    - Pearson correlation penalty

    Parameters
    ----------
    prediction : Tensor [B, 3, H, W] in [0, 1]
    target : Tensor [B, 3, H, W] in [0, 1]
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError("Expected [B, 3, H, W]")

    # Value loss
    value_loss = F.smooth_l1_loss(prediction, target, beta=epsilon)

    # Gradient loss (spatial)
    grad_loss = prediction.new_zeros(())
    if prediction.shape[-1] > 1:
        pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
        target_dx = target[..., :, 1:] - target[..., :, :-1]
        grad_loss = grad_loss + F.l1_loss(pred_dx, target_dx)
    if prediction.shape[-2] > 1:
        pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
        target_dy = target[..., 1:, :] - target[..., :-1, :]
        grad_loss = grad_loss + F.l1_loss(pred_dy, target_dy)

    # Pearson correlation (per-sample, per-channel)
    pred_flat = prediction.flatten(1)
    target_flat = target.flatten(1)
    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    numerator = (pred_centered * target_centered).sum(dim=1)
    denominator = (
        pred_centered.square().sum(dim=1)
        * target_centered.square().sum(dim=1)
    ).clamp_min(1e-12).sqrt()
    pearson = numerator / denominator
    corr_loss = 1.0 - pearson.mean()

    result = value_loss + gradient_weight * grad_loss + pearson_weight * corr_loss
    if not torch.isfinite(result):
        raise FloatingPointError("tiny vision loss is non-finite")
    return result


@torch.no_grad()
def evaluate_tiny_predictions(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Evaluate RGB predictions against targets.

    Returns RGB MAE, RMSE, and per-image Pearson correlation.
    """
    if prediction.shape != target.shape:
        raise ValueError("shape mismatch")
    prediction = prediction.detach().float().cpu().clamp(0, 1)
    target = target.detach().float().cpu().clamp(0, 1)

    maes = []
    rmses = []
    pearsons = []
    for pred_s, target_s in zip(prediction, target, strict=True):
        p_flat = pred_s.flatten()
        t_flat = target_s.flatten()
        maes.append(float((p_flat - t_flat).abs().mean()))
        rmses.append(float((p_flat - t_flat).square().mean().sqrt()))
        if p_flat.std() > 1e-8 and t_flat.std() > 1e-8:
            r = float(torch.corrcoef(torch.stack((p_flat, t_flat)))[0, 1])
        else:
            r = 0.0
        pearsons.append(r)

    return {
        "rgb_mae": sum(maes) / len(maes),
        "rgb_rmse": sum(rmses) / len(rmses),
        "pearson": sum(pearsons) / len(pearsons),
    }


def tiny_overfit_gate_passed(
    metrics: dict[str, float],
    *,
    max_rgb_mae: float = 0.05,
    min_pearson: float = 0.95,
) -> bool:
    """Check if overfit gate is passed (8-sample memorization)."""
    required = ("rgb_mae", "pearson")
    if any(key not in metrics for key in required):
        raise KeyError("metrics incomplete")
    if any(not torch.isfinite(torch.tensor(float(metrics[k]))) for k in required):
        return False
    return metrics["rgb_mae"] <= max_rgb_mae and metrics["pearson"] >= min_pearson


__all__ = [
    "DEFAULT_TASK_INDEX",
    "TINY_TASKS",
    "ConvBlock",
    "DepthwiseSeparableConv",
    "Downsample",
    "TinyVisionBanana",
    "TinyVisionBananaConfig",
    "Upsample",
    "evaluate_tiny_predictions",
    "tiny_overfit_gate_passed",
    "tiny_vision_loss",
]
