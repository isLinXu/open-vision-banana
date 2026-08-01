"""TinyVisionBanana V2 — improved multi-task architecture.

Key improvements over V1:
1. **Per-task output heads** — each task gets a dedicated lightweight decoder
   head, so edge/semantic don't have to fight depth/generation for the same
   conv weights.
2. **Task-conditional residual** — generation benefits from a strong input
   residual; edge/semantic need near-zero residual to avoid being dragged
   toward the input RGB.
3. **Larger capacity** — base_width 48 (≈26M params) with 4 stages.
4. **FiLM task conditioning** — beyond bottleneck embedding, task-dependent
   affine modulation is applied at each decoder stage for stronger task
   separation.

Architecture::

    Input [-1,1] RGB → Stem → Encoder(4 stages) → Bottleneck(+task embed)
        → Decoder(4 stages, skip+FiLM) → Per-task Head → Output [0,1] RGB
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from open_vision_banana.tiny_model import (
    ConvBlock,
    Downsample,
    Upsample,
    _group_count,
    TINY_TASKS,
    DEFAULT_TASK_INDEX,
    evaluate_tiny_predictions,
    tiny_overfit_gate_passed,
)


# ── Config ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class TinyVisionBananaV2Config:
    """Configuration for TinyVisionBananaV2."""

    in_channels: int = 3
    out_channels: int = 3
    base_width: int = 48
    num_stages: int = 4
    block_depth: int = 2
    task_embed_dim: int = 64
    task_embed_init_std: float = 0.02
    use_residual: bool = True
    output_activation: str = "sigmoid"
    resolution: int = 256

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
        if self.output_activation not in ("sigmoid", "none"):
            raise ValueError(f"unsupported output_activation: {self.output_activation!r}")


# ── FiLM conditioning ────────────────────────────────────────────


class FiLMModulation(nn.Module):
    """Feature-wise Linear Modulation: gamma, beta from task embedding."""

    def __init__(self, task_embed_dim: int, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.to_gamma_beta = nn.Linear(task_embed_dim, num_features * 2)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, x: torch.Tensor, task_embed: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W], task_embed: [B, D] → modulated x."""
        gb = self.to_gamma_beta(task_embed)
        gamma, beta = gb.chunk(2, dim=1)
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        return x * (1.0 + gamma) + beta


# ── Per-task output head ─────────────────────────────────────────


class TaskSpecificHead(nn.Module):
    """Lightweight per-task output head.

    2 conv layers + GN + SiLU + 1x1 conv to 3 channels.
    """

    def __init__(self, in_width: int, out_channels: int = 3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_width, in_width, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(in_width), in_width),
            nn.SiLU(),
            nn.Conv2d(in_width, in_width, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(in_width), in_width),
            nn.SiLU(),
            nn.Conv2d(in_width, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ── Model ────────────────────────────────────────────────────────


class TinyVisionBananaV2(nn.Module):
    """V2 standalone lightweight vision-to-vision model.

    Improvements:
    - Per-task output heads (dedicated per task)
    - FiLM conditioning at each decoder stage
    - Task-conditional residual scaling (learned per task)
    """

    def __init__(
        self,
        config: TinyVisionBananaV2Config | None = None,
        *,
        tasks: Sequence[str] = TINY_TASKS,
    ) -> None:
        super().__init__()
        self.config = config or TinyVisionBananaV2Config()
        self.config.validate()
        self.task_names = tuple(tasks)
        self.task_index = {name: i for i, name in enumerate(self.task_names)}

        w = self.config.base_width
        num_stages = self.config.num_stages
        depth = self.config.block_depth

        # Stem
        self.stem = ConvBlock(self.config.in_channels, w, depth=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.encoder_widths: list[int] = []
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

        # Task embedding
        self.task_embed = nn.Embedding(len(self.task_names), self.config.task_embed_dim)
        nn.init.normal_(self.task_embed.weight, std=self.config.task_embed_init_std)

        # Bottleneck task modulation
        self.task_to_bottleneck = nn.Linear(
            self.config.task_embed_dim, bottleneck_width * 2,
        )

        # FiLM at each decoder stage
        self.film_modules = nn.ModuleList()
        for stage in range(num_stages):
            skip_width = self.encoder_widths[num_stages - 1 - stage]
            self.film_modules.append(FiLMModulation(self.config.task_embed_dim, skip_width))

        # Decoder
        self.upsamplers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for stage in range(num_stages):
            skip_width = self.encoder_widths[num_stages - 1 - stage]
            self.upsamplers.append(Upsample(current_width))
            self.decoder_blocks.append(
                ConvBlock(current_width + skip_width, skip_width, depth)
            )
            current_width = skip_width

        # Per-task output heads
        self.task_heads = nn.ModuleDict({
            task: TaskSpecificHead(current_width, self.config.out_channels)
            for task in self.task_names
        })

        # Task-conditional residual scale (learned per task)
        if self.config.use_residual:
            self.residual_scales = nn.Parameter(
                torch.full((len(self.task_names),), 0.1)
            )
            # Initialize: generation gets higher residual, edge/semantic near 0
            init_scales = {"generation": 0.3, "depth": 0.1, "normal": 0.1,
                          "edge": 0.0, "semantic": 0.0}
            for i, task in enumerate(self.task_names):
                if task in init_scales:
                    self.residual_scales.data[i] = init_scales[task]

    def _get_task_embed(self, task_idx: torch.Tensor) -> torch.Tensor:
        """Get task embedding [B, D] from task indices [B]."""
        return self.task_embed(task_idx)

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
        image : Tensor [B, 3, H, W] in [-1, 1]
        task : str or Tensor — task name or task index [B] (long)
        output_size : optional (H, W) to interpolate output

        Returns
        -------
        Tensor [B, 3, H, W] in [0, 1]
        """
        if image.ndim != 4 or image.shape[1] != self.config.in_channels:
            raise ValueError(
                f"Expected image [{image.shape[0]}, {self.config.in_channels}, H, W], "
                f"got {list(image.shape)}"
            )

        # Resolve task
        if isinstance(task, str):
            if task not in self.task_index:
                raise ValueError(f"Unknown task {task!r}. Registered: {self.task_names}")
            task_idx = torch.full(
                (image.shape[0],), self.task_index[task],
                device=image.device, dtype=torch.long,
            )
        else:
            task_idx = task.to(device=image.device, dtype=torch.long)

        task_embed = self._get_task_embed(task_idx)  # [B, D]

        # Stem
        x = self.stem(image)

        # Encoder
        skips: list[torch.Tensor] = []
        for enc_block, down in zip(self.encoder_blocks, self.downsamplers, strict=True):
            x = enc_block(x)
            skips.append(x)
            x = down(x)

        # Bottleneck + task modulation
        x = self.bottleneck(x)
        gb = self.task_to_bottleneck(task_embed)
        gamma, beta = gb.chunk(2, dim=1)
        gamma = gamma.view(-1, x.shape[1], 1, 1)
        beta = beta.view(-1, x.shape[1], 1, 1)
        x = x * (1.0 + gamma) + beta

        # Decoder with FiLM
        for stage, (up, dec_block, film) in enumerate(
            zip(self.upsamplers, self.decoder_blocks, self.film_modules, strict=True)
        ):
            x = up(x)
            skip = skips[num_stages := self.config.num_stages - 1 - stage]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec_block(x)
            x = film(x, task_embed)

        # Per-task head: group samples by task to minimize head calls
        out = x.new_zeros(image.shape[0], self.config.out_channels, x.shape[-2], x.shape[-1])
        unique_tasks = task_idx.unique()
        for t in unique_tasks:
            mask = task_idx == t
            task_name = self.task_names[t.item()]
            head = self.task_heads[task_name]
            out[mask] = head(x[mask])

        # Task-conditional residual
        if self.config.use_residual:
            input_01 = (image + 1.0) / 2.0
            # Per-sample residual scale
            res_scale = self.residual_scales[task_idx]  # [B]
            res_scale = res_scale.view(-1, 1, 1, 1)
            out = out + res_scale * input_01

        if self.config.output_activation == "sigmoid":
            out = torch.sigmoid(out)

        if output_size is not None and out.shape[-2:] != output_size:
            out = F.interpolate(out, size=output_size, mode="bilinear", align_corners=False)

        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Task-aware loss ──────────────────────────────────────────────


def _dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Dice loss for binary/sparse targets."""
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return (1.0 - dice).mean()


def _ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Simplified SSIM loss (per-channel, 7x7 window approximated by avg pool)."""
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    mu_p = F.avg_pool2d(pred, 7, 3, 3)
    mu_t = F.avg_pool2d(target, 7, 3, 3)
    var_p = F.avg_pool2d(pred * pred, 7, 3, 3) - mu_p * mu_p
    var_t = F.avg_pool2d(target * target, 7, 3, 3) - mu_t * mu_t
    cov_pt = F.avg_pool2d(pred * target, 7, 3, 3) - mu_p * mu_t
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_p * mu_t + c1) * (2 * cov_pt + c2)) / \
               ((mu_p ** 2 + mu_t ** 2 + c1) * (var_p + var_t + c2))
    return 1.0 - ssim_map.mean()


def tiny_vision_loss_v2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    task: str | torch.Tensor,
    *,
    gradient_weight: float = 0.5,
    pearson_weight: float = 0.1,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """Task-aware multi-component loss for V2 model.

    Dispatches to task-specific loss:
    - depth/normal: Smooth L1 + gradient + Pearson (original)
    - edge: weighted BCE + Dice + gradient (handles sparse binary targets)
    - semantic: Smooth L1 + Dice + color-matching (handles color-coded segments)
    - generation: Smooth L1 + SSIM + gradient
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError("Expected [B, 3, H, W]")

    # Resolve task name
    if isinstance(task, torch.Tensor):
        # Take first task in batch (mixed-task batches not supported for loss dispatch)
        task_idx = task[0].item()
        task_names = TINY_TASKS
        task = task_names[task_idx]

    # ── Gradient loss (shared) ──
    def _gradient_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        gl = pred.new_zeros(())
        if pred.shape[-1] > 1:
            gl = gl + F.l1_loss(pred[..., :, 1:] - pred[..., :, :-1],
                                tgt[..., :, 1:] - tgt[..., :, :-1])
        if pred.shape[-2] > 1:
            gl = gl + F.l1_loss(pred[..., 1:, :] - pred[..., :-1, :],
                                tgt[..., 1:, :] - tgt[..., :-1, :])
        return gl

    # ── Pearson (shared) ──
    def _pearson_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.flatten(1)
        target_flat = tgt.flatten(1)
        pred_c = pred_flat - pred_flat.mean(dim=1, keepdim=True)
        target_c = target_flat - target_flat.mean(dim=1, keepdim=True)
        num = (pred_c * target_c).sum(dim=1)
        den = (pred_c.square().sum(dim=1) * target_c.square().sum(dim=1)).clamp_min(1e-12).sqrt()
        return 1.0 - (num / den).mean()

    if task == "edge":
        # Edge: sparse binary-ish target (~93% black)
        # Weighted BCE + Dice + gradient
        weight = torch.where(target > 0.3, 5.0, 1.0).to(prediction.device)
        bce = F.binary_cross_entropy(prediction.clamp(1e-6, 1 - 1e-6), target, weight=weight)
        dice = _dice_loss(prediction, target)
        grad = _gradient_loss(prediction, target)
        result = bce + 0.5 * dice + 0.3 * grad
        return result

    elif task == "semantic":
        # Semantic: color-coded segments, high variance
        # Smooth L1 + Dice (per-channel) + gradient
        value_loss = F.smooth_l1_loss(prediction, target, beta=epsilon)
        dice = _dice_loss(prediction, target)
        grad = _gradient_loss(prediction, target)
        result = value_loss + 0.3 * dice + 0.3 * grad
        return result

    elif task == "generation":
        # Generation: near-identity, needs perceptual quality
        # Smooth L1 + SSIM + gradient
        value_loss = F.smooth_l1_loss(prediction, target, beta=epsilon)
        ssim = _ssim_loss(prediction, target)
        grad = _gradient_loss(prediction, target)
        result = value_loss + 0.5 * ssim + 0.3 * grad
        return result

    else:
        # depth / normal: original V1 loss
        value_loss = F.smooth_l1_loss(prediction, target, beta=epsilon)
        grad = _gradient_loss(prediction, target)
        corr = _pearson_loss(prediction, target)
        result = value_loss + gradient_weight * grad + pearson_weight * corr
        return result


__all__ = [
    "TinyVisionBananaV2",
    "TinyVisionBananaV2Config",
    "FiLMModulation",
    "TaskSpecificHead",
    "tiny_vision_loss_v2",
    "evaluate_tiny_predictions",
    "tiny_overfit_gate_passed",
]
