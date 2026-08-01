"""TinyVisionBanana V5 — architecture-level fixes for task isolation & detail preservation.

Key architectural changes over V2 (V4 only changed loss/training, not architecture):

1. **Per-task residual gating (not learnable drift)**
   - Generation: residual ON (identity is the goal)
   - Depth/Normal/Edge/Semantic: residual OFF (prevents RGB contamination)
   - Implemented as a fixed boolean mask, NOT a learnable parameter
   - This permanently fixes the edge "identity mapping" bug where
     residual_scale drifted to 0.1133 and made black output impossible.

2. **Non-zero FiLM initialization**
   - gamma initialized to ~1.0 (identity-like), beta to small random
   - This ensures task conditioning is active from step 0
   - V2's zero init caused all tasks to share identical features in
     early training, leading to semantic→depth task collapse.

3. **Per-task output activation**
   - depth/normal/generation/semantic: tanh (rescaled to [0,1])
     - Avoids sigmoid's gray-default (0.5) that maps to teal/cyan
     - tanh(0) = 0 → rescaled to 0.5, but gradients are symmetric
     - Allows negative logits to produce values < 0.5 naturally
   - edge: sigmoid (kept, since edge is binary-ish [0,1])
     - BUT without residual contamination, sigmoid can now reach 0

4. **Deeper task heads for high-frequency detail**
   - 4 conv layers (was 2) with residual skip within the head
   - Provides more capacity for fine-grained edge/normal prediction

5. **Wider bottleneck (base_width 48 default)**
   - V2 used base_width=32 → bottleneck 16x16x512
   - V5 defaults to base_width=48 → bottleneck 16x16x768
   - More capacity at the information bottleneck

Architecture (unchanged U-Net skeleton):
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
from open_vision_banana.tiny_model_v2 import (
    FiLMModulation,
    _dice_loss,
    _ssim_loss,
)


# ── Config ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class TinyVisionBananaV5Config:
    """Configuration for TinyVisionBananaV5."""

    in_channels: int = 3
    out_channels: int = 3
    base_width: int = 48
    num_stages: int = 4
    block_depth: int = 2
    task_embed_dim: int = 64
    task_embed_init_std: float = 0.02
    resolution: int = 256

    # Per-task residual: which tasks get input RGB skip connection
    # Generation needs it (near-identity); all others must NOT have it
    residual_tasks: tuple[str, ...] = ("generation",)

    # Per-task output activation
    # "tanh" for continuous tasks (depth, normal, generation, semantic)
    # "sigmoid" for binary-ish tasks (edge)
    output_activation: str = "tanh"  # global default
    edge_activation: str = "sigmoid"  # override for edge

    # Head depth (number of conv layers in task head)
    head_depth: int = 4

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
        if self.output_activation not in ("sigmoid", "tanh", "none"):
            raise ValueError(f"unsupported output_activation: {self.output_activation!r}")
        if self.edge_activation not in ("sigmoid", "tanh", "none"):
            raise ValueError(f"unsupported edge_activation: {self.edge_activation!r}")


# ── Non-zero FiLM conditioning ────────────────────────────────────


class NonZeroFiLMModulation(nn.Module):
    """FiLM with non-zero initialization for immediate task conditioning.

    Unlike V2's FiLMModulation (zero init → identity at step 0),
    this initializes gamma≈1 and beta≈0 with small random perturbation,
    so each task gets distinct modulation from the very first forward pass.
    """

    def __init__(self, task_embed_dim: int, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.to_gamma_beta = nn.Linear(task_embed_dim, num_features * 2)
        # Non-zero init: gamma ~ N(1, 0.02), beta ~ N(0, 0.02)
        # This ensures task conditioning is active immediately
        with torch.no_grad():
            nn.init.normal_(self.to_gamma_beta.weight, mean=0.0, std=0.01)
            # Split bias: first half → gamma bias (target ~1), second half → beta bias (target ~0)
            half = num_features
            nn.init.normal_(self.to_gamma_beta.bias[:half], mean=1.0, std=0.02)
            nn.init.normal_(self.to_gamma_beta.bias[half:], mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, task_embed: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W], task_embed: [B, D] → modulated x."""
        gb = self.to_gamma_beta(task_embed)
        gamma, beta = gb.chunk(2, dim=1)
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        return x * (1.0 + gamma) + beta


# ── Deeper per-task output head ───────────────────────────────────


class DeepTaskHead(nn.Module):
    """Deeper per-task output head with internal residual skip.

    4 conv layers (was 2 in V2) with GroupNorm + SiLU + internal residual.
    The extra depth provides more capacity for high-frequency detail
    reconstruction (edges, normal vectors, depth gradients).

    Architecture:
        input → conv1 → GN → SiLU → conv2 → GN → SiLU → +input →
              → conv3 → GN → SiLU → conv4 → GN → SiLU → 1x1 conv → output
    """

    def __init__(self, in_width: int, out_channels: int = 3, depth: int = 4):
        super().__init__()
        self.depth = depth

        layers = []
        for i in range(depth):
            layers.append(nn.Conv2d(in_width, in_width, 3, padding=1, bias=False))
            layers.append(nn.GroupNorm(_group_count(in_width), in_width))
            layers.append(nn.SiLU())
        # Final 1x1 conv to produce output channels
        layers.append(nn.Conv2d(in_width, out_channels, 1))

        self.layers = nn.Sequential(*layers)

        # Internal residual: match channels for skip
        self.residual_proj = nn.Conv2d(in_width, in_width, 1, bias=False) if depth >= 4 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split into pre-residual and post-residual
        if self.residual_proj is not None and self.depth >= 4:
            # Take output before final 1x1 conv
            mid = self.layers[:-1](x)  # All conv+GN+SiLU layers
            res = self.residual_proj(x)
            mid = mid + res
            out = self.layers[-1](mid)  # Final 1x1 conv
            return out
        return self.layers(x)


# ── Model ────────────────────────────────────────────────────────


class TinyVisionBananaV5(nn.Module):
    """V5 lightweight multi-task vision-to-vision model.

    Architectural improvements over V2:
    - Fixed per-task residual mask (not learnable → no drift)
    - Non-zero FiLM initialization (task conditioning active from step 0)
    - Per-task output activation (tanh for continuous, sigmoid for edge)
    - Deeper task heads (4 layers + internal residual)
    """

    def __init__(
        self,
        config: TinyVisionBananaV5Config | None = None,
        *,
        tasks: Sequence[str] = TINY_TASKS,
    ) -> None:
        super().__init__()
        self.config = config or TinyVisionBananaV5Config()
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

        # Bottleneck task modulation (non-zero init)
        self.task_to_bottleneck = nn.Linear(
            self.config.task_embed_dim,
            bottleneck_width * 2,
        )
        with torch.no_grad():
            nn.init.normal_(self.task_to_bottleneck.weight, std=0.01)
            half = bottleneck_width
            nn.init.normal_(self.task_to_bottleneck.bias[:half], mean=1.0, std=0.02)
            nn.init.normal_(self.task_to_bottleneck.bias[half:], mean=0.0, std=0.02)

        # Non-zero FiLM at each decoder stage
        self.film_modules = nn.ModuleList()
        for stage in range(num_stages):
            skip_width = self.encoder_widths[num_stages - 1 - stage]
            self.film_modules.append(NonZeroFiLMModulation(self.config.task_embed_dim, skip_width))

        # Decoder
        self.upsamplers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for stage in range(num_stages):
            skip_width = self.encoder_widths[num_stages - 1 - stage]
            self.upsamplers.append(Upsample(current_width))
            self.decoder_blocks.append(ConvBlock(current_width + skip_width, skip_width, depth))
            current_width = skip_width

        # Deeper per-task output heads
        self.task_heads = nn.ModuleDict(
            {
                task: DeepTaskHead(
                    current_width, self.config.out_channels, depth=self.config.head_depth
                )
                for task in self.task_names
            }
        )

        # Fixed per-task residual mask (NOT learnable)
        # Only generation gets residual; all others are disabled
        self.residual_mask = torch.zeros(len(self.task_names))
        for i, task in enumerate(self.task_names):
            if task in self.config.residual_tasks:
                self.residual_mask[i] = 1.0
        # Register as buffer (not a parameter, moves with model but no gradients)
        self.register_buffer("residual_mask_buf", self.residual_mask)

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
                (image.shape[0],),
                self.task_index[task],
                device=image.device,
                dtype=torch.long,
            )
        else:
            task_idx = task.to(device=image.device, dtype=torch.long)

        task_embed = self.task_embed(task_idx)  # [B, D]

        # Stem
        x = self.stem(image)

        # Encoder
        skips: list[torch.Tensor] = []
        for enc_block, down in zip(self.encoder_blocks, self.downsamplers, strict=True):
            x = enc_block(x)
            skips.append(x)
            x = down(x)

        # Bottleneck + task modulation (non-zero init)
        x = self.bottleneck(x)
        gb = self.task_to_bottleneck(task_embed)
        gamma, beta = gb.chunk(2, dim=1)
        gamma = gamma.view(-1, x.shape[1], 1, 1)
        beta = beta.view(-1, x.shape[1], 1, 1)
        x = x * (1.0 + gamma) + beta

        # Decoder with non-zero FiLM
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

        # Per-task head: group samples by task
        out = x.new_zeros(image.shape[0], self.config.out_channels, x.shape[-2], x.shape[-1])
        unique_tasks = task_idx.unique()
        for t in unique_tasks:
            mask = task_idx == t
            task_name = self.task_names[t.item()]
            head = self.task_heads[task_name]
            out[mask] = head(x[mask])

        # Fixed per-task residual (only generation)
        res_mask = self.residual_mask_buf[task_idx]  # [B], 0 or 1
        res_mask = res_mask.view(-1, 1, 1, 1)
        if res_mask.any():
            input_01 = (image + 1.0) / 2.0
            out = out + res_mask * input_01

        # Per-task output activation
        for t in unique_tasks:
            mask = task_idx == t
            task_name = self.task_names[t.item()]
            if task_name == "edge":
                act = self.config.edge_activation
            else:
                act = self.config.output_activation

            if act == "sigmoid":
                out[mask] = torch.sigmoid(out[mask])
            elif act == "tanh":
                # tanh maps to [-1, 1], rescale to [0, 1]
                out[mask] = (torch.tanh(out[mask]) + 1.0) / 2.0
            # "none" = raw output

        if output_size is not None and out.shape[-2:] != output_size:
            out = F.interpolate(out, size=output_size, mode="bilinear", align_corners=False)

        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── V5 task-aware loss ────────────────────────────────────────────
# Re-export V4 loss functions for use with V5 model.
# The V4 loss is already well-designed (L1, multi-scale gradient, SSIM,
# histogram, ordinal, palette CE/projection). V5's architectural changes
# complement the V4 loss without requiring loss modifications.

from open_vision_banana.tiny_model_v4 import (  # noqa: E402, F401
    tiny_vision_loss_v4 as tiny_vision_loss_v5,
    _extract_palette,
    _palette_ce_loss,
    _palette_projection_loss,
    _multi_scale_gradient_loss,
    _histogram_loss,
    _ordinal_loss,
)


__all__ = [
    "TinyVisionBananaV5",
    "TinyVisionBananaV5Config",
    "NonZeroFiLMModulation",
    "DeepTaskHead",
    "tiny_vision_loss_v5",
    "evaluate_tiny_predictions",
    "tiny_overfit_gate_passed",
    "_extract_palette",
    "_palette_ce_loss",
    "_palette_projection_loss",
    "_multi_scale_gradient_loss",
    "_histogram_loss",
    "_ordinal_loss",
]
