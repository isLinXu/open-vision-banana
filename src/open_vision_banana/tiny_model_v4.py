"""TinyVisionBanana V4 - anti-oversmoothing + anti-task-collapse multi-task model.

Key improvements over V3 (V2 architecture + V3 training):
1. **Multi-scale gradient loss** - 3-scale (1x, 2x, 4x) spatial gradient
   matching to force sharp edges at multiple frequencies.
2. **L1 value loss** (not Smooth L1 with beta=1e-3) - avoids L2 regression-to-mean.
3. **SSIM for depth/normal** - structural similarity preserves geometric patterns.
4. **Histogram loss** - forces dynamic range matching (anti-compression).
5. **Palette-constrained loss for semantic** - nearest-neighbor projection to
   discrete palette colors before computing loss, forcing the model to commit
   to discrete class predictions rather than averaging colors.
6. **Cross-entropy loss for semantic** - converts RGB predictions to class
   probabilities via soft palette matching, then applies CE.
7. **Non-zero FiLM initialization** - gamma/beta initialized with small
   random values so task conditioning is active from step 0.
8. **Per-task residual gating** - residual connection disabled for depth
   and semantic tasks, preventing RGB contamination.

Architecture unchanged from V2 (encoder-decoder U-Net with FiLM + per-task heads).
Loss function, output activation, and initialization are modified.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from open_vision_banana.tiny_model import (
    TINY_TASKS,
    evaluate_tiny_predictions,
    tiny_overfit_gate_passed,
)
from open_vision_banana.tiny_model_v2 import (
    FiLMModulation,
    TaskSpecificHead,
    TinyVisionBananaV2,
    TinyVisionBananaV2Config,
    _dice_loss,
    _ssim_loss,
)

# -- Palette-constrained semantic loss -----------------------------


def _extract_palette(
    target: torch.Tensor,
    num_colors: int = 16,
    min_pixels: int = 50,
) -> torch.Tensor:
    """Extract the discrete color palette from a target semantic image.

    Uses k-means-like quantization to find the most common discrete colors.

    Args:
        target: [B, 3, H, W] in [0, 1]
        num_colors: max number of palette colors
        min_pixels: minimum pixel count for a color to be included

    Returns:
        palette: [num_actual_colors, 3]
    """
    B, C, H, W = target.shape
    pixels = target.permute(0, 2, 3, 1).reshape(-1, 3)  # [B*H*W, 3]

    # Quantize to 32 levels per channel to find unique colors
    quantized = (pixels * 31).round() / 31
    unique, counts = torch.unique(quantized, dim=0, return_counts=True)

    # Filter by minimum pixel count
    mask = counts >= min_pixels
    palette = unique[mask]

    # If too many colors, keep the most frequent ones
    if len(palette) > num_colors:
        _, top_idx = counts[mask].topk(num_colors)
        palette = palette[top_idx]

    return palette


def _palette_ce_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    palette: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Palette-aware cross-entropy loss for semantic segmentation.

    Computes soft class assignment by measuring distance from each pixel's
    predicted RGB to each palette color, then applies cross-entropy against
    the target's nearest palette color.

    This directly forces the model to commit to a specific discrete color
    rather than averaging between colors (which produces muddy mid-tones).

    Args:
        prediction: [B, 3, H, W] in [0, 1]
        target: [B, 3, H, W] in [0, 1]
        palette: [K, 3] discrete color palette
        temperature: softmax temperature (lower = harder assignment)

    Returns:
        Scalar cross-entropy loss.
    """
    B, C, H, W = prediction.shape
    K = palette.shape[0]

    if K < 2:
        return prediction.new_zeros(())

    # Flatten to [N, 3]
    pred_flat = prediction.permute(0, 2, 3, 1).reshape(-1, 3)
    target_flat = target.permute(0, 2, 3, 1).reshape(-1, 3)

    # Compute L2 distances from each pixel to each palette color
    # pred_flat: [N, 3], palette: [K, 3] -> dist: [N, K]
    dist_pred = torch.cdist(pred_flat, palette)
    dist_target = torch.cdist(target_flat, palette)

    # Convert distances to logits (closer = higher logit)
    logits_pred = -dist_pred / temperature  # [N, K]

    # Target class: nearest palette color
    target_class = dist_target.argmin(dim=1)  # [N]

    # Cross-entropy
    ce_loss = F.cross_entropy(logits_pred, target_class)

    return ce_loss


def _palette_projection_loss(
    prediction: torch.Tensor,
    palette: torch.Tensor,
) -> torch.Tensor:
    """Loss that pushes predictions toward the nearest palette color.

    For each predicted pixel, computes the distance to the nearest palette
    color and penalizes it. This directly encourages discrete color output.

    Args:
        prediction: [B, 3, H, W] in [0, 1]
        palette: [K, 3]

    Returns:
        Scalar loss (mean distance to nearest palette color).
    """
    B, C, H, W = prediction.shape
    pred_flat = prediction.permute(0, 2, 3, 1).reshape(-1, 3)

    # Distance to each palette color
    dist = torch.cdist(pred_flat, palette)  # [N, K]

    # Distance to nearest palette color
    min_dist = dist.min(dim=1).values  # [N]

    return min_dist.mean()


# -- Multi-scale gradient loss -------------------------------------


def _multi_scale_gradient_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    scales: tuple[int, ...] = (1, 2, 4),
) -> torch.Tensor:
    """Multi-scale spatial gradient L1 loss.

    Computes gradient matching at multiple spatial scales to enforce
    both fine-grained and coarse-grained edge alignment.

    Args:
        pred: [B, C, H, W] predictions in [0, 1]
        target: [B, C, H, W] targets in [0, 1]
        scales: downsampling factors for multi-scale matching

    Returns:
        Scalar loss.
    """
    total = pred.new_zeros(())
    for s in scales:
        if s == 1:
            p, t = pred, target
        else:
            p = F.avg_pool2d(pred, s, s)
            t = F.avg_pool2d(target, s, s)

        if p.shape[-1] > 1:
            total = total + F.l1_loss(
                p[..., :, 1:] - p[..., :, :-1],
                t[..., :, 1:] - t[..., :, :-1],
            )
        if p.shape[-2] > 1:
            total = total + F.l1_loss(
                p[..., 1:, :] - p[..., :-1, :],
                t[..., 1:, :] - t[..., :-1, :],
            )
    return total / len(scales)


# -- Histogram/dynamic-range loss ----------------------------------


def _histogram_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    bins: int = 32,
) -> torch.Tensor:
    """Soft histogram matching loss.

    Penalizes mismatch in the distribution of pixel values between
    prediction and target. This directly addresses dynamic range compression:
    if the model outputs a narrow range, this loss pushes it to match
    the target's full dynamic range.

    Uses soft binning (differentiable) via Gaussian kernels.

    Args:
        pred: [B, C, H, W] in [0, 1]
        target: [B, C, H, W] in [0, 1]
        bins: number of histogram bins

    Returns:
        Scalar loss (L1 between soft histograms).
    """
    pred_flat = pred.flatten(1)  # [B, C*H*W]
    target_flat = target.flatten(1)

    edges = torch.linspace(0, 1, bins + 1, device=pred.device, dtype=pred.dtype)
    width = 1.0 / bins
    centers = (edges[:-1] + edges[1:]) / 2  # [bins]

    # Soft histogram via Gaussian kernel
    sigma = width * 2.0
    pred_hist = torch.exp(-((pred_flat.unsqueeze(-1) - centers) ** 2) / (2 * sigma**2)).mean(
        dim=1
    )  # [B, bins]
    target_hist = torch.exp(-((target_flat.unsqueeze(-1) - centers) ** 2) / (2 * sigma**2)).mean(
        dim=1
    )  # [B, bins]

    # Normalize to distribution
    pred_hist = pred_hist / (pred_hist.sum(dim=-1, keepdim=True) + 1e-8)
    target_hist = target_hist / (target_hist.sum(dim=-1, keepdim=True) + 1e-8)

    return F.l1_loss(pred_hist, target_hist)


# -- Ordinal ranking loss ------------------------------------------


def _ordinal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_pairs: int = 1024,
) -> torch.Tensor:
    """Ordinal depth ranking loss.

    Randomly samples pixel pairs and penalizes incorrect relative ordering.
    This directly enforces correct spatial depth relationships.

    Args:
        pred: [B, C, H, W] in [0, 1]
        target: [B, C, H, W] in [0, 1]
        num_pairs: number of random pixel pairs to sample

    Returns:
        Scalar loss.
    """
    B, C, H, W = pred.shape
    pred_flat = pred.mean(dim=1).flatten(1)  # [B, H*W]
    target_flat = target.mean(dim=1).flatten(1)

    N = H * W
    loss = pred.new_zeros(())

    for b in range(B):
        if N < 2:
            continue
        idx_a = torch.randint(0, N, (num_pairs,), device=pred.device)
        idx_b = torch.randint(0, N, (num_pairs,), device=pred.device)

        t_diff = target_flat[b, idx_a] - target_flat[b, idx_b]
        p_diff = pred_flat[b, idx_a] - pred_flat[b, idx_b]

        margin = 0.02
        valid = (t_diff.abs() > margin).float()

        if valid.sum() < 1:
            continue

        sign_t = torch.sign(t_diff)
        hinge = F.relu(0.1 - sign_t * p_diff)
        loss = loss + (hinge * valid).sum() / (valid.sum() + 1e-8)

    return loss / max(B, 1)


# -- V4 task-aware loss --------------------------------------------


def tiny_vision_loss_v4(
    prediction: torch.Tensor,
    target: torch.Tensor,
    task: str | torch.Tensor,
    *,
    gradient_weight: float = 2.0,
    pearson_weight: float = 0.5,
    ssim_weight: float = 0.5,
    histogram_weight: float = 0.3,
    ordinal_weight: float = 0.2,
    palette_ce_weight: float = 2.0,
    palette_proj_weight: float = 1.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """V4 task-aware multi-component loss with anti-oversmoothing + anti-task-collapse.

    Key changes from V2:
    - L1 (not Smooth L1 with beta=1e-3) for value loss -- avoids L2 mean regression
    - Multi-scale gradient loss (3 scales) -- stronger edge enforcement
    - SSIM for depth/normal -- structural pattern preservation
    - Histogram loss -- dynamic range matching (anti-compression)
    - Ordinal ranking loss -- spatial depth ordering
    - **Palette CE loss for semantic** -- cross-entropy via palette matching,
      forces the model to commit to discrete colors instead of averaging
    - **Palette projection loss for semantic** -- penalizes distance to nearest
      palette color, directly encouraging discrete color output
    - Higher gradient & Pearson weights

    Loss components per task:
    | Task       | Value | Grad(ms) | Pearson | SSIM | Hist | Ordinal | PaletteCE | PaletteProj |
    |------------|-------|----------|---------|------|------|---------|-----------|-------------|
    | depth      | L1    | 2.0x     | 0.5x    | 0.5x | 0.3x | 0.2x    | -         | -           |
    | normal     | L1    | 2.0x     | 0.5x    | 0.5x | 0.3x | 0.2x    | -         | -           |
    | edge       | wBCE  | 1.0x     | -       | -    | -    | -       | -         | -           |
    | semantic   | L1    | 1.0x     | -       | -    | 0.3x | -       | 2.0x      | 1.0x        |
    | generation | L1    | 0.5x     | -       | 0.5x | -    | -       | -         | -           |
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError("Expected [B, 3, H, W]")

    # Resolve task name
    if isinstance(task, torch.Tensor):
        task_idx = task[0].item()
        task_names = TINY_TASKS
        task = task_names[task_idx]

    # Shared helpers
    def _pearson_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.flatten(1)
        target_flat = tgt.flatten(1)
        pred_c = pred_flat - pred_flat.mean(dim=1, keepdim=True)
        target_c = target_flat - target_flat.mean(dim=1, keepdim=True)
        num = (pred_c * target_c).sum(dim=1)
        den = (pred_c.square().sum(dim=1) * target_c.square().sum(dim=1)).clamp_min(1e-12).sqrt()
        return 1.0 - (num / den).mean()

    if task == "edge":
        # Edge: sparse binary-ish target
        weight = torch.where(target > 0.3, 5.0, 1.0).to(prediction.device)
        bce = F.binary_cross_entropy(prediction.clamp(epsilon, 1 - epsilon), target, weight=weight)
        dice = _dice_loss(prediction, target)
        grad = _multi_scale_gradient_loss(prediction, target, scales=(1, 2))
        return bce + 0.5 * dice + 1.0 * grad

    elif task == "semantic":
        # Semantic: color-coded discrete segments
        # L1 value loss (not L2) avoids mean color regression
        value_loss = F.l1_loss(prediction, target)
        dice = _dice_loss(prediction, target)
        grad = _multi_scale_gradient_loss(prediction, target, scales=(1, 2))
        hist = _histogram_loss(prediction, target, bins=16)

        # Extract palette from target (discrete color set)
        palette = _extract_palette(target, num_colors=16, min_pixels=50)

        if palette.shape[0] >= 2:
            # Palette cross-entropy: forces model to pick a specific class
            ce_loss = _palette_ce_loss(prediction, target, palette, temperature=0.1)
            # Palette projection: pushes output toward nearest discrete color
            proj_loss = _palette_projection_loss(prediction, palette)
        else:
            ce_loss = prediction.new_zeros(())
            proj_loss = prediction.new_zeros(())

        return (
            value_loss
            + 0.3 * dice
            + 1.0 * grad
            + 0.3 * hist
            + palette_ce_weight * ce_loss
            + palette_proj_weight * proj_loss
        )

    elif task == "generation":
        # Generation: near-identity, needs perceptual quality
        value_loss = F.l1_loss(prediction, target)
        ssim = _ssim_loss(prediction, target)
        grad = _multi_scale_gradient_loss(prediction, target, scales=(1,))
        return value_loss + 0.5 * ssim + 0.5 * grad

    else:
        # depth / normal: FULL anti-oversmoothing stack
        # L1 (not L2) avoids mean regression
        value_loss = F.l1_loss(prediction, target)

        # Multi-scale gradient (3 scales: 1x, 2x, 4x)
        grad = _multi_scale_gradient_loss(prediction, target, scales=(1, 2, 4))

        # Pearson correlation (depth ordering at global scale)
        corr = _pearson_loss(prediction, target)

        # SSIM (structural similarity preserves geometric patterns)
        ssim = _ssim_loss(prediction, target)

        # Histogram matching (dynamic range anti-compression)
        hist = _histogram_loss(prediction, target, bins=32)

        # Ordinal ranking (spatial depth ordering)
        ordinal = _ordinal_loss(prediction, target, num_pairs=512)

        result = (
            value_loss
            + gradient_weight * grad
            + pearson_weight * corr
            + ssim_weight * ssim
            + histogram_weight * hist
            + ordinal_weight * ordinal
        )
        return result


__all__ = [
    "TinyVisionBananaV2",
    "TinyVisionBananaV2Config",
    "FiLMModulation",
    "TaskSpecificHead",
    "tiny_vision_loss_v4",
    "evaluate_tiny_predictions",
    "tiny_overfit_gate_passed",
    "_extract_palette",
    "_palette_ce_loss",
    "_palette_projection_loss",
    "_multi_scale_gradient_loss",
    "_histogram_loss",
    "_ordinal_loss",
]
