"""Versioned RGB contracts shared by RES inference and evaluation."""

from __future__ import annotations

import math

import torch

from ..referring_contract import BINARY_REFERRING_ENCODING, LEGACY_REFERRING_ENCODING

WHITE = (255, 255, 255)


def resolve_referring_encoding(
    encoding: str | None = None,
    target_color: tuple[int, int, int] = WHITE,
) -> str:
    """Keep custom-color callers compatible while defaulting white masks to binary."""
    if len(target_color) != 3 or any(
        not isinstance(channel, int) or not 0 <= channel <= 255 for channel in target_color
    ):
        raise ValueError("Referring target_color must contain three integers in [0, 255]")
    if encoding is None:
        encoding = (
            BINARY_REFERRING_ENCODING if tuple(target_color) == WHITE else LEGACY_REFERRING_ENCODING
        )
    if encoding not in (BINARY_REFERRING_ENCODING, LEGACY_REFERRING_ENCODING):
        raise ValueError(f"Unknown referring encoding: {encoding!r}")
    if encoding == BINARY_REFERRING_ENCODING and tuple(target_color) != WHITE:
        raise ValueError(f"{BINARY_REFERRING_ENCODING} requires a white target_color")
    return encoding


# Paper Appendix A only calls referring/semantic post-processing "straightforward"; the
# white/black contract decodes with this intensity threshold and keeps all components.
BINARY_REFERRING_THRESHOLD = 0.5


def decode_referring_mask(
    rgb_image: torch.Tensor,
    target_color: tuple[int, int, int] = WHITE,
    threshold: float | None = None,
    *,
    encoding: str | None = None,
    legacy_largest_component: bool = True,
) -> torch.Tensor:
    """Decode [B,3,H,W] RGB in [0,1] to a boolean [B,1,H,W] mask.

    ``binary_white_black_v1`` selects mean RGB > 0.5 (or the explicit
    intensity threshold), retaining every component. The legacy protocol
    interprets ``threshold`` as an RGB L2 distance and preserves the old
    adaptive threshold and optional SciPy largest-component behavior.
    Non-white target colors select legacy decoding when encoding is omitted.
    """
    encoding = resolve_referring_encoding(encoding, target_color)
    if rgb_image.ndim != 4 or rgb_image.shape[1] != 3:
        raise ValueError("Referring RGB must have shape [B,3,H,W]")
    if threshold is not None and (not math.isfinite(threshold) or threshold < 0):
        raise ValueError("Referring threshold must be finite and nonnegative")
    if encoding == BINARY_REFERRING_ENCODING:
        threshold = BINARY_REFERRING_THRESHOLD if threshold is None else threshold
        if threshold > 1:
            raise ValueError("Binary referring threshold must be in [0,1]")
        return rgb_image.mean(dim=1, keepdim=True) > threshold

    target = (
        torch.tensor(
            target_color,
            dtype=torch.float32,
            device=rgb_image.device,
        ).view(1, 3, 1, 1)
        / 255.0
    )
    distance = torch.norm(rgb_image - target, dim=1, keepdim=True)
    if threshold is None:
        mean = distance.mean(dim=(-2, -1), keepdim=True)
        # A single-pixel image has zero dispersion, rather than NaN.
        std = distance.std(dim=(-2, -1), keepdim=True, correction=int(distance[0].numel() > 1))
        threshold = (mean - 0.5 * std).clamp(min=0.05, max=0.5)
    mask = distance <= threshold
    if not legacy_largest_component:
        return mask
    try:
        import numpy as np
        from scipy.ndimage import label
    except ImportError:
        return mask
    masks = mask.squeeze(1).detach().cpu().numpy()
    cleaned = np.zeros_like(masks)
    for index, sample in enumerate(masks):
        labels, count = label(sample)
        if count:
            largest = int(np.argmax(np.bincount(labels.ravel())[1:])) + 1
            cleaned[index] = labels == largest
    return torch.from_numpy(cleaned).to(device=rgb_image.device).unsqueeze(1)
