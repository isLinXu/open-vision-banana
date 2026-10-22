"""Instance-colour palette shared by the COCO/LVIS and Hypersim instance builders."""

from __future__ import annotations

import itertools

import numpy as np

BACKGROUND = (0, 0, 0)
# Minimum pairwise distance between instance colours and from the background. The
# Appendix A decoder merges colours within tau=14; generated images drift by a few
# levels, so the palette keeps a wide margin.
MIN_COLOR_DISTANCE = 60.0


def color_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    # int() guards against uint8 wrap-around when colours come back from an image array.
    return float(np.sqrt(sum((int(x) - int(y)) ** 2 for x, y in zip(a, b, strict=True))))


def build_instance_palette(
    min_distance: float = MIN_COLOR_DISTANCE,
    background: tuple[int, int, int] = BACKGROUND,
) -> list[tuple[int, int, int]]:
    """Greedy, deterministic palette: 4 levels per channel, far from the background and each other."""
    levels = (255, 128, 192, 64)
    candidates = list(itertools.product(levels, repeat=3))
    # Saturated colours first so the most common small instance counts get the most
    # separable colours.
    candidates.sort(key=lambda c: (-(max(c) - min(c)), -sum(c)))
    palette: list[tuple[int, int, int]] = []
    for color in candidates:
        if color_distance(color, background) < min_distance:
            continue
        if all(color_distance(color, chosen) >= min_distance for chosen in palette):
            palette.append(color)
    return palette


PALETTE = build_instance_palette()

__all__ = ["BACKGROUND", "MIN_COLOR_DISTANCE", "PALETTE", "build_instance_palette", "color_distance"]
