"""Training-time random semantic legends (recipe v2).

Each semantic training sample re-draws its class -> colour legend so the model has
to read the prompt instead of memorising a class/colour binding:

* colours come from a fixed 64-colour palette, the best separated subset (greedy
  farthest point in CIELAB) of a 200-colour candidate grid far from black;
* the request is a random subset of the present classes (70%), all present classes
  (15%), or all present classes plus absent negative classes (15%);
* only requested classes are painted; everything else, including background and
  present-but-unrequested classes, is black, so "not asked for / not there" is black;
* the legend is written as RGB tuples, hex codes, natural language or colour names.

Decoding assigns each pixel to the nearest prompt colour, with black as background.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .openvocab_semantic import BACKGROUND_NAME, NAMED_COLORS, build_prompt

Color = tuple[int, int, int]
BLACK: Color = (0, 0, 0)
LEGEND_STYLES = ("json_tuple", "hex", "natural", "json_named")
DEFAULT_MODE_PROBS = {"subset": 0.70, "all": 0.15, "negatives": 0.15}
# Six levels per channel give 216 colours; the 16 closest to black are dropped (black is background).
_GRID_LEVELS = (0, 51, 102, 153, 204, 255)
PALETTE_CANDIDATES = 200
PALETTE_SIZE = 64
# Named colours usable in a legend: black is the background, and grey/white stay
# well clear of it but some named pairs are close, so ``json_named`` legends also
# enforce a minimum RGB distance.
_NAMED_MIN_DISTANCE = 90.0


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array(
        [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]
    )
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


@lru_cache(maxsize=1)
def separated_palette() -> tuple[Color, ...]:
    """The fixed 64-colour training palette (deterministic, black excluded)."""
    grid = np.array([(r, g, b) for r in _GRID_LEVELS for g in _GRID_LEVELS for b in _GRID_LEVELS])
    lab = _srgb_to_lab(grid)
    black = _srgb_to_lab(np.array(BLACK))
    dist_black = np.linalg.norm(lab - black, axis=1)
    keep = np.argsort(-dist_black, kind="stable")[:PALETTE_CANDIDATES]
    cand, cand_lab = grid[keep], lab[keep]
    # Farthest-point selection; distance to black counts as an already chosen colour.
    nearest = np.linalg.norm(cand_lab - black, axis=1)
    chosen: list[int] = []
    for _ in range(PALETTE_SIZE):
        idx = int(np.argmax(nearest))
        chosen.append(idx)
        nearest = np.minimum(nearest, np.linalg.norm(cand_lab - cand_lab[idx], axis=1))
        nearest[idx] = -1.0
    return tuple(tuple(int(v) for v in cand[i]) for i in chosen)


def _named_pool() -> list[Color]:
    return [c for c in NAMED_COLORS.values() if c != BLACK]


@dataclass(frozen=True)
class LegendSample:
    target: np.ndarray  # uint8 [H, W, 3]
    prompt: str
    mapping: dict[str, Color]  # requested legend, negatives included, no background
    style: str
    mode: str
    painted: tuple[str, ...]
    negatives: tuple[str, ...]


def _draw_mode(rng: random.Random, probs: Mapping[str, float]) -> str:
    total = sum(probs.values())
    roll = rng.random() * total
    for mode, weight in probs.items():
        roll -= weight
        if roll < 0:
            return mode
    return next(reversed(list(probs)))


def _draw_colors(count: int, style: str, rng: random.Random) -> tuple[str, list[Color]]:
    if style == "json_named":
        pool = _named_pool()
        rng.shuffle(pool)
        picked: list[Color] = []
        for color in pool:
            if all(np.linalg.norm(np.subtract(color, other)) >= _NAMED_MIN_DISTANCE for other in picked):
                picked.append(color)
                if len(picked) == count:
                    return style, picked
        style = "json_tuple"
    palette = list(separated_palette())
    if count > len(palette):
        raise ValueError(f"Legend of {count} classes exceeds the {len(palette)}-colour palette")
    return style, rng.sample(palette, count)


def randomize_semantic_legend(
    target: np.ndarray,
    mapping: Mapping[str, Sequence[int]],
    rng: random.Random,
    *,
    vocabulary: Sequence[str] = (),
    mode_probs: Mapping[str, float] | None = None,
    styles: Sequence[str] = LEGEND_STYLES,
    min_area_fraction: float = 5e-4,
    max_classes: int = 24,
    max_negatives: int = 3,
) -> LegendSample | None:
    """Re-legend one semantic target (``uint8 [H, W, 3]``) painted with ``mapping``.

    Returns ``None`` when no class of ``mapping`` is visible and no negative can be
    drawn (the caller keeps the original sample).
    """
    if target.dtype != np.uint8 or target.ndim != 3 or target.shape[2] != 3:
        raise ValueError("target must be uint8 [H, W, 3]")
    probs = dict(mode_probs or DEFAULT_MODE_PROBS)
    packed = (target[..., 0].astype(np.int32) << 16) | (target[..., 1].astype(np.int32) << 8) | target[..., 2]
    min_pixels = max(1, int(round(min_area_fraction * packed.size)))
    present: dict[str, np.ndarray] = {}
    for name, rgb in mapping.items():
        if name == BACKGROUND_NAME:
            continue
        r, g, b = (int(c) for c in rgb)
        mask = packed == ((r << 16) | (g << 8) | b)
        if int(mask.sum()) >= min_pixels:
            present[name] = mask
    names = list(present)
    rng.shuffle(names)
    names = names[:max_classes]

    mode = _draw_mode(rng, probs)
    painted = names[: rng.randint(1, len(names))] if mode == "subset" and names else names
    negatives: list[str] = []
    if mode == "negatives" or not painted:
        mode = "negatives"
        pool = sorted(
            {n for n in mapping if n != BACKGROUND_NAME and n not in present}
            | {n for n in vocabulary if n not in mapping and n != BACKGROUND_NAME}
        )
        room = max(0, max_classes - len(painted))
        if pool and room:
            negatives = rng.sample(pool, min(len(pool), room, rng.randint(1, max_negatives)))
    if not painted and not negatives:
        return None

    style = rng.choice(list(styles))
    legend_names = list(painted) + negatives
    rng.shuffle(legend_names)
    style, colors = _draw_colors(len(legend_names), style, rng)
    legend = dict(zip(legend_names, colors, strict=True))
    out = np.zeros_like(target)
    for name in painted:
        out[present[name]] = legend[name]
    return LegendSample(
        target=out,
        prompt=build_prompt(style, legend),
        mapping=legend,
        style=style,
        mode=mode,
        painted=tuple(painted),
        negatives=tuple(negatives),
    )


def decode_with_black_background(
    rgb: np.ndarray, legend: Mapping[str, Sequence[int]]
) -> tuple[np.ndarray, list[str]]:
    """Nearest prompt colour per pixel; index ``-1`` is the black background.

    Returns (``int32 [H, W]`` index into the returned name list, names).
    """
    names = list(legend)
    colors = np.array([BLACK] + [tuple(legend[n]) for n in names], dtype=np.float32)
    flat = rgb.reshape(-1, 3).astype(np.float32)
    dist = ((flat[:, None, :] - colors[None, :, :]) ** 2).sum(-1)
    index = dist.argmin(1).astype(np.int32) - 1
    return index.reshape(rgb.shape[:2]), names


__all__ = [
    "DEFAULT_MODE_PROBS",
    "LEGEND_STYLES",
    "LegendSample",
    "decode_with_black_background",
    "randomize_semantic_legend",
    "separated_palette",
]
