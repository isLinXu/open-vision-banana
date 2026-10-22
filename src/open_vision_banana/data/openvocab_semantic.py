"""Open-vocabulary semantic-segmentation targets with per-sample color mappings.

The paper's semantic segmentation is *open-vocabulary and prompt-defined*: the
class->color mapping is given in the prompt (named colors, RGB tuples, hex
codes or natural language, Fig. 2) and the model must paint each class with the
color the prompt asks for. At Cityscapes evaluation the same 19-class prompt is
used for every image, including classes absent from the image.

A fixed training palette (one color per class forever, one fixed prompt) cannot
teach that skill: the model can memorize "sky is blue" without reading the
prompt. This module instead re-samples, for every (image, variant):

* which classes are mentioned (all sufficiently visible ones plus absent
  distractors),
* which color each class gets (distinct, well separated from each other and
  from the background so nearest-color decoding is unambiguous),
* the background color,
* the prompt style (JSON with color names / JSON with RGB tuples / natural
  language / hex), mirroring the examples in the paper.

Everything is pure numpy and deterministic given the RNG.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..inference.prompt import PromptBuilder

Color = tuple[int, int, int]

NAMED_COLORS: dict[str, Color] = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "brown": (139, 69, 19),
    "gray": (128, 128, 128),
    "navy": (0, 0, 128),
    "teal": (0, 128, 128),
    "olive": (128, 128, 0),
    "maroon": (128, 0, 0),
    "light purple": (200, 160, 255),
}

# Coordinates used for random RGB colors; mirrors the multiples of 64 seen in
# the paper's examples, e.g. (255, 192, 128), (64, 192, 64), (128, 0, 64).
_LEVELS = (0, 64, 128, 192, 255)

STYLES = ("json_named", "json_tuple", "natural", "hex")
DEFAULT_STYLE_WEIGHTS = {"json_named": 0.25, "json_tuple": 0.30, "natural": 0.25, "hex": 0.20}
BACKGROUND_NAME = "background"


def color_distance(a: Color, b: Color) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def sample_distinct_colors(
    count: int,
    rng: random.Random,
    *,
    min_distance: float = 64.0,
    reserved: Sequence[Color] = (),
    pool: Sequence[Color] | None = None,
) -> list[Color]:
    """Draw ``count`` colors that are pairwise (and vs ``reserved``) >= ``min_distance`` apart.

    ``pool`` restricts candidates (used for named colors); otherwise colors come
    from the 5-level RGB grid. Raises ``ValueError`` if the request is infeasible.
    """
    if pool is None:
        candidates = [(r, g, b) for r in _LEVELS for g in _LEVELS for b in _LEVELS]
    else:
        candidates = list(pool)
    rng.shuffle(candidates)
    chosen: list[Color] = []
    taken = list(reserved)
    for color in candidates:
        if all(color_distance(color, other) >= min_distance for other in taken):
            chosen.append(color)
            taken.append(color)
            if len(chosen) == count:
                return chosen
    raise ValueError(
        f"Cannot place {count} colors with min_distance={min_distance} "
        f"(only {len(chosen)} fit; reserved={len(reserved)})"
    )


def class_phrase(name: str) -> str:
    """Dataset identifiers such as ``traffic_light`` become natural phrases."""
    return name.replace("_", " ").strip()


def build_prompt(style: str, mapping: Mapping[str, Color]) -> str:
    """Render ``mapping`` (phrase -> color, background included) as a prompt."""
    if style == "json_tuple":
        return PromptBuilder.build_semantic_prompt(dict(mapping), template_idx=0)
    if style == "natural":
        return PromptBuilder.build_semantic_prompt(dict(mapping), template_idx=1)
    if style == "hex":
        return PromptBuilder.build_semantic_prompt(dict(mapping), template_idx=2)
    if style == "json_named":
        by_rgb = {rgb: name for name, rgb in NAMED_COLORS.items()}
        parts = []
        for phrase, rgb in mapping.items():
            if rgb not in by_rgb:
                raise ValueError(f"{rgb} is not a named color; json_named needs NAMED_COLORS entries")
            parts.append(f'"{phrase}": "{by_rgb[rgb]}"')
        return PromptBuilder.SEMANTIC_TEMPLATES[0].format(color_map="{" + ", ".join(parts) + "}")
    raise ValueError(f"Unknown style: {style!r}; choose from {STYLES}")


_JSON_TUPLE_RE = re.compile(r'"([^"]+)"\s*:\s*<\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*>')
_JSON_NAMED_RE = re.compile(r'"([^"]+)"\s*:\s*"([^"]+)"')
_NATURAL_RE = re.compile(r"The (.+?) (?:are|is) represented by \((\d+), (\d+), (\d+)\)")
_HEX_RE = re.compile(r"The (.+?) (?:is|are) #([0-9A-Fa-f]{6})")


def parse_prompt_mapping(prompt: str) -> dict[str, Color]:
    """Invert ``build_prompt`` for all four styles: ``phrase -> (R, G, B)``.

    Returns an empty dict when no legend is recognised (e.g. the generic
    Cityscapes-free prompt), which callers treat as "no augmentation possible".
    """
    tuples = _JSON_TUPLE_RE.findall(prompt)
    if tuples:
        return {name: (int(r), int(g), int(b)) for name, r, g, b in tuples}
    named = _JSON_NAMED_RE.findall(prompt)
    if named and all(color in NAMED_COLORS for _, color in named):
        return {name: NAMED_COLORS[color] for name, color in named}
    natural = _NATURAL_RE.findall(prompt)
    if natural:
        return {name: (int(r), int(g), int(b)) for name, r, g, b in natural}
    hexes = _HEX_RE.findall(prompt)
    if hexes:
        return {name: (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)) for name, h in hexes}
    return {}


@dataclass(frozen=True)
class OpenVocabSample:
    target_rgb: np.ndarray  # uint8 [H, W, 3]
    prompt: str
    mapping: dict[str, Color]  # phrase -> color, includes BACKGROUND_NAME
    style: str
    present_classes: tuple[str, ...]  # phrases drawn in the target
    absent_classes: tuple[str, ...]  # distractor phrases in the prompt only


def sample_openvocab_semantic(
    label_map: np.ndarray,
    class_names: Sequence[str],
    rng: random.Random,
    *,
    ignore_index: int | None = 255,
    min_area_fraction: float = 1e-3,
    max_present: int = 12,
    max_absent: int = 6,
    style_weights: Mapping[str, float] | None = None,
    min_color_distance: float = 90.0,
) -> OpenVocabSample:
    """Build one randomized (target, prompt) pair from an integer label map.

    ``label_map[y, x]`` is a class index into ``class_names`` (or
    ``ignore_index``). Mentioned classes are the visible ones above
    ``min_area_fraction`` (largest ``max_present`` kept) plus up to
    ``max_absent`` absent distractors. Everything else, including ignored
    pixels, is painted with the background color.

    ``min_color_distance`` (RGB L2) defaults to 90: on the 5-level grid this
    forbids colors that differ in a single channel by only one step (64), so
    moderate color drift in generated images still decodes to the right class.
    With 12 present + 6 absent + background the grid still has ample room.
    """
    if label_map.ndim != 2:
        raise ValueError(f"label_map must be [H, W], got {label_map.shape}")
    weights = dict(style_weights or DEFAULT_STYLE_WEIGHTS)
    style = rng.choices(list(weights), weights=list(weights.values()), k=1)[0]

    valid = np.ones(label_map.shape, dtype=bool) if ignore_index is None else label_map != ignore_index
    in_range = valid & (label_map >= 0) & (label_map < len(class_names))
    counts = np.bincount(label_map[in_range].astype(np.int64), minlength=len(class_names))
    area = label_map.size
    visible = [i for i in np.argsort(-counts) if counts[i] / area >= min_area_fraction]
    present_ids = visible[:max_present]
    # "" marks an unused label id (gap in the dataset's id space): never a distractor.
    absent_pool = [i for i in range(len(class_names)) if counts[i] == 0 and class_names[i]]
    n_absent = rng.randint(0, min(max_absent, len(absent_pool)))
    absent_ids = rng.sample(absent_pool, n_absent)

    phrases: dict[int, str] = {}
    for idx in [*present_ids, *absent_ids]:
        phrase = class_phrase(class_names[idx])
        if phrase in phrases.values() or phrase == BACKGROUND_NAME:
            continue  # duplicate display names cannot be told apart in a prompt
        phrases[idx] = phrase

    n_colors = len(phrases) + 1  # + background
    if style == "json_named":
        pool = list(NAMED_COLORS.values())
        n_colors = min(n_colors, len(pool))
        keep = n_colors - 1
        phrases = dict(list(phrases.items())[:keep])
        colors = sample_distinct_colors(n_colors, rng, min_distance=min_color_distance, pool=pool)
    else:
        colors = sample_distinct_colors(n_colors, rng, min_distance=min_color_distance)
    background, class_colors = colors[0], colors[1:]

    mapping: dict[str, Color] = {}
    lut_ids = []
    for (idx, phrase), color in zip(phrases.items(), class_colors, strict=True):
        mapping[phrase] = color
        lut_ids.append((idx, color))
    # Background is listed last, like in the paper's Fig. 2 example prompts.
    mapping[BACKGROUND_NAME] = background

    target = np.empty((*label_map.shape, 3), dtype=np.uint8)
    target[...] = background
    for idx, color in lut_ids:
        target[in_range & (label_map == idx)] = color

    present = tuple(phrases[i] for i in present_ids if i in phrases)
    absent = tuple(phrases[i] for i in absent_ids if i in phrases)
    return OpenVocabSample(
        target_rgb=target,
        prompt=build_prompt(style, mapping),
        mapping=mapping,
        style=style,
        present_classes=present,
        absent_classes=absent,
    )


def decode_nearest(target_rgb: np.ndarray, mapping: Mapping[str, Color]) -> np.ndarray:
    """Nearest-color decoding used at evaluation; returns indices into ``mapping`` order."""
    colors = np.asarray(list(mapping.values()), dtype=np.float32)
    flat = target_rgb.reshape(-1, 3).astype(np.float32)
    dist = ((flat[:, None, :] - colors[None, :, :]) ** 2).sum(-1)
    return dist.argmin(axis=1).reshape(target_rgb.shape[:2])


__all__ = [
    "BACKGROUND_NAME",
    "NAMED_COLORS",
    "STYLES",
    "OpenVocabSample",
    "build_prompt",
    "class_phrase",
    "color_distance",
    "decode_nearest",
    "sample_distinct_colors",
    "sample_openvocab_semantic",
]
