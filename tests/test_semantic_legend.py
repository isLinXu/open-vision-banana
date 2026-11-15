import random

import numpy as np
import pytest

from open_vision_banana.data.openvocab_semantic import parse_prompt_mapping
from open_vision_banana.data.semantic_legend import (
    PALETTE_SIZE,
    decode_with_black_background,
    randomize_semantic_legend,
    separated_palette,
)

MAPPING = {"wall": (255, 0, 0), "floor": (0, 255, 0), "chair": (0, 0, 255), "lamp": (255, 255, 0),
           "background": (64, 64, 64)}


def _target():
    t = np.zeros((32, 32, 3), np.uint8)
    t[:] = MAPPING["background"]
    t[:8] = MAPPING["wall"]
    t[8:16] = MAPPING["floor"]
    t[16:24, :16] = MAPPING["chair"]
    return t  # lamp listed but absent


def test_palette_is_fixed_separated_and_not_black():
    palette = separated_palette()
    assert len(palette) == len(set(palette)) == PALETTE_SIZE
    assert palette == separated_palette()
    arr = np.array(palette, np.float32)
    assert np.linalg.norm(arr, axis=1).min() > 100
    pair = np.linalg.norm(arr[:, None] - arr[None], axis=-1) + np.eye(len(arr)) * 1e9
    assert pair.min() >= 50


@pytest.mark.parametrize("seed", range(40))
def test_legend_contract(seed):
    rng = random.Random(seed)
    sample = randomize_semantic_legend(_target(), MAPPING, rng, vocabulary=["sofa", "bed", "wall"])
    assert sample is not None
    assert set(sample.painted) <= {"wall", "floor", "chair"}
    assert not set(sample.negatives) & set(sample.painted)
    assert set(sample.negatives) <= {"lamp", "sofa", "bed"}
    assert set(sample.mapping) == set(sample.painted) | set(sample.negatives)
    assert parse_prompt_mapping(sample.prompt) == sample.mapping
    assert "background" not in sample.prompt
    original = _target()
    for name in ("wall", "floor", "chair"):
        region = (original == MAPPING[name]).all(-1)
        expected = sample.mapping[name] if name in sample.painted else (0, 0, 0)
        assert (sample.target[region] == expected).all()
    assert (sample.target[(original == MAPPING["background"]).all(-1)] == 0).all()
    index, names = decode_with_black_background(sample.target, sample.mapping)
    for name in sample.painted:
        assert (index[(original == MAPPING[name]).all(-1)] == names.index(name)).all()


def test_mode_frequencies_and_styles():
    rng = random.Random(0)
    modes, styles = {}, set()
    for _ in range(2000):
        sample = randomize_semantic_legend(_target(), MAPPING, rng, vocabulary=["sofa"])
        modes[sample.mode] = modes.get(sample.mode, 0) + 1
        styles.add(sample.style)
    assert abs(modes["subset"] / 2000 - 0.70) < 0.04
    assert abs(modes["negatives"] / 2000 - 0.15) < 0.03
    assert styles == {"json_tuple", "hex", "natural", "json_named"}


def test_nothing_visible():
    empty = np.full((8, 8, 3), 64, np.uint8)
    assert randomize_semantic_legend(empty, {"background": (64, 64, 64)}, random.Random(0)) is None
    sample = randomize_semantic_legend(empty, {"background": (64, 64, 64), "wall": (255, 0, 0)},
                                       random.Random(0), vocabulary=["sofa"])
    assert sample.mode == "negatives" and not sample.painted and (sample.target == 0).all()
    with pytest.raises(ValueError):
        randomize_semantic_legend(empty.astype(np.float32), MAPPING, random.Random(0))


def test_large_named_legend_falls_back_to_tuples():
    from open_vision_banana.data import semantic_legend

    style, colors = semantic_legend._draw_colors(30, "json_named", random.Random(0))
    assert style == "json_tuple" and len(set(colors)) == 30
    with pytest.raises(ValueError):
        semantic_legend._draw_colors(PALETTE_SIZE + 1, "hex", random.Random(0))
