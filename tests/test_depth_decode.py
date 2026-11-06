"""Depth RGB inversion used when a checkpoint emits closer=brighter gray."""

import torch

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.evaluation.evaluator import decode_depth_rgb, relative_luminance_depth


def test_invert_treats_bright_gray_as_near():
    mapper = DepthRGBMapper(colormap="grayscale")
    bright = torch.ones(1, 3, 2, 2)
    dark = torch.zeros(1, 3, 2, 2)
    bright_depth = decode_depth_rgb(bright, mapper, invert=True)
    dark_depth = decode_depth_rgb(dark, mapper, invert=True)
    assert float(bright_depth.mean()) < float(dark_depth.mean())


def test_default_grayscale_treats_dark_as_near():
    mapper = DepthRGBMapper(colormap="grayscale")
    bright = torch.full((1, 3, 2, 2), 0.8)
    dark = torch.full((1, 3, 2, 2), 0.2)
    assert float(decode_depth_rgb(dark, mapper).mean()) < float(
        decode_depth_rgb(bright, mapper).mean()
    )


def test_relative_luminance_is_closer_brighter():
    rgb = torch.tensor([0.2, 0.8], dtype=torch.float32).view(1, 1, 1, 2).expand(1, 3, 1, 2)
    relative = relative_luminance_depth(rgb)
    assert relative.shape == (1, 1, 1, 2)
    assert float(relative[0, 0, 0, 0]) > float(relative[0, 0, 0, 1])


def test_oracle_depth_decode_picks_the_encoding_colormap():
    """A viridis-painted depth map is decoded as viridis, not cube_edge."""
    from open_vision_banana.evaluation.evaluator import oracle_depth_decode

    names = ["cube_edge", "plasma", "inferno", "viridis", "grayscale"]
    mappers = {name: DepthRGBMapper(colormap=name) for name in names}
    depth = torch.rand(2, 1, 32, 32) * 8.0 + 0.5
    rgb = mappers["viridis"].depth_to_rgb(depth)

    decoded, chosen = oracle_depth_decode(rgb, mappers)

    assert chosen == ["viridis", "viridis"]
    assert torch.allclose(decoded, depth, atol=0.2)


def test_oracle_depth_decode_recovers_each_trained_colormap():
    """The oracle identifies whichever palette the model actually painted."""
    from open_vision_banana.evaluation.evaluator import oracle_depth_decode

    names = ["cube_edge", "viridis", "grayscale"]
    mappers = {name: DepthRGBMapper(colormap=name) for name in names}
    depth = torch.rand(1, 1, 16, 16) * 6.0 + 0.5
    for name in names:
        rgb = mappers[name].depth_to_rgb(depth)
        _, chosen = oracle_depth_decode(rgb, mappers)
        assert chosen == [name]
