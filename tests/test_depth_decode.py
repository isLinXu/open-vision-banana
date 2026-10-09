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
