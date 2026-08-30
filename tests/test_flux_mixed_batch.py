"""Mixed vision/generation micro-batches for FLUX paper-method training."""

from __future__ import annotations

import torch

from open_vision_banana.training.tuner import (
    _collate_fn,
    mix_rectified_flow_latents,
    normalize_target_pixels,
)


def test_collate_pads_generation_image_and_keeps_noise_flow():
    vision = {
        "prompt": "depth",
        "image": torch.ones(3, 4, 4),
        "target_rgb": torch.full((3, 4, 4), 0.25),
        "target_rgb_range": "0_1",
        "task": "depth",
        "colormap": "cube_edge",
    }
    generation = {
        "prompt": "a photo",
        "image": None,
        "target_rgb": torch.full((3, 4, 4), -0.5),
        "target_rgb_range": "-1_1",
        "task": "generation",
        "flow_source": "noise",
    }
    batch = _collate_fn([vision, generation])

    assert batch["image"].shape == (2, 3, 4, 4)
    assert torch.equal(batch["image"][0], vision["image"])
    assert torch.equal(batch["image"][1], torch.zeros(3, 4, 4))
    assert batch["flow_source"] == ["input", "noise"]
    assert batch["task"] == ["depth", "generation"]
    assert batch["colormap"] == ["cube_edge", None]


def test_collate_all_generation_keeps_image_none():
    sample = {
        "prompt": "a photo",
        "image": None,
        "target_rgb": torch.zeros(3, 2, 2),
        "target_rgb_range": "-1_1",
    }
    batch = _collate_fn([sample, dict(sample)])
    assert batch["image"] is None
    assert batch["flow_source"] == ["noise", "noise"]


def test_normalize_target_pixels_is_per_sample():
    target = torch.stack([torch.full((3, 2, 2), 0.25), torch.full((3, 2, 2), -0.5)])
    pixels = normalize_target_pixels(target, ["0_1", "-1_1"])
    assert torch.allclose(pixels[0], torch.full((3, 2, 2), -0.5))
    assert torch.allclose(pixels[1], torch.full((3, 2, 2), -0.5))


def test_mixed_flow_keeps_noise_branch_for_generation():
    target = torch.zeros(2, 2, 2, 2)
    source = torch.full((2, 2, 2, 2), 7.0)
    noise = torch.full((2, 2, 2, 2), 3.0)
    timesteps = torch.tensor([0.25, 0.25])
    noisy, velocity = mix_rectified_flow_latents(
        target,
        noise,
        timesteps,
        input_latents=source,
        flow_source=["input", "noise"],
        use_img2img_flow=True,
        flow_sigma=0.0,
    )
    assert torch.allclose(noisy[0], torch.full((2, 2, 2), 5.25))
    assert torch.allclose(velocity[0], torch.full((2, 2, 2), -7.0))
    assert torch.allclose(noisy[1], torch.full((2, 2, 2), 0.75))
    assert torch.allclose(velocity[1], torch.full((2, 2, 2), -3.0))
