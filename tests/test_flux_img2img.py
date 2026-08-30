"""FLUX training-matched Euler img2img used by VisionBanana.generate()."""

from __future__ import annotations

from unittest.mock import patch

import torch
from PIL import Image

from open_vision_banana.inference.flux_img2img import (
    images_to_flux_source,
    pack_flux_latents,
    unpack_flux_latents,
)
from open_vision_banana.inference.generator import BaseGenerator
from open_vision_banana.model_registry import ModelConfig


def test_images_to_flux_source_preserves_minus_one_one() -> None:
    source = torch.linspace(-1.0, 1.0, 48).reshape(1, 3, 4, 4)
    out = images_to_flux_source(source, "cpu")
    assert out.shape == (1, 3, 4, 4)
    assert torch.allclose(out, source)


def test_images_to_flux_source_maps_unit_interval_and_pil() -> None:
    unit = torch.zeros(3, 2, 2)
    out = images_to_flux_source(unit, "cpu")
    assert out.shape == (1, 3, 2, 2)
    assert torch.allclose(out, torch.full_like(out, -1.0))

    pil = Image.new("RGB", (2, 2), (255, 255, 255))
    pil_out = images_to_flux_source(pil, "cpu")
    assert pil_out.shape == (1, 3, 2, 2)
    assert torch.allclose(pil_out, torch.ones_like(pil_out))


def test_pack_unpack_roundtrip() -> None:
    latent = torch.arange(16 * 8 * 8, dtype=torch.float32).view(1, 16, 8, 8)
    packed = pack_flux_latents(latent, 1, 16, 8, 8)
    assert packed.shape == (1, 16, 64)
    restored = unpack_flux_latents(packed, 8, 8)
    assert torch.equal(restored, latent)


def test_generate_routes_flux_image_to_training_flow() -> None:
    generator = BaseGenerator.__new__(BaseGenerator)
    generator.model_config = ModelConfig(
        name="flux-schnell",
        model_id="black-forest-labs/FLUX.1-schnell",
        family="flux",
        num_text_encoders=2,
    )
    generator.device = "cpu"
    generator.dtype = torch.float32
    generator._t2i_pipe = object()
    generator._prompt_cache = {}
    generator.source_conditioning_config = None

    sentinel = torch.zeros(1, 3, 4, 4)
    with patch.object(generator, "_generate_flux_img2img", return_value=sentinel) as mocked:
        with patch.object(generator, "_get_i2i_pipe") as i2i:
            out = generator.generate(
                prompt="Generate a metric depth visualization of this image.",
                image=torch.zeros(1, 3, 4, 4),
                num_inference_steps=4,
            )
    assert out is sentinel
    mocked.assert_called_once()
    i2i.assert_not_called()
