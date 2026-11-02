"""FLUX training-matched Euler img2img used by VisionBanana.generate()."""

from __future__ import annotations

from unittest.mock import patch

import torch
from PIL import Image

from open_vision_banana.inference.flux_img2img import (
    flux_guidance_tensor,
    flow_matching_inference,
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


def test_flux_dev_forward_receives_guidance_and_pooled_projections() -> None:
    class _Cfg:
        guidance_embeds = True

    class _Transformer:
        config = _Cfg()
        calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            hidden = kwargs["hidden_states"]
            return (hidden,)

    class _VAE:
        def encode(self, images):
            return type(
                "Out",
                (),
                {
                    "latent_dist": type(
                        "D", (), {"mode": lambda self=None: torch.zeros(1, 16, 4, 4)}
                    )()
                },
            )()

        def decode(self, latents, return_dict=True):
            sample = torch.zeros(1, 3, 32, 32)
            if return_dict:
                return type("Out", (), {"sample": sample})()
            return (sample,)

        def parameters(self):
            yield torch.zeros(1)

        config = type("C", (), {"scaling_factor": 1.0, "shift_factor": 0.0})()

    transformer = _Transformer()
    pipe = type(
        "Pipe",
        (),
        {
            "transformer": transformer,
            "vae": _VAE(),
            "encode_prompt": lambda *a, **k: (
                torch.zeros(1, 2, 8),
                torch.zeros(1, 768),
                torch.zeros(2, 3),
            ),
            "text_encoder": None,
            "text_encoder_2": None,
        },
    )()
    config = ModelConfig(
        name="flux-dev",
        model_id="black-forest-labs/FLUX.1-dev",
        family="flux",
        num_text_encoders=2,
        vae_scaling_factor=1.0,
        vae_has_latent_dist=True,
    )
    out = flow_matching_inference(
        pipe=pipe,
        config=config,
        input_images=torch.zeros(1, 3, 32, 32),
        prompts=["depth"],
        device=torch.device("cpu"),
        model_dtype=torch.float32,
        num_steps=1,
        guidance=50.0,
    )
    assert out.shape == (1, 3, 32, 32)
    call = transformer.calls[0]
    assert call["pooled_projections"].shape == (1, 768)
    assert torch.allclose(call["guidance"], torch.tensor([50.0]))
    schnell = type("T", (), {"config": type("C", (), {"guidance_embeds": False})()})()
    assert flux_guidance_tensor(schnell, 1, torch.device("cpu"), torch.float32) is None


def test_flux_decode_skips_meta_vae_parameters() -> None:
    class _Cfg:
        guidance_embeds = False

    class _Transformer:
        config = _Cfg()

        def __call__(self, **kwargs):
            return (kwargs["hidden_states"],)

    class _VAE:
        def encode(self, images):
            return type(
                "Out",
                (),
                {
                    "latent_dist": type(
                        "D", (), {"mode": lambda self=None: torch.zeros(1, 16, 4, 4)}
                    )()
                },
            )()

        def decode(self, latents, return_dict=True):
            assert latents.device.type != "meta"
            sample = torch.zeros(1, 3, 32, 32, device=latents.device)
            return (sample,) if not return_dict else type("Out", (), {"sample": sample})()

        def parameters(self):
            yield torch.zeros(1, device="meta")

        config = type("C", (), {"scaling_factor": 1.0, "shift_factor": 0.0})()

    pipe = type(
        "Pipe",
        (),
        {
            "transformer": _Transformer(),
            "vae": _VAE(),
            "encode_prompt": lambda *a, **k: (
                torch.zeros(1, 2, 8),
                torch.zeros(1, 768),
                torch.zeros(2, 3),
            ),
            "text_encoder": None,
            "text_encoder_2": None,
        },
    )()
    config = ModelConfig(
        name="flux-dev",
        model_id="black-forest-labs/FLUX.1-dev",
        family="flux",
        num_text_encoders=2,
        vae_scaling_factor=1.0,
        vae_has_latent_dist=True,
    )
    out = flow_matching_inference(
        pipe=pipe,
        config=config,
        input_images=torch.zeros(1, 3, 32, 32),
        prompts=["depth"],
        device=torch.device("cpu"),
        model_dtype=torch.float32,
        num_steps=1,
    )
    assert out.shape == (1, 3, 32, 32)


def test_encode_text_flux_leaves_quantized_t5_in_place() -> None:
    from open_vision_banana.inference.flux_img2img import encode_text_flux

    class _Enc:
        is_loaded_in_8bit = True

        def to(self, *_args, **_kwargs):
            raise AssertionError("quantized T5 must not be recast")

        def parameters(self):
            yield torch.zeros(1)

    pipe = type(
        "Pipe",
        (),
        {
            "text_encoder": None,
            "text_encoder_2": _Enc(),
            "encode_prompt": lambda *a, **k: (
                torch.zeros(1, 2, 8),
                torch.zeros(1, 768),
                torch.zeros(2, 3),
            ),
        },
    )()
    text = encode_text_flux(pipe, ["depth"], torch.device("cpu"))
    assert text["pooled_prompt_embeds"].shape == (1, 768)
