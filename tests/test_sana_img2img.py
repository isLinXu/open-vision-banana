"""SANA flow-matching img2img used by VisionBanana.generate()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from open_vision_banana.inference.sana_img2img import (
    flow_matching_inference,
    images_to_sana_source,
    pipeline_components_from_sana,
    resolve_sana_img2img_settings,
)
from open_vision_banana.model_registry import ModelConfig, PipelineComponents


def test_images_to_sana_source_preserves_minus_one_one() -> None:
    source = torch.linspace(-1.0, 1.0, 48).reshape(1, 3, 4, 4)
    out = images_to_sana_source(source, "cpu")
    assert out.shape == (1, 3, 4, 4)
    assert torch.allclose(out, source)


def test_images_to_sana_source_maps_unit_interval_and_pil() -> None:
    unit = torch.zeros(3, 2, 2)
    out = images_to_sana_source(unit, "cpu")
    assert out.shape == (1, 3, 2, 2)
    assert torch.allclose(out, torch.full_like(out, -1.0))

    pil = Image.new("RGB", (2, 2), (255, 255, 255))
    pil_out = images_to_sana_source(pil, "cpu")
    assert pil_out.shape == (1, 3, 2, 2)
    assert torch.allclose(pil_out, torch.ones_like(pil_out))


def test_resolve_sana_img2img_settings_rewrites_schnell_defaults() -> None:
    steps, cfg = resolve_sana_img2img_settings(
        num_inference_steps=1,
        guidance_scale=0.0,
        default_steps=20,
    )
    assert steps == 20
    assert cfg == 1.0
    steps, cfg = resolve_sana_img2img_settings(
        num_inference_steps=20,
        guidance_scale=1.5,
        default_steps=20,
    )
    assert steps == 20
    assert cfg == 1.5


def test_flow_matching_inference_uses_scheduler_not_pipeline_image_kwarg() -> None:
    class DummyTok:
        def __call__(self, prompts, **_kwargs):
            batch = len(prompts)
            return type(
                "Tok",
                (),
                {
                    "input_ids": torch.zeros(batch, 4, dtype=torch.long),
                    "attention_mask": torch.ones(batch, 4),
                },
            )()

    class DummyTE:
        def __call__(self, input_ids, attention_mask=None):
            del attention_mask
            return type(
                "Out",
                (),
                {"last_hidden_state": torch.zeros(input_ids.shape[0], 4, 8)},
            )()

    class DummyVAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dtype = torch.float32
            self.config = type("C", (), {"scaling_factor": 1.0, "shift_factor": 0.0})()

        def encode(self, images):
            return type("E", (), {"latent": torch.zeros(images.shape[0], 4, 2, 2)})()

        def decode(self, latents):
            return type("D", (), {"sample": torch.zeros(latents.shape[0], 3, 8, 8)})()

    class DummySched:
        def set_timesteps(self, num_steps, device=None):
            del num_steps, device
            self.timesteps = torch.tensor([500.0, 0.0])

        def step(self, pred, timestep, latents, return_dict=False):
            del pred, timestep, return_dict
            return (latents,)

    class DummyBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = type("C", (), {"timestep_scale": 1.0})()

        def forward(self, latents, **_kwargs):
            return (torch.zeros_like(latents),)

    components = PipelineComponents(
        unet=DummyBackbone(),
        vae=DummyVAE(),
        text_encoder=DummyTE(),
        text_encoder_2=None,
        tokenizer=DummyTok(),
        config=ModelConfig(name="sana", model_id="x", family="sana", num_text_encoders=1),
        scheduler=DummySched(),
    )
    images = torch.zeros(1, 3, 8, 8)
    out = flow_matching_inference(
        peft_model=DummyBackbone(),
        vae=DummyVAE(),
        components=components,
        input_images=images,
        prompts=["Depth estimation: convert this image to a depth map"],
        device=torch.device("cpu"),
        model_dtype=torch.float32,
        num_steps=2,
        guidance_scale=1.0,
    )
    assert out.shape == (1, 3, 8, 8)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_flow_matching_inference_broadcasts_single_prompt_to_image_batch() -> None:
    class DummyTok:
        def __call__(self, prompts, **_kwargs):
            # The evaluator intentionally supplies one shared prompt for a
            # multi-image batch.
            assert prompts == ["depth"]
            return type(
                "Tok",
                (),
                {
                    "input_ids": torch.zeros(1, 4, dtype=torch.long),
                    "attention_mask": torch.ones(1, 4),
                },
            )()

    class DummyTE:
        def __call__(self, input_ids, attention_mask=None):
            del attention_mask
            return type(
                "Out",
                (),
                {"last_hidden_state": torch.zeros(input_ids.shape[0], 4, 8)},
            )()

    class DummyVAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dtype = torch.float32
            self.config = type("C", (), {"scaling_factor": 1.0, "shift_factor": 0.0})()

        def encode(self, images):
            return type("E", (), {"latent": torch.zeros(images.shape[0], 4, 2, 2)})()

        def decode(self, latents):
            return type("D", (), {"sample": torch.zeros(latents.shape[0], 3, 8, 8)})()

    class DummySched:
        def set_timesteps(self, num_steps, device=None):
            del num_steps, device
            self.timesteps = torch.tensor([500.0])

        def step(self, pred, timestep, latents, return_dict=False):
            del pred, timestep, return_dict
            return (latents,)

    class StrictBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = type("C", (), {"timestep_scale": 1.0})()

        def forward(self, latents, encoder_hidden_states, encoder_attention_mask, **_kwargs):
            assert encoder_hidden_states.shape[0] == latents.shape[0] == 2
            assert encoder_attention_mask.shape == (2, 4)
            return (torch.zeros_like(latents),)

    components = PipelineComponents(
        unet=StrictBackbone(),
        vae=DummyVAE(),
        text_encoder=DummyTE(),
        text_encoder_2=None,
        tokenizer=DummyTok(),
        config=ModelConfig(name="sana", model_id="x", family="sana", num_text_encoders=1),
        scheduler=DummySched(),
    )
    out = flow_matching_inference(
        peft_model=StrictBackbone(),
        vae=DummyVAE(),
        components=components,
        input_images=torch.zeros(2, 3, 8, 8),
        prompts=["depth"],
        device=torch.device("cpu"),
        model_dtype=torch.float32,
        num_steps=1,
        guidance_scale=1.0,
    )
    assert out.shape == (2, 3, 8, 8)


def test_generate_routes_sana_image_to_flow_matching() -> None:
    from open_vision_banana.inference.generator import BaseGenerator

    with patch.object(BaseGenerator, "_load_model"):
        gen = BaseGenerator(model_name="dummy", dtype=torch.float32, device="cpu")
    gen.model_config = ModelConfig(
        name="sana-600m-512px",
        model_id="Efficient-Large-Model/Sana_600M_512px_diffusers",
        family="sana",
        num_text_encoders=1,
        default_num_inference_steps=20,
        text_encoder_max_length=300,
    )
    gen._t2i_pipe = MagicMock()
    gen._get_i2i_pipe = MagicMock(side_effect=AssertionError("must not use Diffusers img2img"))

    expected = torch.zeros(1, 3, 8, 8)
    with patch.object(BaseGenerator, "_generate_sana_img2img", return_value=expected) as routed:
        out = gen.generate(
            prompt="depth",
            image=torch.zeros(1, 3, 8, 8),
            num_inference_steps=1,
            guidance_scale=0.0,
        )
    routed.assert_called_once()
    assert out is expected
    gen._get_i2i_pipe.assert_not_called()


def test_pipeline_components_from_sana_reads_transformer() -> None:
    pipe = MagicMock()
    pipe.transformer = object()
    pipe.vae = object()
    pipe.text_encoder = object()
    pipe.tokenizer = object()
    pipe.scheduler = object()
    config = ModelConfig(name="sana", model_id="x", family="sana", num_text_encoders=1)
    components = pipeline_components_from_sana(pipe, config)
    assert components.transformer is pipe.transformer
    assert components.scheduler is pipe.scheduler
