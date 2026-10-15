"""Qwen-Image-2.1 training conditioning, with stub pipelines and backbones on CPU."""

import types

import pytest
import torch

from open_vision_banana.model_registry import get_model_config, resolve_model_config
from open_vision_banana.training import qwen_image21
from open_vision_banana.training.edit_forward import edit_latent_layout
from open_vision_banana.training.flow_matching import FlowScheduler
from open_vision_banana.training.model_backend import ModelBackend

B, C, H, W = 1, 64, 4, 6


def _backend(module=None, pipe=None):
    backend = object.__new__(ModelBackend)
    backend.device = torch.device("cpu")
    backend.dtype = torch.float32
    backend.model_name = "qwen-image-2.1"
    backend.model_config = get_model_config("qwen-image-2.1")
    backend.family = backend.model_config.family
    backend.peft_model = module
    backend.flow_scheduler = FlowScheduler()
    backend._qwen21_pipeline = pipe
    return backend


class _Vae:
    def __init__(self):
        self.config = types.SimpleNamespace(z_dim=C, latents_mean=[0.5] * C, latents_std=[2.0] * C)
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.decoded = None

    def decode(self, z, return_dict=False):
        self.decoded = z
        return (torch.zeros(z.shape[0], 4, 1, 16 * z.shape[-2], 16 * z.shape[-1]),)


class _Pipe:
    def __init__(self):
        self.vae = _Vae()
        self.encoded = None
        self.prompt_calls = []

    def _encode_vae_image(self, image, generator):
        self.encoded = image
        return torch.ones(image.shape[0], C, 1, image.shape[-2] // 16, image.shape[-1] // 16)

    def _get_qwen_prompt_embeds(self, prompt, image, device):
        self.prompt_calls.append((prompt, image, device))
        pad = torch.tensor([[False, True, True, False, False]])
        return torch.randn(1, 5, 8), torch.ones(1, 5, dtype=torch.long), pad


def test_registry_entry_and_layout():
    config = get_model_config("qwen-image-2.1")
    assert config.pipeline_class == "QwenImage21Pipeline"
    assert config.family == "qwen-image-2.1"
    assert config.instruction_editing
    assert edit_latent_layout(config) == qwen_image21.LAYOUT
    assert "to_out.0" in config.lora_target_modules
    assert resolve_model_config("/models/Qwen-Image-2.1").name == "qwen-image-2.1"


def test_vae_encode_adds_opaque_alpha_and_drops_frame_axis():
    pipe = _Pipe()
    images = torch.rand(2, 3, 32, 48) * 2 - 1
    latents = qwen_image21.vae_encode(pipe, images)
    assert pipe.encoded.shape == (2, 4, 1, 32, 48)
    torch.testing.assert_close(pipe.encoded[:, :3, 0], images)
    assert torch.all(pipe.encoded[:, 3] == 1)
    assert latents.shape == (2, C, 2, 3)


def test_vae_decode_undoes_normalization_and_keeps_rgb():
    pipe = _Pipe()
    out = qwen_image21.vae_decode(pipe, torch.zeros(1, C, 2, 3))
    assert out.shape == (1, 3, 32, 48)
    torch.testing.assert_close(pipe.vae.decoded, torch.full((1, C, 1, 2, 3), 0.5))


def test_encode_condition_feeds_the_source_to_the_text_encoder():
    pipe = _Pipe()
    images = torch.zeros(1, 3, 32, 32)
    cond = qwen_image21.encode_condition(
        pipe, ["depth"], images, torch.device("cpu"), torch.float32
    )
    prompt, pils, _ = pipe.prompt_calls[0]
    assert prompt == ["depth"] and len(pils) == 1 and pils[0].size == (32, 32)
    assert cond["attention_mask"] is None
    assert cond["image_pad_mask"].sum() == 2


def test_encode_condition_rejects_batches():
    with pytest.raises(ValueError, match="batch_size 1"):
        qwen_image21.encode_condition(
            _Pipe(), ["a", "b"], torch.zeros(2, 3, 32, 32), torch.device("cpu"), torch.float32
        )


def test_edit_forward_puts_source_first_feeds_sigma_and_negates():
    calls = []

    def backbone(**kwargs):
        calls.append(kwargs)
        return (kwargs["hidden_states"] * 3,)

    noisy, context = torch.randn(B, C, H, W), torch.randn(B, C, H, W)
    pad = torch.tensor([[False, True, True, False]])
    text = {"encoder_hidden_states": torch.randn(B, 4, 8), "image_pad_mask": pad}
    out = qwen_image21.edit_forward(backbone, noisy, context, torch.tensor([0.25]), text)

    kwargs = calls[0]
    hidden = kwargs["hidden_states"]
    torch.testing.assert_close(hidden[:, : H * W], context.reshape(B, C, -1).transpose(1, 2))
    torch.testing.assert_close(hidden[:, H * W :], noisy.reshape(B, C, -1).transpose(1, 2))
    torch.testing.assert_close(kwargs["timestep"], torch.tensor([0.75]))
    assert kwargs["img_shapes"] == [[(1, H, W), (1, H, W)]]
    assert kwargs["img_mask"].shape == (B, 4 + H * W // 4)
    assert kwargs["img_mask"][:, 4:].all()
    assert kwargs["encoder_hidden_states_mask"] is None
    torch.testing.assert_close(out, -3 * noisy)


def test_backend_routes_encode_and_forward_through_the_qwen21_layout():
    def backbone(**kwargs):
        return (kwargs["hidden_states"],)

    pipe = _Pipe()
    backend = _backend(module=backbone, pipe=pipe)
    images = torch.zeros(1, 3, 32, 32)
    latents = backend.encode_image(images)
    assert latents.shape == (1, C, 2, 2) and pipe.encoded.shape[1] == 4
    cond = backend.encode_qwen21_edit(["depth"], images)
    noisy = torch.randn(1, C, 2, 2)
    out = backend.forward(noisy, torch.tensor([0.5]), cond, 1, context_latents=latents)
    torch.testing.assert_close(out, -noisy)


def test_cfg_dropout_is_refused_for_qwen21():
    backend = _backend()
    cond = {
        "encoder_hidden_states": torch.zeros(1, 2, 8),
        "image_pad_mask": torch.zeros(1, 2, dtype=torch.bool),
    }
    assert backend.apply_cfg_dropout(cond, 0.0, 1) is cond
    with pytest.raises(NotImplementedError):
        backend.apply_cfg_dropout(cond, 0.5, 1)
