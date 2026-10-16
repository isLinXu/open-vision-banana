"""Qwen-Image-2.1 training conditioning, with stub pipelines, VAEs and backbones on CPU."""

import types

import pytest
import torch
from PIL import Image

from open_vision_banana.model_registry import (
    get_model_config,
    resolve_model_config,
    vae_decode,
    vae_encode,
)
from open_vision_banana.training import qwen_image21
from open_vision_banana.training.edit_forward import edit_latent_layout, qwen21_edit_forward
from open_vision_banana.training.flow_matching import FlowScheduler
from open_vision_banana.training.model_backend import ModelBackend

C = 64
CONFIG = get_model_config("qwen-image-2.1")


class _Vae:
    """AutoencoderKLQwenImage21 stand-in: 16x down, 64 latent channels, mean 0.5, std 2."""

    def __init__(self):
        self.config = types.SimpleNamespace(z_dim=C, latents_mean=[0.5] * C, latents_std=[2.0] * C)
        self.encoded = None
        self.decoded = None

    def encode(self, x):
        self.encoded = x
        mode = torch.full((x.shape[0], C, 1, x.shape[-2] // 16, x.shape[-1] // 16), 4.5)
        return types.SimpleNamespace(latent_dist=types.SimpleNamespace(mode=lambda: mode))

    def decode(self, z, return_dict=False):
        self.decoded = z
        return (torch.zeros(z.shape[0], 4, 1, 16 * z.shape[-2], 16 * z.shape[-1]),)


class _Pipe:
    def __init__(self):
        self.calls = []

    def _get_qwen_prompt_embeds(self, prompt, image, device):
        self.calls.append((prompt, image))
        pad = torch.tensor([[False, True, True, False, False]])
        return torch.randn(1, 5, 8), torch.ones(1, 5, dtype=torch.long), pad


def _backend(module=None, pipe=None, vae=None):
    backend = object.__new__(ModelBackend)
    backend.device = torch.device("cpu")
    backend.dtype = torch.float32
    backend.model_name = "qwen-image-2.1"
    backend.model_config = CONFIG
    backend.family = CONFIG.family
    backend.peft_model = module
    backend.flow_scheduler = FlowScheduler()
    backend.pipe = pipe
    backend.vae = vae
    backend._qwen21_logged = False
    return backend


def test_registry_entry_and_layout():
    assert CONFIG.pipeline_class == "QwenImage21Pipeline"
    assert CONFIG.family == "qwen-image-2.1"
    assert CONFIG.instruction_editing and CONFIG.vae_latent_channels == C
    assert edit_latent_layout(CONFIG) == "qwen21"
    assert "to_out.0" in CONFIG.lora_target_modules
    assert resolve_model_config("/models/Qwen-Image-2.1").name == "qwen-image-2.1"


def test_vae_encode_adds_opaque_alpha_and_normalizes_per_channel():
    vae = _Vae()
    images = torch.rand(2, 3, 32, 48) * 2 - 1
    latents = vae_encode(vae, images, CONFIG)
    assert vae.encoded.shape == (2, 4, 1, 32, 48)
    torch.testing.assert_close(vae.encoded[:, :3, 0], images)
    assert torch.all(vae.encoded[:, 3] == 1)
    torch.testing.assert_close(latents, torch.full((2, C, 2, 3), 2.0))


def test_vae_decode_undoes_normalization_and_keeps_rgb():
    vae = _Vae()
    out = vae_decode(vae, torch.zeros(1, C, 2, 3), CONFIG)
    assert out.shape == (1, 3, 32, 48)
    torch.testing.assert_close(vae.decoded, torch.full((1, C, 1, 2, 3), 0.5))


def test_build_pipeline_requires_the_processor():
    with pytest.raises(RuntimeError, match="Qwen3VLProcessor"):
        qwen_image21.build_pipeline(types.SimpleNamespace(processor=None))


def test_encode_condition_feeds_the_source_to_the_text_encoder():
    pipe = _Pipe()
    source = Image.new("RGB", (32, 32))
    cond = qwen_image21.encode_condition(pipe, ["depth"], [source], "cpu", torch.float32)
    assert pipe.calls == [(["depth"], [source])]
    assert set(cond) == {"encoder_hidden_states", "image_pad_mask"}
    assert cond["image_pad_mask"].sum() == 2


def test_encode_condition_rejects_batches():
    with pytest.raises(ValueError, match="batch_size 1"):
        qwen_image21.encode_condition(_Pipe(), ["a", "b"], [None, None], "cpu", torch.float32)


def test_edit_forward_puts_source_first_feeds_sigma_and_negates():
    calls = []

    def backbone(**kwargs):
        calls.append(kwargs)
        return (kwargs["hidden_states"] * 3,)

    h, w = 4, 6
    noisy, context = torch.randn(1, C, h, w), torch.randn(1, C, h, w)
    text = {
        "encoder_hidden_states": torch.randn(1, 4, 8),
        "image_pad_mask": torch.tensor([[False, True, True, False]]),
    }
    out = qwen21_edit_forward(backbone, noisy, context, torch.tensor([0.25]), text)

    kwargs = calls[0]
    hidden = kwargs["hidden_states"]
    torch.testing.assert_close(hidden[:, : h * w], context.flatten(2).transpose(1, 2))
    torch.testing.assert_close(hidden[:, h * w :], noisy.flatten(2).transpose(1, 2))
    torch.testing.assert_close(kwargs["timestep"], torch.tensor([0.75]))
    assert kwargs["img_shapes"] == [[(1, h, w), (1, h, w)]]
    assert kwargs["img_mask"].shape == (1, 4 + h * w // 4) and kwargs["img_mask"][:, 4:].all()
    assert kwargs["encoder_hidden_states_mask"] is None
    torch.testing.assert_close(out, -3 * noisy)


def test_backend_encodes_with_the_source_and_routes_the_qwen21_forward():
    def backbone(**kwargs):
        return (kwargs["hidden_states"],)

    pipe = _Pipe()
    backend = _backend(module=backbone, pipe=pipe, vae=_Vae())
    images = torch.zeros(1, 3, 32, 32)
    latents = backend.encode_image(images)
    assert latents.shape == (1, C, 2, 2)
    cond = backend.encode_edit_text(["depth"], images)
    prompt, pils = pipe.calls[0]
    assert prompt == ["depth"] and pils[0].size == (32, 32)
    assert backend._qwen21_logged
    noisy = torch.randn(1, C, 2, 2)
    out = backend.forward(noisy, torch.tensor([0.5]), cond, 1, context_latents=latents)
    torch.testing.assert_close(out, -noisy)


def test_cfg_dropout_blanks_the_prompt_and_keeps_the_source():
    pipe = _Pipe()
    backend = _backend(pipe=pipe)
    images = torch.zeros(1, 3, 32, 32)
    backend.encode_edit_text(["depth"], images, cfg_dropout_rate=0.0)
    backend.encode_edit_text(["depth"], images, cfg_dropout_rate=1.0)
    assert [call[0] for call in pipe.calls] == [["depth"], [""]]
    assert all(len(call[1]) == 1 for call in pipe.calls)


@pytest.mark.parametrize(
    ("batch_size", "resolution", "message"),
    [(4, 512, "batch_size 1"), (1, 528, "multiple of 32"), (1, 224, "multiple of 32")],
)
def test_training_config_is_checked_before_loading(batch_size, resolution, message):
    config = types.SimpleNamespace(batch_size=batch_size, resolution=resolution)
    with pytest.raises(ValueError, match=message):
        qwen_image21.check_training_config(config)
    qwen_image21.check_training_config(types.SimpleNamespace(batch_size=1, resolution=512))


def test_quantized_loading_is_refused(monkeypatch):
    def load_pipeline(*args, **kwargs):
        raise AssertionError("the model must not load")

    monkeypatch.setattr("open_vision_banana.training.model_backend.load_pipeline", load_pipeline)
    with pytest.raises(ValueError, match="Quantized loading"):
        ModelBackend("qwen-image-2.1", device=torch.device("cpu"), quantize="4bit")


class _FakeQwen21Pipeline:
    """Stands in for diffusers.QwenImage21Pipeline in loader tests."""

    def __init__(self, **modules):
        self.modules = modules

    @classmethod
    def from_pretrained(cls, source, **kwargs):
        return types.SimpleNamespace(
            transformer=torch.nn.Linear(2, 2),
            vae=torch.nn.Linear(2, 2),
            text_encoder=torch.nn.Linear(2, 2),
            processor="qwen3-vl-processor",
            scheduler="scheduler",
        )


def test_load_pipeline_keeps_the_processor(monkeypatch):
    import diffusers

    from open_vision_banana.model_registry import load_pipeline

    monkeypatch.setattr(diffusers, "QwenImage21Pipeline", _FakeQwen21Pipeline, raising=False)
    components = load_pipeline(CONFIG, device="cpu", dtype=torch.float32)
    assert components.processor == "qwen3-vl-processor"
    assert components.scheduler == "scheduler"


def test_build_pipeline_wraps_the_loaded_components(monkeypatch):
    import diffusers

    monkeypatch.setattr(diffusers, "QwenImage21Pipeline", _FakeQwen21Pipeline, raising=False)
    parts = types.SimpleNamespace(scheduler="s", vae="v", text_encoder="t", processor="p", unet="u")
    pipe = qwen_image21.build_pipeline(parts)
    assert pipe.modules == {
        "scheduler": "s",
        "vae": "v",
        "text_encoder": "t",
        "processor": "p",
        "transformer": "u",
    }


def test_backend_builds_the_prompt_pipeline_once(monkeypatch):
    from open_vision_banana.training import model_backend

    components = types.SimpleNamespace(unet=torch.nn.Linear(2, 2), vae=_Vae(), processor="p")
    built = []
    monkeypatch.setattr(model_backend, "load_pipeline", lambda config, **kwargs: components)
    monkeypatch.setattr(model_backend, "apply_lora", lambda backbone, **kwargs: backbone)
    monkeypatch.setattr(
        model_backend.qwen_image21, "build_pipeline", lambda parts: built.append(parts) or "pipe"
    )
    backend = ModelBackend("qwen-image-2.1", device=torch.device("cpu"), dtype=torch.float32)
    assert backend.pipe == "pipe" and built == [components]
    assert backend._qwen21_logged is False


def test_trainer_rejects_batches_before_loading_the_model(monkeypatch, tmp_path):
    import open_vision_banana.training.trainer as module
    from open_vision_banana.training.config import TrainingConfig

    loaded = []
    monkeypatch.setattr(module.Trainer, "_build_datasets", lambda self: (None, None))
    monkeypatch.setattr(module, "ModelBackend", lambda **kwargs: loaded.append(kwargs))
    config = TrainingConfig(
        model_name="qwen-image-2.1", batch_size=4, device="cpu", output_dir=str(tmp_path)
    )
    with pytest.raises(ValueError, match="batch_size 1"):
        module.Trainer(config)
    assert not loaded


def test_encode_edit_text_keeps_the_other_layouts(monkeypatch):
    calls = []
    hunyuan = object.__new__(ModelBackend)
    hunyuan.model_config = get_model_config("hunyuan-image3-instruct")
    monkeypatch.setattr(
        ModelBackend, "encode_hunyuan_edit", lambda self, p, i: calls.append(("hunyuan", p)) or {}
    )
    hunyuan.encode_edit_text(["depth"], torch.zeros(1, 3, 8, 8), cfg_dropout_rate=0.5)

    text_only = object.__new__(ModelBackend)
    text_only.model_config = get_model_config("sana-1600m")
    text_only.device, text_only.dtype = torch.device("cpu"), torch.float32
    monkeypatch.setattr(
        ModelBackend, "encode_text", lambda self, p, d, t: calls.append(("text", p)) or {}
    )
    monkeypatch.setattr(
        ModelBackend,
        "apply_cfg_dropout",
        lambda self, cond, rate, n: calls.append(("dropout", rate, n)) or cond,
    )
    text_only.encode_edit_text(["depth"], torch.zeros(1, 3, 8, 8), cfg_dropout_rate=0.1)
    assert calls == [("hunyuan", ["depth"]), ("text", ["depth"]), ("dropout", 0.1, 1)]
