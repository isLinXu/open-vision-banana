"""Training forwards for instruction editors, using stub backbones on CPU."""

import pytest
import torch

from open_vision_banana.model_registry import get_model_config
from open_vision_banana.training.flow_matching import FlowScheduler, pack_latents
from open_vision_banana.training.model_backend import ModelBackend

B, C, H, W = 2, 4, 8, 8
TOKENS = (H // 2) * (W // 2)


def _backend(name, module):
    backend = object.__new__(ModelBackend)
    backend.device = torch.device("cpu")
    backend.dtype = torch.float32
    backend.model_name = name
    backend.model_config = get_model_config(name)
    backend.family = backend.model_config.family
    backend.peft_model = module
    backend.flow_scheduler = FlowScheduler()
    return backend


def _text(batch=B, dim=8, length=3):
    return {"encoder_hidden_states": torch.randn(batch, length, dim)}


class _Record:
    def __init__(self, producer):
        self.calls = []
        self._producer = producer

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._producer(args, kwargs)


def test_qwen_edit_appends_packed_source_and_img_shapes():
    def produce(_args, kwargs):
        hidden = kwargs["hidden_states"]
        assert hidden.shape == (B, TOKENS * 2, C * 4)
        assert kwargs["img_shapes"] == [[(1, H // 2, W // 2), (1, H // 2, W // 2)]] * B
        assert kwargs["timestep"].shape == (B,)
        return (hidden[:, :TOKENS],)

    backend = _backend("qwen-image-edit", _Record(produce))
    noisy, context = torch.randn(B, C, H, W), torch.randn(B, C, H, W)
    out = backend.forward(noisy, torch.tensor([0.2, 0.7]), _text(), B, context_latents=context)
    # Native frame: sigma = 1 - t in, noise - x0 out, so the repo velocity is its negation.
    torch.testing.assert_close(backend.peft_model.calls[0][1]["timestep"], torch.tensor([0.8, 0.3]))
    torch.testing.assert_close(out, -noisy)
    packed = backend.peft_model.calls[0][1]["hidden_states"]
    torch.testing.assert_close(packed[:, :TOKENS], pack_latents(noisy, B))
    torch.testing.assert_close(packed[:, TOKENS:], pack_latents(context, B))


def test_qwen_edit_2509_uses_the_same_packed_layout():
    def produce(_args, kwargs):
        return (kwargs["hidden_states"][:, :TOKENS],)

    backend = _backend("qwen-image-edit-2509", _Record(produce))
    out = backend.forward(
        torch.randn(B, C, H, W), torch.zeros(B), _text(), B, context_latents=torch.randn(B, C, H, W)
    )
    assert out.shape == (B, C, H, W)
    assert "img_shapes" in backend.peft_model.calls[0][1]


@pytest.mark.parametrize(
    ("name", "scale"),
    [
        ("qwen-image-edit", 1.0),
        ("longcat-image-edit", 1.0),
        ("flux2-klein", 1.0),
        ("joyimage-edit", 1000.0),
    ],
)
def test_flow_editors_feed_sigma_and_negate_the_prediction(name, scale):
    # Native frame: sigma = 1 - t in, noise - x0 out, so the repo velocity is its negation.
    backend = _backend(name, _Record(lambda _a, kw: (torch.ones_like(kw["hidden_states"]),)))
    t = torch.tensor([0.1, 0.9])
    out = backend.forward(
        torch.randn(B, C, H, W), t, _text(), B, context_latents=torch.randn(B, C, H, W)
    )
    torch.testing.assert_close(backend.peft_model.calls[0][1]["timestep"], (1.0 - t) * scale)
    torch.testing.assert_close(out, -torch.ones(B, C, H, W))


def test_longcat_edit_marks_source_modality_two():
    def produce(_args, kwargs):
        ids = kwargs["img_ids"]
        assert ids.shape == (TOKENS * 2, 3)
        assert torch.all(ids[:TOKENS, 0] == 1)
        assert torch.all(ids[TOKENS:, 0] == 2)
        return (kwargs["hidden_states"][:, :TOKENS],)

    backend = _backend("longcat-image-edit", _Record(produce))
    out = backend.forward(
        torch.randn(B, C, H, W), torch.full((B,), 0.4), _text(length=5), B,
        context_latents=torch.randn(B, C, H, W),
    )
    assert out.shape == (B, C, H, W)
    ids = backend.peft_model.calls[0][1]["img_ids"]
    assert torch.all(ids[:TOKENS, 1] >= 5)


def test_flux2_klein_flattens_and_offsets_reference_time():
    def produce(_args, kwargs):
        hidden = kwargs["hidden_states"]
        ids = kwargs["img_ids"]
        assert hidden.shape == (B, H * W * 2, C)
        assert ids.shape == (B, H * W * 2, 4)
        assert torch.all(ids[:, : H * W, 0] == 0)
        assert torch.all(ids[:, H * W :, 0] == 10)
        assert kwargs["guidance"] is None
        return (hidden[:, : H * W],)

    backend = _backend("flux2-klein", _Record(produce))
    out = backend.forward(
        torch.randn(B, C, H, W), torch.tensor([0.1, 0.9]), _text(), B,
        context_latents=torch.randn(B, C, H, W),
    )
    assert out.shape == (B, C, H, W)


def test_joy_edit_puts_the_source_in_the_first_slot():
    def produce(_args, kwargs):
        hidden = kwargs["hidden_states"]
        assert hidden.shape == (B, 2, C, 1, H, W)
        assert torch.allclose(kwargs["timestep"], torch.tensor([750.0, 750.0]))
        return (hidden,)

    backend = _backend("joyimage-edit", _Record(produce))
    context = torch.randn(B, C, H, W)
    out = backend.forward(
        torch.randn(B, C, H, W), torch.full((B,), 0.25), _text(), B, context_latents=context
    )
    assert out.shape == (B, C, H, W)
    hidden = backend.peft_model.calls[0][1]["hidden_states"]
    torch.testing.assert_close(hidden[:, 0, :, 0], context)

    plus = _backend("joyimage-edit-plus", _Record(produce))
    plus_out = plus.forward(
        torch.randn(B, C, H, W), torch.full((B,), 0.25), _text(), B, context_latents=context
    )
    assert plus_out.shape == (B, C, H, W)


def test_instruct_pix2pix_concatenates_channels_and_scales_time():
    class _UNet:
        def __init__(self):
            self.seen = None

        def __call__(self, hidden, timesteps, encoder_hidden_states, **kwargs):
            self.seen = (hidden, timesteps, kwargs)
            return type("Out", (), {"sample": hidden[:, :C]})()

    unet = _UNet()
    backend = _backend("instruct-pix2pix", unet)
    noisy, context = torch.randn(B, C, H, W), torch.randn(B, C, H, W)
    out = backend.forward(noisy, torch.full((B,), 0.5), _text(), B, context_latents=context)
    assert out.shape == (B, C, H, W)
    hidden, times, _kwargs = unet.seen
    torch.testing.assert_close(hidden[:, :C], noisy)
    torch.testing.assert_close(hidden[:, C:], context)
    torch.testing.assert_close(times, torch.full((B,), 500.0))

    sdxl = _UNet()
    backend = _backend("sdxl-instruct-pix2pix", sdxl)
    pooled = torch.randn(B, 6)
    text = {**_text(), "pooled_embeds": pooled}
    backend.forward(noisy, torch.full((B,), 0.5), text, B, context_latents=context)
    added = sdxl.seen[2]["added_cond_kwargs"]
    torch.testing.assert_close(added["text_embeds"], pooled)
    assert added["time_ids"].shape == (B, 6)


def test_hunyuan_edit_forward_passes_clean_condition_at_timestep_zero():
    class _Model:
        def __init__(self):
            self.kwargs = None

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return type("Out", (), {"diffusion_prediction": kwargs["images"]})()

    model = _Model()
    backend = _backend("hunyuan-image3-instruct", model)
    noisy, context = torch.randn(B, C, H, W), torch.randn(B, C, H, W)
    text = {
        "encoder_hidden_states": torch.zeros(B, 1, 4),
        "hunyuan_inputs": {
            "input_ids": torch.zeros(B, 6, dtype=torch.long),
            "image_mask": torch.zeros(B, 6),
            "timesteps_index": torch.zeros(B, 2, dtype=torch.long),
            "cond_vae_image_mask": torch.zeros(B, 6),
            "cond_timesteps_index": torch.zeros(B, 2, dtype=torch.long),
        },
    }
    out = backend.forward(noisy, torch.tensor([0.3, 0.8]), text, B, context_latents=context)
    torch.testing.assert_close(out, -noisy)
    torch.testing.assert_close(model.kwargs["timesteps"], torch.tensor([700.0, 200.0]))
    assert model.kwargs["mode"] == "gen_image"
    assert model.kwargs["first_step"] is True
    assert model.kwargs["use_cache"] is False
    torch.testing.assert_close(model.kwargs["images"], noisy)
    torch.testing.assert_close(model.kwargs["cond_vae_images"], context)
    torch.testing.assert_close(model.kwargs["cond_timesteps"], torch.zeros(B))

    distil = _backend("hunyuan-image3-instruct-distil", model)
    distil.forward(noisy, torch.tensor([0.3, 0.8]), text, B, context_latents=context)
    assert model.kwargs["mode"] == "gen_image"


def test_edit_forward_requires_the_source_image():
    backend = _backend("qwen-image-edit", _Record(lambda *_: None))
    try:
        backend.forward(torch.randn(1, C, H, W), torch.tensor([0.5]), _text(1), 1)
    except ValueError as exc:
        assert "context_latents" in str(exc)
    else:
        raise AssertionError("missing source image was accepted")


def test_hunyuan_encode_builds_the_token_batch_and_uses_its_vae():
    class _Model:
        def __init__(self):
            self.prepared = None
            self.kwargs = None

        def prepare_model_inputs(self, prompt, image, mode, bot_task, image_size, device):
            self.prepared = (prompt, image, mode, bot_task, image_size, device)
            batch = len(prompt)
            return {
                "input_ids": torch.arange(batch * 4).reshape(batch, 4),
                "attention_mask": torch.ones(batch, 4),
                "image_mask": torch.zeros(batch, 4),
                "timesteps_index": torch.zeros(batch, 2, dtype=torch.long),
                "cond_vae_image_mask": torch.ones(batch, 4),
                "cond_timesteps_index": torch.zeros(batch, 2, dtype=torch.long),
            }

        def vae_encode(self, images):
            return torch.zeros(images.shape[0]), torch.zeros(images.shape[0], 4, 8, 8)

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return type("Out", (), {"diffusion_prediction": kwargs["images"]})()

    model = _Model()
    backend = _backend("hunyuan-image3-instruct", model)
    images = torch.zeros(2, 3, 16, 16)
    text = backend.encode_hunyuan_edit(["make it blue", "make it red"], images)
    assert text["hunyuan_inputs"]["input_ids"].shape == (2, 4)
    assert model.prepared[2] == "gen_image"
    assert model.prepared[3] == "image"
    assert len(model.prepared[1]) == 2 and len(model.prepared[1][0]) == 1
    latents = backend.encode_image(images)
    assert latents.shape == (2, 4, 8, 8)
    noisy = torch.randn(2, 4, 8, 8)
    out = backend.forward(noisy, torch.tensor([0.2, 0.4]), text, 2, context_latents=latents)
    torch.testing.assert_close(out, -noisy)
    assert backend.apply_cfg_dropout(text, 0.5, 2) is text
    try:
        backend.encode_text(["x"], torch.device("cpu"), torch.float32)
    except ValueError as exc:
        assert "encode_hunyuan_edit" in str(exc)
    else:
        raise AssertionError("hunyuan encode_text ran without the source image")


def test_hunyuan_forward_requires_token_batch():
    backend = _backend("hunyuan-image3-instruct", _Record(lambda *_: None))
    try:
        backend.forward(
            torch.randn(1, C, H, W), torch.tensor([0.5]), _text(1), 1,
            context_latents=torch.randn(1, C, H, W),
        )
    except ValueError as exc:
        assert "hunyuan_inputs" in str(exc)
    else:
        raise AssertionError("hunyuan forward ran without token inputs")
