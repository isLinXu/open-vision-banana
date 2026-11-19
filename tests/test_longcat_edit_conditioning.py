"""LongCat-Image-Edit training conditioning follows LongCatImageEditPipeline (source image in the prompt)."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from open_vision_banana.training.model_backend import (  # noqa: E402
    ModelBackend,
    encode_longcat_edit_condition,
    longcat_edit_pipeline_from_components,
)


class FakePipe:
    """Records what the pipeline encoder receives; embeddings depend on the prompt and image size."""

    def __init__(self):
        self.calls = []
        self.image_processor = SimpleNamespace(
            resize=lambda image, height, width: image.resize((width, height))
        )

    def encode_prompt(self, prompt, image):
        self.calls.append((prompt, image.size))
        value = float(len(prompt[0]) + image.size[0])
        return torch.full((1, 6, 4), value), torch.arange(18.0).reshape(6, 3)


def _images(n, size=64):
    return [Image.new("RGB", (size, size), color=(10 * i, 0, 0)) for i in range(n)]


def test_each_sample_is_encoded_with_its_half_size_source():
    pipe = FakePipe()
    cond = encode_longcat_edit_condition(
        pipe, ["depth", "segment it"], _images(2), torch.device("cpu"), torch.float32
    )
    assert pipe.calls == [(["depth"], (32, 32)), (["segment it"], (32, 32))]
    assert cond["encoder_hidden_states"].shape == (2, 6, 4)
    assert cond["encoder_hidden_states"][1, 0, 0].item() == len("segment it") + 32
    assert torch.equal(cond["text_ids"], torch.arange(18.0).reshape(6, 3))


def test_backend_routes_longcat_edits_through_the_pipeline(monkeypatch):
    backend = object.__new__(ModelBackend)
    backend.model_config = SimpleNamespace(pipeline_class="LongCatImageEditPipeline")
    backend.pipe = FakePipe()
    backend.device, backend.dtype = torch.device("cpu"), torch.float32
    backend.encode_text = lambda *a, **k: pytest.fail("generic text path used for LongCat")
    # torch.rand -> 0 drops every prompt
    monkeypatch.setattr(torch, "rand", lambda n, device=None: torch.zeros(n))
    images = torch.zeros(2, 3, 64, 64)
    cond = backend.encode_edit_text(["depth", "segment it"], images, cfg_dropout_rate=0.5)
    assert [prompt for prompt, _ in backend.pipe.calls] == [[""], [""]]  # empty prompt, same source
    assert cond["encoder_hidden_states"].shape == (2, 6, 4)


def test_pipeline_is_built_from_the_loaded_components(monkeypatch):
    seen = {}

    class FakeLongCat:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    import diffusers

    monkeypatch.setattr(diffusers, "LongCatImageEditPipeline", FakeLongCat, raising=False)
    parts = SimpleNamespace(
        scheduler="s", vae="v", text_encoder="te", tokenizer="tok", processor="proc", unet="tr"
    )
    longcat_edit_pipeline_from_components(parts)
    assert seen == {
        "scheduler": "s",
        "vae": "v",
        "text_encoder": "te",
        "tokenizer": "tok",
        "text_processor": "proc",
        "transformer": "tr",
    }
