"""FLUX.1 training encodes prompts through the pipeline load_pipeline built, at the registry T5 length."""

from types import SimpleNamespace

import torch

from open_vision_banana.model_registry import get_model_config
from open_vision_banana.training import model_backend


class FakePipe:
    def __init__(self):
        self.calls = []

    def encode_prompt(self, **kwargs):
        self.calls.append(kwargs)
        n = len(kwargs["prompt"])
        return torch.zeros(n, 3, 4), torch.zeros(n, 4), torch.zeros(3, 3)


def test_backend_keeps_the_pipeline_load_pipeline_built(monkeypatch):
    pipe = FakePipe()

    def load_pipeline(config, **kwargs):
        return SimpleNamespace(unet=torch.nn.Linear(2, 2), vae=None, config=config, pipe=pipe)

    monkeypatch.setattr(model_backend, "load_pipeline", load_pipeline)
    monkeypatch.setattr(model_backend, "apply_lora", lambda backbone, *args, **kwargs: backbone)
    backend = model_backend.ModelBackend("flux-kontext", torch.device("cpu"), dtype=torch.float32)
    assert backend.pipe is pipe

    cond = backend.encode_text(["depth", "segment"], torch.device("cpu"), torch.float32)
    # T5 at the registry length, which the manifest preflight checks legends against
    assert pipe.calls[0]["max_sequence_length"] == 512
    assert get_model_config("flux-kontext").text_encoder_max_length == 512
    assert cond["encoder_hidden_states"].shape == (2, 3, 4)
    assert cond["pooled_projections"].shape == (2, 4)
