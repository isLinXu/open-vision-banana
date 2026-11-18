"""The FLUX.1 Kontext prompt length the preflight checks is the one training encodes with."""

import inspect

from open_vision_banana.model_registry import get_model_config
from open_vision_banana.training.model_backend import ModelBackend


def test_kontext_length_matches_the_t5_encoding_used_in_training():
    t5_length = inspect.signature(ModelBackend._flux_encode_text).parameters["max_length"].default
    assert get_model_config("flux-kontext").text_encoder_max_length == t5_length == 512


def test_flux_pipeline_is_rebuilt_from_loaded_components(monkeypatch):
    import diffusers

    from open_vision_banana.model_registry import PipelineComponents
    from open_vision_banana.training.model_backend import flux_pipeline_from_components

    seen = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(diffusers, "FluxKontextPipeline", FakePipeline, raising=False)
    parts = {k: object() for k in ("transformer", "vae", "clip", "t5", "tok", "tok2", "sched")}
    components = PipelineComponents(
        unet=parts["transformer"],
        vae=parts["vae"],
        text_encoder=parts["clip"],
        text_encoder_2=parts["t5"],
        tokenizer=parts["tok"],
        tokenizer_2=parts["tok2"],
        scheduler=parts["sched"],
    )
    pipe = flux_pipeline_from_components(components, "FluxKontextPipeline")
    assert isinstance(pipe, FakePipeline)
    assert seen == {
        "scheduler": parts["sched"],
        "vae": parts["vae"],
        "text_encoder": parts["clip"],
        "tokenizer": parts["tok"],
        "text_encoder_2": parts["t5"],
        "tokenizer_2": parts["tok2"],
        "transformer": parts["transformer"],
    }
