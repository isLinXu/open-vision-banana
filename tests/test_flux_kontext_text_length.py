"""The FLUX.1 Kontext prompt length the preflight checks is the one training encodes with."""

import inspect

from open_vision_banana.model_registry import get_model_config
from open_vision_banana.training.model_backend import ModelBackend


def test_kontext_length_matches_the_t5_encoding_used_in_training():
    t5_length = inspect.signature(ModelBackend._flux_encode_text).parameters["max_length"].default
    assert get_model_config("flux-kontext").text_encoder_max_length == t5_length == 512
