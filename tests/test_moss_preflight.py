"""MOSS-Image passes the manifest preflight checks that only look at its registry entry."""

from open_vision_banana.data.validation import _is_attention_lora_target
from open_vision_banana.model_registry import get_model_config


def test_moss_projection_names_count_as_attention_targets():
    assert _is_attention_lora_target("proj.q")
    assert _is_attention_lora_target("qkv")
    assert _is_attention_lora_target("to_q")  # the existing markers still match
    assert not _is_attention_lora_target("proj.ffn_down")
    assert not _is_attention_lora_target("proj_out")
    assert any(
        _is_attention_lora_target(t) for t in get_model_config("moss-image").lora_target_modules
    )
