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


def test_cfg_dropout_pads_unpadded_condition_lengths(monkeypatch):
    """MOSS encodes without padding: the empty-prompt branch is shorter than the batch's prompts."""
    import pytest

    torch = pytest.importorskip("torch")
    from open_vision_banana.training.model_backend import ModelBackend

    backend = object.__new__(ModelBackend)
    backend.device, backend.dtype = torch.device("cpu"), torch.float32
    backend.encode_text = lambda prompts, device, dtype: {
        "encoder_hidden_states": torch.full((len(prompts), 3, 4), -1.0),
        "attention_mask": torch.ones(len(prompts), 3, dtype=torch.bool),
    }
    cond = {
        "encoder_hidden_states": torch.ones(2, 7, 4),
        "attention_mask": torch.ones(2, 7, dtype=torch.bool),
    }
    monkeypatch.setattr(torch, "rand", lambda n, device=None: torch.tensor([0.0, 0.9]))
    out = backend.apply_cfg_dropout(cond, 0.5, 2)
    assert out["encoder_hidden_states"].shape == (2, 7, 4)
    assert out["encoder_hidden_states"][0, :3].eq(-1).all()  # dropped sample: empty prompt
    assert out["encoder_hidden_states"][0, 3:].eq(0).all()  # padded tokens
    assert out["attention_mask"][0].tolist() == [True] * 3 + [False] * 4
    assert out["encoder_hidden_states"][1].eq(1).all()  # kept sample untouched
