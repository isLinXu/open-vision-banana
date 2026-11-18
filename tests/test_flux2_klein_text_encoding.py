"""FLUX.2 [klein] text conditioning used by training matches Flux2KleinPipeline.encode_prompt."""

import dataclasses
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("diffusers.pipelines.flux2.pipeline_flux2_klein")
transformers = pytest.importorskip("transformers")

from open_vision_banana.model_registry import PipelineComponents, encode_text, get_model_config  # noqa: E402


class _ChatTokenizer:
    """Tiny stand-in with the two calls the pipeline makes: apply_chat_template and __call__."""

    def __init__(self):
        self.templates = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt and not enable_thinking
        text = "<user>" + messages[0]["content"] + "<assistant>"
        self.templates.append(text)
        return text

    def __call__(self, text, return_tensors, padding, truncation, max_length):
        assert padding == "max_length" and truncation
        ids = [3 + (ord(c) % 40) for c in text][:max_length]
        mask = [1] * len(ids) + [0] * (max_length - len(ids))
        ids = ids + [0] * (max_length - len(ids))
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])}


def _qwen3(layers=28, hidden=8):
    config = transformers.Qwen3Config(vocab_size=64, hidden_size=hidden, intermediate_size=16,
                                      num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=1,
                                      head_dim=4, max_position_embeddings=64)
    torch.manual_seed(0)
    return transformers.Qwen3ForCausalLM(config).eval()


def test_registry_uses_the_pipeline_sequence_length():
    assert get_model_config("flux2-klein").text_encoder_max_length == 512


def test_flux2_klein_stacks_hidden_layers_9_18_27():
    config = dataclasses.replace(get_model_config("flux2-klein"), text_encoder_max_length=16)
    encoder, tokenizer = _qwen3(), _ChatTokenizer()
    components = PipelineComponents(unet=MagicMock(), vae=MagicMock(), text_encoder=encoder, text_encoder_2=None,
                                    tokenizer=tokenizer, config=config)
    hidden, second = encode_text(components, ["a cat", "segment the dog"], torch.device("cpu"), torch.float32)

    assert second is None
    assert hidden.shape == (2, 16, 3 * 8)  # joint_attention_dim = 3 x hidden_size, not the vocabulary
    assert len(tokenizer.templates) == 2  # chat template applied per prompt
    ids = torch.cat([tokenizer(t, return_tensors="pt", padding="max_length", truncation=True,
                               max_length=16)["input_ids"] for t in tokenizer.templates])
    mask = (ids != 0).long()
    with torch.no_grad():
        states = encoder(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False).hidden_states
    expected = torch.cat([states[9], states[18], states[27]], dim=-1)
    torch.testing.assert_close(hidden, expected)
