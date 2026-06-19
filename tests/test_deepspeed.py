"""Tests for DeepSpeed configuration utilities."""

import pytest

from open_vision_banana.training.deepspeed_utils import (
    DEFAULT_DEEPSPEED_CONFIG,
    load_deepspeed_config,
    merge_training_into_config,
)


class TestDeepSpeedConfig:
    def test_default_config_exists(self):
        assert DEFAULT_DEEPSPEED_CONFIG.exists()

    def test_load_zero3_config(self):
        cfg = load_deepspeed_config(DEFAULT_DEEPSPEED_CONFIG)
        assert cfg["zero_optimization"]["stage"] == 3
        assert cfg["bf16"]["enabled"] is True

    def test_merge_training_params(self):
        base = load_deepspeed_config(DEFAULT_DEEPSPEED_CONFIG)
        merged = merge_training_into_config(
            base,
            micro_batch_size=4,
            gradient_accumulation_steps=2,
            lr=1e-4,
            max_grad_norm=0.5,
        )
        assert merged["train_micro_batch_size_per_gpu"] == 4
        assert merged["gradient_accumulation_steps"] == 2
        assert merged["gradient_clipping"] == 0.5
        assert merged["optimizer"]["params"]["lr"] == 1e-4

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_deepspeed_config("/nonexistent/deepspeed.json")
