"""LoRA overrides are local to a backend, and logs report actual trainable names."""

import logging
from types import SimpleNamespace

import pytest
import torch

from open_vision_banana import model_registry
from open_vision_banana.model_registry import ModelConfig, get_model_config
from open_vision_banana.training import model_backend


@pytest.fixture
def lora_setup(monkeypatch):
    config = ModelConfig(
        name="test-lora-backend",
        model_id="test/lora-backend",
        family="flux",
        num_text_encoders=2,
        backbone_type="transformer",
        lora_target_modules=["to_q", "to_v"],
        in_context_conditioning=True,
    )
    monkeypatch.setattr(model_registry, "REGISTRY", {config.name: config})
    calls = []

    def components(config):
        return SimpleNamespace(unet=torch.nn.Linear(2, 2), vae=None, config=config)

    def load_pipeline(config, **kwargs):
        return components(config)

    def load_quantized(backend, quantize):
        backend.components = components(backend.model_config)

    def apply_lora(backbone, config, rank, alpha, use_rslora):
        calls.append(
            SimpleNamespace(
                config=config,
                targets=list(config.lora_target_modules),
                rank=rank,
                alpha=alpha,
                use_rslora=use_rslora,
            )
        )
        backbone.requires_grad_(False)
        backbone.register_parameter("lora_A", torch.nn.Parameter(torch.ones(2)))
        backbone.register_parameter("auxiliary", torch.nn.Parameter(torch.ones(1)))
        return backbone

    monkeypatch.setattr(model_backend, "load_pipeline", load_pipeline)
    monkeypatch.setattr(model_backend.ModelBackend, "_load_quantized", load_quantized)
    monkeypatch.setattr(model_backend, "apply_lora", apply_lora)

    def make_backend(**kwargs):
        return model_backend.ModelBackend(
            config.name, torch.device("cpu"), dtype=torch.float32, **kwargs
        )

    return SimpleNamespace(config=config, calls=calls, make_backend=make_backend)


def test_full_tune_mode_skips_lora(lora_setup):
    backend = lora_setup.make_backend(tune_mode="full")
    assert lora_setup.calls == []
    assert backend.peft_model is backend.backbone
    assert all(parameter.requires_grad for parameter in backend.backbone.parameters())
    assert hasattr(backend.peft_model, "save_pretrained")


@pytest.mark.parametrize("quantize", [None, "4bit"])
def test_override_reaches_apply_lora_on_each_loading_path(lora_setup, quantize):
    backend = lora_setup.make_backend(
        quantize=quantize,
        lora_targets=["to_k"],
        lora_rank=4,
        lora_alpha=8,
        use_rslora=False,
    )

    call = lora_setup.calls[-1]
    assert call.targets == ["to_k"]
    assert call.config is backend.model_config
    assert (call.rank, call.alpha, call.use_rslora) == (4, 8, False)
    assert backend.model_config.in_context_conditioning


def test_override_does_not_change_registry_or_next_backend(lora_setup):
    overridden = lora_setup.make_backend(lora_targets=["to_k"])
    default = lora_setup.make_backend()

    assert get_model_config(lora_setup.config.name).lora_target_modules == ["to_q", "to_v"]
    assert lora_setup.calls[0].targets == ["to_k"]
    assert lora_setup.calls[1].targets == ["to_q", "to_v"]
    assert overridden.model_config is not default.model_config


@pytest.mark.parametrize("targets", [None, ["to_k"]])
def test_backend_targets_do_not_alias_registry_or_caller(lora_setup, targets):
    backend = lora_setup.make_backend(lora_targets=targets)
    expected = ["to_q", "to_v"] if targets is None else ["to_k"]

    assert backend.model_config is not lora_setup.config
    assert backend.model_config.lora_target_modules == expected
    assert backend.model_config.lora_target_modules is not lora_setup.config.lora_target_modules
    if targets is not None:
        assert backend.model_config.lora_target_modules is not targets
        targets.append("caller_only")
        assert backend.model_config.lora_target_modules == expected
    backend.model_config.lora_target_modules.append("backend_only")
    assert lora_setup.config.lora_target_modules == ["to_q", "to_v"]


def test_empty_override_is_not_replaced_with_registry_defaults(lora_setup):
    lora_setup.make_backend(lora_targets=[])

    assert lora_setup.calls[-1].targets == []
    assert lora_setup.config.lora_target_modules == ["to_q", "to_v"]


def test_logs_actual_trainable_parameter_names(lora_setup, caplog):
    with caplog.at_level(logging.INFO, logger=model_backend.__name__):
        backend = lora_setup.make_backend()

    expected = [
        name for name, parameter in backend.peft_model.named_parameters() if parameter.requires_grad
    ]
    records = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Trainable parameter names:")
    ]
    assert len(records) == 1
    assert records[0] == f"Trainable parameter names: {expected}"
    assert "weight" not in records[0]
    assert "bias" not in records[0]
    assert "Trainable params: 3 / 9 (33.33%)" in caplog.text


def test_full_mode_trains_the_backbone_itself_without_lora(lora_setup):
    backend = lora_setup.make_backend(train_mode="full")

    assert lora_setup.calls == []
    assert backend.peft_model is backend.backbone
    assert backend.flow_scheduler is not None
