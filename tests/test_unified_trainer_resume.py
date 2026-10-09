"""--resume must restore the unified trainer's state instead of being ignored."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from open_vision_banana.training.callbacks import TrainingCallbacks
from open_vision_banana.training.config import TrainingConfig, apply_cli_overrides, load_config
from open_vision_banana.training.distributed import DistContext
from open_vision_banana.training.trainer import Trainer
from open_vision_banana.training.utils import EMAModel


def make_trainer(tmp_path, seed: int, **config) -> Trainer:
    torch.manual_seed(seed)
    model = torch.nn.Linear(4, 3)
    trainer = object.__new__(Trainer)
    trainer.config = TrainingConfig(**config)
    trainer.backend = SimpleNamespace(peft_model=model)
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda s: 0.5**s)
    trainer.ema = EMAModel(model, decay=0.9)
    trainer.callbacks = TrainingCallbacks(output_dir=tmp_path / f"out{seed}")
    trainer.dist = DistContext()
    return trainer


def train_and_save(trainer: Trainer, steps: int = 3):
    model = trainer.backend.peft_model
    for _ in range(steps):
        model(torch.randn(2, 4)).sum().backward()
        trainer.optimizer.step()
        trainer.lr_scheduler.step()
        trainer.optimizer.zero_grad()
        trainer.ema.update(model)
    trainer.callbacks.best_val_loss = 0.25
    return trainer.callbacks.on_checkpoint(
        steps, model, trainer.optimizer, trainer.lr_scheduler, ema=trainer.ema,
        selection_protocol=trainer._selection_protocol(),
    )


def weights(trainer: Trainer) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in trainer.backend.peft_model.named_parameters()}


@pytest.mark.parametrize("as_file", [False, True])
def test_resume_restores_full_state(tmp_path, as_file):
    source = make_trainer(tmp_path, seed=0)
    ckpt_dir = train_and_save(source)
    resumed = make_trainer(tmp_path, seed=1)

    step = resumed._resume(str(ckpt_dir / "lora.pt" if as_file else ckpt_dir))

    assert step == 3
    for name, tensor in weights(source).items():
        torch.testing.assert_close(weights(resumed)[name], tensor)
        torch.testing.assert_close(resumed.ema.shadow_params[name], source.ema.shadow_params[name])
    assert resumed.ema.step == 3
    assert resumed.lr_scheduler.last_epoch == 3
    assert resumed.lr_scheduler.get_last_lr() == source.lr_scheduler.get_last_lr()
    state = resumed.optimizer.state_dict()["state"]
    assert state and all(s["step"] == 3 for s in state.values())
    assert resumed.callbacks.best_val_loss == 0.25
    # A new seed for the resumed segment, so it does not replay the first steps' draws.
    assert torch.initial_seed() == resumed.config.seed + 3


def test_resume_without_ema_starts_shadow_from_loaded_weights(tmp_path):
    source = make_trainer(tmp_path, seed=0)
    ckpt_dir = train_and_save(source)
    ckpt = torch.load(ckpt_dir / "lora.pt")
    del ckpt["ema_state_dict"]
    torch.save(ckpt, ckpt_dir / "lora.pt")
    resumed = make_trainer(tmp_path, seed=1)

    resumed._resume(str(ckpt_dir))

    for name, tensor in weights(source).items():
        torch.testing.assert_close(resumed.ema.shadow_params[name], tensor)


def test_resume_weights_only_resets_schedule(tmp_path):
    source = make_trainer(tmp_path, seed=0)
    ckpt_dir = train_and_save(source)
    resumed = make_trainer(tmp_path, seed=1, resume_weights_only=True)

    assert resumed._resume(str(ckpt_dir)) == 0

    for name, tensor in weights(source).items():
        torch.testing.assert_close(weights(resumed)[name], tensor)
        torch.testing.assert_close(resumed.ema.shadow_params[name], tensor)
    assert resumed.ema.step == 0
    assert resumed.lr_scheduler.last_epoch == 0
    assert not resumed.optimizer.state_dict()["state"]
    assert resumed.callbacks.best_val_loss == float("inf")


@pytest.mark.parametrize(
    "other_model",
    [torch.nn.Sequential(torch.nn.Linear(4, 3)), torch.nn.Linear(4, 5)],
    ids=["other-keys", "other-shape"],
)
def test_resume_rejects_mismatched_lora(tmp_path, other_model):
    ckpt_dir = train_and_save(make_trainer(tmp_path, seed=0))
    other = make_trainer(tmp_path, seed=1)
    other.backend.peft_model = other_model
    with pytest.raises(ValueError, match="does not match"):
        other._resume(str(ckpt_dir))


def test_train_continues_from_resumed_step(tmp_path):
    source = make_trainer(tmp_path, seed=0)
    ckpt_dir = train_and_save(source)
    trainer = make_trainer(tmp_path, seed=1, steps=5, grad_accum=1, save_every=100, log_every=100)
    trainer.start_step = trainer._resume(str(ckpt_dir))
    model = trainer.backend.peft_model
    trainer.trainable_params = list(model.parameters())
    trainer.train_loader = DataLoader(list(range(4)), batch_size=1)
    trainer.val_loader = None
    calls = []

    def fake_step(batch, global_step):
        calls.append(global_step)
        loss = model(torch.ones(1, 4)).sum()
        return {
            "loss": loss,
            "total_loss": 0.0,
            "flow_loss": 0.0,
            "depth_aux": 0.0,
            "normal_aux": 0.0,
        }

    trainer._train_step = fake_step
    trainer.train()

    assert calls == [3, 4]  # two steps left of five, not five more
    assert trainer.ema.step == 5
    assert (trainer.callbacks.output_dir / "checkpoint-5_final" / "lora.pt").is_file()


def test_resume_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        make_trainer(tmp_path, seed=0)._resume(str(tmp_path / "nope"))


def test_yaml_checkpoint_resume_keys_are_read(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "checkpoint:\n  resume: out/checkpoint-2000/lora.pt\n  resume_weights_only: true\n",
        encoding="utf-8",
    )
    config = load_config(config_path=cfg)
    assert config.resume == "out/checkpoint-2000/lora.pt"
    assert config.resume_weights_only is True
    assert load_config(config_path=None).resume_weights_only is False
    cfg.write_text("resume: top.pt\nresume_weights_only: true\n", encoding="utf-8")
    config = load_config(config_path=cfg)
    assert (config.resume, config.resume_weights_only) == ("top.pt", True)


def test_cli_resume_flags_override():
    args = argparse.Namespace(resume="ckpt", resume_weights_only=True)
    config = apply_cli_overrides(TrainingConfig(), args)
    assert config.resume == "ckpt"
    assert config.resume_weights_only is True


def test_training_without_ema(tmp_path):
    trainer = make_trainer(tmp_path, seed=0, steps=1, grad_accum=1, ema_enabled=False)
    trainer.ema = None
    trainer.start_step = 0
    trainer.trainable_params = list(trainer.backend.peft_model.parameters())
    trainer.train_loader = DataLoader(list(range(2)), batch_size=1)
    trainer.val_loader = None
    trainer._train_step = lambda batch, step: {
        "loss": trainer.backend.peft_model(torch.ones(1, 4)).sum(),
        "total_loss": 0.0, "flow_loss": 0.0, "depth_aux": 0.0, "normal_aux": 0.0,
    }
    trainer.train()
    checkpoint = torch.load(trainer.callbacks.output_dir / "checkpoint-1_final/lora.pt")
    assert "ema_state_dict" not in checkpoint


def test_changed_validation_protocol_resets_best_score(tmp_path):
    source = make_trainer(tmp_path, seed=0)
    path = train_and_save(source)
    source_payload = torch.load(path / "lora.pt")
    source_payload["selection_protocol"] = "old_crop_loss"
    torch.save(source_payload, path / "lora.pt")
    resumed = make_trainer(tmp_path, seed=1)
    resumed._resume(str(path))
    assert resumed.callbacks.best_val_loss == float("inf")


def test_missing_manifest_fails_before_model_loading(tmp_path, monkeypatch):
    import open_vision_banana.training.trainer as module

    called = []
    monkeypatch.setattr(module, "ModelBackend", lambda **kwargs: called.append(kwargs))
    config = TrainingConfig(pair_manifest=str(tmp_path / "missing.jsonl"), device="cpu")
    with pytest.raises(FileNotFoundError):
        Trainer(config)
    assert not called


def test_validation_protocol_changes_with_manifest_content(tmp_path):
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text("first")
    trainer = make_trainer(tmp_path, seed=0, pair_manifest=str(manifest))
    first = trainer._selection_protocol()
    manifest.write_text("second")
    assert trainer._selection_protocol() != first
