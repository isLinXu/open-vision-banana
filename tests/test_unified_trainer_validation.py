"""Validation must measure the training objective, and the same one at every call."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from open_vision_banana.training.config import TrainingConfig
from open_vision_banana.training.trainer import Trainer

B, C, H, W = 2, 4, 8, 8


def make_trainer(**config) -> tuple[Trainer, list[dict]]:
    flow_calls: list[dict] = []

    def prepare_flow_batch(target, source, batch_size, **kwargs):
        flow_calls.append(kwargs)
        noise = torch.randn_like(target)
        return {
            "timesteps": torch.rand(batch_size),
            "noisy_latents": noise,
            "velocity_target": target,
        }

    backend = SimpleNamespace(
        peft_model=torch.nn.Linear(1, 1),
        encode_image=lambda pixels: torch.zeros(pixels.shape[0], C, H, W),
        prepare_flow_batch=prepare_flow_batch,
        encode_text=lambda *a: {},
        encode_edit_text=lambda *a, **k: {},
        forward=lambda noisy, *a: noisy,
    )
    losses = SimpleNamespace(
        compute_flow_loss=lambda pred, target, t, tasks, device: (
            (pred - target).square().mean() + t.mean(),
            None,
            None,
        )
    )
    trainer = object.__new__(Trainer)
    trainer.config = TrainingConfig(**config)
    trainer.device = torch.device("cpu")
    trainer.dtype = torch.float32
    trainer.backend = backend
    trainer.losses = losses
    batch = {
        "prompt": ["p"] * B,
        "target_rgb": torch.rand(B, 3, 16, 16),
        "image": torch.rand(B, 3, 16, 16),
        "task": ["depth"] * B,
    }
    trainer.val_loader = [batch, batch, batch]
    return trainer, flow_calls


def test_validation_uses_training_flow_settings():
    trainer, flow_calls = make_trainer(logit_normal=False, flow_shift=3.0, flow_sigma=0.1)
    trainer.validate()
    assert len(flow_calls) == 3
    for kwargs in flow_calls:
        assert kwargs["logit_normal"] is False
        assert kwargs["flow_shift"] == 3.0
        assert kwargs["flow_sigma"] == 0.1


def test_validation_loss_is_repeatable_and_leaves_training_rng_alone():
    trainer, _ = make_trainer(seed=7)
    torch.manual_seed(123)
    first = trainer.validate()
    after_first = torch.rand(3)
    second = trainer.validate()

    assert first == second  # same noise and timesteps at every evaluation
    torch.manual_seed(123)
    torch.testing.assert_close(after_first, torch.rand(3))  # as if validate() never drew


def test_validation_covers_late_tasks_and_macro_averages():
    trainer, _ = make_trainer()
    depth = trainer.val_loader[0]
    normal = dict(depth, task=["normal"] * B)
    trainer.val_loader = [depth] * 25 + [normal]
    trainer.losses.compute_flow_loss = lambda pred, target, t, tasks, device: (
        torch.tensor(1.0 if tasks[0] == "depth" else 3.0), None, None
    )
    assert trainer.validate() == 2.0
    assert trainer.validation_metrics["sample_counts"] == {"depth": 40, "normal": 2}


def test_validation_restores_mode_and_rejects_nonfinite():
    trainer, _ = make_trainer()
    trainer.backend.peft_model.eval()
    trainer.validate()
    assert not trainer.backend.peft_model.training
    trainer.backend.peft_model.train()
    trainer.losses.compute_flow_loss = lambda *a: (torch.tensor(float("nan")), None, None)
    with pytest.raises(FloatingPointError):
        trainer.validate()
    assert trainer.backend.peft_model.training


def test_empty_validation_is_not_zero_loss():
    trainer, _ = make_trainer()
    trainer.val_loader = []
    with pytest.raises(ValueError):
        trainer.validate()


def test_macro_loss_weights_samples_not_batches():
    trainer, _ = make_trainer()
    batch = trainer.val_loader[0]
    tail = {k: v[:1] for k, v in batch.items()}
    trainer.val_loader = [batch, tail]
    trainer.losses.compute_flow_loss = lambda pred, target, t, tasks, device: (
        torch.tensor(2.0 if len(tasks) == 2 else 8.0), None, None
    )
    assert trainer.validate() == 4.0


def test_validation_budget_is_configurable():
    trainer, calls = make_trainer(val_max_samples_per_task=3)
    trainer.validate()
    assert len(calls) == 2
    assert trainer.validation_metrics["sample_counts"] == {"depth": 3}


def test_validation_subset_does_not_read_discarded_samples():
    from torch.utils.data import DataLoader, Dataset

    from open_vision_banana.training.trainer import _collate_fn

    trainer, _ = make_trainer(val_max_samples_per_task=2)
    example = trainer.val_loader[0]
    reads = []

    class RecordedDataset(Dataset):
        samples = [{"task": "depth"}] * 100

        def __len__(self):
            return 100

        def __getitem__(self, index):
            reads.append(index)
            return {k: v[0] for k, v in example.items()}

    trainer.val_loader = DataLoader(RecordedDataset(), batch_size=2, collate_fn=_collate_fn)
    trainer.validate()
    assert reads == [0, 1]
