"""Best adapters must contain validated weights without changing resume state."""

import json

import pytest
import torch

from open_vision_banana.training.callbacks import TrainingCallbacks
from open_vision_banana.training.utils import EMAModel


class TinyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0]))
        self.frozen = torch.nn.Parameter(torch.tensor([7.0]), requires_grad=False)
        self.exported_weights = []
        self.fail_export = False

    def save_pretrained(self, path):
        self.exported_weights.append(self.weight.detach().clone())
        if self.fail_export:
            raise OSError("adapter export failed")
        from pathlib import Path

        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({"weight": self.weight.detach().clone()}, directory / "adapter.pt")


@pytest.fixture
def setup(tmp_path):
    model = TinyAdapter()
    optimizer = torch.optim.AdamW([model.weight], lr=0.01)
    model.weight.sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    with torch.no_grad():
        model.weight.fill_(2.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    ema = EMAModel(model)
    ema.shadow_params["weight"].fill_(5.0)
    callbacks = TrainingCallbacks(tmp_path, task_prompts={"depth": "depth prompt"})
    return callbacks, model, optimizer, scheduler, ema


def read_checkpoint(directory):
    return (
        torch.load(directory / "lora.pt", weights_only=True),
        json.loads((directory / "metadata.json").read_text()),
        torch.load(directory / "lora_adapter" / "adapter.pt", weights_only=True),
    )


def test_best_exports_validated_ema_but_keeps_online_resume_state(setup):
    callbacks, model, optimizer, scheduler, ema = setup
    # Reproduce the trainer's validation swap and restoration before the callback.
    backup = ema.apply_shadow(model)
    val_loss = model.weight.square().mean().item()
    ema.restore_shadow(model, backup)

    assert callbacks.on_validate(3, val_loss, model, optimizer, scheduler, ema=ema)
    checkpoint, metadata, adapter = read_checkpoint(callbacks.output_dir / "checkpoint-3_best")
    torch.testing.assert_close(adapter["weight"], ema.shadow_params["weight"])
    torch.testing.assert_close(checkpoint["lora_state_dict"]["weight"], torch.tensor([2.0]))
    torch.testing.assert_close(model.weight, torch.tensor([2.0]))
    torch.testing.assert_close(model.frozen, torch.tensor([7.0]))
    torch.testing.assert_close(checkpoint["ema_state_dict"]["weight"], torch.tensor([5.0]))
    saved_optimizer = checkpoint["optimizer_state_dict"]
    live_optimizer = optimizer.state_dict()
    assert saved_optimizer["param_groups"] == live_optimizer["param_groups"]
    for name, value in live_optimizer["state"][0].items():
        torch.testing.assert_close(saved_optimizer["state"][0][name], value)
    assert checkpoint["scheduler_state_dict"] == scheduler.state_dict()
    assert checkpoint["best_val_loss"] == metadata["best_val_loss"] == val_loss
    assert callbacks.best_val_loss == val_loss
    assert metadata["checkpoint_weight_source"] == "online"
    assert metadata["adapter_weight_source"] == "ema"
    assert metadata["validation_weight_source"] == "ema"
    assert metadata["selection_protocol"] == "ema_validation_loss"
    assert metadata["task_prompts"] == {"depth": "depth prompt"}


def test_regular_checkpoint_keeps_online_adapter_and_resume_payload(setup):
    callbacks, model, optimizer, scheduler, ema = setup
    directory = callbacks.on_checkpoint(3, model, optimizer, scheduler, ema=ema)
    checkpoint, metadata, adapter = read_checkpoint(directory)
    torch.testing.assert_close(adapter["weight"], torch.tensor([2.0]))
    torch.testing.assert_close(checkpoint["lora_state_dict"]["weight"], model.weight)
    assert metadata["checkpoint_weight_source"] == metadata["adapter_weight_source"] == "online"
    assert metadata["validation_weight_source"] is None
    assert metadata["selection_protocol"] is None


@pytest.mark.parametrize("failure", ["adapter", "payload", "metadata", "apply"])
def test_failed_best_save_restores_online_and_does_not_commit_best(setup, monkeypatch, failure):
    callbacks, model, optimizer, scheduler, ema = setup
    callbacks.best_val_loss = 10.0
    if failure == "adapter":
        model.fail_export = True
    elif failure == "payload":
        def fail_save(*args, **kwargs):
            raise OSError("payload save failed")

        monkeypatch.setattr(torch, "save", fail_save)
    elif failure == "metadata":
        from pathlib import Path

        def fail_write(*args, **kwargs):
            raise OSError("metadata save failed")

        monkeypatch.setattr(Path, "write_text", fail_write)
    else:
        def fail_apply(module):
            with torch.no_grad():
                module.weight.fill_(5.0)
            raise OSError("partial EMA apply failed")

        monkeypatch.setattr(ema, "apply_shadow", fail_apply)

    with pytest.raises(OSError):
        callbacks.on_validate(3, 1.0, model, optimizer, scheduler, ema=ema)
    assert callbacks.best_val_loss == 10.0
    torch.testing.assert_close(model.weight, torch.tensor([2.0]))
    torch.testing.assert_close(ema.shadow_params["weight"], torch.tensor([5.0]))
    if failure == "adapter":
        torch.testing.assert_close(model.exported_weights[-1], torch.tensor([5.0]))


@pytest.mark.parametrize("empty_ema", [False, True])
def test_best_without_ema_exports_online_and_records_custom_protocol(setup, empty_ema):
    callbacks, model, optimizer, scheduler, ema = setup
    if empty_ema:
        ema.shadow_params.clear()
    else:
        ema = None
    assert callbacks.on_validate(
        3, 1.0, model, optimizer, scheduler, ema=ema,
        selection_protocol="heldout_depth_v1",
    )
    _, metadata, adapter = read_checkpoint(callbacks.output_dir / "checkpoint-3_best")
    torch.testing.assert_close(adapter["weight"], torch.tensor([2.0]))
    assert metadata["adapter_weight_source"] == metadata["validation_weight_source"] == "online"
    assert metadata["selection_protocol"] == "heldout_depth_v1"


@pytest.mark.parametrize("loss", [2.0, 3.0, float("nan"), float("inf"), -float("inf")])
def test_non_improving_or_nonfinite_loss_does_not_export(setup, loss):
    callbacks, model, optimizer, scheduler, ema = setup
    callbacks.best_val_loss = 2.0
    assert not callbacks.on_validate(3, loss, model, optimizer, scheduler, ema=ema)
    assert callbacks.best_val_loss == 2.0
    assert model.exported_weights == []
    assert not (callbacks.output_dir / "checkpoint-3_best").exists()


def test_regular_adapter_export_failure_remains_nonfatal(setup):
    callbacks, model, optimizer, scheduler, ema = setup
    model.fail_export = True
    directory = callbacks.on_checkpoint(3, model, optimizer, scheduler, ema=ema)
    assert (directory / "lora.pt").exists()
    assert (directory / "metadata.json").exists()
    torch.testing.assert_close(model.weight, torch.tensor([2.0]))
