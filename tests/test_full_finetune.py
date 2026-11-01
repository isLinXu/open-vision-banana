"""Full-parameter (FSDP2) training mode: config contract, sharding / checkpoint helpers, Trainer paths.

The helpers run on CPU in a one-rank gloo group; the Qwen-Image-2.1 path itself needs GPUs.
"""

import json
import logging
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import yaml
from safetensors.torch import load_file
from torch.distributed.fsdp import FSDPModule

from open_vision_banana.training import distributed, full_finetune
from open_vision_banana.training.callbacks import TrainingCallbacks
from open_vision_banana.training.config import TrainingConfig, build_arg_parser, load_config
from open_vision_banana.training.distributed import DistContext
from open_vision_banana.training.trainer import Trainer
from open_vision_banana.training.utils import EMAModel


def test_train_mode_defaults_to_lora_and_parses_from_yaml_and_cli(tmp_path):
    assert TrainingConfig().train_mode == "lora"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"training": {"mode": "full"}}))
    assert load_config(config_path=path).train_mode == "full"
    args = build_arg_parser().parse_args(["--config", str(path), "--train-mode", "lora"])
    assert load_config(args).train_mode == "lora"


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"train_mode": "partial"}, "train_mode must be lora or full"),
        ({"train_mode": "full", "quantize": "4bit"}, "unquantized backbone"),
        ({"train_mode": "full", "resume": "x", "resume_weights_only": True}, "own sharded state"),
    ],
)
def test_train_mode_rejections(fields, message):
    with pytest.raises(ValueError, match=message):
        TrainingConfig(**fields).validate()


def test_full_mode_ignores_an_inert_weights_only_flag():
    # The #88 configs set resume_weights_only without a resume path.
    TrainingConfig(train_mode="full", resume_weights_only=True).validate()


def test_plain_tensor_helpers_without_a_group():
    # Runs before the module's group fixture exists (file order).
    params = [torch.nn.Parameter(torch.zeros(2))]
    params[0].grad = torch.tensor([3.0, 4.0])
    assert distributed.clip_grad_norm(params, float("inf")) == pytest.approx(5.0)
    assert torch.equal(params[0].grad, torch.tensor([3.0, 4.0]))
    if not dist.is_initialized():
        assert distributed.broadcast_float(0.5) == 0.5


def test_ema_applied_restores_the_online_weights_even_on_error():
    model = torch.nn.Linear(2, 2)
    ema = EMAModel(model, decay=0.9)
    online = {n: p.detach().clone() for n, p in model.named_parameters()}
    for shadow in ema.shadow_params.values():
        shadow.fill_(7.0)
    with pytest.raises(RuntimeError), ema.applied(model):
        assert all(torch.all(p == 7.0) for p in model.parameters())
        raise RuntimeError
    for name, param in model.named_parameters():
        assert torch.equal(param, online[name])


def test_on_validation_loss_saves_only_finite_improvements(tmp_path, caplog):
    callbacks = TrainingCallbacks(output_dir=tmp_path)
    saved = []
    with caplog.at_level(logging.INFO):
        assert callbacks.on_validation_loss(1, 0.5, lambda: saved.append(1))
        assert not callbacks.on_validation_loss(2, 0.7, lambda: saved.append(2))
        assert not callbacks.on_validation_loss(3, float("nan"), lambda: saved.append(3))
    assert saved == [1] and callbacks.best_val_loss == 0.5
    assert "New best val_loss=0.5000" in caplog.text


class _Config(dict):
    """Stands in for a diffusers FrozenDict config."""


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Config(num_layers=2, inner_dim=8)
        self.proj_in = torch.nn.Linear(4, 8)
        self.transformer_blocks = torch.nn.ModuleList(torch.nn.Linear(8, 8) for _ in range(2))
        self.proj_out = torch.nn.Linear(8, 4)

    def forward(self, x):
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = x + torch.tanh(block(x))
        return self.proj_out(x)


@pytest.fixture(scope="module")
def group():
    distributed.ensure_process_group()
    distributed.ensure_process_group()  # a second call keeps the existing group
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _sharded(seed=0):
    torch.manual_seed(seed)
    model = TinyBackbone().to(torch.bfloat16).requires_grad_(False)
    record = full_finetune.shard_backbone(model)
    return model, record


def test_shard_backbone_unfreezes_fp32_masters_and_records_the_class(group):
    model, record = _sharded()
    assert record == {"class_name": "TinyBackbone", "config": {"num_layers": 2, "inner_dim": 8}}
    assert sum(isinstance(m, FSDPModule) for m in model.modules()) == 3  # two blocks + the root
    params = list(model.parameters())
    assert params and all(p.requires_grad and p.dtype == torch.float32 for p in params)
    out = model(torch.randn(3, 4))
    assert out.dtype == torch.bfloat16  # bf16 compute on fp32 shards


def test_shard_backbone_rejects_a_model_without_blocks(group):
    with pytest.raises(ValueError, match="cannot shard"):
        full_finetune.shard_backbone(torch.nn.Linear(2, 2))


def test_grad_norm_matches_the_unsharded_norm_and_clips(group):
    model, _ = _sharded()
    model(torch.randn(5, 4)).float().pow(2).mean().backward()
    grads = {n: p.grad.full_tensor().clone() for n, p in model.named_parameters()}
    reference = float(
        torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads.values()]))
    )
    norm = distributed.clip_grad_norm(list(model.parameters()), reference / 2)
    assert norm == pytest.approx(reference, rel=1e-6)  # the pre-clip norm
    for name, param in model.named_parameters():
        assert torch.allclose(param.grad.full_tensor(), grads[name] / 2, rtol=1e-4)
    assert distributed.broadcast_float(1.25) == 1.25


def test_save_transformer_writes_a_diffusers_directory(group, tmp_path):
    model, record = _sharded()
    out = tmp_path / "checkpoint-2" / "transformer"
    full_finetune.save_transformer(model, out, record, is_main=True)
    config = json.loads((out / "config.json").read_text())
    assert config["_class_name"] == "TinyBackbone" and config["num_layers"] == 2
    weights = {}
    for shard in out.glob("diffusion_pytorch_model*.safetensors"):
        weights.update(load_file(shard))
    expected = {name: p.full_tensor() for name, p in model.named_parameters()}
    assert set(weights) == set(expected)
    for name, tensor in weights.items():
        assert tensor.dtype == torch.bfloat16
        assert torch.equal(tensor, expected[name].to(torch.bfloat16))
    assert not out.with_name("transformer.tmp").exists()


def _train(model, optimizer, ema, scheduler, steps, seed):
    torch.manual_seed(seed)
    for _ in range(steps):
        model(torch.randn(4, 4)).float().pow(2).mean().backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)


def _parts(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0 / (1 + s))
    return optimizer, scheduler, EMAModel(model, decay=0.9, warmup_steps=0)


def test_train_state_round_trip_continues_exactly(group, tmp_path):
    model, _ = _sharded(seed=1)
    optimizer, scheduler, ema = _parts(model)
    _train(model, optimizer, ema, scheduler, 3, seed=7)
    full_finetune.save_train_state(
        tmp_path / "train_state",
        model,
        optimizer,
        scheduler,
        ema,
        3,
        {"best_val_loss": 0.5},
        is_main=True,
    )
    _train(model, optimizer, ema, scheduler, 2, seed=8)
    reference = {n: p.full_tensor().clone() for n, p in model.named_parameters()}
    reference_ema = {n: t.full_tensor().clone() for n, t in ema.shadow_params.items()}

    other, _ = _sharded(seed=2)
    o_optimizer, o_scheduler, o_ema = _parts(other)
    extra = full_finetune.load_train_state(
        tmp_path / "train_state", other, o_optimizer, o_scheduler, o_ema
    )
    assert extra["step"] == 3 and extra["best_val_loss"] == 0.5
    assert o_ema.step == 3 and o_scheduler.last_epoch == 3
    _train(other, o_optimizer, o_ema, o_scheduler, 2, seed=8)
    for name, tensor in other.named_parameters():
        assert torch.equal(tensor.full_tensor(), reference[name])
    for name, tensor in o_ema.shadow_params.items():
        assert torch.equal(tensor.full_tensor(), reference_ema[name])


def test_ema_swap_around_a_no_grad_forward_needs_and_survives_a_reshard(group):
    model, _ = _sharded(seed=3)
    optimizer, scheduler, ema = _parts(model)
    _train(model, optimizer, ema, scheduler, 2, seed=4)
    online = {n: p.full_tensor().clone() for n, p in model.named_parameters()}
    backup = ema.apply_shadow(model)
    with torch.no_grad():
        model(torch.randn(2, 4))  # the root unit stays gathered after a no-grad forward
    full_finetune.reshard(model)
    ema.restore_shadow(model, backup)
    for name, param in model.named_parameters():
        assert torch.equal(param.full_tensor(), online[name])


def test_save_train_state_replaces_the_previous_one(group, tmp_path):
    model, _ = _sharded()
    optimizer, scheduler, ema = _parts(model)
    path = tmp_path / "train_state"
    for step in (1, 2):
        _train(model, optimizer, ema, scheduler, 1, seed=step)
        full_finetune.save_train_state(
            path, model, optimizer, scheduler, ema, step, {}, is_main=True
        )
    assert torch.load(path / "extra.pt", weights_only=False)["step"] == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["train_state"]
    with pytest.raises(FileNotFoundError):
        full_finetune.load_train_state(tmp_path / "missing", model, optimizer, scheduler, ema)


def _full_trainer(tmp_path, seed=0, ema=True, **config):
    """A Trainer in full mode around TinyBackbone, without loading a pipeline."""
    model, record = _sharded(seed)
    trainer = object.__new__(Trainer)
    trainer.config = TrainingConfig(train_mode="full", **config)
    trainer.backend = SimpleNamespace(peft_model=model)
    trainer.full_record = record
    trainer.optimizer, trainer.lr_scheduler, trainer.ema = _parts(model)
    if not ema:
        trainer.ema = None
    trainer.callbacks = TrainingCallbacks(output_dir=tmp_path / f"out{seed}")
    trainer.callbacks.task_prompts = {"depth": "Generate a depth map."}
    trainer.dist = DistContext()
    trainer.validation_metrics = {}
    return trainer


def _transformer_weights(ckpt_dir):
    weights = {}
    for shard in (ckpt_dir / "transformer").glob("diffusion_pytorch_model*.safetensors"):
        weights.update(load_file(shard))
    return weights


def test_full_trainer_saves_online_weights_with_metadata_and_train_state(group, tmp_path):
    trainer = _full_trainer(tmp_path)
    assert trainer.full
    model = trainer.backend.peft_model
    _train(model, trainer.optimizer, trainer.ema, trainer.lr_scheduler, 2, seed=1)
    trainer._save_full(2)
    ckpt = trainer.callbacks.output_dir / "checkpoint-2"
    meta = json.loads((ckpt / "metadata.json").read_text())
    assert meta["train_mode"] == "full" and meta["transformer_weight_source"] == "online"
    assert meta["val_loss"] is None and meta["validation_weight_source"] is None
    assert meta["task_prompts"] == {"depth": "Generate a depth map."}
    assert meta["selection_protocol"] == trainer._selection_protocol()
    for name, tensor in _transformer_weights(ckpt).items():
        assert torch.equal(
            tensor, dict(model.named_parameters())[name].full_tensor().to(torch.bfloat16)
        )
    extra = torch.load(
        trainer.callbacks.output_dir / "train_state" / "extra.pt", weights_only=False
    )
    assert extra["step"] == 2 and extra["selection_protocol"] == meta["selection_protocol"]


@pytest.mark.parametrize("ema", [True, False])
def test_full_validation_runs_on_ema_weights_and_keeps_one_best(group, tmp_path, ema):
    trainer = _full_trainer(tmp_path, seed=4, ema=ema)
    model = trainer.backend.peft_model
    _train(model, trainer.optimizer, trainer.ema, trainer.lr_scheduler, 3, seed=5)
    online = {n: p.full_tensor().clone() for n, p in model.named_parameters()}
    expected = trainer.ema.shadow_params if ema else dict(model.named_parameters())
    expected = {n: t.full_tensor().clone() for n, t in expected.items()}
    losses = iter([0.4, 0.6, 0.3])

    def validate():
        for name, param in model.named_parameters():
            assert torch.equal(param.full_tensor(), expected[name])
        with torch.no_grad():
            model(torch.randn(2, 4))  # leaves the root unit gathered
        return next(losses)

    trainer.validate = validate
    for step in (10, 20, 30):
        trainer._validate_full(step)
        for name, param in model.named_parameters():
            assert torch.equal(param.full_tensor(), online[name])  # resharded, online weights back
    out = trainer.callbacks.output_dir
    assert sorted(p.name for p in out.glob("checkpoint-*_best")) == ["checkpoint-30_best"]
    assert trainer.callbacks.best_val_loss == 0.3
    meta = json.loads((out / "checkpoint-30_best" / "metadata.json").read_text())
    source = "ema" if ema else "online"
    assert (
        meta["transformer_weight_source"] == source and meta["validation_weight_source"] == source
    )
    assert meta["val_loss"] == meta["best_val_loss"] == 0.3
    for name, tensor in _transformer_weights(out / "checkpoint-30_best").items():
        assert torch.equal(tensor, expected[name].to(torch.bfloat16))
    assert not (out / "train_state").exists()  # a best export does not refresh the resume state


@pytest.mark.parametrize("same_protocol", [True, False])
def test_full_resume_restores_state_step_and_comparable_best(
    group, tmp_path, caplog, same_protocol
):
    trainer = _full_trainer(tmp_path, seed=6)
    model = trainer.backend.peft_model
    _train(model, trainer.optimizer, trainer.ema, trainer.lr_scheduler, 3, seed=7)
    trainer.callbacks.best_val_loss = 0.25
    trainer._save_full(3)

    other = _full_trainer(tmp_path, seed=8, **({} if same_protocol else {"lora_rank": 8}))
    with caplog.at_level(logging.WARNING):
        step = other._resume(str(trainer.callbacks.output_dir))  # the run dir holding train_state
    assert step == 3 and other.ema.step == 3 and other.lr_scheduler.last_epoch == 3
    for name, param in other.backend.peft_model.named_parameters():
        assert torch.equal(param.full_tensor(), dict(model.named_parameters())[name].full_tensor())
    if same_protocol:
        assert other.callbacks.best_val_loss == 0.25
    else:
        assert other.callbacks.best_val_loss == float("inf")
        assert "Reset best score" in caplog.text
