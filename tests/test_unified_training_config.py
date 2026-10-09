"""Effective configuration contract for the unified training entry point."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from open_vision_banana.training.config import (
    TrainingConfig,
    apply_cli_overrides,
    build_arg_parser,
    load_config,
)
from open_vision_banana.training.optimization import SchedulerFactory

ROOT = Path(__file__).resolve().parents[1]


def write_config(tmp_path, raw):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_string_loss_weights_override_defaults(tmp_path):
    path = write_config(tmp_path, {"loss": {"task_loss_weights": "depth=1,normal=2,semantic=1.5"}})
    assert load_config(config_path=path).task_loss_weights == {
        "depth": 1.0, "normal": 2.0, "semantic": 1.5,
    }


def test_real_yaml_normalizes_sampling_ema_and_targets():
    config = load_config(config_path=ROOT / "configs/sana_1600m_3task_full_clean.yaml")
    assert config.task_loss_weights == {"depth": 1.0, "normal": 2.0, "semantic": 1.5}
    assert config.task_sampling == "sqrt"
    assert config.task_sampling_weights == {}
    assert config.ema_enabled is True
    assert config.lora_targets == ["linear", "proj_out", "conv_inverted", "conv_point"]


def test_unspecified_cli_preserves_yaml_without_reading_sys_argv(tmp_path, monkeypatch):
    path = write_config(tmp_path, {
        "training": {"steps": 123, "gradient_checkpointing": False},
        "flow": {"img2img_flow": False}, "ema": {"enabled": False},
        "wandb": {"enabled": True},
    })
    monkeypatch.setattr(sys, "argv", ["pytest", "--config", "/missing.yaml"])
    args = build_arg_parser().parse_args(["--config", str(path)])
    direct = load_config(args)
    staged = apply_cli_overrides(load_config(config_path=path), args)
    assert direct == staged
    assert direct.steps == 123
    assert direct.gradient_checkpointing is False
    assert direct.img2img_flow is False
    assert direct.ema_enabled is False
    assert direct.wandb is True


@pytest.mark.parametrize("field,section,key,flag", [
    ("rslora", "lora", "rs_lora", "rslora"),
    ("img2img_flow", "flow", "img2img_flow", "img2img-flow"),
    ("logit_normal", "flow", "logit_normal_timesteps", "logit-normal"),
    ("min_snr", "flow", "min_snr", "min-snr"),
    ("gradient_checkpointing", "training", "gradient_checkpointing", "gradient-checkpointing"),
    ("ema_enabled", "ema", "enabled", "ema"),
    ("wandb", "wandb", "enabled", "wandb"),
    ("resume_weights_only", "checkpoint", "resume_weights_only", "resume-weights-only"),
])
@pytest.mark.parametrize("enabled", [False, True])
def test_boolean_cli_overrides_both_directions(tmp_path, field, section, key, flag, enabled):
    path = write_config(tmp_path, {section: {key: not enabled}})
    option = f"--{flag}" if enabled else f"--no-{flag}"
    args = build_arg_parser().parse_args(["--config", str(path), option])
    assert getattr(load_config(args), field) is enabled


def test_cli_sampling_loss_and_targets_are_independent(tmp_path):
    path = write_config(tmp_path, {"data": {"task_sampling": "uniform", "task_weights": {"depth": 3}}})
    args = build_arg_parser().parse_args([
        "--config", str(path), "--task-sampling", "sqrt",
        "--task-sampling-weights", "normal=4", "--task-loss-weights", "depth=2",
        "--lora-targets", "to_q", "to_v", "--warmup-steps", "0",
    ])
    config = load_config(args)
    assert config.task_sampling == "sqrt"
    assert config.task_sampling_weights == {"normal": 4.0}
    assert config.task_loss_weights == {"depth": 2.0}
    assert config.lora_targets == ["to_q", "to_v"]
    assert config.warmup_steps == 0


@pytest.mark.parametrize("raw,match", [
    ({"typo": 1}, "typo"),
    ({"training": {"stepz": 3}}, "training.stepz"),
    ({"training": {"epochs": 3}}, "training.epochs"),
    ({"data": {"manifest": "guess.jsonl"}}, "data.manifest"),
    ({"data": {"vision_ratio": 0.1}}, "vision_ratio"),
    ({"data": {"task_sampling": "bogus"}}, "task_sampling"),
    ({"data": {"task_weights": "bogus"}}, "task_sampling"),
    ({"data": {"task_sampling": "uniform", "task_weights": "sqrt"}}, "conflict"),
    ({"ema": {"enabled": "false"}}, "ema_enabled"),
    ({"lora": {"target_modules": "to_q"}}, "lora_targets"),
    ({"training": {"steps": 0}}, "steps"),
    ({"training": {"steps": True}}, "steps"),
    ({"training": {"lr": float("nan")}}, "lr"),
    ({"lora": {"dropout": 0.1}}, "dropout"),
    ({"model": {"model_id": "ignored/model"}}, "model_id"),
    ({"data": {"tasks": []}}, "tasks"),
])
def test_rejects_invalid_or_unsupported_yaml(tmp_path, raw, match):
    with pytest.raises(ValueError, match=match):
        load_config(config_path=write_config(tmp_path, raw))


@pytest.mark.parametrize("weights", ["depth", "depth=1,depth=2", "depth=nan", "=1", "depth=-1", {"depth": True}])
def test_rejects_invalid_loss_weights(tmp_path, weights):
    with pytest.raises(ValueError, match="task_loss_weights"):
        load_config(config_path=write_config(tmp_path, {"loss": {"task_loss_weights": weights}}))


def test_cli_can_repair_yaml_values_before_effective_validation(tmp_path):
    path = write_config(tmp_path, {"data": {"vision_ratio": 0.1}, "training": {"steps": 0}})
    args = build_arg_parser().parse_args(["--config", str(path), "--vision-ratio", "1", "--steps", "2"])
    assert load_config(args).steps == 2


def test_default_sampling_preserves_proportional_traversal():
    assert load_config().task_sampling == "proportional"
    assert load_config().ema_enabled is True
    assert TrainingConfig().parse_task_weights() == TrainingConfig().task_loss_weights


def test_print_effective_config_exits_without_constructing_trainer_or_writing_output(tmp_path):
    output = tmp_path / "must-not-exist"
    path = write_config(tmp_path, {"checkpoint": {"output_dir": str(output)}})
    code = (
        "import runpy,sys; "
        "import open_vision_banana.training.config; "
        "sys.modules['open_vision_banana.training.trainer']=None; "
        f"sys.argv=['train.py','--config',{str(path)!r},'--print-effective-config']; "
        f"runpy.run_path({str(ROOT / 'scripts/train.py')!r},run_name='__main__')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    assert config["output_dir"] == str(output)
    assert config["task_sampling"] == "proportional"
    assert config["ema_enabled"] is True
    assert not any(key.startswith("_") for key in config)
    assert not output.exists()


def test_entry_rejects_mixing_before_importing_trainer(tmp_path):
    path = write_config(tmp_path, {"data": {"vision_ratio": 0.1}})
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/train.py"), "--config", str(path)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "vision_ratio" in result.stderr


@pytest.mark.parametrize("name", ["cosine", "cosine_restarts", "polynomial", "constant", "one_cycle"])
def test_scheduler_whitelist_matches_factory(tmp_path, name):
    path = write_config(tmp_path, {"scheduler": {"type": name}})
    config = load_config(config_path=path)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=config.lr)
    scheduler = SchedulerFactory.create(
        optimizer, total_steps=100, warmup_steps=5, scheduler_type=config.scheduler,
    )
    assert scheduler.optimizer is optimizer
    args = build_arg_parser().parse_args(["--scheduler", name])
    assert load_config(args).scheduler == name


@pytest.mark.parametrize("name", ["linear", "unknown"])
def test_rejects_scheduler_not_supported_by_factory(name):
    args = build_arg_parser().parse_args(["--scheduler", name])
    with pytest.raises(ValueError, match="scheduler"):
        load_config(args)


def test_validation_budget_defaults_to_40():
    config = load_config()
    assert config.val_max_samples_per_task == 40
    assert config.to_dict()["val_max_samples_per_task"] == 40


def test_validation_budget_yaml_and_explicit_cli_priority(tmp_path):
    path = write_config(tmp_path, {
        "validation": {"max_samples_per_task": 12},
        "data": {"max_samples_per_task": 100},
    })
    assert load_config(config_path=path).val_max_samples_per_task == 12
    args = build_arg_parser().parse_args(["--config", str(path)])
    assert load_config(args).val_max_samples_per_task == 12
    args = build_arg_parser().parse_args([
        "--config", str(path), "--val-max-samples-per-task", "8",
    ])
    config = load_config(args)
    assert config.val_max_samples_per_task == 8
    assert config.max_samples_per_task == 100
    assert apply_cli_overrides(load_config(config_path=path), args) == config


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "12", None])
def test_rejects_invalid_validation_budget(tmp_path, budget):
    path = write_config(tmp_path, {"validation": {"max_samples_per_task": budget}})
    with pytest.raises(ValueError, match="val_max_samples_per_task"):
        load_config(config_path=path)


@pytest.mark.parametrize("source", ["yaml", "cli", "direct"])
def test_rejects_unconsumed_tar_index(tmp_path, source):
    with pytest.raises(ValueError, match="tar_index.*not implemented"):
        if source == "yaml":
            path = write_config(tmp_path, {"data": {"tar_index": "index.json"}})
            load_config(config_path=path)
        elif source == "cli":
            load_config(build_arg_parser().parse_args(["--tar-index", "index.json"]))
        else:
            TrainingConfig(tar_index="index.json").validate()


def test_null_tar_index_remains_compatible(tmp_path):
    path = write_config(tmp_path, {"data": {"tar_index": None}})
    assert load_config(config_path=path).tar_index is None


@pytest.mark.parametrize("empty", [None, "", "   "])
def test_empty_top_resume_preserves_checkpoint_resume(tmp_path, empty):
    path = write_config(tmp_path, {
        "resume": empty, "checkpoint": {"resume": "output/checkpoint-100"},
    })
    assert load_config(config_path=path).resume == "output/checkpoint-100"


@pytest.mark.parametrize("empty", [None, "", "   "])
def test_top_resume_fills_empty_checkpoint_resume(tmp_path, empty):
    path = write_config(tmp_path, {
        "resume": "output/checkpoint-100", "checkpoint": {"resume": empty},
    })
    assert load_config(config_path=path).resume == "output/checkpoint-100"


def test_conflicting_resume_paths_raise_before_cli_override(tmp_path):
    path = write_config(tmp_path, {
        "resume": "output/checkpoint-200", "checkpoint": {"resume": "output/checkpoint-100"},
    })
    args = build_arg_parser().parse_args(["--config", str(path), "--resume", "output/checkpoint-300"])
    with pytest.raises(ValueError, match="conflict.*resume.*checkpoint.resume"):
        load_config(args)


@pytest.mark.parametrize("top,nested", [(True, False), (False, True)])
def test_conflicting_resume_booleans_raise(tmp_path, top, nested):
    path = write_config(tmp_path, {
        "resume_weights_only": top, "checkpoint": {"resume_weights_only": nested},
    })
    with pytest.raises(ValueError, match="conflict.*resume_weights_only.*checkpoint.resume_weights_only"):
        load_config(config_path=path)


@pytest.mark.parametrize("enabled", [False, True])
def test_matching_resume_aliases_are_accepted(tmp_path, enabled):
    path = write_config(tmp_path, {
        "resume": "output/checkpoint-100", "resume_weights_only": enabled,
        "checkpoint": {"resume": "output/checkpoint-100", "resume_weights_only": enabled},
    })
    config = load_config(config_path=path)
    assert config.resume == "output/checkpoint-100"
    assert config.resume_weights_only is enabled


@pytest.mark.parametrize("enabled", [False, True])
def test_null_top_resume_boolean_does_not_override_checkpoint(tmp_path, enabled):
    path = write_config(tmp_path, {
        "resume_weights_only": None, "checkpoint": {"resume_weights_only": enabled},
    })
    assert load_config(config_path=path).resume_weights_only is enabled


def test_false_top_resume_boolean_fills_null_checkpoint(tmp_path):
    path = write_config(tmp_path, {
        "resume_weights_only": False, "checkpoint": {"resume_weights_only": None},
    })
    assert load_config(config_path=path).resume_weights_only is False


def test_cli_resume_overrides_unambiguous_yaml(tmp_path):
    path = write_config(tmp_path, {
        "resume": None, "checkpoint": {"resume": "output/checkpoint-100", "resume_weights_only": True},
    })
    args = build_arg_parser().parse_args([
        "--config", str(path), "--resume", "output/checkpoint-200", "--no-resume-weights-only",
    ])
    config = load_config(args)
    assert config.resume == "output/checkpoint-200"
    assert config.resume_weights_only is False
