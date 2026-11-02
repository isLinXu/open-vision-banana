from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from _paths import ARCHIVE_SCRIPTS


def load_audit_script():
    path = ARCHIVE_SCRIPTS / "audit_v3_multitask_dataset.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_monitor_script():
    path = ARCHIVE_SCRIPTS / "monitor_multitask_checkpoints.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_task(root: Path, names: list[str]) -> None:
    task = root / "depth"
    (task / "images").mkdir(parents=True)
    (task / "targets").mkdir(parents=True)
    for name in names:
        (task / "images" / name).write_bytes(b"image")
        (task / "targets" / name).write_bytes(b"target")


def test_audit_reports_counts_and_stable_manifest_checksum(tmp_path: Path):
    module = load_audit_script()
    names = ["a.png", "b.png", "c.png"]
    _make_task(tmp_path, names)
    manifest = {"train": {"depth": names[:2]}, "val": {"depth": names[2:]}}
    (tmp_path / "train_val_split.json").write_text(json.dumps(manifest), encoding="utf-8")

    first = module.audit_v3_dataset(tmp_path, tasks=["depth"])
    second = module.audit_v3_dataset(tmp_path, tasks=["depth"])

    assert first["manifest_checksum"] == second["manifest_checksum"]
    assert first["tasks"]["depth"]["train_count"] == 2
    assert first["tasks"]["depth"]["val_count"] == 1


def test_audit_rejects_unsupported_task(tmp_path: Path):
    module = load_audit_script()
    with pytest.raises(ValueError, match="unsupported"):
        module.audit_v3_dataset(tmp_path, tasks=["keypoint"])


def test_audit_rejects_overlap_and_missing_pairs(tmp_path: Path):
    module = load_audit_script()
    _make_task(tmp_path, ["a.png"])
    (tmp_path / "train_val_split.json").write_text(
        json.dumps({"train": {"depth": ["a.png"]}, "val": {"depth": ["a.png"]}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlap"):
        module.audit_v3_dataset(tmp_path, tasks=["depth"])


def test_runner_builds_all_task_command_after_audit():
    path = ARCHIVE_SCRIPTS / "train_multitask_lora.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    args = type(
        "Args",
        (),
        {
            "model": "ssd-1b",
            "data_dir": Path("data"),
            "output_dir": Path("output"),
            "max_steps": 2,
            "epochs": 1,
            "batch_size": 1,
            "grad_accum": 1,
            "resolution": 64,
            "rank": 4,
            "alpha": 8,
            "lr": 1e-4,
            "max_per_task": 2,
            "max_generation": 2,
            "save_every": 1,
            "log_every": 1,
            "seed": 42,
            "device": "cpu",
            "gradient_checkpointing": True,
        },
    )()

    command = module.build_training_command(args)
    assert command[1].endswith("train_sdxl_mps.py")
    assert "--task" in command and command[command.index("--task") + 1] == "all"
    assert "--gradient-checkpointing" in command
    assert "--no-ema" in command
    assert "--source-conditioned" in command
    assert command[command.index("--prediction-type") + 1] == "epsilon"


def test_runner_scope_is_explicitly_all_five_tasks():
    module = load_audit_script()
    assert {"depth", "normal", "semantic", "generation", "edge"} == module.SUPPORTED_TASKS


def test_checkpoint_monitor_accepts_depth_only_scope():
    module = load_monitor_script()

    assert module.parse_task_names("depth") == ["depth"]
    quality = module._quality_judgement(
        {"validation_losses": {"depth": 0.5}},
        [{"task": "depth", "metrics": {"rgb_mae": 0.2}}],
        None,
        ["depth"],
    )

    assert quality["visual_inference_complete"] is True
    assert quality["visual_inference_tasks"] == ["depth"]


def test_checkpoint_monitor_rejects_unknown_tasks():
    module = load_monitor_script()

    with pytest.raises(ValueError, match="Unsupported monitoring tasks"):
        module.parse_task_names("depth,keypoint")


def test_checkpoint_monitor_always_resumes_paused_training():
    module = load_monitor_script()
    sent_signals = []

    with (
        patch.object(module, "_pid_alive", return_value=True),
        patch.object(module.os, "kill", side_effect=lambda pid, sig: sent_signals.append((pid, sig))),
        pytest.raises(RuntimeError, match="inference failed"),
        module.pause_process(123),
    ):
        raise RuntimeError("inference failed")

    assert sent_signals == [
        (123, module.signal.SIGSTOP),
        (123, module.signal.SIGCONT),
    ]


def test_runner_command_carries_schedule_and_validation_controls():
    path = ARCHIVE_SCRIPTS / "train_multitask_lora.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    args = type(
        "Args",
        (),
        {
            "model": "ssd-1b", "data_dir": Path("data"), "output_dir": Path("output"),
            "max_steps": 2, "epochs": 1, "batch_size": 1, "grad_accum": 1,
            "resolution": 64, "rank": 4, "alpha": 8, "lr": 1e-4,
            "max_per_task": None, "max_val_per_task": 2, "max_generation": None,
            "save_every": 1, "log_every": 1, "seed": 42, "device": "cpu",
            "gradient_checkpointing": False, "task_ratios": '{"depth": 1, "normal": 1}',
            "task_schedule": True, "no_balance_data": True, "eval_every": 2,
        },
    )()
    command = module.build_training_command(args)
    assert "--task-schedule" in command
    assert command[command.index("--eval-every") + 1] == "2"
    assert "--task-ratios" in command


def test_task_schedule_sampler_covers_each_task_first():
    path = ARCHIVE_SCRIPTS / "train_sdxl_mps.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    sampler = module.TaskScheduleSampler([2, 2, 2], ["depth", "normal", "semantic"], 3, seed=7)
    first_cycle = list(sampler)
    assert {value // 2 for value in first_cycle} == {0, 1, 2}
