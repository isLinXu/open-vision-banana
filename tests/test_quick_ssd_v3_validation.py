"""Regression tests for the SSD-1B v3 quick-validation controls."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.nn as nn
from _paths import ARCHIVE_SCRIPTS
from PIL import Image

from open_vision_banana.training.tuner import InstructionTuner, VisionTaskDataset

ROOT = Path(__file__).parents[1]


def load_script(name: str):
    path = ARCHIVE_SCRIPTS / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_data_dir_is_exclusive(tmp_path: Path):
    module = load_script("train_sdxl_mps.py")
    selected = tmp_path / "vb_snvc_train_v3"
    selected.mkdir()

    assert module.resolve_data_dirs(str(selected)) == [selected.resolve()]


def test_dedicated_source_training_uses_all_samples_by_default():
    module = load_script("train_sdxl_mps.py")

    assert module.resolve_max_per_task(
        None,
        source_conditioned=True,
        explicit_data_dir=True,
    ) is None
    assert module.resolve_max_per_task(
        None,
        source_conditioned=False,
        explicit_data_dir=False,
    ) == 300
    assert module.resolve_max_per_task(
        17,
        source_conditioned=True,
        explicit_data_dir=True,
    ) == 17


def test_source_training_defaults_enable_real_x0_supervision():
    module = load_script("train_sdxl_mps.py")
    perception = SimpleNamespace(
        task="depth",
        source_conditioned=None,
        x0_loss_weight=None,
        pixel_loss_weight=None,
    )
    generation = SimpleNamespace(
        task="generation",
        source_conditioned=None,
        x0_loss_weight=None,
        pixel_loss_weight=None,
    )

    module.resolve_source_training_defaults(perception)
    module.resolve_source_training_defaults(generation)

    assert perception.source_conditioned is True
    assert perception.x0_loss_weight == 1.0
    assert perception.pixel_loss_weight == 0.5
    assert generation.source_conditioned is False
    assert generation.x0_loss_weight == 0.0
    assert generation.pixel_loss_weight == 0.0


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("depth", ("depth",)),
        ("normal", ("normal",)),
        ("segmentation", ("semantic",)),
        ("edge", ("edge",)),
        ("generation", ("generation",)),
        ("all", ("depth", "normal", "semantic", "edge", "generation")),
    ],
)
def test_validation_tasks_follow_training_selection(task: str, expected: tuple[str, ...]):
    module = load_script("train_sdxl_mps.py")

    assert module.validation_task_names(task) == expected


def test_manifest_split_excludes_declared_validation_samples(tmp_path: Path):
    module = load_script("train_sdxl_mps.py")
    (tmp_path / "train_val_split.json").write_text(
        json.dumps(
            {
                "train": {"depth": ["train.jpg"]},
                "val": {"depth": ["val.jpg"]},
            }
        ),
        encoding="utf-8",
    )
    dataset = SimpleNamespace(
        samples=[
            {"image_path": str(tmp_path / "train.jpg")},
            {"image_path": str(tmp_path / "val.jpg")},
        ]
    )

    removed = module.filter_dataset_to_manifest_split(
        dataset,
        tmp_path,
        "depth",
    )

    assert removed == 1
    assert Path(dataset.samples[0]["image_path"]).name == "train.jpg"


def test_relative_depth_evaluator_reads_validation_split(tmp_path: Path):
    module = load_script("evaluate_v3_relative_depth.py")
    (tmp_path / "train_val_split.json").write_text(
        json.dumps(
            {
                "train": {"depth": ["train.jpg"]},
                "val": {"depth": ["b.jpg", "a.jpg"]},
            }
        ),
        encoding="utf-8",
    )

    assert module.validation_depth_names(tmp_path) == ["b.jpg", "a.jpg"]
    assert module.depth_names(tmp_path, "train") == ["train.jpg"]


def test_find_pairs_matches_stems_and_extensions(tmp_path: Path):
    module = load_script("quick_validate_ssd_v3.py")
    image_dir = tmp_path / "images"
    target_dir = tmp_path / "targets"
    image_dir.mkdir()
    target_dir.mkdir()
    Image.new("RGB", (8, 8), "black").save(image_dir / "b.jpg")
    Image.new("RGB", (8, 8), "white").save(target_dir / "b.png")
    Image.new("RGB", (8, 8), "black").save(image_dir / "a.jpg")
    Image.new("RGB", (8, 8), "white").save(target_dir / "a.png")

    pairs = module.find_pairs(tmp_path, 1)

    assert pairs == [(image_dir / "a.jpg", target_dir / "a.png")]


def test_quick_validation_defaults_to_epsilon(monkeypatch: pytest.MonkeyPatch):
    module = load_script("quick_validate_ssd_v3.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quick_validate_ssd_v3.py",
            "--data-dir",
            "data",
            "--output-dir",
            "output",
        ],
    )

    assert module.parse_args().prediction_type == "epsilon"


def test_quick_validation_accepts_nested_source_conditioned_adapter(tmp_path: Path):
    module = load_script("quick_validate_ssd_v3.py")
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "adapter").mkdir(parents=True)
    (checkpoint / "adapter" / "adapter_model.safetensors").write_bytes(b"weights")

    assert module.validate_checkpoint_path(checkpoint) == checkpoint


def test_tuner_stops_at_exact_optimizer_step(tmp_path: Path):
    class FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_k = nn.Conv2d(3, 8, 1)
            self.to_v = nn.Conv2d(3, 8, 1)
            self.out = nn.Conv2d(8, 3, 1)

        def forward(self, hidden_states, timestep=None, **kwargs):
            return self.out(self.to_k(hidden_states) + self.to_v(hidden_states))

    class FakeGenerator:
        transformer = FakeTransformer()

    image_dir = tmp_path / "images"
    target_dir = tmp_path / "targets"
    image_dir.mkdir()
    target_dir.mkdir()
    Image.new("RGB", (8, 8), "black").save(image_dir / "a.png")
    Image.new("RGB", (8, 8), "white").save(target_dir / "a.png")
    dataset = VisionTaskDataset(str(tmp_path), resolution=8, task="depth")
    tuner = InstructionTuner(
        model=FakeGenerator(),
        lora_rank=2,
        lora_alpha=4,
        allow_component_fallback=True,
        use_ema=False,
        use_gradient_checkpointing=False,
    )

    summary = tuner.train(
        train_dataset=dataset,
        epochs=5,
        batch_size=1,
        max_steps=2,
        output_dir=None,
        log_every=1,
    )

    assert summary["global_step"] == 2
    assert summary["reached_max_steps"] is True


def test_resumed_scheduler_keeps_absolute_max_step_horizon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_k = nn.Conv2d(3, 8, 1)
            self.to_v = nn.Conv2d(3, 8, 1)
            self.out = nn.Conv2d(8, 3, 1)

        def forward(self, hidden_states, timestep=None, **kwargs):
            return self.out(self.to_k(hidden_states) + self.to_v(hidden_states))

    class FakeGenerator:
        transformer = FakeTransformer()

    image_dir = tmp_path / "images"
    target_dir = tmp_path / "targets"
    image_dir.mkdir()
    target_dir.mkdir()
    for index in range(4):
        Image.new("RGB", (8, 8), "black").save(image_dir / f"{index}.png")
        Image.new("RGB", (8, 8), "white").save(target_dir / f"{index}.png")

    dataset = VisionTaskDataset(str(tmp_path), resolution=8, task="depth")
    tuner = InstructionTuner(
        model=FakeGenerator(),
        lora_rank=2,
        lora_alpha=4,
        allow_component_fallback=True,
        use_ema=False,
        use_gradient_checkpointing=False,
    )
    captured: dict[str, int] = {}

    def capture_scheduler(scheduler_type: str, total_steps: int, warmup_steps: int):
        captured["total_steps"] = total_steps
        raise RuntimeError("scheduler captured")

    monkeypatch.setattr(tuner, "_create_scheduler", capture_scheduler)

    with pytest.raises(RuntimeError, match="scheduler captured"):
        tuner.train(
            train_dataset=dataset,
            epochs=5,
            batch_size=1,
            start_step=4,
            start_epoch=1,
            max_steps=20,
            output_dir=None,
        )

    assert captured["total_steps"] == 20
