"""Tests for experiment manifests and strict preflight validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from open_vision_banana.config import DataConfig, VisionBananaConfig
from open_vision_banana.data.split import split_vision_datasets
from open_vision_banana.reproducibility import (
    build_experiment_manifest,
    configured_tasks,
    fingerprint_path,
)


def test_legacy_task_names_are_canonicalized():
    config = VisionBananaConfig(
        data=DataConfig(tasks={"depth_estimation": {}, "normal_estimation": {}})
    )
    assert configured_tasks(config) == ["depth", "normal"]


def test_metadata_fingerprint_changes_with_file(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("one", encoding="utf-8")
    first = fingerprint_path(tmp_path)
    file_path.write_text("a different value", encoding="utf-8")
    second = fingerprint_path(tmp_path)
    assert first["sha256"] != second["sha256"]
    assert second["file_count"] == 1


def test_manifest_contains_effective_config_and_dataset_hash(tmp_path: Path):
    task_root = tmp_path / "depth"
    task_root.mkdir()
    (task_root / "sample.bin").write_bytes(b"sample")
    config = VisionBananaConfig(
        data=DataConfig(
            root=str(tmp_path),
            tasks=["depth"],
            task_dirs={"depth": "depth"},
        )
    )
    manifest = build_experiment_manifest(config, repo_root=tmp_path)
    assert manifest["config_sha256"]
    assert manifest["datasets"]["task:depth"]["file_count"] == 1


def test_manifest_rejects_missing_data(tmp_path: Path):
    config = VisionBananaConfig(
        data=DataConfig(
            root=str(tmp_path),
            tasks=["depth"],
            task_dirs={"depth": "missing"},
        )
    )
    with pytest.raises(FileNotFoundError, match="task:depth"):
        build_experiment_manifest(config, repo_root=tmp_path)


def test_source_split_keeps_tasks_separate_without_source_leakage(tmp_path: Path):
    class SourceDataset:
        def __init__(self, task: str, paths: list[Path]):
            self.TASK_TYPE = task
            self.samples = [{"image": path} for path in paths]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return {"image": str(self.samples[index]["image"]), "task": self.TASK_TYPE}

    paths = []
    for index in range(20):
        path = tmp_path / f"source-{index}.bin"
        path.write_bytes(f"source-{index}".encode())
        paths.append(path)
    datasets = [SourceDataset("depth", paths), SourceDataset("normal", paths)]

    train_datasets, val_dataset = split_vision_datasets(
        datasets, val_ratio=0.2, seed=42, source_level=True
    )
    assert len(train_datasets) == 2
    assert val_dataset is not None

    train_sources = {
        datasets[dataset_index].samples[int(local_index)]["image"]
        for dataset_index, subset in enumerate(train_datasets)
        for local_index in subset.indices
    }
    val_sources = {Path(val_dataset[index]["image"]) for index in range(len(val_dataset))}
    assert train_sources.isdisjoint(val_sources)
