import json
from types import SimpleNamespace

import pytest
from PIL import Image

from open_vision_banana.training.config import TrainingConfig
from open_vision_banana.training.distributed import DistContext
from open_vision_banana.training.trainer import ManifestVisionDataset, Trainer


def test_uniform_sampler_probabilities_and_rank_partition():
    from open_vision_banana.training.sampling import TaskWeightedSampler

    groups = {"depth": list(range(100)), "normal": list(range(100, 104)), "semantic": [104]}
    full = TaskWeightedSampler(groups, strategy="uniform", seed=7, num_samples=1000)
    assert full.task_probabilities == pytest.approx(dict.fromkeys(groups, 1 / 3))
    ranks = [
        TaskWeightedSampler(
            groups, strategy="uniform", seed=7, num_samples=1000, rank=r, world_size=2
        )
        for r in range(2)
    ]
    draws = list(full)
    assert list(ranks[0]) == draws[::2]
    assert list(ranks[1]) == draws[1::2]
    assert list(full) == draws
    full.set_epoch(1)
    assert list(full) != draws


def test_sqrt_probabilities_and_invalid_weights():
    from open_vision_banana.training.sampling import TaskWeightedSampler

    groups = {"depth": list(range(100)), "normal": [100, 101, 102, 103]}
    sampler = TaskWeightedSampler(groups, strategy="sqrt")
    assert sampler.task_probabilities == pytest.approx({"depth": 10 / 12, "normal": 2 / 12})
    with pytest.raises(ValueError):
        TaskWeightedSampler(groups, weights={"depth": float("nan")})


def make_manifest(tmp_path, counts):
    Image.new("RGB", (8, 8)).save(tmp_path / "image.png")
    path = tmp_path / "pairs.jsonl"
    rows = [
        {"input": "image.png", "target": "image.png", "task": t, "split": "train"}
        for t, n in counts.items()
        for _ in range(n)
    ]
    path.write_text("\n".join(map(json.dumps, rows)))
    return path


def test_uniform_strategy_is_consumed_by_trainer(tmp_path):
    from open_vision_banana.training.sampling import TaskWeightedSampler

    path = make_manifest(tmp_path, {"depth": 8, "normal": 2})
    config = TrainingConfig(
        pair_manifest=str(path), tasks=["depth", "normal"], task_sampling="uniform", resolution=8
    )
    loader, _ = Trainer._build_datasets(SimpleNamespace(config=config, dist=DistContext()))
    assert isinstance(loader.sampler, TaskWeightedSampler)


def test_empty_requested_task_and_empty_loader_fail(tmp_path):
    path = make_manifest(tmp_path, {"depth": 1})
    for tasks, batch in [(["depth", "normal"], 1), (["depth"], 2)]:
        config = TrainingConfig(pair_manifest=str(path), tasks=tasks, batch_size=batch)
        with pytest.raises(ValueError):
            Trainer._build_datasets(SimpleNamespace(config=config, dist=DistContext()))


def test_manifest_missing_file_is_not_silently_dropped(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps(
            {"task": "depth", "split": "train", "input": "missing.png", "target": "missing.png"}
        )
    )
    with pytest.raises(FileNotFoundError):
        ManifestVisionDataset(path, 8, "train", {"depth"})


def test_sampling_weights_accept_task_alias(tmp_path):
    path = make_manifest(tmp_path, {"seg": 4})
    config = TrainingConfig(
        pair_manifest=str(path), tasks=["seg"], resolution=8, task_sampling_weights={"seg": 2.0}
    )
    loader, _ = Trainer._build_datasets(SimpleNamespace(config=config, dist=DistContext()))
    assert loader.sampler.task_probabilities == {"semantic": 1.0}
