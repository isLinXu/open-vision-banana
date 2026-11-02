"""Tests for the frozen DINOv2 normal-regression gate helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from _paths import ARCHIVE_SCRIPTS


def load_module():
    scripts = ARCHIVE_SCRIPTS
    sys.path.insert(0, str(scripts))
    path = scripts / "normal_dinov2_probe.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_patch_targets_average_only_valid_normals() -> None:
    module = load_module()
    normal = np.zeros((4, 4, 3), dtype=np.float32)
    normal[..., 2] = 1.0
    normal[0:2, 0:2] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    valid = np.ones((4, 4), dtype=bool)
    valid[0, 0] = False
    target, mask = module.patch_normal_targets(normal, valid, grid_size=2, min_valid_fraction=0.5)
    assert mask.tolist() == [True, True, True, True]
    assert np.allclose(target[0], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(np.linalg.norm(target, axis=-1), 1.0)


def test_patch_targets_reject_sparse_invalid_patch() -> None:
    module = load_module()
    normal = np.zeros((4, 4, 3), dtype=np.float32)
    normal[..., 2] = 1.0
    valid = np.zeros((4, 4), dtype=bool)
    valid[0, 0] = True
    _, mask = module.patch_normal_targets(normal, valid, grid_size=2, min_valid_fraction=0.5)
    assert mask.tolist() == [False, False, False, False]


def test_expanded_metrics_are_zero_for_identical_unit_normals(tmp_path: Path) -> None:
    module = load_module()
    normal_path = tmp_path / "frame_normal.png"
    rgb = np.full((4, 4, 4), (128, 128, 255, 255), dtype=np.uint8)
    from PIL import Image

    Image.fromarray(rgb, mode="RGBA").save(normal_path)
    pair = module.NormalPair("diode", tmp_path / "frame.png", normal_path)
    prediction = np.zeros((1, 1, 3), dtype=np.float32)
    prediction[..., 2] = 1.0
    result = module.expanded_metrics(prediction, [pair], image_size=4, tolerance=0.12)
    # The 8-bit [128, 128, 255] encoding is a near-front vector, not exact
    # [0, 0, 1], so the metric should only contain a sub-degree quantisation error.
    assert result["mean_angular_error_deg"] == pytest.approx(0.318, abs=0.02)


def test_per_sample_cache_storage_is_not_a_batch_view(tmp_path: Path) -> None:
    import torch

    batch = torch.zeros(32, 256, 384)
    cache_path = tmp_path / "feature.pt"
    torch.save(batch[0].clone(), cache_path)
    assert cache_path.stat().st_size < 1_000_000
