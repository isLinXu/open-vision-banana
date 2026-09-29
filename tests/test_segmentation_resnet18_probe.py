"""Tests for the discriminative segmentation gate's label operations."""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest
from _paths import ARCHIVE_SCRIPTS


def load_module():
    path = ARCHIVE_SCRIPTS / "segmentation_resnet18_probe.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_patch_mode_uses_dominant_valid_label() -> None:
    module = load_module()
    labels = np.array([
        [1, 1, 2, 2],
        [1, 255, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 255],
    ], dtype=np.uint8)
    assert np.array_equal(module.patch_mode_labels(labels, 2), np.array([1, 2, 3, 4], dtype=np.uint8))


def test_expanded_grid_restores_requested_shape() -> None:
    module = load_module()
    labels = np.array([[1, 2, 3, 4]], dtype=np.uint8)
    expanded = module.expand_grid_labels(labels, 2, 4)
    assert expanded.shape == (1, 4, 4)
    assert np.array_equal(expanded[0, :2, :2], np.full((2, 2), 1, dtype=np.uint8))
    assert np.array_equal(expanded[0, 2:, 2:], np.full((2, 2), 4, dtype=np.uint8))


def test_metrics_perfect_prediction_is_one() -> None:
    module = load_module()
    labels = np.array([[1, 2], [module.IGNORE_LABEL, 2]], dtype=np.uint8)
    result = module.metrics(labels, labels)
    assert result["pixel_accuracy"] == pytest.approx(1.0)
    assert result["miou"] == pytest.approx(1.0)
