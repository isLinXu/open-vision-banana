"""Tests for high-resolution normals derived from relative depth."""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest
from _paths import ARCHIVE_SCRIPTS


def load_module():
    path = ARCHIVE_SCRIPTS / "infer_depth_normal.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_constant_depth_is_forward_facing_normal() -> None:
    module = load_module()
    normal = module.normal_from_relative_depth(np.ones((8, 8), dtype=np.float32))
    assert np.allclose(normal[..., 0], 0.0)
    assert np.allclose(normal[..., 1], 0.0)
    assert np.allclose(normal[..., 2], 1.0)


def test_depth_slope_produces_unit_tilted_normals() -> None:
    module = load_module()
    depth = np.tile(np.linspace(0.0, 1.0, 32, dtype=np.float32), (32, 1))
    normal = module.normal_from_relative_depth(depth)
    assert np.allclose(np.linalg.norm(normal, axis=-1), 1.0, atol=1e-5)
    # A [0,1] slope over 32 pixels is approximately one unit per normalised
    # image coordinate, producing a visibly tilted, but not horizontal, plane.
    assert float(normal[..., 0].mean()) < -0.7
    assert 0.6 < float(normal[..., 2].mean()) < 0.75


def test_normal_rgb_forward_direction_is_project_convention() -> None:
    module = load_module()
    rgb = np.asarray(module.normal_to_rgb(np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)))
    assert tuple(rgb[0, 0]) == (128, 128, 255)


def test_normal_strength_must_be_positive() -> None:
    module = load_module()
    with pytest.raises(ValueError, match="positive"):
        module.normal_from_relative_depth(np.ones((4, 4), dtype=np.float32), 0.0)
