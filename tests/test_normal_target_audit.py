"""Unit tests for normal target audit math."""

from __future__ import annotations

import importlib.util
import sys

import pytest
import torch
from _paths import ARCHIVE_SCRIPTS


def load_module():
    path = ARCHIVE_SCRIPTS / "audit_normal_targets.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_decode_raw_normals_matches_normal_rgb_contract() -> None:
    module = load_module()
    rgb = torch.tensor([[[0.5]], [[0.5]], [[1.0]]])
    raw = module.decode_raw_normals(rgb)
    assert torch.allclose(raw, torch.tensor([[[0.0]], [[0.0]], [[1.0]]]))


def test_constant_normal_has_zero_neighbour_variation() -> None:
    module = load_module()
    normal = torch.zeros(3, 4, 4)
    normal[2] = 1.0
    variation = module.neighbour_angular_variation(normal)
    assert variation.numel() == 24
    assert float(variation.max()) == pytest.approx(0.0, abs=1e-5)


def test_angular_error_detects_orthogonal_directions() -> None:
    module = load_module()
    angles = module.angular_error_degrees(
        torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([0.0, 1.0, 0.0])
    )
    assert float(angles.item()) == pytest.approx(90.0, abs=1e-4)
