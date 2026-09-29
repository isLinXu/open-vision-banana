"""Tests for real-normal data discovery and validity filtering."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from _paths import ARCHIVE_SCRIPTS
from PIL import Image


def load_module():
    path = ARCHIVE_SCRIPTS / "audit_real_normal_datasets.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_diode_discovery_requires_matching_rgb_input(tmp_path: Path) -> None:
    module = load_module()
    normal = tmp_path / "split" / "frame_normal.png"
    normal.parent.mkdir()
    Image.new("RGBA", (4, 4), (128, 128, 255, 255)).save(normal)
    assert module.discover_diode(tmp_path) == []
    Image.new("RGB", (4, 4), "white").save(normal.with_name("frame.png"))
    pairs = module.discover_diode(tmp_path)
    assert len(pairs) == 1
    assert pairs[0].input_path.name == "frame.png"


def test_decode_excludes_opaque_black_non_unit_background(tmp_path: Path) -> None:
    module = load_module()
    pixels = np.array([
        [[128, 128, 255, 255], [0, 0, 0, 255]],
        [[128, 128, 255, 0], [128, 255, 128, 255]],
    ], dtype=np.uint8)
    path = tmp_path / "normal.png"
    Image.fromarray(pixels, mode="RGBA").save(path)
    normal, valid = module.decode_normal(path, resolution=2, normal_length_tolerance=0.12)
    assert valid.tolist() == [[True, False], [False, True]]
    assert np.allclose(np.linalg.norm(normal[valid], axis=-1), 1.0)


def test_neighbour_angles_ignore_invalid_boundaries() -> None:
    module = load_module()
    normal = np.array([[[0, 0, 1], [0, 0, 1]], [[0, 1, 0], [0, 1, 0]]], dtype=np.float32)
    valid = np.array([[True, False], [True, True]])
    angles = module.neighbour_angles(normal, valid)
    # The invalid top-right point removes its two edges, while the valid left
    # column still contributes a 90-degree vertical edge.
    assert np.array_equal(angles, np.array([0.0, 90.0], dtype=np.float32))
