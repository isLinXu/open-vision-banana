"""Regression tests for the audited v5 pseudo-depth target encoding."""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
from _paths import ARCHIVE_SCRIPTS
from PIL import Image


def load_script_module():
    path = ARCHIVE_SCRIPTS / "infer_depth_pseudo.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_depth_to_rgb_uses_v5_rainbow_endpoints() -> None:
    module = load_script_module()
    rgb = np.asarray(module.depth_to_rgb(np.array([[0.0, 1.0]], dtype=np.float32)))
    assert tuple(rgb[0, 0]) == (0, 0, 255)
    assert tuple(rgb[0, 1]) == (255, 0, 0)


def test_normalization_handles_constant_depth() -> None:
    module = load_script_module()
    normalized = module.normalize_depth_to_uint8(np.full((2, 3), 4.0, dtype=np.float32))
    assert normalized.dtype == np.uint8
    assert np.array_equal(normalized, np.full((2, 3), 128, dtype=np.uint8))
    rgb = np.asarray(module.depth_to_rgb(normalized))
    assert np.array_equal(rgb, np.full((2, 3, 3), 128, dtype=np.uint8))


def test_rgb_metrics_are_exact_for_equal_images() -> None:
    module = load_script_module()
    image = Image.new("RGB", (4, 4), (12, 34, 56))
    metrics = module.rgb_metrics(image, image)
    assert metrics["rgb_mae_0_255"] == 0.0
    assert metrics["rgb_rmse_0_255"] == 0.0
    assert metrics["rgb_psnr_db"] is None
    assert metrics["rgb_psnr_is_infinite"] is True
    assert metrics["exact_rgb_pixel_fraction"] == 1.0
    assert metrics["max_channel_abs_error"] == 0.0


def test_evenly_spaced_selection_has_no_repeats() -> None:
    module = load_script_module()
    assert module.select_evenly_spaced_indices(5, 20) == [0, 1, 2, 3, 4]
    assert module.select_evenly_spaced_indices(10, 3) == [0, 4, 9]
