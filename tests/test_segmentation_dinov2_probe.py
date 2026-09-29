"""Unit checks for the local DINOv2 segmentation gate configuration."""

from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from _paths import ARCHIVE_SCRIPTS


def load_module():
    scripts = ARCHIVE_SCRIPTS
    sys.path.insert(0, str(scripts))
    path = scripts / "segmentation_dinov2_probe.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dino_gate_requires_patch_aligned_input() -> None:
    module = load_module()
    args = Namespace(
        image_size=225,
        steps=1,
        batch_size=1,
        eval_batch_size=1,
        train_samples=1,
        val_samples=1,
        dino_repository=Path("."),
        dino_weights=Path(__file__),
    )
    with pytest.raises(ValueError, match="multiple of 14"):
        module.validate_args(args)


def test_dino_gate_accepts_224_patch_grid_when_local_paths_exist(tmp_path: Path) -> None:
    module = load_module()
    weight = tmp_path / "weights.pth"
    weight.write_bytes(b"placeholder")
    args = Namespace(
        image_size=224,
        steps=1,
        batch_size=1,
        eval_batch_size=1,
        train_samples=1,
        val_samples=1,
        dino_repository=tmp_path,
        dino_weights=weight,
    )
    module.validate_args(args)


def test_cached_slice_uses_independent_storage(tmp_path: Path) -> None:
    """A per-sample feature cache must not serialize its source batch."""
    batch = torch.zeros(32, 256, 384)
    sample = batch[0].clone()
    cache_path = tmp_path / "feature.pt"
    torch.save(sample, cache_path)
    assert cache_path.stat().st_size < 1_000_000
