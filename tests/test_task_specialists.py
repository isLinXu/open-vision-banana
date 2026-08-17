"""Tests for the unified task-specialist CLI helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "infer_task_specialists.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_tasks_deduplicates_preserving_order() -> None:
    module = load_module()
    assert module.parse_tasks("normal,depth,normal") == ("normal", "depth")


def test_parse_tasks_accepts_validated_semantic_output() -> None:
    module = load_module()
    assert module.parse_tasks("depth,semantic") == ("depth", "semantic")


def test_parse_tasks_rejects_unknown_task() -> None:
    module = load_module()
    with pytest.raises(ValueError, match="unsupported"):
        module.parse_tasks("depth,unknown")


def test_checkpoint_metadata_requires_dino_aligned_size() -> None:
    module = load_module()
    with pytest.raises(ValueError, match="invalid DINO image size"):
        module.checkpoint_image_size({"image_size": 225}, "semantic")
