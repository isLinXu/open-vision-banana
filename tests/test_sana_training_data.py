"""Regression tests for SANA v5 data routing."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _train_module():
    path = Path(__file__).parents[1] / "scripts" / "train_sana_512.py"
    spec = importlib.util.spec_from_file_location("train_sana_512_data_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mixed_flow_sources_keep_noise_for_generation():
    module = _train_module()
    source = torch.full((2, 2, 2, 2), 7.0)
    target = torch.zeros_like(source)
    torch.manual_seed(0)
    selected = module.select_sana_flow_sources(source, target, ["input", "noise"])

    assert torch.equal(selected[0], source[0])
    assert not torch.equal(selected[1], source[1])
    assert selected[1].std().item() > 0


def test_collate_adds_input_flow_default():
    module = _train_module()
    batch = module.flexible_collate(
        [
            {"image": torch.zeros(3, 2, 2), "target_rgb": torch.ones(3, 2, 2)},
            {"image": torch.zeros(3, 2, 2), "target_rgb": torch.ones(3, 2, 2)},
        ]
    )
    assert list(batch["flow_source"]) == ["input", "input"]
