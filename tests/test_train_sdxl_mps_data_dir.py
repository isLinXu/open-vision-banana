"""Tests for explicit dataset source selection in train_sdxl_mps.py."""

from __future__ import annotations

import importlib.util

import pytest
import torch
from _paths import ARCHIVE_SCRIPTS

SCRIPT_PATH = ARCHIVE_SCRIPTS / "train_sdxl_mps.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("train_sdxl_mps", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_data_dirs_preserves_legacy_defaults(monkeypatch, tmp_path):
    module = _load_script_module()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)

    assert module.resolve_data_dirs(None) == [
        tmp_path / "data" / "vb_zimage_train",
        tmp_path / "data" / "vb_snvc_train",
    ]


def test_resolve_data_dirs_explicit_source_is_exclusive(tmp_path):
    module = _load_script_module()
    selected = tmp_path / "vb_snvc_train"
    selected.mkdir()

    assert module.resolve_data_dirs(str(selected)) == [selected.resolve()]


def test_resolve_data_dirs_rejects_missing_directory(tmp_path):
    module = _load_script_module()

    with pytest.raises(NotADirectoryError, match="Training data directory not found"):
        module.resolve_data_dirs(str(tmp_path / "missing"))


class _LabelDataset(torch.utils.data.Dataset):
    def __init__(self, label: str, length: int = 20):
        self.label = label
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {"label": self.label, "index": index}


def test_paper_aligned_training_defaults():
    module = _load_script_module()
    assert module.PAPER_LORA_RANK == 64
    assert module.PAPER_LORA_ALPHA == 128
    assert pytest.approx(0.1) == module.PAPER_VISION_RATIO
    assert module.MODEL_ID_MAP["flux-schnell"] == "flux-schnell"


def test_flux_defaults_to_img2img_flow_sdxl_stays_off():
    module = _load_script_module()
    assert module.resolve_use_img2img_flow("flux-schnell", None) is True
    assert module.resolve_use_img2img_flow("ssd-1b", None) is False
    assert module.resolve_use_img2img_flow("sdxl", None) is False
    assert module.resolve_use_img2img_flow("ssd-1b", True) is True
    assert module.resolve_use_img2img_flow("flux-schnell", False) is False


def test_generation_is_outer_ninety_percent_branch():
    module = _load_script_module()
    combined, sampler = module.combine_training_datasets(
        [_LabelDataset("vision")],
        [_LabelDataset("generation")],
        balance_data=True,
        vision_ratio=0.1,
        seed=7,
    )

    assert sampler is None
    assert combined.vision_ratio == pytest.approx(0.1)
    labels = [combined[index]["label"] for index in range(1000)]
    vision_fraction = labels.count("vision") / len(labels)
    assert 0.07 < vision_fraction < 0.13
