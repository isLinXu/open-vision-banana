"""Tests for decoded paper benchmark data and pixel-level aggregation."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from open_vision_banana.data import build_dataset
from open_vision_banana.evaluation.benchmark import (
    DepthAccumulator,
    NormalAccumulator,
    SemanticAccumulator,
)


def test_depth_accumulator_is_pixel_weighted() -> None:
    accumulator = DepthAccumulator()
    accumulator.update(torch.ones(1, 1, 1, 3), torch.ones(1, 1, 1, 3))
    accumulator.update(
        torch.tensor([[[[2.0]]]]),
        torch.tensor([[[[1.0]]]]),
    )
    metrics = accumulator.compute()
    assert metrics["valid_pixels"] == 4
    assert metrics["AbsRel"] == 0.25
    assert metrics["delta_1.25"] == 0.75


def test_normal_accumulator_reports_decoded_angles() -> None:
    gt = torch.tensor([[[[0.0]], [[0.0]], [[1.0]]]])
    pred = torch.tensor([[[[0.0]], [[1.0]], [[0.0]]]])
    accumulator = NormalAccumulator()
    accumulator.update(pred, gt)
    metrics = accumulator.compute()
    assert metrics["valid_pixels"] == 1
    assert metrics["mean_angle_deg"] == 90.0


def test_semantic_accumulator_uses_global_confusion() -> None:
    accumulator = SemanticAccumulator(num_classes=2)
    accumulator.update(torch.tensor([[[0, 1]]]), torch.tensor([[[0, 0]]]))
    accumulator.update(torch.tensor([[[1, 1]]]), torch.tensor([[[1, 1]]]))
    metrics = accumulator.compute()
    assert metrics["valid_pixels"] == 4
    assert metrics["mIoU"] == pytest.approx(7 / 12)


def test_cityscapes_label_ids_map_to_train_ids(tmp_path: Path) -> None:
    image_dir = tmp_path / "leftImg8bit" / "val" / "demo"
    label_dir = tmp_path / "gtFine" / "val" / "demo"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (2, 2), (10, 20, 30)).save(image_dir / "demo_000000_000000_leftImg8bit.png")
    labels = np.array([[7, 8], [33, 255]], dtype=np.uint8)
    Image.fromarray(labels).save(label_dir / "demo_000000_000000_gtFine_labelIds.png")

    dataset = build_dataset("semantic", str(tmp_path), split="val", resolution=2)
    item = dataset[0]
    assert item["gt_labels"].tolist() == [[0, 1], [18, 255]]
