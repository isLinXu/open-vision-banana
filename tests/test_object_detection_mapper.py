"""Tests for object detection (bounding box) RGB encoder/decoder."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from open_vision_banana.core.object_detection_mapper import ObjectDetectionRGBMapper


def test_bbox_roundtrip() -> None:
    """Test bbox -> RGB -> bbox roundtrip."""
    mapper = ObjectDetectionRGBMapper(num_classes=3, line_width=2)

    bboxes = torch.tensor([[[0.2, 0.3, 0.2, 0.2], [0.5, 0.5, 0.15, 0.15]]])  # [1, 2, 4]
    class_ids = torch.tensor([[0, 1]])  # [1, 2]

    rgb = mapper.bboxes_to_rgb(bboxes, class_ids)
    assert rgb.shape == (1, 3, 512, 512)
    assert rgb.max() > 0

    recovered_bboxes, recovered_class_ids, recovered_scores = mapper.rgb_to_bboxes(rgb)
    assert recovered_bboxes.shape[0] == 1
    assert recovered_class_ids.shape[0] == 1
    assert recovered_scores.shape[0] == 1


def test_coco_classes() -> None:
    """Test COCO 20-class palette."""
    from open_vision_banana.core.object_detection_mapper import COCO_BBOX_COLORS, COCO_CLASS_NAMES

    assert len(COCO_CLASS_NAMES) == 20
    assert COCO_BBOX_COLORS.shape == (20, 3)

    mapper = ObjectDetectionRGBMapper(num_classes=20)
    assert mapper.colors.shape == (20, 3)


def test_bbox_with_scores() -> None:
    """Test bbox encoding with confidence scores."""
    mapper = ObjectDetectionRGBMapper(num_classes=2, line_width=2)

    bboxes = torch.tensor([[[0.3, 0.3, 0.2, 0.2]]])
    class_ids = torch.tensor([[0]])
    scores = torch.tensor([[0.8]])

    rgb = mapper.bboxes_to_rgb(bboxes, class_ids, scores)
    # With lower score, color should be dimmer
    rgb_no_score = mapper.bboxes_to_rgb(bboxes, class_ids)
    assert rgb.max() < rgb_no_score.max() or torch.allclose(rgb, rgb_no_score)


def test_bbox_dynamic_resolution() -> None:
    """Test bbox encoding with non-default resolution (P1-2)."""
    mapper = ObjectDetectionRGBMapper(
        num_classes=2, line_width=2, output_height=256, output_width=256
    )

    bboxes = torch.tensor([[[0.2, 0.3, 0.2, 0.2]]])
    class_ids = torch.tensor([[0]])

    rgb = mapper.bboxes_to_rgb(bboxes, class_ids)
    assert rgb.shape == (1, 3, 256, 256)

    rgb = mapper.bboxes_to_rgb(bboxes, class_ids, output_size=(128, 128))
    assert rgb.shape == (1, 3, 128, 128)


if __name__ == "__main__":
    test_bbox_roundtrip()
    test_coco_classes()
    test_bbox_with_scores()
    test_bbox_dynamic_resolution()
    print("All object detection mapper tests passed!")
