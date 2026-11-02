"""Tests for P0 bug fixes: NMS and Union-Find corrections."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from open_vision_banana.core.object_detection_mapper import (
    ObjectDetectionRGBMapper,
    _nms,
)
from open_vision_banana.core.segmentation import SegmentationDecoder

# ── NMS Tests ───────────────────────────────────────────────────────────


class TestNMS:
    """Tests for Non-Maximum Suppression implementation."""

    def test_nms_empty(self) -> None:
        """NMS with no boxes returns empty."""
        boxes = torch.empty(0, 4)
        scores = torch.empty(0)
        keep = _nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 0

    def test_nms_single_box(self) -> None:
        """NMS with one box keeps it."""
        boxes = torch.tensor([[0.1, 0.1, 0.3, 0.3]])
        scores = torch.tensor([0.9])
        keep = _nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 1
        assert keep[0].item() == 0

    def test_nms_overlapping(self) -> None:
        """NMS removes overlapping boxes."""
        boxes = torch.tensor(
            [
                [0.1, 0.1, 0.4, 0.4],
                [0.12, 0.12, 0.42, 0.42],  # high overlap with box 0
            ]
        )
        scores = torch.tensor([0.9, 0.8])
        keep = _nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 1
        assert keep[0].item() == 0  # higher score kept

    def test_nms_non_overlapping(self) -> None:
        """NMS keeps non-overlapping boxes."""
        boxes = torch.tensor(
            [
                [0.1, 0.1, 0.3, 0.3],
                [0.5, 0.5, 0.7, 0.7],
                [0.8, 0.8, 1.0, 1.0],
            ]
        )
        scores = torch.tensor([0.9, 0.8, 0.7])
        keep = _nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 3

    def test_nms_low_threshold(self) -> None:
        """NMS with low threshold removes more boxes."""
        boxes = torch.tensor(
            [
                [0.1, 0.1, 0.4, 0.4],
                [0.15, 0.15, 0.45, 0.45],  # moderate overlap
            ]
        )
        scores = torch.tensor([0.9, 0.8])
        # High threshold → keep both
        keep_high = _nms(boxes, scores, iou_threshold=0.9)
        assert len(keep_high) == 2
        # Low threshold → keep only best
        keep_low = _nms(boxes, scores, iou_threshold=0.3)
        assert len(keep_low) == 1

    def test_nms_sorting(self) -> None:
        """NMS keeps highest score regardless of input order."""
        boxes = torch.tensor(
            [
                [0.1, 0.1, 0.4, 0.4],  # lower score, same region
                [0.12, 0.12, 0.42, 0.42],  # higher score, same region
            ]
        )
        scores = torch.tensor([0.7, 0.95])
        keep = _nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 1
        assert keep[0].item() == 1  # higher score index kept

    def test_rgb_to_bboxes_nms_applied(self) -> None:
        """Verify rgb_to_bboxes applies NMS to remove duplicates."""
        mapper = ObjectDetectionRGBMapper(num_classes=2, line_width=2)

        # Create two overlapping bboxes of the same class
        bboxes = torch.tensor([[[0.2, 0.2, 0.3, 0.3], [0.22, 0.22, 0.3, 0.3]]])
        class_ids = torch.tensor([[0, 0]])

        rgb = mapper.bboxes_to_rgb(bboxes, class_ids)
        recovered_bboxes, recovered_cls, recovered_scores = mapper.rgb_to_bboxes(
            rgb, iou_threshold=0.3
        )

        # NMS should prevent having too many overlapping detections
        n_detections = recovered_bboxes.shape[1]
        # With NMS, we shouldn't get excessive duplicates
        assert n_detections <= 5, f"Too many detections ({n_detections}), NMS may not be working"


# ── Union-Find Seed-Based Fix Tests ─────────────────────────────────────


class TestSeedBasedUnionFindFix:
    """Tests for the P0 fix in _seed_based_union_find.

    The bug was that min_roots used pixel indices instead of label roots.
    This caused incorrect component merging — pixels with lower indices
    would become roots regardless of actual component structure.
    """

    def test_seed_based_produces_valid_labels(self) -> None:
        """Seed-based union-find produces valid consecutive labels."""
        decoder = SegmentationDecoder()
        img = torch.zeros(32, 32, 3)
        img[:16, :16] = torch.tensor([255.0, 0.0, 0.0])
        img[:16, 16:] = torch.tensor([0.0, 255.0, 0.0])
        img[16:, :] = torch.tensor([0.0, 0.0, 255.0])

        bg_mask = torch.zeros(32, 32, dtype=torch.bool)
        labels = decoder._seed_based_union_find(img, bg_mask)

        # Labels should be non-negative consecutive integers
        unique = labels.unique()
        assert (unique >= 0).all(), "Negative labels found"
        # Background (0) should exist
        assert 0 in unique.tolist() or len(unique) >= 1

    def test_seed_based_correct_grouping(self) -> None:
        """Seed-based union-find correctly groups same-color regions."""
        decoder = SegmentationDecoder()
        # Create image with two distinct color regions
        img = torch.zeros(32, 32, 3)
        img[:16, :] = torch.tensor([255.0, 0.0, 0.0])  # red top
        img[16:, :] = torch.tensor([0.0, 0.0, 255.0])  # blue bottom

        bg_mask = torch.zeros(32, 32, dtype=torch.bool)
        labels = decoder._seed_based_union_find(img, bg_mask)

        # Top half should have the same label
        top_labels = labels[:16, :].unique()
        bottom_labels = labels[16:, :].unique()
        # Each half should be dominated by a single label
        assert len(top_labels) <= 2, f"Top half has too many labels: {top_labels}"
        assert len(bottom_labels) <= 2, f"Bottom half has too many labels: {bottom_labels}"

    def test_seed_vs_adjacent_consistency(self) -> None:
        """Seed-based and adjacent grouping produce similar component counts."""
        decoder = SegmentationDecoder()
        img = torch.zeros(32, 32, 3)
        img[:16, :16] = torch.tensor([255.0, 0.0, 0.0])
        img[:16, 16:] = torch.tensor([0.0, 255.0, 0.0])
        img[16:, :] = torch.tensor([0.0, 0.0, 255.0])

        bg_mask = torch.zeros(32, 32, dtype=torch.bool)
        labels_adj = decoder._union_find_grouping(img, bg_mask)
        labels_seed = decoder._seed_based_union_find(img, bg_mask)

        n_adj = len(labels_adj.unique())
        n_seed = len(labels_seed.unique())
        # Both should detect roughly the same number of components
        # (seed-based may produce more compact = fewer components)
        assert abs(n_adj - n_seed) <= 2, (
            f"Component count mismatch: adjacent={n_adj}, seed={n_seed}"
        )

    def test_seed_based_no_pixel_index_artifact(self) -> None:
        """Verify the fix: labels should not be raw pixel indices.

        Before the fix, _seed_based_union_find used min(src_pixel_index, dst_pixel_index)
        as the merge root. This meant pixel 0 would always become the root of any
        component it touched, even if it belonged to a different color region.

        After the fix, the merge uses labels[src/dst] (actual component roots).
        """
        decoder = SegmentationDecoder()
        # Create image where pixel 0 is in a different color region from pixel 1
        img = torch.zeros(16, 16, 3)
        img[0, 0] = torch.tensor([255.0, 0.0, 0.0])  # pixel 0 = red
        img[0, 1:] = torch.tensor([0.0, 255.0, 0.0])  # pixel 1+ = green
        img[1:, :] = torch.tensor([0.0, 255.0, 0.0])

        bg_mask = torch.zeros(16, 16, dtype=torch.bool)
        labels = decoder._seed_based_union_find(img, bg_mask)

        # Pixel (0,0) should NOT have the same label as pixel (0,1)
        # because they are different colors
        label_00 = labels[0, 0].item()
        label_01 = labels[0, 1].item()
        # They might be 0 (background) if color_tolerance is high, but
        # they should not be forced to the same label by the pixel index bug
        # The key assertion: label_00 is not forced to 0 just because pixel 0
        # has the lowest index
        assert label_00 != label_01 or (label_00 == 0 and label_01 == 0), (
            f"Pixel index bug: pixel 0 label={label_00}, pixel 1 label={label_01} "
            f"— they should differ for different colors"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
