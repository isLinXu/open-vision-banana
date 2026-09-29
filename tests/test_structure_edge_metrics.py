from __future__ import annotations

import sys

import cv2
import numpy as np
from _paths import ARCHIVE_SCRIPTS

sys.path.insert(0, str(ARCHIVE_SCRIPTS))
from build_sana_structure_binary_targets import build_binary_structure_map
from eval_sana_structure_checkpoints import edge_counts, edge_scores, select_evenly_spaced_indices
from infer_structure_binary import read_target_binary


def test_identical_edges_have_perfect_scores() -> None:
    target = np.zeros((9, 9), dtype=bool)
    target[4, 2:7] = True

    scores = edge_scores(edge_counts(target, target, tolerance_px=0.0))

    assert scores == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_tolerance_matches_small_spatial_offset() -> None:
    target = np.zeros((9, 9), dtype=bool)
    prediction = np.zeros((9, 9), dtype=bool)
    target[4, 2:7] = True
    prediction[6, 2:7] = True

    assert edge_scores(edge_counts(prediction, target, tolerance_px=1.0))["f1"] == 0.0
    assert edge_scores(edge_counts(prediction, target, tolerance_px=2.0))["f1"] == 1.0


def test_extra_edges_reduce_precision() -> None:
    target = np.zeros((9, 9), dtype=bool)
    prediction = np.zeros((9, 9), dtype=bool)
    target[4, 4] = True
    prediction[4, 4] = True
    prediction[0, 0] = True

    scores = edge_scores(edge_counts(prediction, target, tolerance_px=0.0))

    assert scores["precision"] == 0.5
    assert scores["recall"] == 1.0
    assert scores["f1"] == 2.0 / 3.0


def test_evenly_spaced_indices_cover_bounds() -> None:
    assert select_evenly_spaced_indices(10, 4) == [0, 3, 6, 9]
    assert select_evenly_spaced_indices(3, 10) == [0, 1, 2]


def test_binary_target_builder_accepts_opencv_lsd_layouts(tmp_path) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.line(image, (4, 4), (59, 59), (255, 255, 255), 2)
    source = tmp_path / "input.png"
    assert cv2.imwrite(str(source), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    target = build_binary_structure_map(source, 64, 0.08, 3)

    assert target.shape == (64, 64, 3)
    assert target.dtype == np.uint8
    assert set(np.unique(target).tolist()) <= {0, 255}


def test_structure_target_reader_uses_binary_luminance(tmp_path) -> None:
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    image[1, 2] = (255, 255, 255)
    source = tmp_path / "target.png"
    assert cv2.imwrite(str(source), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    result = read_target_binary(source, 4)

    assert result.shape == (4, 4)
    assert result.dtype == bool
    assert result.any()
