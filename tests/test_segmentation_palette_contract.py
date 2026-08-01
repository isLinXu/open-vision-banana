"""Tests for dataset-bound semantic palette loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from open_vision_banana.inference.decoder import (
    VisualizationDecoder,
    load_segmentation_color_map,
)


def test_v3_unified_palette_loads_from_dataset_root(tmp_path: Path):
    palette_dir = tmp_path / "segmentation"
    palette_dir.mkdir()
    (palette_dir / "_unified_palette.json").write_text(
        json.dumps({"palette": [[0, 0, 0], [12, 34, 56]], "palette_size": 2}),
        encoding="utf-8",
    )

    color_map = load_segmentation_color_map(tmp_path)
    decoder = VisualizationDecoder(semantic_palette_path=tmp_path)
    result = decoder.decode_semantic(torch.zeros(1, 3, 2, 2))

    assert color_map == {"palette_0": (0, 0, 0), "palette_1": (12, 34, 56)}
    assert result["class_names"] == ["palette_0", "palette_1"]


def test_palette_info_resolves_sibling_unified_palette(tmp_path: Path):
    (tmp_path / "palette_info.json").write_text(
        json.dumps({"palette": "unified_palette", "palette_size": 2}),
        encoding="utf-8",
    )
    (tmp_path / "_unified_palette.json").write_text(
        json.dumps({"palette": [[1, 2, 3], [4, 5, 6]]}),
        encoding="utf-8",
    )

    assert load_segmentation_color_map(tmp_path / "palette_info.json") == {
        "palette_0": (1, 2, 3),
        "palette_1": (4, 5, 6),
    }


def test_legacy_palette_info_loads_embedded_palette_mapping(tmp_path: Path):
    palette_info = tmp_path / "palette_info.json"
    palette_info.write_text(
        json.dumps(
            {
                "palette": "ADE20K_3",
                "num_classes": 3,
                "palette_mapping": {
                    "0": [0, 0, 0],
                    "1": [12, 34, 56],
                    "2": [78, 90, 123],
                },
            }
        ),
        encoding="utf-8",
    )

    assert load_segmentation_color_map(palette_info) == {
        "class_0": (0, 0, 0),
        "class_1": (12, 34, 56),
        "class_2": (78, 90, 123),
    }


def test_palette_loader_preserves_duplicate_legacy_colors_and_rejects_invalid(
    tmp_path: Path,
):
    palette = tmp_path / "palette.json"
    palette.write_text(json.dumps({"palette": [[0, 0, 0], [0, 0, 0]]}), encoding="utf-8")
    assert load_segmentation_color_map(palette) == {
        "palette_0": (0, 0, 0),
        "palette_1": (0, 0, 0),
    }

    palette.write_text(json.dumps({"palette": [[0, 0, 256]]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid"):
        load_segmentation_color_map(palette)
