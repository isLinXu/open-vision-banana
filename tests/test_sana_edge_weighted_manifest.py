"""Tests for reproducible high-depth-edge manifest construction."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch
from _paths import ARCHIVE_SCRIPTS
from PIL import Image

from open_vision_banana.core.depth_mapper import DepthRGBMapper


def _builder_module():
    path = ARCHIVE_SCRIPTS / "build_sana_depth_edge_weighted_manifest.py"
    spec = importlib.util.spec_from_file_location("edge_weighted_manifest_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_turbo_depth(path: Path, values: torch.Tensor) -> None:
    mapper = DepthRGBMapper(colormap="turbo")
    rgb = mapper.normalized_to_rgb(values.unsqueeze(0).unsqueeze(0)).squeeze(0)
    Image.fromarray((rgb.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")).save(path)


def test_builder_duplicates_only_top_edge_train_row_and_preserves_val(tmp_path: Path) -> None:
    module = _builder_module()
    flat = tmp_path / "flat.png"
    edged = tmp_path / "edged.png"
    val = tmp_path / "val.png"
    _write_turbo_depth(flat, torch.full((16, 16), 0.4))
    _write_turbo_depth(edged, torch.cat((torch.zeros(16, 8), torch.ones(16, 8)), dim=1))
    _write_turbo_depth(val, torch.linspace(0.0, 1.0, 16).repeat(16, 1))

    source = tmp_path / "pairs.jsonl"
    rows = [
        {"task": "depth", "split": "train", "input": str(flat), "target": str(flat)},
        {"task": "depth", "split": "train", "input": str(edged), "target": str(edged)},
        {"task": "depth", "split": "val", "input": str(val), "target": str(val)},
        {"task": "normal", "split": "train", "input": str(flat), "target": str(flat)},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "weighted.jsonl"
    stats = tmp_path / "stats.json"

    summary = module.build_weighted_manifest(
        source,
        output,
        stats,
        edge_fraction=0.5,
        high_edge_repeat=2,
        resolution=16,
        edge_threshold=0.02,
    )

    weighted = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(weighted) == 4
    assert [row["split"] for row in weighted] == ["train", "train", "val", "train"]
    assert weighted[-1]["input"] == str(edged)
    assert weighted[-1]["sampling"]["copy_index"] == 2
    assert summary["reweighting"]["effective_train_rows"] == 3
    assert json.loads(stats.read_text(encoding="utf-8"))["base_split_counts"] == {"train": 2, "val": 1}
