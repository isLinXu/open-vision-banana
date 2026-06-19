"""Tests for Hypersim data preparation script."""

from pathlib import Path

import numpy as np
from PIL import Image

from scripts.prepare_hypersim import find_hypersim_pairs, prepare_hypersim


def _write_pair(tmp: Path, idx: int) -> None:
    color = tmp / f"scene_{idx}_color.png"
    depth = tmp / f"scene_{idx}_depth.png"
    Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(color)
    Image.fromarray(np.random.randint(100, 5000, (32, 32), dtype=np.uint16)).save(depth)


class TestPrepareHypersim:
    def test_find_pairs(self, tmp_path: Path):
        _write_pair(tmp_path, 0)
        _write_pair(tmp_path, 1)
        pairs = find_hypersim_pairs(tmp_path)
        assert len(pairs) == 2

    def test_prepare_writes_layout(self, tmp_path: Path):
        source = tmp_path / "raw"
        source.mkdir()
        _write_pair(source, 0)
        out = tmp_path / "prepared"
        count = prepare_hypersim(source, out, resolution=16)
        assert count == 1
        assert (out / "images").is_dir()
        assert (out / "targets").is_dir()
        assert len(list((out / "images").glob("*.png"))) == 1
        assert len(list((out / "targets").glob("*.png"))) == 1
