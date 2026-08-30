"""Tests for Hypersim data preparation script."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from open_vision_banana.data.datasets import HypersimDepthDataset
from scripts.prepare_hypersim import (
    _load_depth_meters,
    find_hypersim_pairs,
    prepare_hypersim,
    write_metric_depth_pairs_jsonl,
)


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
        assert (out / "metric_depth").is_dir()
        assert len(list((out / "images").glob("*.png"))) == 1
        assert len(list((out / "targets").glob("*.png"))) == 1
        metric_path = next((out / "metric_depth").glob("*.npy"))
        metric_depth = np.load(metric_path)
        assert metric_depth.dtype == np.float32
        assert metric_depth.shape == (16, 16)
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["metric_depth"]["invalid_value"] == "NaN"
        assert manifest["data_contract"]["valid_pixels"] == 16 * 16

    def test_prepare_rebuilds_legacy_output_without_metric_depth(self, tmp_path: Path):
        source = tmp_path / "raw"
        source.mkdir()
        _write_pair(source, 0)
        out = tmp_path / "prepared"
        (out / "images").mkdir(parents=True)
        (out / "targets").mkdir()
        Image.new("RGB", (32, 32)).save(out / "images" / "000000_scene_0.png")
        Image.new("RGB", (32, 32)).save(out / "targets" / "000000_scene_0.png")

        assert prepare_hypersim(source, out) == 1
        assert len(list((out / "metric_depth").glob("*.npy"))) == 1

    def test_prepared_metric_depth_is_loaded_as_ground_truth(self, tmp_path: Path):
        source = tmp_path / "raw"
        source.mkdir()
        _write_pair(source, 0)
        out = tmp_path / "prepared"
        assert prepare_hypersim(source, out, resolution=16) == 1

        dataset = HypersimDepthDataset(out, resolution=16)
        sample = dataset[0]
        assert sample["gt_depth"].shape == (1, 16, 16)
        assert sample["valid_mask"].shape == (16, 16)
        assert sample["valid_mask"].all()
        assert sample["target_rgb"].shape == (3, 16, 16)

    def test_invalid_prepared_metric_depth_shape_is_rejected(self, tmp_path: Path):
        root = tmp_path / "prepared"
        (root / "images").mkdir(parents=True)
        (root / "targets").mkdir()
        (root / "metric_depth").mkdir()
        Image.new("RGB", (2, 2)).save(root / "images" / "000000_frame.png")
        Image.new("RGB", (2, 2)).save(root / "targets" / "000000_frame.png")
        np.save(root / "metric_depth" / "000000_frame.npy", np.zeros((1, 2, 2), dtype=np.float32))

        dataset = HypersimDepthDataset(root, resolution=2)
        with pytest.raises(ValueError, match="must be 2-D"):
            _ = dataset[0]

    def test_hdf5_depth_is_loaded_as_metric_meters(self, tmp_path: Path):
        h5py = pytest.importorskip("h5py")
        depth_path = tmp_path / "frame.0000.depth_meters.hdf5"
        with h5py.File(depth_path, "w") as handle:
            handle.create_dataset("dataset", data=np.array([[1.5, 2.0]], dtype=np.float32))

        depth = _load_depth_meters(depth_path)
        assert depth.shape == (1, 1, 1, 2)
        assert depth.flatten().tolist() == [1.5, 2.0]

    def test_hdf5_float16_depth_is_copied_to_float32(self, tmp_path: Path):
        h5py = pytest.importorskip("h5py")
        depth_path = tmp_path / "frame.0000.depth_meters.hdf5"
        with h5py.File(depth_path, "w") as handle:
            handle.create_dataset("dataset", data=np.array([[1.5, 2.0]], dtype=np.float16))

        depth = _load_depth_meters(depth_path)
        assert depth.shape == (1, 1, 1, 2)
        assert depth.dtype == torch.float32
        assert depth.flatten().tolist() == pytest.approx([1.5, 2.0], abs=1e-3)

    def test_official_hypersim_hdf5_name_is_paired(self, tmp_path: Path):
        color_path = tmp_path / "frame.0000.color.jpg"
        Image.new("RGB", (2, 2)).save(color_path)
        depth_path = tmp_path / "frame.0000.depth_meters.hdf5"
        depth_path.touch()

        assert find_hypersim_pairs(tmp_path) == [(color_path, depth_path)]

    def test_official_preview_and_geometry_dirs_are_paired(self, tmp_path: Path):
        preview = tmp_path / "ai_001_001" / "images" / "scene_cam_00_final_preview"
        geometry = tmp_path / "ai_001_001" / "images" / "scene_cam_00_geometry_hdf5"
        preview.mkdir(parents=True)
        geometry.mkdir(parents=True)
        color_path = preview / "frame.0000.color.jpg"
        Image.new("RGB", (2, 2)).save(color_path)
        depth_path = geometry / "frame.0000.depth_meters.hdf5"
        depth_path.touch()

        assert find_hypersim_pairs(tmp_path) == [(color_path, depth_path)]

    def test_geometry_preview_png_is_ignored_when_hdf5_exists(self, tmp_path: Path):
        preview = tmp_path / "images" / "scene_cam_00_final_preview"
        geometry = tmp_path / "images" / "scene_cam_00_geometry_hdf5"
        geometry_preview = tmp_path / "images" / "scene_cam_00_geometry_preview"
        preview.mkdir(parents=True)
        geometry.mkdir(parents=True)
        geometry_preview.mkdir(parents=True)
        color_path = preview / "frame.0000.color.jpg"
        Image.new("RGB", (2, 2)).save(color_path)
        Image.new("RGB", (2, 2)).save(geometry_preview / "frame.0000.color.jpg")
        Image.new("RGBA", (2, 2)).save(geometry_preview / "frame.0000.depth_meters.png")
        depth_path = geometry / "frame.0000.depth_meters.hdf5"
        depth_path.touch()

        assert find_hypersim_pairs(tmp_path) == [(color_path, depth_path)]

    def test_exclude_scenes_skips_named_directories(self, tmp_path: Path):
        keep = tmp_path / "ai_001_003"
        skip = tmp_path / "ai_001_001"
        keep.mkdir()
        skip.mkdir()
        _write_pair(keep, 0)
        _write_pair(skip, 1)
        pairs = find_hypersim_pairs(tmp_path, exclude_scenes={"ai_001_001"})
        assert [color.parent.name for color, _depth in pairs] == ["ai_001_003"]

    def test_write_metric_depth_pairs_jsonl_uses_eval_prompt(self, tmp_path: Path):
        source = tmp_path / "raw"
        source.mkdir()
        _write_pair(source, 0)
        out = tmp_path / "prepared"
        assert prepare_hypersim(source, out, resolution=8) == 1
        assert write_metric_depth_pairs_jsonl(out) == 1
        row = json.loads((out / "pairs.jsonl").read_text().splitlines()[0])
        assert row["task"] == "metric_depth"
        assert row["prompt"].startswith("Generate a metric depth visualization")

    def test_start_index_offsets_output_names(self, tmp_path: Path):
        source = tmp_path / "raw"
        source.mkdir()
        Image.fromarray(np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)).save(
            source / "scene_0_color.png"
        )
        Image.fromarray(np.random.randint(100, 5000, (8, 8), dtype=np.uint16)).save(
            source / "scene_0_depth.png"
        )
        out = tmp_path / "prepared"
        assert prepare_hypersim(source, out, resolution=8, start_index=5) == 1
        assert (out / "images" / "000005_scene_0.png").is_file()
        assert (out / "metric_depth" / "000005_scene_0.npy").is_file()
