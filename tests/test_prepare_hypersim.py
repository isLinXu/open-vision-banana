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

        depth, converted = _load_depth_meters(depth_path, ray_distance="never")
        assert not converted
        assert depth.shape == (1, 1, 1, 2)
        assert depth.flatten().tolist() == [1.5, 2.0]

    def test_hdf5_float16_depth_is_copied_to_float32(self, tmp_path: Path):
        h5py = pytest.importorskip("h5py")
        depth_path = tmp_path / "frame.0000.depth_meters.hdf5"
        with h5py.File(depth_path, "w") as handle:
            handle.create_dataset("dataset", data=np.array([[1.5, 2.0]], dtype=np.float16))

        depth, _ = _load_depth_meters(depth_path, ray_distance="never")
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


class TestHypersimRayDistanceToPlanarDepth:
    """Official depth_meters.hdf5 is Euclidean ray distance; the paper's task is planar z."""

    def test_focal_rule_matches_official_constant(self):
        from open_vision_banana.data.depth_geometry import hypersim_focal_px

        assert hypersim_focal_px(1024) == pytest.approx(886.81, abs=0.01)
        assert hypersim_focal_px(512) == pytest.approx(443.405, abs=0.01)

    def test_conversion_keeps_centre_and_shrinks_corners(self):
        from open_vision_banana.data.depth_geometry import ray_distance_to_planar_depth

        distance = torch.full((1, 1, 768, 1024), 10.0)
        distance[0, 0, 5, 5] = float("nan")
        distance[0, 0, 6, 6] = 0.0
        z = ray_distance_to_planar_depth(distance)
        assert z[0, 0, 384, 512] == pytest.approx(10.0, abs=1e-3)  # on-axis pixel unchanged
        corner_ratio = 10.0 / float(z[0, 0, 0, 0])
        assert corner_ratio == pytest.approx((511.5**2 + 383.5**2 + 886.81**2) ** 0.5 / 886.81, abs=1e-3)
        assert corner_ratio > 1.2  # the bias the unconverted pipeline baked in
        assert torch.isnan(z[0, 0, 5, 5]) and z[0, 0, 6, 6] == 0.0  # invalid markers survive
        # a flat wall at constant z is recovered exactly from its ray distances
        f = 886.81
        us = torch.arange(1024) - 511.5
        vs = torch.arange(768) - 383.5
        wall = 3.0 * torch.sqrt(vs.view(-1, 1) ** 2 + us.view(1, -1) ** 2 + f**2) / f
        back = ray_distance_to_planar_depth(wall.view(1, 1, 768, 1024), f)
        assert torch.allclose(back, torch.full_like(back, 3.0), atol=1e-4)

    def test_auto_mode_converts_only_official_hdf5(self, tmp_path: Path):
        h5py = pytest.importorskip("h5py")
        hdf5 = tmp_path / "frame.0000.depth_meters.hdf5"
        with h5py.File(hdf5, "w") as handle:
            handle.create_dataset("dataset", data=np.full((4, 8), 2.0, dtype=np.float32))
        depth, converted = _load_depth_meters(hdf5)
        assert converted and depth.shape == (1, 1, 4, 8)
        assert float(depth.max()) < 2.0 and float(depth.min()) > 1.5
        depth_never, converted_never = _load_depth_meters(hdf5, ray_distance="never")
        assert not converted_never and float(depth_never.min()) == 2.0

        png = tmp_path / "frame.0000.depth.png"
        Image.fromarray(np.full((4, 8), 2000, dtype=np.uint16)).save(png)
        depth_png, converted_png = _load_depth_meters(png)
        assert not converted_png and float(depth_png.min()) == 2.0  # mm export: planar, untouched
        depth_forced, converted_forced = _load_depth_meters(png, ray_distance="always", focal_px=10.0)
        assert converted_forced and float(depth_forced.max()) < 2.0
        with pytest.raises(ValueError, match="ray_distance"):
            _load_depth_meters(png, ray_distance="sometimes")

    def test_prepare_records_conversion_in_manifest_and_metric_depth(self, tmp_path: Path):
        h5py = pytest.importorskip("h5py")
        source = tmp_path / "raw" / "ai_001_001" / "images"
        (source / "scene_cam_00_final_preview").mkdir(parents=True)
        (source / "scene_cam_00_geometry_hdf5").mkdir()
        Image.new("RGB", (16, 8), (90, 90, 90)).save(source / "scene_cam_00_final_preview" / "frame.0000.color.jpg")
        with h5py.File(source / "scene_cam_00_geometry_hdf5" / "frame.0000.depth_meters.hdf5", "w") as handle:
            handle.create_dataset("dataset", data=np.full((8, 16), 4.0, dtype=np.float32))

        out = tmp_path / "prepared"
        assert prepare_hypersim(tmp_path / "raw", out) == 1
        metric = np.load(next((out / "metric_depth").glob("*.npy")))
        assert metric.shape == (8, 16)
        assert metric[4, 8] == pytest.approx(4.0, abs=0.02)  # near the optical axis
        assert metric[0, 0] < metric[4, 8]  # corners are closer in z than along the ray
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["metric_depth"]["convention"] == "planar_z_depth"
        assert manifest["depth_geometry"]["ray_distance_converted_pairs"] == 1
        assert manifest["depth_geometry"]["ray_distance_mode"] == "auto"
        # The encoded RGB target decodes back to the planar depth, not the ray distance
        from open_vision_banana.core.depth_mapper import DepthRGBMapper

        target = np.asarray(Image.open(next((out / "targets").glob("*.png"))).convert("RGB"), np.float32) / 255.0
        decoded = DepthRGBMapper().rgb_to_depth(torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0))[0, 0]
        assert float(decoded[0, 0]) < float(decoded[4, 8]) <= 4.05


def test_prepare_synthetic_depth_hypersim_path_finds_hdf5_and_converts(tmp_path: Path):
    """The multi-dataset preparer used to look for depth_meters.png only; it now shares the
    Hypersim scanner and the ray-distance conversion."""
    import argparse

    h5py = pytest.importorskip("h5py")
    from open_vision_banana.core.depth_mapper import DepthRGBMapper
    from scripts.prepare_synthetic_depth import prepare_hypersim as prepare_multi

    raw = tmp_path / "raw" / "ai_001_001" / "images"
    (raw / "scene_cam_00_final_preview").mkdir(parents=True)
    (raw / "scene_cam_00_geometry_hdf5").mkdir()
    Image.new("RGB", (64, 32)).save(raw / "scene_cam_00_final_preview" / "frame.0000.color.jpg")
    with h5py.File(raw / "scene_cam_00_geometry_hdf5" / "frame.0000.depth_meters.hdf5", "w") as handle:
        handle.create_dataset("dataset", data=np.full((32, 64), 4.0, dtype=np.float32))

    def decoded_corner_and_centre(mode: str) -> tuple[float, float]:
        out = tmp_path / f"out_{mode}"
        args = argparse.Namespace(resolution=64, max_samples=None, overwrite=False, ray_distance=mode, focal_px=None)
        assert prepare_multi(tmp_path / "raw", out, args) == 1
        rgb = np.asarray(Image.open(next((out / "targets").glob("*.png"))).convert("RGB"), np.float32) / 255.0
        dec = DepthRGBMapper().rgb_to_depth(torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0))[0, 0]
        return float(dec[0, 0]), float(dec[32, 32])

    corner, centre = decoded_corner_and_centre("auto")
    assert centre == pytest.approx(4.0, abs=0.05) and corner < 3.6
    corner_raw, centre_raw = decoded_corner_and_centre("never")
    assert corner_raw == pytest.approx(centre_raw, abs=0.05)
