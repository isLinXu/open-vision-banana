"""Tests for data/datasets.py — dataset loaders and utility functions."""

import json
import pickle

import numpy as np
import pytest
import torch
from PIL import Image

from open_vision_banana.data.datasets import (
    CityscapesSegDataset,
    HypersimDepthDataset,
    RefCOCOgDataset,
    SACoInstanceDataset,
    ScanNetNormalDataset,
    _fill_polygon,
    _pil_to_tensor,
    build_dataset,
)

# ──────────────────────────────────────────────────────────────────
# Utility function tests
# ──────────────────────────────────────────────────────────────────


class TestPilToTensor:
    """Test _pil_to_tensor helper."""

    def test_rgb_image(self):
        """RGB image should convert to [3, H, W] tensor."""
        img = Image.new("RGB", (8, 4), color=(255, 128, 0))
        t = _pil_to_tensor(img)
        assert t.shape == (3, 4, 8)
        assert t.dtype == torch.uint8

    def test_grayscale_image(self):
        """Grayscale image should convert to [1, H, W] tensor."""
        img = Image.new("L", (8, 4), color=128)
        t = _pil_to_tensor(img)
        assert t.shape == (1, 4, 8)

    def test_values(self):
        """Pixel values should match the PIL image."""
        img = Image.new("RGB", (1, 1), color=(100, 200, 50))
        t = _pil_to_tensor(img)
        assert t[0, 0, 0].item() == 100
        assert t[1, 0, 0].item() == 200
        assert t[2, 0, 0].item() == 50


class TestFillPolygon:
    """Test _fill_polygon scanline fill."""

    def test_triangle(self):
        """A triangle should fill the expected area."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        pts = np.array([[2, 2], [8, 2], [5, 8]], dtype=np.float32)
        _fill_polygon(mask, pts)
        assert mask.sum() > 0
        # Center should be filled
        assert mask[4, 5] == 1

    def test_too_few_points(self):
        """Fewer than 3 points should be a no-op."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        pts = np.array([[1, 1], [5, 5]], dtype=np.float32)
        _fill_polygon(mask, pts)
        assert mask.sum() == 0

    def test_rectangle(self):
        """A rectangle should fill exactly the right area."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        pts = np.array([[2, 2], [8, 2], [8, 8], [2, 8]], dtype=np.float32)
        _fill_polygon(mask, pts)
        # Rectangle from (2,2) to (8,8) — interior should be filled
        assert mask[5, 5] == 1
        # Outside should be empty
        assert mask[0, 0] == 0


# ──────────────────────────────────────────────────────────────────
# build_dataset factory tests
# ──────────────────────────────────────────────────────────────────


class TestBuildDataset:
    """Test the build_dataset factory function."""

    def test_unknown_task_raises(self, tmp_path):
        """Unknown task should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown task"):
            build_dataset("unknown", str(tmp_path))

    def test_creates_depth_dataset(self, tmp_path):
        """build_dataset should create a HypersimDepthDataset for 'depth'."""
        ds = build_dataset("depth", str(tmp_path))
        assert isinstance(ds, HypersimDepthDataset)
        assert ds.TASK_TYPE == "depth"

    def test_creates_normal_dataset(self, tmp_path):
        """build_dataset should create a ScanNetNormalDataset for 'normal'."""
        ds = build_dataset("normal", str(tmp_path))
        assert isinstance(ds, ScanNetNormalDataset)

    def test_creates_semantic_dataset(self, tmp_path):
        """build_dataset should create a CityscapesSegDataset for 'semantic'."""
        ds = build_dataset("semantic", str(tmp_path))
        assert isinstance(ds, CityscapesSegDataset)

    def test_creates_instance_dataset(self, tmp_path):
        """build_dataset should create a SACoInstanceDataset for 'instance'."""
        ds = build_dataset("instance", str(tmp_path))
        assert isinstance(ds, SACoInstanceDataset)

    def test_creates_referring_dataset(self, tmp_path):
        """build_dataset should create a RefCOCOgDataset for 'referring'."""
        ds = build_dataset("referring", str(tmp_path))
        assert isinstance(ds, RefCOCOgDataset)

    def test_passes_resolution(self, tmp_path):
        """Resolution should be passed through."""
        ds = build_dataset("depth", str(tmp_path), resolution=256)
        assert ds.resolution == 256


# ──────────────────────────────────────────────────────────────────
# BaseVisionDataset tests
# ──────────────────────────────────────────────────────────────────


class TestBaseVisionDataset:
    """Test BaseVisionDataset base class."""

    def test_init_nonexistent_dir(self, tmp_path):
        """Non-existent directory should not raise, just log warning."""
        ds = HypersimDepthDataset(root_dir=str(tmp_path / "nonexistent"))
        assert len(ds) == 0

    def test_task_mix_weight(self, tmp_path):
        """task_mix_weight should be stored."""
        ds = HypersimDepthDataset(root_dir=str(tmp_path), task_mix_weight=0.5)
        assert ds.task_mix_weight == 0.5

    def test_load_image(self, tmp_path):
        """_load_image should return tensor in [-1, 1]."""
        # Create a test image
        img = Image.new("RGB", (512, 512), color=(128, 64, 200))
        img_path = tmp_path / "test.png"
        img.save(img_path)

        ds = HypersimDepthDataset(root_dir=str(tmp_path))
        tensor = ds._load_image(img_path)
        assert tensor.shape == (3, 512, 512)
        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0

    def test_load_image_resizes(self, tmp_path):
        """_load_image should resize to resolution."""
        img = Image.new("RGB", (256, 256), color=(128, 64, 200))
        img_path = tmp_path / "test.png"
        img.save(img_path)

        ds = HypersimDepthDataset(root_dir=str(tmp_path), resolution=128)
        tensor = ds._load_image(img_path)
        assert tensor.shape == (3, 128, 128)


# ──────────────────────────────────────────────────────────────────
# HypersimDepthDataset tests
# ──────────────────────────────────────────────────────────────────


class TestHypersimDepthDataset:
    """Test HypersimDepthDataset."""

    def test_scan_finds_pairs(self, tmp_path):
        """_scan should find color/depth pairs."""
        # Create mock data
        for i in range(3):
            Image.new("RGB", (512, 512)).save(tmp_path / f"scene_{i}_color.png")
            Image.new("L", (512, 512)).save(tmp_path / f"scene_{i}_depth.png")

        ds = HypersimDepthDataset(root_dir=str(tmp_path))
        assert len(ds) == 3

    def test_getitem(self, tmp_path):
        """__getitem__ should return dict with prompt, image, target_rgb."""
        Image.new("RGB", (512, 512), color=(100, 100, 100)).save(tmp_path / "scene_0_color.png")

        # Create a depth image with some variation
        depth_arr = (np.ones((512, 512), dtype=np.float32) * 5.0 * 1000).astype(np.uint16)
        Image.fromarray(depth_arr).save(tmp_path / "scene_0_depth.png")

        ds = HypersimDepthDataset(root_dir=str(tmp_path), resolution=64)
        item = ds[0]
        assert "prompt" in item
        assert "image" in item
        assert "target_rgb" in item
        assert item["image"].shape == (3, 64, 64)
        assert item["target_rgb"].shape == (3, 64, 64)
        assert item["target_rgb"].min() >= 0.0
        assert item["target_rgb"].max() <= 1.0


# ──────────────────────────────────────────────────────────────────
# ScanNetNormalDataset tests
# ──────────────────────────────────────────────────────────────────


class TestScanNetNormalDataset:
    """Test ScanNetNormalDataset."""

    def test_scan_finds_pairs(self, tmp_path):
        """_scan should find color/normal pairs."""
        for i in range(2):
            Image.new("RGB", (512, 512)).save(tmp_path / f"scene_{i}_color.png")
            Image.new("RGB", (512, 512), color=(128, 128, 128)).save(
                tmp_path / f"scene_{i}_normal.png"
            )

        ds = ScanNetNormalDataset(root_dir=str(tmp_path))
        assert len(ds) == 2

    def test_getitem(self, tmp_path):
        """__getitem__ should return normal-encoded target_rgb."""
        Image.new("RGB", (64, 64)).save(tmp_path / "scene_0_color.png")
        Image.new("RGB", (64, 64), color=(128, 128, 128)).save(tmp_path / "scene_0_normal.png")

        ds = ScanNetNormalDataset(root_dir=str(tmp_path), resolution=64)
        item = ds[0]
        assert item["image"].shape == (3, 64, 64)
        assert item["target_rgb"].shape == (3, 64, 64)


# ──────────────────────────────────────────────────────────────────
# CityscapesSegDataset tests
# ──────────────────────────────────────────────────────────────────


class TestCityscapesSegDataset:
    """Test CityscapesSegDataset."""

    def test_scan_finds_pairs(self, tmp_path):
        """_scan should find image/label pairs in Cityscapes layout."""
        img_dir = tmp_path / "leftImg8bit" / "train" / "cityA"
        img_dir.mkdir(parents=True)
        gt_dir = tmp_path / "gtFine" / "train" / "cityA"
        gt_dir.mkdir(parents=True)

        for i in range(2):
            Image.new("RGB", (64, 64)).save(img_dir / f"image_{i}_leftImg8bit.png")
            Image.new("L", (64, 64)).save(gt_dir / f"image_{i}_gtFine_labelIds.png")

        ds = CityscapesSegDataset(root_dir=str(tmp_path), resolution=64)
        assert len(ds) == 2

    def test_label_to_rgb(self, tmp_path):
        """_label_to_rgb should convert label indices to RGB."""
        ds = CityscapesSegDataset(root_dir=str(tmp_path), resolution=8)
        label = torch.zeros(8, 8, dtype=torch.long)
        label[0, 0] = 0  # road
        label[1, 1] = 1  # sidewalk
        rgb = ds._label_to_rgb(label)
        assert rgb.shape == (3, 8, 8)

    def test_palette_cached(self, tmp_path):
        """_get_palette should cache the color palette."""
        CityscapesSegDataset(root_dir=str(tmp_path))
        palette1 = CityscapesSegDataset._get_palette()
        palette2 = CityscapesSegDataset._get_palette()
        assert palette1 is palette2  # Same object (cached)


# ──────────────────────────────────────────────────────────────────
# SACoInstanceDataset tests
# ──────────────────────────────────────────────────────────────────


class TestSACoInstanceDataset:
    """Test SACoInstanceDataset."""

    def test_scan_finds_pairs(self, tmp_path):
        """_scan should find color/instance pairs."""
        for i in range(2):
            Image.new("RGB", (64, 64)).save(tmp_path / f"obj_{i}_color.png")
            Image.new("RGB", (64, 64)).save(tmp_path / f"obj_{i}_instance.png")

        ds = SACoInstanceDataset(root_dir=str(tmp_path))
        assert len(ds) == 2

    def test_getitem(self, tmp_path):
        """__getitem__ should return instance-encoded target_rgb."""
        Image.new("RGB", (64, 64)).save(tmp_path / "obj_0_color.png")
        Image.new("RGB", (64, 64), color=(255, 0, 0)).save(tmp_path / "obj_0_instance.png")

        ds = SACoInstanceDataset(root_dir=str(tmp_path), resolution=64)
        item = ds[0]
        assert item["image"].shape == (3, 64, 64)
        assert item["target_rgb"].shape == (3, 64, 64)


# ──────────────────────────────────────────────────────────────────
# RefCOCOgDataset tests
# ──────────────────────────────────────────────────────────────────


class TestRefCOCOgDataset:
    """Test RefCOCOgDataset."""

    def test_scan_no_annotation(self, tmp_path):
        """_scan should handle missing annotation file gracefully."""
        ds = RefCOCOgDataset(root_dir=str(tmp_path))
        assert len(ds) == 0

    def test_scan_with_mock_data(self, tmp_path):
        """_scan should load mock RefCOCOg annotations."""
        # Create image
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = Image.new("RGB", (64, 64))
        img.save(img_dir / "123.jpg")

        # Create annotation
        refs = [
            {
                "image_id": 123,
                "ann_id": 1,
                "sentences": ["a cat sitting on a chair"],
            }
        ]
        with open(tmp_path / "refs(g).p", "wb") as f:
            pickle.dump(refs, f)

        # Create instances.json with polygon segmentation
        instances = {
            "annotations": [
                {
                    "id": 1,
                    "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]],
                    "bbox": [10, 10, 40, 40],
                }
            ]
        }
        with open(tmp_path / "instances.json", "w") as f:
            json.dump(instances, f)

        ds = RefCOCOgDataset(root_dir=str(tmp_path), resolution=32)
        assert len(ds) == 1

        item = ds[0]
        assert "prompt" in item
        assert "image" in item
        assert "target_rgb" in item
        assert item["target_rgb"].shape == (3, 32, 32)

    def test_load_mask_no_annotation(self, tmp_path):
        """_load_mask should return black mask when annotation is missing."""
        ds = RefCOCOgDataset(root_dir=str(tmp_path), resolution=16)
        mask = ds._load_mask(ann_id=None)
        assert mask.shape == (3, 16, 16)
        assert mask.sum() == 0

    def test_load_mask_polygon(self, tmp_path):
        """_load_mask should decode polygon segmentation."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        Image.new("RGB", (64, 64)).save(img_dir / "123.jpg")

        refs = [{"image_id": 123, "ann_id": 1, "sentences": ["cat"]}]
        with open(tmp_path / "refs(g).p", "wb") as f:
            pickle.dump(refs, f)

        instances = {
            "annotations": [
                {
                    "id": 1,
                    "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]],
                    "bbox": [10, 10, 40, 40],
                }
            ]
        }
        with open(tmp_path / "instances.json", "w") as f:
            json.dump(instances, f)

        ds = RefCOCOgDataset(root_dir=str(tmp_path), resolution=32)
        mask = ds._load_mask(ann_id=1)
        # Some pixels should be white (filled polygon)
        assert mask.sum() > 0

    def test_load_mask_rle(self, tmp_path):
        """_load_mask should decode RLE segmentation."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        Image.new("RGB", (64, 64)).save(img_dir / "123.jpg")

        refs = [{"image_id": 123, "ann_id": 1, "sentences": ["cat"]}]
        with open(tmp_path / "refs(g).p", "wb") as f:
            pickle.dump(refs, f)

        # RLE: uncompressed list format
        # 4x4 mask with a 2x2 filled region
        rle_counts = [4, 2, 2, 2, 2, 4]  # 4 zeros, 2 ones, 2 zeros, 2 ones, 2 zeros, 4 zeros
        instances = {
            "annotations": [
                {
                    "id": 1,
                    "segmentation": {
                        "counts": rle_counts,
                        "size": [4, 4],
                    },
                    "bbox": [0, 0, 4, 4],
                }
            ]
        }
        with open(tmp_path / "instances.json", "w") as f:
            json.dump(instances, f)

        ds = RefCOCOgDataset(root_dir=str(tmp_path), resolution=4)
        mask = ds._load_mask(ann_id=1)
        assert mask.shape == (3, 4, 4)

    def test_load_mask_no_segmentation(self, tmp_path):
        """_load_mask should return black mask when seg is None."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        Image.new("RGB", (64, 64)).save(img_dir / "123.jpg")

        refs = [{"image_id": 123, "ann_id": 1, "sentences": ["cat"]}]
        with open(tmp_path / "refs(g).p", "wb") as f:
            pickle.dump(refs, f)

        instances = {"annotations": [{"id": 1, "bbox": [0, 0, 10, 10]}]}
        with open(tmp_path / "instances.json", "w") as f:
            json.dump(instances, f)

        ds = RefCOCOgDataset(root_dir=str(tmp_path), resolution=8)
        mask = ds._load_mask(ann_id=1)
        assert mask.sum() == 0


class TestETH3DOfficialLayout:
    def test_scans_official_scene_and_loads_binary_depth(self, tmp_path):
        from open_vision_banana.data.benchmark_io import infer_planar_hw, load_eth3d_depth
        from open_vision_banana.data.datasets import ETH3DDataset

        height, width = 4, 6
        assert infer_planar_hw(height * width) == (height, width)
        rgb_dir = tmp_path / "courtyard" / "images" / "dslr_images_undistorted"
        depth_dir = tmp_path / "courtyard" / "ground_truth_depth" / "dslr_images"
        rgb_dir.mkdir(parents=True)
        depth_dir.mkdir(parents=True)
        Image.new("RGB", (width, height), color=(10, 20, 30)).save(rgb_dir / "DSC_0001.JPG")
        depth = np.array(
            [
                [1.5, 2.0, 3.0, 4.0, 5.0, np.inf],
                [1.5, 2.0, 3.0, 4.0, 5.0, np.inf],
                [1.5, 2.0, 3.0, 4.0, 5.0, np.inf],
                [1.5, 2.0, 3.0, 4.0, 5.0, np.inf],
            ],
            dtype=np.float32,
        )
        depth.tofile(depth_dir / "DSC_0001.JPG")

        loaded = load_eth3d_depth(depth_dir / "DSC_0001.JPG")
        assert loaded.shape == (1, height, width)
        assert torch.isinf(loaded[0, 0, -1])

        ds = ETH3DDataset(root_dir=str(tmp_path), split="val", resolution=8)
        assert len(ds) == 1
        item = ds[0]
        assert item["gt_depth"].shape == (1, 8, 8)
        assert bool(item["valid_mask"].any())

    def test_flat_layout_still_preferred(self, tmp_path):
        from open_vision_banana.data.datasets import ETH3DDataset

        img_dir = tmp_path / "images"
        depth_dir = tmp_path / "depths"
        img_dir.mkdir()
        depth_dir.mkdir()
        Image.new("RGB", (8, 8)).save(img_dir / "a.png")
        np.save(depth_dir / "a.npy", np.ones((8, 8), dtype=np.float32))
        official_rgb = tmp_path / "scene" / "images" / "dslr_images_undistorted"
        official_depth = tmp_path / "scene" / "ground_truth_depth" / "dslr_images"
        official_rgb.mkdir(parents=True)
        official_depth.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(official_rgb / "b.JPG")
        np.ones((4, 6), dtype=np.float32).tofile(official_depth / "b.JPG")

        ds = ETH3DDataset(root_dir=str(tmp_path), resolution=8)
        assert len(ds) == 1
        assert ds.samples[0]["image"].name == "a.png"


class TestKITTIOfficialLayout:
    def test_scans_annotated_drive_when_rgb_present(self, tmp_path):
        from open_vision_banana.data.datasets import KITTIDataset

        drive = tmp_path / "val" / "2011_09_26_drive_0002_sync"
        rgb_dir = drive / "image_02" / "data"
        depth_dir = drive / "proj_depth" / "groundtruth" / "image_02"
        rgb_dir.mkdir(parents=True)
        depth_dir.mkdir(parents=True)
        Image.new("RGB", (16, 8), color=(1, 2, 3)).save(rgb_dir / "0000000005.png")
        Image.new("I;16", (16, 8), color=256).save(depth_dir / "0000000005.png")

        ds = KITTIDataset(root_dir=str(tmp_path), split="val", resolution=8)
        assert len(ds) == 1

    def test_depth_only_drive_is_not_a_pair(self, tmp_path):
        from open_vision_banana.data.datasets import KITTIDataset

        depth_dir = (
            tmp_path
            / "val"
            / "2011_09_26_drive_0002_sync"
            / "proj_depth"
            / "groundtruth"
            / "image_02"
        )
        depth_dir.mkdir(parents=True)
        Image.new("I;16", (16, 8), color=256).save(depth_dir / "0000000005.png")
        ds = KITTIDataset(root_dir=str(tmp_path), split="val", resolution=8)
        assert len(ds) == 0


class TestDIODENormalPairing:
    def test_requires_distinct_normal_file(self, tmp_path):
        from open_vision_banana.data.datasets import DIODENormalDataset

        val = tmp_path / "val" / "indoor"
        val.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(val / "frame_rgb.png")
        np.save(val / "frame_normal.npy", np.ones((8, 8, 3), dtype=np.float32))
        Image.new("RGB", (8, 8)).save(val / "orphan.png")
        ds = DIODENormalDataset(root_dir=str(tmp_path), split="val", resolution=8)
        assert len(ds) == 1
        assert ds.samples[0]["image"].name == "frame_rgb.png"

    def test_missing_val_split_is_empty(self, tmp_path):
        from open_vision_banana.data.datasets import DIODENormalDataset

        Image.new("RGB", (8, 8)).save(tmp_path / "frame.png")
        ds = DIODENormalDataset(root_dir=str(tmp_path), split="val", resolution=8)
        assert len(ds) == 0


class TestCityscapesFlattenedVal:
    def test_scan_finds_export_val_pairs(self, tmp_path):
        img_dir = tmp_path / "leftImg8bit" / "val" / "export"
        gt_dir = tmp_path / "gtFine" / "val" / "export"
        img_dir.mkdir(parents=True)
        gt_dir.mkdir(parents=True)
        Image.new("RGB", (32, 16)).save(img_dir / "export_000000_000000_leftImg8bit.png")
        Image.new("L", (32, 16)).save(gt_dir / "export_000000_000000_gtFine_labelIds.png")
        ds = CityscapesSegDataset(root_dir=str(tmp_path), split="val", resolution=16)
        assert len(ds) == 1
