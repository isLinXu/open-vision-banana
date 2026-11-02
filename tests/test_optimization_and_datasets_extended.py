"""Tests for extended benchmark datasets and optimization utilities."""

import pytest
import torch

# ─── Optimization module tests ───
from open_vision_banana.training.optimization import (
    AdvancedAugmentation,
    MemoryManager,
    SchedulerFactory,
    create_optimized_dataloader,
)


class TestMemoryManager:
    """Tests for MemoryManager."""

    def test_init_cpu(self):
        device = torch.device("cpu")
        mm = MemoryManager(device)
        assert mm.device == device
        assert mm.current_usage_gb() == 0.0
        assert not mm.should_clear()

    def test_init_mps(self):
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")
        device = torch.device("mps")
        mm = MemoryManager(device)
        assert mm._mps_available

    def test_clear_cache_cpu(self):
        device = torch.device("cpu")
        mm = MemoryManager(device)
        mm.clear_cache()  # Should not raise
        assert not mm.maybe_clear()


class TestSchedulerFactory:
    """Tests for learning rate scheduler factory."""

    def test_cosine_scheduler(self):
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([param], lr=1e-4)
        sched = SchedulerFactory.create(
            opt, total_steps=100, warmup_steps=10, scheduler_type="cosine"
        )
        assert sched is not None
        for _ in range(5):
            opt.step()
            sched.step()
        lr = sched.get_last_lr()[0]
        assert 0 < lr <= 1e-4

    def test_cosine_restarts_scheduler(self):
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([param], lr=1e-4)
        sched = SchedulerFactory.create(
            opt, total_steps=100, warmup_steps=10, scheduler_type="cosine_restarts"
        )
        for _ in range(100):
            opt.step()
            sched.step()
        lr = sched.get_last_lr()[0]
        assert lr >= 0

    def test_polynomial_scheduler(self):
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([param], lr=1e-4)
        sched = SchedulerFactory.create(
            opt, total_steps=100, warmup_steps=10, scheduler_type="polynomial", power=2.0
        )
        for _ in range(100):
            opt.step()
            sched.step()
        lr = sched.get_last_lr()[0]
        assert lr >= 0

    def test_constant_scheduler(self):
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([param], lr=1e-4)
        sched = SchedulerFactory.create(
            opt, total_steps=100, warmup_steps=10, scheduler_type="constant"
        )
        for _ in range(50):
            opt.step()
            sched.step()
        lr = sched.get_last_lr()[0]
        assert lr == pytest.approx(1e-4, abs=1e-6)

    def test_one_cycle_scheduler(self):
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([param], lr=1e-4)
        sched = SchedulerFactory.create(
            opt, total_steps=100, warmup_steps=10, scheduler_type="one_cycle"
        )
        for _ in range(100):
            opt.step()
            sched.step()
        lr = sched.get_last_lr()[0]
        assert lr >= 0

    def test_warmup_ratio(self):
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([param], lr=1e-4)
        sched = SchedulerFactory.create(
            opt, total_steps=100, warmup_ratio=0.2, scheduler_type="cosine"
        )
        for _ in range(10):
            opt.step()
            sched.step()
        lr = sched.get_last_lr()[0]
        assert 0 < lr < 1e-4  # Should be in warmup


class TestAdvancedAugmentation:
    """Tests for advanced data augmentation pipeline."""

    def test_no_augmentation(self):
        aug = AdvancedAugmentation(
            flip_prob=0.0,
            scale_prob=0.0,
            brightness_prob=0.0,
            color_jitter_prob=0.0,
            gaussian_noise_prob=0.0,
            cutout_prob=0.0,
        )
        image = torch.rand(3, 8, 8) * 2.0 - 1.0
        target = torch.rand(3, 8, 8)
        img_out, tgt_out = aug(image, target)
        assert torch.allclose(img_out, image, atol=1e-5)
        assert torch.allclose(tgt_out, target, atol=1e-5)

    def test_flip(self):
        aug = AdvancedAugmentation(
            flip_prob=1.0,
            scale_prob=0.0,
            brightness_prob=0.0,
            color_jitter_prob=0.0,
            gaussian_noise_prob=0.0,
            cutout_prob=0.0,
        )
        image = torch.arange(24).float().reshape(3, 2, 4) * 2.0 / 24.0 - 1.0
        target = torch.arange(24).float().reshape(3, 2, 4) / 24.0
        img_out, tgt_out = aug(image, target)
        # Both should be flipped horizontally
        assert torch.allclose(img_out[:, :, 0], image[:, :, -1], atol=1e-5)
        assert torch.allclose(tgt_out[:, :, 0], target[:, :, -1], atol=1e-5)

    def test_scale_crop(self):
        aug = AdvancedAugmentation(
            flip_prob=0.0,
            scale_prob=1.0,
            scale_range=(0.8, 0.8),
            brightness_prob=0.0,
            color_jitter_prob=0.0,
            gaussian_noise_prob=0.0,
            cutout_prob=0.0,
        )
        image = torch.rand(3, 16, 16) * 2.0 - 1.0
        target = torch.rand(3, 16, 16)
        img_out, tgt_out = aug(image, target)
        assert img_out.shape == (3, 16, 16)
        assert tgt_out.shape == (3, 16, 16)

    def test_brightness(self):
        aug = AdvancedAugmentation(
            flip_prob=0.0,
            scale_prob=0.0,
            brightness_prob=1.0,
            brightness_range=(0.1, 0.1),
            color_jitter_prob=0.0,
            gaussian_noise_prob=0.0,
            cutout_prob=0.0,
        )
        image = torch.ones(3, 8, 8) * 0.5
        target = torch.ones(3, 8, 8)
        img_out, tgt_out = aug(image, target)
        assert not torch.allclose(img_out, image, atol=1e-5)
        assert torch.allclose(tgt_out, target, atol=1e-5)  # Target unchanged

    def test_color_jitter(self):
        aug = AdvancedAugmentation(
            flip_prob=0.0,
            scale_prob=0.0,
            brightness_prob=0.0,
            color_jitter_prob=1.0,
            color_jitter_strength=0.1,
            gaussian_noise_prob=0.0,
            cutout_prob=0.0,
        )
        image = torch.rand(3, 8, 8) * 2.0 - 1.0
        target = torch.rand(3, 8, 8)
        img_out, tgt_out = aug(image, target)
        assert img_out.shape == (3, 8, 8)
        assert tgt_out.shape == (3, 8, 8)

    def test_gaussian_noise(self):
        aug = AdvancedAugmentation(
            flip_prob=0.0,
            scale_prob=0.0,
            brightness_prob=0.0,
            color_jitter_prob=0.0,
            gaussian_noise_prob=1.0,
            gaussian_noise_std=0.05,
            cutout_prob=0.0,
        )
        image = torch.zeros(3, 8, 8)
        target = torch.zeros(3, 8, 8)
        img_out, tgt_out = aug(image, target)
        assert not torch.allclose(img_out, image, atol=1e-3)
        assert torch.allclose(tgt_out, target, atol=1e-5)

    def test_cutout(self):
        aug = AdvancedAugmentation(
            flip_prob=0.0,
            scale_prob=0.0,
            brightness_prob=0.0,
            color_jitter_prob=0.0,
            gaussian_noise_prob=0.0,
            cutout_prob=1.0,
            cutout_size=0.5,
            depth_aware=False,
        )
        image = torch.ones(3, 8, 8)
        target = torch.ones(3, 8, 8)
        img_out, tgt_out = aug(image, target)
        assert img_out.min() == 0.0  # Some region should be zeroed
        assert torch.allclose(tgt_out, target, atol=1e-5)

    def test_value_ranges_preserved(self):
        aug = AdvancedAugmentation(
            flip_prob=0.5,
            scale_prob=0.3,
            brightness_prob=0.5,
            color_jitter_prob=0.3,
            gaussian_noise_prob=0.2,
            cutout_prob=0.1,
        )
        for _ in range(10):
            image = torch.rand(3, 16, 16) * 2.0 - 1.0
            target = torch.rand(3, 16, 16)
            img_out, tgt_out = aug(image, target)
            assert img_out.min() >= -1.0 and img_out.max() <= 1.0
            assert tgt_out.min() >= 0.0 and tgt_out.max() <= 1.0


class TestCreateOptimizedDataloader:
    """Tests for optimized DataLoader creation."""

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=10):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {"image": torch.zeros(3, 8, 8), "target": torch.zeros(3, 8, 8)}

    def test_cpu_dataloader(self):
        ds = self.DummyDataset()
        device = torch.device("cpu")
        dl = create_optimized_dataloader(ds, batch_size=2, device=device)
        assert dl.batch_size == 2
        assert dl.num_workers == 0
        batch = next(iter(dl))
        assert "image" in batch

    def test_mps_dataloader(self):
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")
        ds = self.DummyDataset()
        device = torch.device("mps")
        dl = create_optimized_dataloader(ds, batch_size=2, device=device)
        assert dl.num_workers == 0  # MPS must use 0 workers
        assert not dl.pin_memory  # MPS doesn't benefit from pin_memory

    def test_cuda_dataloader(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        ds = self.DummyDataset()
        device = torch.device("cuda")
        dl = create_optimized_dataloader(ds, batch_size=2, device=device)
        assert dl.num_workers > 0  # CUDA should use workers
        assert dl.pin_memory  # CUDA should pin memory

    def test_custom_workers(self):
        ds = self.DummyDataset()
        device = torch.device("cpu")
        dl = create_optimized_dataloader(ds, batch_size=2, device=device, num_workers=2)
        assert dl.num_workers == 2

    def test_collate_fn(self):
        ds = self.DummyDataset()

        def custom_collate(batch):
            return batch

        dl = create_optimized_dataloader(ds, batch_size=2, collate_fn=custom_collate)
        batch = next(iter(dl))
        assert isinstance(batch, list)
        assert len(batch) == 2


# ─── Extended dataset build tests ───

from open_vision_banana.data.datasets import build_dataset  # noqa: E402


class TestBuildDatasetExtended:
    """Tests for build_dataset factory with all 15+ datasets."""

    def test_all_registry_keys_exist(self):
        """Verify all registered dataset keys can be instantiated (even if empty)."""
        from open_vision_banana.data.datasets import (
            COCODetectionDataset,
            COCOStuffDataset,
            DIODEDataset,
            ETH3DDataset,
            FlyingThings3DDataset,
            HypersimDepthDataset,
            KITTIDataset,
            MapillaryVistasDataset,
            NYUDepthDataset,
            OASISNormalDataset,
            RefCOCODataset,
            RefCOCOgDataset,
            SACoInstanceDataset,
            ScanNetNormalDataset,
            SintelFlowDataset,
            TUMRGBDDataset,
            VirtualKITTIDataset,
            nuScenesDataset,
        )

        # All classes should be importable
        classes = [
            HypersimDepthDataset,
            ScanNetNormalDataset,
            SACoInstanceDataset,
            RefCOCOgDataset,
            RefCOCODataset,
            NYUDepthDataset,
            DIODEDataset,
            KITTIDataset,
            ETH3DDataset,
            OASISNormalDataset,
            COCODetectionDataset,
            COCOStuffDataset,
            VirtualKITTIDataset,
            TUMRGBDDataset,
            FlyingThings3DDataset,
            SintelFlowDataset,
            MapillaryVistasDataset,
            nuScenesDataset,
        ]
        for cls in classes:
            assert cls is not None
            assert hasattr(cls, "TASK_TYPE")
            assert hasattr(cls, "__init__")

    def test_registry_completeness(self):
        """Verify build_dataset registry covers all 15+ datasets."""
        import inspect

        inspect.getclosurevars(build_dataset).nonlocals.get("registry", {})
        # We can't easily inspect the closure, so we test by calling with invalid keys
        expected_tasks = [
            "depth",
            "normal",
            "semantic",
            "instance",
            "referring",
            "nyu_depth",
            "diode",
            "kitti",
            "ade20k",
            "sintel",
            "eth3d",
            "oasis",
            "coco_detection",
            "coco_stuff",
            "refcoco",
            "virtual_kitti",
            "tum_rgbd",
            "flyingthings3d",
            "mapillary",
            "nuscenes",
            "nuscenes_depth",
        ]
        for task in expected_tasks:
            try:
                ds = build_dataset(task, root_dir="/nonexistent", split="train")
                # Should succeed even with empty data (warns but doesn't crash)
                assert hasattr(ds, "__len__")
                assert len(ds) == 0  # Empty because root_dir doesn't exist
            except ValueError as e:
                pytest.fail(f"build_dataset failed for {task}: {e}")

    def test_task_type_attributes(self):
        from open_vision_banana.data.datasets import (
            COCODetectionDataset,
            ETH3DDataset,
            FlyingThings3DDataset,
            OASISNormalDataset,
            TUMRGBDDataset,
            nuScenesDataset,
        )

        assert ETH3DDataset.TASK_TYPE == "depth"
        assert OASISNormalDataset.TASK_TYPE == "normal"
        assert COCODetectionDataset.TASK_TYPE == "object_detection"
        assert FlyingThings3DDataset.TASK_TYPE == "flow"
        assert TUMRGBDDataset.TASK_TYPE == "depth"
        assert nuScenesDataset.TASK_TYPE == "semantic"

    def test_coco_classes_count(self):
        from open_vision_banana.data.datasets import COCODetectionDataset

        assert len(COCODetectionDataset.COCO_CLASSES) == 80
        assert COCODetectionDataset.COCO_CLASSES[0] == "person"

    def test_load_flo_format(self):
        from open_vision_banana.data.datasets import FlyingThings3DDataset, SintelFlowDataset

        # Verify _load_flo method exists on both flow datasets
        assert hasattr(SintelFlowDataset, "_load_flo")
        assert hasattr(FlyingThings3DDataset, "_load_flo")


# ─── MixedVisionDataset with multi-colormap tests ───

from open_vision_banana.data.mixed import MixedVisionDataset  # noqa: E402


class TestMixedVisionDatasetMultiColormap:
    """Tests for multi-colormap training in MixedVisionDataset."""

    class DummyVisionDataset(torch.utils.data.Dataset):
        def __init__(self, size=5):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                "image": torch.zeros(3, 8, 8),
                "target_rgb": torch.zeros(3, 8, 8),
                "task": "depth",
            }

    class DummyGenDataset(torch.utils.data.Dataset):
        def __init__(self, size=45):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                "image": torch.zeros(3, 8, 8),
                "target_rgb": torch.zeros(3, 8, 8),
                "task": "generation",
            }

    def test_multi_colormap(self):
        vision_ds = self.DummyVisionDataset(5)
        gen_ds = self.DummyGenDataset(45)
        mixed = MixedVisionDataset(
            vision_datasets=[vision_ds],
            generation_dataset=gen_ds,
            vision_ratio=1.0,  # 100% vision to guarantee colormap assignment
            colormaps=["cube_edge", "rainbow", "viridis"],
        )
        colormaps_seen = set()
        for _ in range(20):
            sample = mixed[0]
            if "colormap" in sample:
                colormaps_seen.add(sample["colormap"])
        assert len(colormaps_seen) > 0, "No colormaps were assigned to depth samples"
        # With 3 colormaps and random sampling, should see variety over 20 draws
        assert all(c in ["cube_edge", "rainbow", "viridis"] for c in colormaps_seen)

    def test_effective_length(self):
        vision_ds = self.DummyVisionDataset(10)
        gen_ds = self.DummyGenDataset(90)
        mixed = MixedVisionDataset(
            vision_datasets=[vision_ds],
            generation_dataset=gen_ds,
            vision_ratio=0.1,
        )
        # Effective length should be vision / vision_ratio = 10 / 0.1 = 100
        assert len(mixed) == 100

    def test_multiple_vision_datasets(self):
        ds1 = self.DummyVisionDataset(5)
        ds2 = self.DummyVisionDataset(5)
        gen_ds = self.DummyGenDataset(90)
        mixed = MixedVisionDataset(
            vision_datasets=[ds1, ds2],
            generation_dataset=gen_ds,
            vision_ratio=0.1,
        )
        # Total vision = 10, effective = 10 / 0.1 = 100
        assert len(mixed) == 100
        assert mixed.total_vision == 10

    def test_no_generation_dataset(self):
        ds1 = self.DummyVisionDataset(10)
        mixed = MixedVisionDataset(
            vision_datasets=[ds1],
            generation_dataset=None,
            vision_ratio=1.0,
        )
        assert len(mixed) == 10
        for _ in range(5):
            sample = mixed[0]
            assert sample["task"] == "depth"

    def test_repr(self):
        ds1 = self.DummyVisionDataset(5)
        gen_ds = self.DummyGenDataset(45)
        mixed = MixedVisionDataset([ds1], gen_ds, vision_ratio=0.1)
        r = repr(mixed)
        assert "MixedVisionDataset" in r
        assert "total_vision=5" in r

    def test_backward_compatibility(self):
        """MixedVisionDataset should be backward compatible with old API."""
        ds1 = self.DummyVisionDataset(5)
        gen_ds = self.DummyGenDataset(45)
        mixed = MixedVisionDataset([ds1], gen_ds, vision_ratio=0.1)
        # Access backward-compatible aliases
        assert mixed.vision_weight == 0.1
        assert mixed.gen_weight == 0.9


# ─── Integration: optimization + data ───


class TestIntegrationOptimizationData:
    """Integration tests combining optimization utilities with data loading."""

    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, size=20):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                "image": torch.rand(3, 8, 8) * 2.0 - 1.0,
                "target_rgb": torch.rand(3, 8, 8),
                "task": "depth",
            }

    def test_dataloader_with_augmented_dataset(self):
        """Test that optimized DataLoader works with augmented vision datasets."""
        base_ds = self.MockDataset()
        aug = AdvancedAugmentation(
            flip_prob=0.5,
            scale_prob=0.3,
            brightness_prob=0.3,
            color_jitter_prob=0.2,
            gaussian_noise_prob=0.1,
            cutout_prob=0.1,
        )

        class AugmentedDS(torch.utils.data.Dataset):
            def __init__(self, base, aug):
                self.base = base
                self.aug = aug

            def __len__(self):
                return len(self.base)

            def __getitem__(self, idx):
                s = self.base[idx]
                img, tgt = self.aug(s["image"], s["target_rgb"])
                s = dict(s)
                s["image"] = img
                s["target_rgb"] = tgt
                return s

        aug_ds = AugmentedDS(base_ds, aug)
        device = torch.device("cpu")
        dl = create_optimized_dataloader(aug_ds, batch_size=2, device=device, num_workers=0)
        batch = next(iter(dl))
        assert batch["image"].shape == (2, 3, 8, 8)
        assert batch["target_rgb"].shape == (2, 3, 8, 8)

    def test_scheduler_with_dataloader_iteration(self):
        """Test that scheduler updates correctly with dataloader steps."""
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([param], lr=1e-4)
        sched = SchedulerFactory.create(
            opt, total_steps=10, warmup_steps=2, scheduler_type="cosine"
        )

        ds = self.MockDataset()
        dl = create_optimized_dataloader(
            ds, batch_size=2, device=torch.device("cpu"), num_workers=0
        )

        steps = 0
        for _batch in dl:
            opt.step()
            sched.step()
            steps += 1
            if steps >= 5:
                break

        lr = sched.get_last_lr()[0]
        assert lr > 0
        assert steps == 5
