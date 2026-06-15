"""Tests for data loading and configuration."""

import pytest
import torch
import tempfile
from pathlib import Path

from open_vision_banana.data.datasets import (
    BaseVisionDataset,
    build_dataset,
)
from open_vision_banana.data.mixed import MixedVisionDataset
from open_vision_banana.config import (
    VisionBananaConfig,
    ModelConfig,
    LoRAConfig,
    OptimizerConfig,
    DataConfig,
    TrainingConfig,
    DepthMappingConfig,
    NormalMappingConfig,
    SegmentationConfig,
)


class TestBaseVisionDataset:
    """Tests for base dataset class."""

    def test_nonexistent_root(self):
        ds = BaseVisionDataset(root_dir="/nonexistent/path")
        assert len(ds) == 0

    def test_abstract_scan(self):
        ds = BaseVisionDataset.__new__(BaseVisionDataset)
        ds.root_dir = Path("/tmp")
        ds.samples = []
        # _scan is a no-op in base class, overridden in subclasses
        ds._scan()
        assert len(ds) == 0


class TestBuildDataset:
    """Tests for dataset factory function."""

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            build_dataset(task="unknown", root_dir="/tmp")

    def test_depth_task_creates_dataset(self):
        ds = build_dataset(task="depth", root_dir="/nonexistent")
        assert ds.TASK_TYPE == "depth"

    def test_normal_task_creates_dataset(self):
        ds = build_dataset(task="normal", root_dir="/nonexistent")
        assert ds.TASK_TYPE == "normal"

    def test_semantic_task_creates_dataset(self):
        ds = build_dataset(task="semantic", root_dir="/nonexistent")
        assert ds.TASK_TYPE == "semantic"

    def test_referring_task_creates_dataset(self):
        ds = build_dataset(task="referring", root_dir="/nonexistent")
        assert ds.TASK_TYPE == "referring"


class TestMixedVisionDataset:
    """Tests for mixed vision/generation dataset."""

    def test_empty_datasets(self):
        ds1 = BaseVisionDataset(root_dir="/nonexistent1")
        ds2 = BaseVisionDataset(root_dir="/nonexistent2")
        mixed = MixedVisionDataset(vision_datasets=[ds1, ds2])
        assert len(mixed) == 0

    def test_vision_ratio(self):
        ds1 = BaseVisionDataset(root_dir="/nonexistent1")
        mixed = MixedVisionDataset(
            vision_datasets=[ds1],
            vision_ratio=0.1,
        )
        assert mixed.vision_weight == 0.1
        assert mixed.gen_weight == 0.9


class TestConfigSystem:
    """Comprehensive tests for the configuration system."""

    def test_all_defaults(self):
        cfg = VisionBananaConfig()
        assert cfg.model.name == "black-forest-labs/FLUX.1-schnell"
        assert cfg.model.dtype == "bfloat16"
        assert cfg.lora.rank == 64
        assert cfg.lora.alpha == 128
        assert cfg.lora.rs_lora is True
        assert cfg.lora.target_modules == ["to_k", "to_v"]
        assert cfg.lora.dropout == 0.0
        assert cfg.optimizer.lr == 5e-5
        assert cfg.optimizer.weight_decay == 0.01
        assert cfg.optimizer.scheduler == "cosine"
        assert cfg.optimizer.warmup_steps == 500
        assert cfg.optimizer.max_grad_norm == 1.0
        assert cfg.data.vision_ratio == 0.1
        assert cfg.data.resolution == 512
        assert cfg.data.batch_size == 8
        assert cfg.data.num_workers == 4
        assert cfg.data.augmentation is True
        assert cfg.training.epochs == 10
        assert cfg.training.use_amp is True
        assert cfg.training.use_ema is True
        assert cfg.training.ema_decay == 0.9999
        assert cfg.depth_mapping.lam == -3.0
        assert abs(cfg.depth_mapping.c - 10.0/3.0) < 1e-6
        assert cfg.segmentation.color_tolerance == 14.0
        assert cfg.segmentation.min_area_fraction == 2e-4

    def test_yaml_roundtrip_full(self):
        cfg = VisionBananaConfig()
        cfg.lora.rank = 32
        cfg.lora.alpha = 64
        cfg.optimizer.lr = 1e-4
        cfg.depth_mapping.lam = -5.0
        cfg.segmentation.color_tolerance = 20.0

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            cfg.to_yaml(f.name)
            loaded = VisionBananaConfig.from_yaml(f.name)

        assert loaded.lora.rank == 32
        assert loaded.lora.alpha == 64
        assert loaded.optimizer.lr == 1e-4
        assert loaded.depth_mapping.lam == -5.0
        assert loaded.segmentation.color_tolerance == 20.0
        Path(f.name).unlink()

    def test_to_dict(self):
        cfg = VisionBananaConfig()
        d = cfg.to_dict()
        assert "model" in d
        assert "lora" in d
        assert "optimizer" in d
        assert "data" in d
        assert "training" in d
        assert "depth_mapping" in d
        assert "normal_mapping" in d
        assert "segmentation" in d

    def test_from_dict_with_extra_keys(self):
        """Config should ignore unknown keys gracefully."""
        d = {
            "model": {"name": "test", "dtype": "float16", "extra_key": "ignored"},
            "data": {"vision_ratio": 0.2, "generation_ratio": 0.8},
        }
        cfg = VisionBananaConfig.from_dict(d)
        assert cfg.model.name == "test"
        assert cfg.data.vision_ratio == 0.2

    def test_subconfig_independence(self):
        """Modifying one sub-config shouldn't affect defaults."""
        cfg1 = VisionBananaConfig()
        cfg1.lora.rank = 16
        cfg2 = VisionBananaConfig()
        assert cfg2.lora.rank == 64

    def test_depth_mapping_config(self):
        cfg = DepthMappingConfig(lam=-5.0, c=5.0)
        d = cfg.to_dict()
        loaded = DepthMappingConfig.from_dict(d)
        assert loaded.lam == -5.0
        assert loaded.c == 5.0

    def test_normal_mapping_config(self):
        cfg = NormalMappingConfig()
        d = cfg.to_dict()
        assert d == {}

    def test_segmentation_config_roundtrip(self):
        cfg = SegmentationConfig(color_tolerance=20.0)
        d = cfg.to_dict()
        loaded = SegmentationConfig.from_dict(d)
        assert loaded.color_tolerance == 20.0

    def test_model_config_roundtrip(self):
        cfg = ModelConfig(name="test-model", dtype="float32")
        d = cfg.to_dict()
        loaded = ModelConfig.from_dict(d)
        assert loaded.name == "test-model"

    def test_lora_config_roundtrip(self):
        cfg = LoRAConfig(rank=32, alpha=64, rs_lora=False)
        d = cfg.to_dict()
        loaded = LoRAConfig.from_dict(d)
        assert loaded.rank == 32
        assert loaded.rs_lora is False

    def test_optimizer_config_roundtrip(self):
        cfg = OptimizerConfig(lr=1e-4, scheduler="linear")
        d = cfg.to_dict()
        loaded = OptimizerConfig.from_dict(d)
        assert loaded.lr == 1e-4
        assert loaded.scheduler == "linear"

    def test_data_config_roundtrip(self):
        cfg = DataConfig(vision_ratio=0.2, resolution=1024)
        d = cfg.to_dict()
        loaded = DataConfig.from_dict(d)
        assert loaded.vision_ratio == 0.2
        assert loaded.resolution == 1024

    def test_training_config_roundtrip(self):
        cfg = TrainingConfig(epochs=20, use_ema=False)
        d = cfg.to_dict()
        loaded = TrainingConfig.from_dict(d)
        assert loaded.epochs == 20
        assert loaded.use_ema is False
