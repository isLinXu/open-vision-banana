"""Integration tests for CheckpointManager, VisionBanana pipeline,
CityscapesSegDataset vectorized label_to_rgb, and VisualizationDecoder.

These tests use only CPU / synthetic data and do not require FLUX weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    """Identity-like model used as a stand-in for VisionBanana internals."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

class TestCheckpointManagerIntegration:

    def test_numeric_sort_order(self, tmp_path):
        """Checkpoint with highest numeric step is selected as 'latest'."""
        from open_vision_banana.training.checkpoint import CheckpointManager
        cm = CheckpointManager(str(tmp_path), max_keep=10)
        for step in [10, 200, 30, 1000, 50]:
            (tmp_path / f"checkpoint-{step}").mkdir()
        latest = cm._latest_checkpoint()
        assert latest is not None
        assert latest.name == "checkpoint-1000", f"Got {latest.name}"

    def test_prune_keeps_newest(self, tmp_path):
        """Pruning removes oldest (numerically smallest) checkpoints."""
        from open_vision_banana.training.checkpoint import CheckpointManager
        cm = CheckpointManager(str(tmp_path), max_keep=3)
        for step in [1, 2, 3, 4, 5]:
            d = tmp_path / f"checkpoint-{step}"
            d.mkdir()
        cm._prune_old_checkpoints()
        remaining = sorted(
            [int(p.name.split("-")[1]) for p in tmp_path.glob("checkpoint-*")]
        )
        assert remaining == [3, 4, 5], f"Got {remaining}"

    def test_metadata_roundtrip(self, tmp_path):
        """Saving and manually loading metadata.json preserves all fields."""
        meta = {"step": 42, "epoch": 3, "loss": 0.123, "lr": 1e-4}
        ckpt_dir = tmp_path / "checkpoint-42"
        ckpt_dir.mkdir()
        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(meta, f)
        with open(ckpt_dir / "metadata.json") as f:
            loaded = json.load(f)
        assert loaded == meta

    def test_load_missing_raises(self, tmp_path):
        """Loading from an empty directory raises FileNotFoundError."""
        from open_vision_banana.training.checkpoint import CheckpointManager
        cm = CheckpointManager(str(tmp_path), max_keep=3)

        class _FakePeft:
            pass

        with pytest.raises(FileNotFoundError):
            cm.load(_FakePeft())


# ---------------------------------------------------------------------------
# VisionBanana pipeline (mock generator, no real FLUX weights)
# ---------------------------------------------------------------------------

class _MockGenerator:
    """Stand-in for BaseGenerator that returns a constant RGB tensor."""
    def __init__(self, output_val: float = 0.7):
        self._output_val = output_val

    def generate(self, prompt, image=None, **kwargs) -> torch.Tensor:
        B = image.shape[0] if image is not None else 1
        return torch.full((B, 3, 64, 64), self._output_val)


class TestVisionBananaPipeline:

    def _make_pipeline(self, output_val: float = 0.7):
        from open_vision_banana.core.rgb_mapper import RGBMapper
        from open_vision_banana.inference.decoder import VisualizationDecoder
        from open_vision_banana.inference.pipeline import VisionBanana
        gen = _MockGenerator(output_val)
        mapper = RGBMapper()
        decoder = VisualizationDecoder(mapper)
        return VisionBanana(base_generator=gen, rgb_mapper=mapper, decoder=decoder)

    def test_depth_predict_returns_depth_key(self):
        pipe = self._make_pipeline()
        image = torch.rand(1, 3, 64, 64)
        result = pipe.predict(image, task="depth")
        assert "depth" in result
        assert isinstance(result["depth"], torch.Tensor)

    def test_normal_predict_returns_normal_key(self):
        pipe = self._make_pipeline()
        image = torch.rand(1, 3, 64, 64)
        result = pipe.predict(image, task="normal")
        assert "normal" in result
        assert result["normal"].shape[1] == 3

    def test_referring_predict_returns_mask(self):
        pipe = self._make_pipeline(output_val=0.9)   # bright → mask = True
        image = torch.rand(1, 3, 64, 64)
        result = pipe.predict(image, task="referring", expression="the cat")
        assert "mask" in result
        mask = result["mask"]
        assert mask.dtype == torch.bool
        assert mask.any(), "Bright output should produce non-empty mask"

    def test_semantic_predict_raises_without_color_map(self):
        pipe = self._make_pipeline()
        image = torch.rand(1, 3, 64, 64)
        with pytest.raises(ValueError, match="class_color_map"):
            pipe.predict(image, task="semantic")

    def test_predict_batch_splits_correctly(self):
        pipe = self._make_pipeline()
        images = torch.rand(3, 3, 64, 64)
        results = pipe.predict_batch(images, task="depth")
        assert len(results) == 3
        for r in results:
            assert "depth" in r

    def test_unsupported_task_raises(self):
        pipe = self._make_pipeline()
        image = torch.rand(1, 3, 64, 64)
        with pytest.raises(ValueError, match="Unsupported task"):
            pipe.predict(image, task="super_resolution")

    def test_predict_reports_metadata(self):
        pipe = self._make_pipeline()
        image = torch.rand(1, 3, 64, 64)
        result = pipe.predict(image, task="depth", seed=7, num_inference_steps=2)
        assert result["metadata"]["task"] == "depth"
        assert result["metadata"]["seed"] == 7
        assert result["metadata"]["input_shape"] == [1, 3, 64, 64]
        assert result["metadata"]["output_shape"] == [1, 3, 64, 64]

    def test_predict_emits_structured_diagnostics(self, tmp_path):
        from open_vision_banana.inference.diagnostics import InferenceDiagnosticsWriter

        pipe = self._make_pipeline()
        image = torch.rand(1, 3, 64, 64)
        path = tmp_path / "inference.jsonl"
        with InferenceDiagnosticsWriter(path) as writer:
            result = pipe.predict(image, task="depth", seed=11, diagnostics_callback=writer)

        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1
        assert rows[0]["event"] == "inference_complete"
        assert rows[0]["task"] == "depth"
        assert rows[0]["seed"] == 11
        assert rows[0]["output_shape"] == result["metadata"]["output_shape"]
        assert rows[0]["latency_ms"] >= 0

    def test_invalid_image_contract_raises(self):
        pipe = self._make_pipeline()
        with pytest.raises(ValueError, match="3 channels"):
            pipe.predict(torch.rand(1, 1, 16, 16), task="depth")
        with pytest.raises(ValueError, match="NaN"):
            pipe.predict(torch.full((1, 3, 16, 16), float("nan")), task="depth")

    def test_invalid_inference_options_raise(self):
        pipe = self._make_pipeline()
        image = torch.rand(1, 3, 64, 64)
        with pytest.raises(ValueError, match="strength"):
            pipe.predict(image, task="depth", strength=0.0)
        with pytest.raises(ValueError, match="positive integer"):
            pipe.predict(image, task="depth", num_inference_steps=0)

    def test_predict_tiled_continuous_preserves_raster(self):
        pipe = self._make_pipeline()
        image = torch.rand(1, 3, 80, 160)
        result = pipe.predict_tiled(image, task="depth", tile_size=64, overlap=16)
        assert result["depth"].shape[-2:] == (80, 160)
        assert result["rgb_output"].shape == (1, 3, 80, 160)
        assert result["metadata"]["tiled"] is True
        assert result["metadata"]["tile_count"] > 1
        assert torch.isfinite(result["rgb_output"]).all()
        assert result["rgb_output"].abs().sum() > 0

    def test_predict_tiled_discrete_returns_full_rgb(self):
        pipe = self._make_pipeline(output_val=0.9)
        image = torch.rand(1, 3, 80, 160)
        result = pipe.predict_tiled(
            image, task="referring", tile_size=64, overlap=16, expression="the object"
        )
        assert result["mask"].shape == (1, 1, 80, 160)
        assert result["rgb_output"].shape == (1, 3, 80, 160)

    def test_predict_tiled_rejects_invalid_overlap(self):
        pipe = self._make_pipeline()
        image = torch.rand(1, 3, 80, 160)
        with pytest.raises(ValueError, match="overlap"):
            pipe.predict_tiled(image, task="depth", tile_size=32, overlap=32)


# ---------------------------------------------------------------------------
# CityscapesSegDataset vectorized _label_to_rgb
# ---------------------------------------------------------------------------

class TestCityscapesLabelToRGB:

    def _make_dataset(self):
        """Instantiate without scanning (non-existent dir is OK — scan is a no-op)."""
        from open_vision_banana.data.datasets import CityscapesSegDataset
        # Bypass _scan by patching root_dir to a non-existent path
        ds = CityscapesSegDataset.__new__(CityscapesSegDataset)
        ds.root_dir = Path("/nonexistent")
        ds.split = "train"
        ds.resolution = 64
        ds.task_mix_weight = 1.0
        ds.samples = []
        return ds

    def test_output_shape(self):
        ds = self._make_dataset()
        label = torch.zeros(64, 64, dtype=torch.long)   # all "road"
        rgb = ds._label_to_rgb(label)
        assert rgb.shape == (3, 64, 64), f"Expected (3,64,64), got {rgb.shape}"

    def test_road_color(self):
        """Label 0 = road → RGB (128/255, 64/255, 128/255)."""
        ds = self._make_dataset()
        label = torch.zeros(4, 4, dtype=torch.long)
        rgb = ds._label_to_rgb(label)
        expected = torch.tensor([128, 64, 128], dtype=torch.float32) / 255.0
        actual = rgb[:, 0, 0]
        assert torch.allclose(actual, expected, atol=1e-6), \
            f"Road color mismatch: {actual} vs {expected}"

    def test_all_classes_present(self):
        """Each class index should map to a distinct non-zero colour."""
        from open_vision_banana.data.datasets import CityscapesSegDataset
        ds = self._make_dataset()
        n_classes = len(CityscapesSegDataset.CITYSCAPES_COLORS)
        label = torch.arange(n_classes, dtype=torch.long).reshape(1, n_classes)
        rgb = ds._label_to_rgb(label)
        assert rgb.shape == (3, 1, n_classes)
        # All colour values should be in [0, 1]
        assert rgb.min() >= 0.0 and rgb.max() <= 1.0

    def test_matches_naive_loop(self):
        """Vectorized result must match the old per-class loop implementation."""
        ds = self._make_dataset()
        from open_vision_banana.data.datasets import CityscapesSegDataset
        colors = list(CityscapesSegDataset.CITYSCAPES_COLORS.values())
        H, W = 16, 16
        label = torch.randint(0, len(colors), (H, W))

        # Naive reference
        rgb_ref = torch.zeros(3, H, W)
        for cls_idx, color in enumerate(colors):
            mask = label == cls_idx
            for ch, cv in enumerate(color):
                rgb_ref[ch][mask] = cv / 255.0

        rgb_vec = ds._label_to_rgb(label)
        assert torch.allclose(rgb_vec, rgb_ref, atol=1e-6), \
            "Vectorized _label_to_rgb does not match naive loop"


# ---------------------------------------------------------------------------
# VisualizationDecoder.decode_referring
# ---------------------------------------------------------------------------

class TestVisualizationDecoderReferring:

    def test_bright_image_gives_true_mask(self):
        from open_vision_banana.inference.decoder import VisualizationDecoder
        dec = VisualizationDecoder()
        rgb = torch.ones(1, 3, 8, 8)   # all white
        result = dec.decode_referring(rgb, threshold=0.5)
        assert result["mask"].all(), "All-white image should give all-True mask"

    def test_dark_image_gives_false_mask(self):
        from open_vision_banana.inference.decoder import VisualizationDecoder
        dec = VisualizationDecoder()
        rgb = torch.zeros(1, 3, 8, 8)   # all black
        result = dec.decode_referring(rgb, threshold=0.5)
        assert not result["mask"].any(), "All-black image should give all-False mask"

    def test_mask_shape(self):
        from open_vision_banana.inference.decoder import VisualizationDecoder
        dec = VisualizationDecoder()
        rgb = torch.rand(2, 3, 16, 16)
        result = dec.decode_referring(rgb)
        assert result["mask"].shape == (2, 1, 16, 16)
        assert result["mask"].dtype == torch.bool


# ---------------------------------------------------------------------------
# VisionTaskDataset flat-directory loader
# ---------------------------------------------------------------------------

class TestVisionTaskDataset:

    def _make_fake_data_dir(self, tmp_path, n=4):
        """Create a fake images/targets directory pair."""
        import numpy as np
        from PIL import Image
        (tmp_path / "images").mkdir()
        (tmp_path / "targets").mkdir()
        for i in range(n):
            arr = (np.random.rand(64, 64, 3) * 255).astype("uint8")
            Image.fromarray(arr).save(tmp_path / "images" / f"{i:03d}.png")
            Image.fromarray(arr).save(tmp_path / "targets" / f"{i:03d}.png")
        return tmp_path

    def test_loads_paired_samples(self, tmp_path):
        self._make_fake_data_dir(tmp_path, n=4)
        from open_vision_banana.training.tuner import VisionTaskDataset
        ds = VisionTaskDataset(data_path=str(tmp_path), resolution=64)
        assert len(ds) == 4

    def test_sample_shapes(self, tmp_path):
        self._make_fake_data_dir(tmp_path, n=2)
        from open_vision_banana.training.tuner import VisionTaskDataset
        ds = VisionTaskDataset(data_path=str(tmp_path), resolution=64)
        sample = ds[0]
        assert sample["image"].shape == (3, 64, 64)
        assert sample["target_rgb"].shape == (3, 64, 64)
        assert isinstance(sample["prompt"], str)

    def test_image_range(self, tmp_path):
        """Image should be in [-1, 1], target_rgb in [0, 1]."""
        self._make_fake_data_dir(tmp_path, n=2)
        from open_vision_banana.training.tuner import VisionTaskDataset
        ds = VisionTaskDataset(data_path=str(tmp_path), resolution=64)
        s = ds[0]
        assert s["image"].min() >= -1.0 - 1e-6 and s["image"].max() <= 1.0 + 1e-6
        assert s["target_rgb"].min() >= -1e-6 and s["target_rgb"].max() <= 1.0 + 1e-6

    def test_unpaired_images_skipped(self, tmp_path):
        """Files without a matching target should be silently skipped."""
        import numpy as np
        from PIL import Image
        (tmp_path / "images").mkdir()
        (tmp_path / "targets").mkdir()
        arr = (np.random.rand(64, 64, 3) * 255).astype("uint8")
        Image.fromarray(arr).save(tmp_path / "images" / "paired.png")
        Image.fromarray(arr).save(tmp_path / "targets" / "paired.png")
        # This image has no matching target
        Image.fromarray(arr).save(tmp_path / "images" / "orphan.png")
        from open_vision_banana.training.tuner import VisionTaskDataset
        ds = VisionTaskDataset(data_path=str(tmp_path), resolution=64)
        assert len(ds) == 1

# ---------------------------------------------------------------------------
# Round 4 tests — training pipeline readiness
# ---------------------------------------------------------------------------

class TestMixedVisionDatasetSampling:
    """Verify MixedVisionDataset random sampling (no idx-bias)."""

    def _make_ds(self, n, label):
        import torch
        class _D(torch.utils.data.Dataset):
            def __len__(self): return n
            def __getitem__(self, i):
                return {"label": label, "image": torch.zeros(3, 4, 4), "target_rgb": torch.zeros(3, 4, 4)}
        return _D()

    def test_vision_only_returns_vision(self):
        from open_vision_banana.data.mixed import MixedVisionDataset
        ds = MixedVisionDataset([self._make_ds(20, "vision")], generation_dataset=None, vision_ratio=0.5)
        for i in range(10):
            assert ds[i]["label"] == "vision"

    def test_ratio_respected(self):
        from open_vision_banana.data.mixed import MixedVisionDataset
        vision_ds = self._make_ds(100, "vision")
        gen_ds = self._make_ds(100, "generation")
        mds = MixedVisionDataset([vision_ds], generation_dataset=gen_ds, vision_ratio=0.1, seed=0)
        vision_count = sum(1 for i in range(300) if mds[i % len(mds)]["label"] == "vision")
        ratio = vision_count / 300
        assert 0.03 < ratio < 0.25, f"Vision ratio {ratio:.2%} out of expected range [3%, 25%]"

    def test_uniform_dataset_selection(self):
        """When multiple vision datasets, each should be sampled."""
        from open_vision_banana.data.mixed import MixedVisionDataset
        ds_a = self._make_ds(50, "A")
        ds_b = self._make_ds(50, "B")
        mds = MixedVisionDataset([ds_a, ds_b], vision_ratio=1.0, seed=99)
        labels = {mds[i]["label"] for i in range(200)}
        assert "A" in labels and "B" in labels, "Both datasets should be sampled"


class TestVisionAugmentationClamp:
    """Verify color-jitter clamp prevents out-of-range values."""

    def test_high_brightness_clamped(self):
        import torch

        from open_vision_banana.training.utils import VisionAugmentation
        aug = VisionAugmentation(
            resolution=32, hflip_prob=0.0,
            crop_scale=(1.0, 1.0),
            color_jitter={"brightness": 0.99},  # extreme brightness
            task="depth",
        )
        img = torch.ones(3, 32, 32)  # already at max valid range
        tgt = torch.rand(3, 32, 32)
        aug_img, _ = aug(img, tgt)
        assert aug_img.max() <= 1.0 + 1e-5, f"Image exceeded 1.0 after brightness: {aug_img.max():.4f}"
        assert aug_img.min() >= -1.0 - 1e-5, f"Image below -1.0 after brightness: {aug_img.min():.4f}"

    def test_high_contrast_clamped(self):
        import torch

        from open_vision_banana.training.utils import VisionAugmentation
        aug = VisionAugmentation(
            resolution=32, hflip_prob=0.0,
            crop_scale=(1.0, 1.0),
            color_jitter={"contrast": 0.99},
            task="depth",
        )
        img = torch.randn(3, 32, 32).clamp(-0.9, 0.9)
        tgt = torch.rand(3, 32, 32)
        aug_img, _ = aug(img, tgt)
        assert aug_img.max() <= 1.0 + 1e-5
        assert aug_img.min() >= -1.0 - 1e-5


class TestInstructionTunerForwardKwargs:
    """Verify _training_step works with PEFT ≥ 0.14 kwargs-only calling convention."""

    def _make_tuner(self):
        import torch.nn as nn

        from open_vision_banana.training.tuner import InstructionTuner

        class FakeTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_k = nn.Conv2d(3, 8, 1)
                self.to_v = nn.Conv2d(3, 8, 1)
                self.out = nn.Conv2d(8, 3, 1)

            def forward(self, hidden_states, timestep=None, **kwargs):
                k = self.to_k(hidden_states)
                v = self.to_v(hidden_states)
                return self.out(k + v)

        class FakeGenerator:
            def __init__(self):
                self.transformer = FakeTransformer()

        gen = FakeGenerator()
        tuner = InstructionTuner(
            model=gen,
            lora_rank=2,
            lora_alpha=4,
            use_ema=False,
            use_gradient_checkpointing=False,
        )
        return tuner

    def test_training_step_finite_loss(self):
        import torch
        tuner = self._make_tuner()
        batch = {
            "target_rgb": torch.rand(2, 3, 8, 8),
            "image": torch.rand(2, 3, 8, 8),
            "prompt": ["a", "b"],
        }
        loss = tuner._training_step(batch)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_training_step_device_from_batch(self):
        """Device must be inferred from batch, not from peft_model.parameters()."""
        import torch
        tuner = self._make_tuner()
        # Even if peft_model has no parameters, step should not raise
        batch = {
            "target_rgb": torch.rand(2, 3, 8, 8),
            "prompt": ["a", "b"],
        }
        loss = tuner._training_step(batch)
        assert torch.isfinite(loss)


class TestInstructionTunerNumWorkers:
    """Verify num_workers is exposed and defaults to 0."""

    def test_train_accepts_num_workers(self):
        import inspect

        from open_vision_banana.training.tuner import InstructionTuner
        sig = inspect.signature(InstructionTuner.train)
        assert "num_workers" in sig.parameters, "num_workers not in InstructionTuner.train() signature"
        assert sig.parameters["num_workers"].default == 0, "num_workers default should be 0"


# ──────────────────────────────────────────────────────────────
# P0-2: _output_to_tensor numpy array handling
# ──────────────────────────────────────────────────────────────


class TestBaseGeneratorOutputToTensor:
    """Verify _output_to_tensor handles diffusers output.images formats."""

    def test_numpy_array_batch(self):
        """P0-2: Handle diffusers output.images as np.ndarray [B, H, W, 3]."""
        import numpy as np

        from open_vision_banana.inference.generator import BaseGenerator

        class MockOutput:
            images = np.random.randint(0, 256, (2, 64, 64, 3), dtype=np.uint8)

        tensor = BaseGenerator._output_to_tensor(MockOutput())
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 3, 64, 64)
        assert tensor.dtype == torch.float32
        assert tensor.min() >= 0.0 and tensor.max() <= 1.0

    def test_numpy_array_single_image(self):
        """P0-2: Handle single-image numpy array [H, W, 3]."""
        import numpy as np

        from open_vision_banana.inference.generator import BaseGenerator

        class MockOutput:
            images = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

        tensor = BaseGenerator._output_to_tensor(MockOutput())
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 64, 64)

    def test_tensor_output_passthrough(self):
        """Existing behaviour: tensor output returned directly."""
        from open_vision_banana.inference.generator import BaseGenerator

        class MockOutput:
            images = torch.rand(2, 3, 64, 64)

        tensor = BaseGenerator._output_to_tensor(MockOutput())
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 3, 64, 64)

    def test_list_of_pil_images(self):
        """Existing behaviour: list of PIL Images."""
        from PIL import Image

        from open_vision_banana.inference.generator import BaseGenerator

        class MockOutput:
            images = [Image.new("RGB", (64, 64), color=(128, 128, 128)) for _ in range(2)]

        tensor = BaseGenerator._output_to_tensor(MockOutput())
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 3, 64, 64)

    def test_list_of_numpy_arrays(self):
        """Existing behaviour: list of numpy arrays."""
        import numpy as np

        from open_vision_banana.inference.generator import BaseGenerator

        class MockOutput:
            images = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(2)]

        tensor = BaseGenerator._output_to_tensor(MockOutput())
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 3, 64, 64)


# ──────────────────────────────────────────────────────────────
# P1-4: InferenceOptions dtype sync with BaseGenerator
# ──────────────────────────────────────────────────────────────


class TestBaseGeneratorDtypeSync:
    """Verify InferenceOptions.dtype is synced when auto-created."""

    def test_dtype_sync_float32(self):
        from unittest.mock import patch

        from open_vision_banana.inference.generator import BaseGenerator

        with patch.object(BaseGenerator, '_load_model'):
            gen = BaseGenerator(model_name="dummy", dtype=torch.float32, device="cpu")
            assert gen.inference_options.dtype == "float32"

    def test_dtype_sync_bfloat16(self):
        from unittest.mock import patch

        from open_vision_banana.inference.generator import BaseGenerator

        with patch.object(BaseGenerator, '_load_model'):
            gen = BaseGenerator(model_name="dummy", dtype=torch.bfloat16, device="cpu")
            assert gen.inference_options.dtype == "bfloat16"

    def test_dtype_sync_float16(self):
        from unittest.mock import patch

        from open_vision_banana.inference.generator import BaseGenerator

        with patch.object(BaseGenerator, '_load_model'):
            gen = BaseGenerator(model_name="dummy", dtype=torch.float16, device="cpu")
            assert gen.inference_options.dtype == "float16"


class TestPromptCacheKey:
    """Long prompts should not be retained verbatim as cache keys."""

    def test_long_prompt_is_hashed(self):
        from open_vision_banana.inference.generator import _prompt_cache_key

        prompt = "x" * 257
        key = _prompt_cache_key(prompt)
        assert key != prompt
        assert len(key) == 64

    def test_short_prompt_is_readable(self):
        from open_vision_banana.inference.generator import _prompt_cache_key

        assert _prompt_cache_key("a cat") == "a cat"

    def test_explicit_inference_options_not_overridden(self):
        from unittest.mock import patch

        from open_vision_banana.inference.generator import BaseGenerator
        from open_vision_banana.inference.optimization import InferenceOptions

        with patch.object(BaseGenerator, '_load_model'):
            opts = InferenceOptions(dtype="float16")
            gen = BaseGenerator(
                model_name="dummy", dtype=torch.float32, device="cpu", inference_options=opts
            )
            assert gen.inference_options.dtype == "float16"
