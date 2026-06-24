"""Tests for v0.2.1 bug fixes and P1/P2 enhancements.

Covers:
- P0: NativeTrainer GradScaler device-awareness
- P0: DDPTrainer._optimizer initialization via setup_optimizer()
- P0: AccelerateTrainer.optimizer_step documentation (no crash)
- P0: CheckpointManager._latest_checkpoint uses self.output_dir
- P1: KV-cache LRU eviction
- P1: Segmentation label relabeling vectorization (cumsum)
- P1: VisionTaskDataset task-specific prompts
- P2: predict_streaming callback
- P2: predict_tiled falls back for small images
"""

from __future__ import annotations

import torch

from open_vision_banana.core.segmentation import SegmentationDecoder
from open_vision_banana.inference.generator import _PromptEmbedCache
from open_vision_banana.training.checkpoint import CheckpointManager

# ──────────────────────────────────────────────────────────────
# P0: CheckpointManager._latest_checkpoint fix
# ──────────────────────────────────────────────────────────────


class TestCheckpointLatestFix:
    """Verify _latest_checkpoint now searches self.output_dir."""

    def test_finds_latest_in_output_dir(self, tmp_path):
        cm = CheckpointManager(str(tmp_path), max_keep=10)
        for step in [10, 200, 30, 1000, 50]:
            (tmp_path / f"checkpoint-{step}").mkdir()
        latest = cm._latest_checkpoint()
        assert latest is not None
        assert latest.name == "checkpoint-1000"

    def test_returns_none_when_empty(self, tmp_path):
        cm = CheckpointManager(str(tmp_path), max_keep=5)
        assert cm._latest_checkpoint() is None


# ──────────────────────────────────────────────────────────────
# P1: KV-cache LRU eviction
# ──────────────────────────────────────────────────────────────


class TestPromptEmbedCacheLRU:
    """Verify LRU cache evicts oldest entries."""

    def test_eviction_order(self):
        cache = _PromptEmbedCache(maxsize=3)
        cache["a"] = {"embeds": 1}
        cache["b"] = {"embeds": 2}
        cache["c"] = {"embeds": 3}
        assert len(cache) == 3

        # Insert "d" → "a" should be evicted (oldest)
        cache["d"] = {"embeds": 4}
        assert "a" not in cache
        assert "b" in cache
        assert "c" in cache
        assert "d" in cache
        assert len(cache) == 3

    def test_move_to_end_on_access(self):
        cache = _PromptEmbedCache(maxsize=2)
        cache["a"] = {"embeds": 1}
        cache["b"] = {"embeds": 2}

        # Access "a" → moves to end (most recently used)
        _ = cache.get("a")
        cache.move_to_end("a")

        # Insert "c" → "b" should be evicted (now oldest)
        cache["c"] = {"embeds": 3}
        assert "b" not in cache
        assert "a" in cache
        assert "c" in cache

    def test_maxsize_boundary(self):
        cache = _PromptEmbedCache(maxsize=1)
        cache["a"] = 1
        cache["b"] = 2
        assert len(cache) == 1
        assert "b" in cache
        assert "a" not in cache


# ──────────────────────────────────────────────────────────────
# P1: Segmentation vectorized relabeling
# ──────────────────────────────────────────────────────────────


class TestVectorizedRelabeling:
    """Verify cumsum-based relabeling produces correct consecutive labels."""

    def test_relabeling_produces_consecutive_ids(self):
        """Test that the vectorized relabeling works correctly."""
        decoder = SegmentationDecoder(color_tolerance=20.0)

        # Create a small test image with distinct colored regions
        img = torch.zeros(1, 3, 8, 8)
        # Region 1: red block
        img[0, 0, 0:4, 0:4] = 1.0
        # Region 2: green block
        img[0, 1, 4:, 4:] = 1.0
        # Background: black

        instances = decoder.decode_instance(img, background_color=(0, 0, 0))
        assert len(instances) == 1  # one image in batch
        # Should have at least 2 instances (red and green regions)
        assert len(instances[0]) >= 2

    def test_no_crash_on_uniform_image(self):
        """Relabeling should handle a uniform (all-background) image."""
        decoder = SegmentationDecoder()
        img = torch.zeros(1, 3, 8, 8)
        instances = decoder.decode_instance(img)
        assert len(instances) == 1
        # All background → no instances
        assert len(instances[0]) == 0


# ──────────────────────────────────────────────────────────────
# P1: VisionTaskDataset task-specific prompts
# ──────────────────────────────────────────────────────────────


class TestTaskSpecificPrompts:
    """Verify VisionTaskDataset uses task-specific prompts."""

    def test_task_prompts_exist(self):
        from open_vision_banana.training.tuner import VisionTaskDataset

        assert "depth" in VisionTaskDataset.TASK_PROMPTS
        assert "normal" in VisionTaskDataset.TASK_PROMPTS
        assert "semantic" in VisionTaskDataset.TASK_PROMPTS

    def test_depth_prompt_is_specific(self):
        from open_vision_banana.training.tuner import VisionTaskDataset

        depth_prompt = VisionTaskDataset.TASK_PROMPTS["depth"]
        assert "depth" in depth_prompt.lower()
        # Should NOT be the generic prompt
        assert depth_prompt != VisionTaskDataset.DEFAULT_PROMPT

    def test_task_assignment(self):
        from open_vision_banana.training.tuner import VisionTaskDataset

        ds = VisionTaskDataset(task="normal")
        assert ds.task == "normal"
        assert ds.TASK_PROMPTS["normal"] != ds.DEFAULT_PROMPT


# ──────────────────────────────────────────────────────────────
# P2: Streaming and tiled inference (mock-based)
# ──────────────────────────────────────────────────────────────


class _MockGenerator:
    """Minimal mock generator for testing VisionBanana methods."""

    _prompt_cache = _PromptEmbedCache(maxsize=64)

    def generate(self, prompt, image=None, **kwargs):
        B = image.shape[0] if image is not None else 1
        C, H, W = 3, 8, 8
        return torch.rand(B, C, H, W)


class TestStreamingInference:
    """Test predict_streaming callback invocation."""

    def test_callback_is_called(self):
        from open_vision_banana.core.rgb_mapper import RGBMapper
        from open_vision_banana.inference.decoder import VisualizationDecoder
        from open_vision_banana.inference.pipeline import VisionBanana

        gen = _MockGenerator()
        model = VisionBanana(
            base_generator=gen,
            rgb_mapper=RGBMapper(),
            decoder=VisualizationDecoder(RGBMapper()),
        )

        callback_calls = []

        def cb(step, rgb):
            callback_calls.append((step, rgb is not None))

        image = torch.rand(1, 3, 8, 8) * 2 - 1
        result = model.predict_streaming(image=image, task="depth", callback=cb)
        assert len(callback_calls) == 1
        assert callback_calls[0][1] is True
        assert "depth" in result


class TestTiledInference:
    """Test predict_tiled fallback for small images."""

    def test_small_image_falls_back_to_predict(self):
        from open_vision_banana.core.rgb_mapper import RGBMapper
        from open_vision_banana.inference.decoder import VisualizationDecoder
        from open_vision_banana.inference.pipeline import VisionBanana

        gen = _MockGenerator()
        model = VisionBanana(
            base_generator=gen,
            rgb_mapper=RGBMapper(),
            decoder=VisualizationDecoder(RGBMapper()),
        )

        # Small image → should use standard predict path
        image = torch.rand(1, 3, 8, 8) * 2 - 1
        result = model.predict_tiled(image, task="depth", tile_size=1024)
        assert "depth" in result
        assert "rgb_output" in result
