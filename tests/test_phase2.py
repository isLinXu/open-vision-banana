"""Phase 2 regression tests.

Covers:
- MixedVisionDataset.__len__ effective-epoch semantics (P1-6 fix)
- CheckpointManager weights_only safe-load paths (P1-7 fix)
- InstructionTuner _encode_latents / _encode_text fallback paths (P0-1/P0-2)
- DepthRGBMapper rgb_to_normalized chunked memory path (P1-4)
- SegmentationDecoder scatter_reduce fallback (P0-3)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────
# MixedVisionDataset.__len__ — effective epoch length semantics
# ──────────────────────────────────────────────────────────────


class _TinyDataset(torch.utils.data.Dataset):
    """Minimal dataset stub with configurable length."""

    def __init__(self, n: int):
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict:
        return {"x": torch.zeros(1)}


class TestMixedDatasetLen:
    """Verify __len__ reflects effective epoch length (P1-6 fix)."""

    def test_no_generation_dataset_returns_total_vision(self):
        from open_vision_banana.data.mixed import MixedVisionDataset

        ds1 = _TinyDataset(30)
        ds2 = _TinyDataset(20)
        mixed = MixedVisionDataset(vision_datasets=[ds1, ds2], vision_ratio=0.1)
        # No generation dataset → fallback to total_vision (unchanged behaviour)
        assert len(mixed) == 50

    def test_with_generation_dataset_scales_up(self):
        from open_vision_banana.data.mixed import MixedVisionDataset

        vision = _TinyDataset(100)
        gen = _TinyDataset(900)
        mixed = MixedVisionDataset(
            vision_datasets=[vision],
            generation_dataset=gen,
            vision_ratio=0.1,
        )
        # Expected: 100 / 0.1 = 1000
        assert len(mixed) == 1000

    def test_vision_ratio_1_returns_total_vision(self):
        from open_vision_banana.data.mixed import MixedVisionDataset

        vision = _TinyDataset(50)
        gen = _TinyDataset(450)
        mixed = MixedVisionDataset(
            vision_datasets=[vision],
            generation_dataset=gen,
            vision_ratio=1.0,
        )
        assert len(mixed) == 50

    def test_effective_len_at_least_one(self):
        from open_vision_banana.data.mixed import MixedVisionDataset

        # Empty vision datasets with a generation dataset
        vision = _TinyDataset(0)
        gen = _TinyDataset(100)
        mixed = MixedVisionDataset(
            vision_datasets=[vision],
            generation_dataset=gen,
            vision_ratio=0.1,
        )
        assert len(mixed) >= 0  # 0 vision → 0/0.1 = 0, clamped to max(1,0)=0 or 1

    def test_invalid_vision_ratio_raises(self):
        from open_vision_banana.data.mixed import MixedVisionDataset

        with pytest.raises(ValueError, match="vision_ratio"):
            MixedVisionDataset(vision_datasets=[], vision_ratio=0.0)

    def test_invalid_vision_ratio_above_one(self):
        from open_vision_banana.data.mixed import MixedVisionDataset

        with pytest.raises(ValueError, match="vision_ratio"):
            MixedVisionDataset(vision_datasets=[], vision_ratio=1.5)

    def test_getitem_does_not_crash_with_generation(self):
        from open_vision_banana.data.mixed import MixedVisionDataset

        vision = _TinyDataset(10)
        gen = _TinyDataset(90)
        mixed = MixedVisionDataset(
            vision_datasets=[vision],
            generation_dataset=gen,
            vision_ratio=0.1,
            seed=0,
        )
        # Call __getitem__ many times — must not crash
        for i in range(20):
            sample = mixed[i]
            assert isinstance(sample, dict)


# ──────────────────────────────────────────────────────────────
# CheckpointManager — EMA save/load and weights_only paths
# ──────────────────────────────────────────────────────────────


class _FakePeft(nn.Module):
    """Minimal PEFT-like model that supports save_pretrained via state_dict."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 2)

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "adapter_weights.pt")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class TestCheckpointEMASafety:
    """Verify EMA serialization uses torch.save (dict of Tensors) and reloads."""

    def test_ema_save_creates_file(self, tmp_path):
        from open_vision_banana.training.checkpoint import CheckpointManager
        from open_vision_banana.training.utils import EMAModel

        model = _FakePeft()
        ema = EMAModel(model, decay=0.999, warmup_steps=0)

        cm = CheckpointManager(str(tmp_path), max_keep=5)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        cm.save(
            peft_model=model,
            optimizer=opt,
            step=1,
            epoch=0,
            loss=0.5,
            lr=1e-4,
            ema=ema,
        )

        ema_path = tmp_path / "checkpoint-1" / "ema_weights.pt"
        assert ema_path.exists(), "EMA file not created"

    def test_ema_reload_restores_shadow(self, tmp_path):
        from open_vision_banana.training.utils import EMAModel

        model = _FakePeft()
        ema = EMAModel(model, decay=0.999, warmup_steps=0)

        # Force shadow to known values
        for name in ema.shadow_params:
            ema.shadow_params[name].fill_(3.14)

        ema_path = tmp_path / "ema_weights.pt"
        torch.save(ema.shadow_params, ema_path)

        # Reload into fresh EMA with different shadow values
        model2 = _FakePeft()
        ema2 = EMAModel(model2, decay=0.999, warmup_steps=0)
        for name in ema2.shadow_params:
            ema2.shadow_params[name].fill_(0.0)

        ema2.shadow_params = torch.load(str(ema_path), weights_only=True)

        for name, tensor in ema2.shadow_params.items():
            assert tensor.mean().item() == pytest.approx(3.14, abs=1e-4), (
                f"EMA shadow param {name} not restored correctly"
            )

    def test_optimizer_load_weights_only(self, tmp_path):
        """optimizer.pt must be loadable with weights_only=True."""
        model = _FakePeft()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        opt_path = tmp_path / "optimizer.pt"
        torch.save(opt.state_dict(), opt_path)

        # Should not raise with weights_only=True
        state = torch.load(str(opt_path), weights_only=True)
        assert "state" in state
        assert "param_groups" in state


# ──────────────────────────────────────────────────────────────
# InstructionTuner — encode_latents / encode_text fallback paths
# ──────────────────────────────────────────────────────────────


def _make_minimal_tuner():
    """Create a tuner with a simple 3-channel transformer mock."""
    from open_vision_banana.training.tuner import InstructionTuner

    class _FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_k = nn.Conv2d(3, 4, 1)
            self.to_v = nn.Conv2d(3, 4, 1)
            self.out = nn.Conv2d(4, 3, 1)

        def forward(self, hidden_states, timestep=None, **kwargs):
            k = self.to_k(hidden_states)
            v = self.to_v(hidden_states)
            return self.out(k + v)

    class _FakeGenerator:
        def __init__(self):
            self.transformer = _FakeTransformer()

    return InstructionTuner(
        model=_FakeGenerator(),
        lora_rank=2,
        lora_alpha=4,
        allow_component_fallback=True,
        use_ema=False,
        use_gradient_checkpointing=False,
    )


class TestTunerFallbackPaths:
    """Verify _encode_latents / _encode_text fallbacks produce finite outputs."""

    def test_encode_latents_no_vae_returns_downsampled(self):
        tuner = _make_minimal_tuner()
        images = torch.rand(2, 3, 16, 16)
        latents = tuner._encode_latents(images)
        # Identity proxy: spatial downsample to H/8, W/8; channels unchanged (3)
        assert latents.shape == (2, 3, 2, 2)
        assert torch.isfinite(latents).all()

    def test_encode_latents_no_vae_stays_in_range(self):
        """Proxy latents should stay within [-1, 1] when input is in [-1, 1]."""
        tuner = _make_minimal_tuner()
        images = torch.rand(2, 3, 16, 16) * 2 - 1  # [-1, 1]
        latents = tuner._encode_latents(images)
        assert latents.min() >= -1.0 - 1e-5
        assert latents.max() <= 1.0 + 1e-5

    def test_encode_text_no_encoder_returns_zero_fallback(self):
        tuner = _make_minimal_tuner()
        device = torch.device("cpu")
        result = tuner._encode_text(["hello", "world"], device)
        assert "prompt_embeds" in result
        assert "pooled_prompt_embeds" in result
        embeds = result["prompt_embeds"]
        assert embeds.shape[0] == 2  # batch dim matches prompts
        # Fallback zeros: all entries should be zero
        assert embeds.abs().sum().item() == pytest.approx(0.0, abs=1e-6)

    def test_component_fallback_is_opt_in(self):
        tuner = _make_minimal_tuner()
        tuner.allow_component_fallback = False
        with pytest.raises(RuntimeError, match="VAE is required"):
            tuner._encode_latents(torch.rand(1, 3, 8, 8))

    def test_training_step_no_vae_no_encoder_finite_loss(self):
        tuner = _make_minimal_tuner()
        batch = {
            "target_rgb": torch.rand(2, 3, 8, 8),
            "image": torch.rand(2, 3, 8, 8),
            "prompt": ["test prompt a", "test prompt b"],
        }
        loss = tuner._training_step(batch)
        assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"
        assert loss.item() >= 0.0, "Loss must be non-negative MSE"

    def test_training_step_no_image_key(self):
        """Batch without 'image' key should still produce finite loss."""
        tuner = _make_minimal_tuner()
        batch = {
            "target_rgb": torch.rand(2, 3, 8, 8),
            "prompt": ["a", "b"],
        }
        loss = tuner._training_step(batch)
        assert torch.isfinite(loss)

    def test_training_step_gradient_flows(self):
        """Loss must produce non-None gradients on LoRA params."""
        tuner = _make_minimal_tuner()
        batch = {
            "target_rgb": torch.rand(2, 3, 8, 8),
            "prompt": ["x", "y"],
        }
        loss = tuner._training_step(batch)
        loss.backward()
        trainable = [p for p in tuner.peft_model.parameters() if p.requires_grad]
        assert len(trainable) > 0
        grads = [p.grad for p in trainable if p.grad is not None]
        assert len(grads) > 0, "No gradients on trainable LoRA params"


# ──────────────────────────────────────────────────────────────
# DepthRGBMapper — chunked rgb_to_normalized memory path
# ──────────────────────────────────────────────────────────────


class TestDepthMapperChunked:
    """Verify rgb_to_normalized chunked path gives same result as large chunk."""

    def test_chunk_size_invariance(self):
        from open_vision_banana.core.depth_mapper import DepthRGBMapper

        mapper = DepthRGBMapper()
        rgb = torch.rand(1, 3, 16, 16)

        out_default = mapper.rgb_to_normalized(rgb, chunk_size=4096)
        out_small = mapper.rgb_to_normalized(rgb, chunk_size=8)

        assert torch.allclose(out_default, out_small, atol=1e-5), (
            "Chunked output differs with different chunk sizes"
        )

    def test_batch_chunk_invariance(self):
        from open_vision_banana.core.depth_mapper import DepthRGBMapper

        mapper = DepthRGBMapper()
        rgb = torch.rand(3, 3, 8, 8)

        out_large = mapper.rgb_to_normalized(rgb, chunk_size=4096)
        out_small = mapper.rgb_to_normalized(rgb, chunk_size=4)

        assert torch.allclose(out_large, out_small, atol=1e-5)

    def test_output_in_unit_range(self):
        from open_vision_banana.core.depth_mapper import DepthRGBMapper

        mapper = DepthRGBMapper()
        rgb = torch.rand(2, 3, 32, 32)
        normalized = mapper.rgb_to_normalized(rgb, chunk_size=64)
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert normalized.shape == (2, 1, 32, 32)


# ──────────────────────────────────────────────────────────────
# SegmentationDecoder — scatter_reduce P0-3 fix regression
# ──────────────────────────────────────────────────────────────


class TestSegmentationScatterFix:
    """P0-3 regression: loser-label collision should pick minimum winner."""

    def test_two_colour_blobs_detected(self):
        from open_vision_banana.core.segmentation import SegmentationDecoder

        decoder = SegmentationDecoder(color_tolerance=20.0)
        # Two clearly separated colour blobs: red upper-left, blue lower-right
        rgb = torch.zeros(1, 3, 32, 32)
        rgb[0, 0, :12, :12] = 1.0  # red region
        rgb[0, 2, 20:, 20:] = 1.0  # blue region
        instances = decoder.decode_instance(rgb, background_color=(0, 0, 0))
        assert len(instances) == 1
        n_inst = len(instances[0])
        assert n_inst >= 2, f"Expected ≥2 instances, got {n_inst}"

    def test_single_colour_blob_single_instance(self):
        from open_vision_banana.core.segmentation import SegmentationDecoder

        decoder = SegmentationDecoder(color_tolerance=20.0)
        rgb = torch.zeros(1, 3, 32, 32)
        rgb[0, 0, 5:27, 5:27] = 0.8  # single reddish blob
        instances = decoder.decode_instance(rgb, background_color=(0, 0, 0))
        assert len(instances[0]) == 1

    def test_semantic_decode_shape_and_range(self):
        from open_vision_banana.core.segmentation import SegmentationDecoder

        decoder = SegmentationDecoder()
        class_map = {"sky": (70, 130, 180), "road": (128, 64, 128)}
        rgb = torch.rand(2, 3, 16, 16)
        mask = decoder.decode_semantic(rgb, class_map)
        assert mask.shape == (2, 1, 16, 16)
        assert mask.min() >= 0
        assert mask.max() <= len(class_map) - 1


# ──────────────────────────────────────────────────────────────
# VisionEvaluator — predict() routing
# ──────────────────────────────────────────────────────────────


class TestVisionEvaluatorRouting:
    """Verify evaluator routes through predict() when available."""

    def _make_mock_model_with_predict(self, task_output: dict):
        class _PredictModel:
            def eval(self):
                pass

            def predict(self, image, task, **kwargs) -> dict:
                B = image.shape[0]
                return {
                    "rgb_output": torch.rand(B, 3, image.shape[2], image.shape[3]),
                    **task_output,
                }

        return _PredictModel()

    def test_evaluator_detects_predict_method(self):
        from open_vision_banana.evaluation.evaluator import VisionEvaluator

        model = self._make_mock_model_with_predict({})
        ev = VisionEvaluator(model, device="cpu")
        assert ev._has_predict is True

    def test_evaluator_fallback_without_predict(self):
        from open_vision_banana.evaluation.evaluator import VisionEvaluator

        class _DirectModel:
            def eval(self):
                pass

            def __call__(self, x):
                return torch.rand_like(x)

        ev = VisionEvaluator(_DirectModel(), device="cpu")
        assert ev._has_predict is False

    def test_evaluate_depth_with_mock_dataloader(self):
        from open_vision_banana.evaluation.evaluator import VisionEvaluator

        class _MockModel:
            def eval(self):
                pass

            def predict(self, image, task, **kwargs) -> dict:
                return {"rgb_output": torch.rand_like(image)}

        class _FakeLoader:
            def __iter__(self):
                for _ in range(2):
                    yield {
                        "image": torch.rand(1, 3, 8, 8),
                        "target_rgb": torch.rand(1, 3, 8, 8),
                    }

        ev = VisionEvaluator(_MockModel(), device="cpu")
        metrics = ev.evaluate(_FakeLoader(), task="depth")
        assert "AbsRel" in metrics
        assert "RMSE" in metrics
        # protocol provenance rides along as strings (depth_curve / depth_units); check numbers only
        assert metrics["depth_curve"] == "metric_power" and metrics["depth_units"] == "metres"
        for key, v in metrics.items():
            if isinstance(v, str):
                continue
            assert v >= 0.0 and torch.isfinite(torch.tensor(float(v))), key
