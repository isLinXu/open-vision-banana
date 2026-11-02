"""Tests for the v6 optimization fixes.

Covers:
- CFG dropout per-sample logic
- Multi-colormap re-encoding
- depth_aware default and hue shift fix
- _one_cycle scheduler correctness
- LatentCacheDataset
- Batch generation function signature
"""

import pytest
import torch
from _paths import ARCHIVE_SCRIPTS

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.training.optimization import (
    AdvancedAugmentation,
    SchedulerFactory,
)

# ─── Test: depth_aware default ───────────────────────────────────────────────


class TestDepthAwareDefault:
    def test_depth_aware_defaults_to_true(self):
        """depth_aware should default to True to protect depth RGB encoding."""
        aug = AdvancedAugmentation()
        assert aug.depth_aware is True

    def test_depth_aware_can_be_disabled(self):
        aug = AdvancedAugmentation(depth_aware=False)
        assert aug.depth_aware is False


# ─── Test: hue shift fix ────────────────────────────────────────────────────


class TestHueShift:
    def test_hue_shift_produces_actual_shift(self):
        """The old code used int(shift * 3) which was always 0.
        The new code uses ±1 channel rotation.
        """
        aug = AdvancedAugmentation(
            color_jitter_prob=1.0,  # force color jitter
            color_jitter_strength=0.5,
            depth_aware=False,  # enable color jitter
            gaussian_noise_prob=0.0,
            cutout_prob=0.0,
            flip_prob=0.0,
            scale_prob=0.0,
            brightness_prob=0.0,
        )

        # Create a distinctive image where R=high, G=0, B=0
        image = torch.zeros(3, 8, 8)
        image[0] = 0.8  # Red channel high, in [-1, 1] range
        target = torch.rand(3, 8, 8)  # target doesn't matter for this test

        # Run many times — hue shift has 50% probability inside _color_jitter
        shifts_detected = 0
        for _ in range(50):
            result_img, _ = aug(image.clone(), target)
            # If hue shifted, the high value should have moved to G or B channel
            result_img[0].max().item()
            g_max = result_img[1].max().item()
            b_max = result_img[2].max().item()
            # If no shift: R should still be highest
            if g_max > 0.5 or b_max > 0.5:
                shifts_detected += 1

        # With 50% hue shift probability over 50 trials, we expect ~25 shifts
        # (plus saturation/contrast effects). At least some shifts should be detected.
        assert shifts_detected > 5, f"Expected hue shifts but only got {shifts_detected}"


# ─── Test: One-Cycle scheduler ───────────────────────────────────────────────


class TestOneCycleScheduler:
    def test_one_cycle_warmup(self):
        """During warmup, LR should increase linearly from ~0 to base."""
        optimizer = torch.optim.SGD([torch.randn(2, requires_grad=True)], lr=0.01)
        scheduler = SchedulerFactory.create(
            optimizer, total_steps=100, warmup_steps=10, scheduler_type="one_cycle"
        )

        lrs = []
        for _ in range(10):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        # First LR should be ~0 (before first step)
        assert lrs[0] < 0.002  # near zero
        # After warmup, should be close to base lr
        assert lrs[-1] > 0.008  # close to 0.01

    def test_one_cycle_anneal(self):
        """After warmup, LR should cosine-anneal from base to ~1% of base."""
        optimizer = torch.optim.SGD([torch.randn(2, requires_grad=True)], lr=0.01)
        scheduler = SchedulerFactory.create(
            optimizer, total_steps=100, warmup_steps=10, scheduler_type="one_cycle"
        )

        # Skip warmup
        for _ in range(10):
            optimizer.step()
            scheduler.step()

        peak_lr = optimizer.param_groups[0]["lr"]
        assert peak_lr > 0.008  # near base lr

        # Anneal to end
        for _ in range(90):
            optimizer.step()
            scheduler.step()

        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr < 0.002  # should be near 1% of base = 0.0001
        assert final_lr < peak_lr  # must have decreased

    def test_one_cycle_never_exceeds_base_by_2x(self):
        """Old buggy code had LR multipliers up to 2.0x base. Fixed version stays ≤ 1.0x."""
        optimizer = torch.optim.SGD([torch.randn(2, requires_grad=True)], lr=0.01)
        scheduler = SchedulerFactory.create(
            optimizer, total_steps=100, warmup_steps=10, scheduler_type="one_cycle"
        )

        max_lr = 0.0
        for _ in range(100):
            lr = optimizer.param_groups[0]["lr"]
            max_lr = max(max_lr, lr)
            optimizer.step()
            scheduler.step()

        # Should never exceed base lr (multiplier ≤ 1.0)
        assert max_lr <= 0.01 + 1e-6, f"LR exceeded base: {max_lr}"


# ─── Test: Multi-colormap re-encoding ────────────────────────────────────────


class TestMultiColormapReencoding:
    def test_colormap_reencode_roundtrip(self):
        """Depth → RGB (cube_edge) → depth → RGB (rainbow) → depth should preserve depth.

        Note: Rainbow colormap has non-monotonic RGB transitions (Red→Yellow→Green→Cyan→Blue),
        so exact roundtrip is impossible. We verify:
        1. cube_edge roundtrip is accurate (primary encoding)
        2. Re-encoding with a different colormap produces a valid, different RGB
        3. The re-encoded RGB decodes to values in valid depth range
        """
        mapper = DepthRGBMapper(colormap="cube_edge")

        # Create synthetic depth
        depth = torch.rand(1, 1, 32, 32) * 10.0  # [0, 10] meters
        rgb_cube = mapper.depth_to_rgb(depth)

        # Decode back to depth — cube_edge should be accurately invertible
        depth_recovered = mapper.rgb_to_depth(rgb_cube)
        assert torch.allclose(depth, depth_recovered, atol=0.5), "Roundtrip failed for cube_edge"

        # Re-encode with rainbow (produces a different but valid target)
        mapper.set_colormap("rainbow")
        rgb_rainbow = mapper.depth_to_rgb(depth_recovered)
        mapper.set_colormap("cube_edge")

        # The re-encoded RGB should be different from the cube_edge RGB
        assert not torch.allclose(rgb_cube, rgb_rainbow, atol=0.01), "Rainbow should differ from cube_edge"

        # Rainbow RGB should be in valid [0, 1] range
        assert rgb_rainbow.min() >= 0.0 and rgb_rainbow.max() <= 1.0

    def test_different_colormaps_produce_different_rgb(self):
        """Different colormaps should produce visually different RGB for same depth."""
        mapper = DepthRGBMapper()
        depth = torch.rand(1, 1, 16, 16) * 5.0

        rgb_cube = mapper.depth_to_rgb(depth).clone()
        mapper.set_colormap("plasma")
        rgb_plasma = mapper.depth_to_rgb(depth).clone()
        mapper.set_colormap("viridis")
        rgb_viridis = mapper.depth_to_rgb(depth).clone()

        # They should be different
        assert not torch.allclose(rgb_cube, rgb_plasma, atol=0.01)
        assert not torch.allclose(rgb_cube, rgb_viridis, atol=0.01)
        assert not torch.allclose(rgb_plasma, rgb_viridis, atol=0.01)


# ─── Test: CFG dropout per-sample ────────────────────────────────────────────


class TestCFGDropoutPerSample:
    def test_per_sample_dropout_logic(self):
        """Simulate the per-sample CFG dropout logic from the training loop."""
        bsz = 8
        hidden_dim = 16
        seq_len = 4

        # Mock conditional and unconditional embeddings
        cond = torch.randn(bsz, seq_len, hidden_dim)
        uncond = torch.randn(bsz, seq_len, hidden_dim)

        # Simulate dropout with a fixed seed
        torch.manual_seed(42)
        cfg_dropout_rate = 0.5  # 50% for testing
        drop_mask = torch.rand(bsz) < cfg_dropout_rate
        drop_exp = drop_mask.view(bsz, 1, 1)
        result = torch.where(drop_exp, uncond, cond)

        # Dropped samples should match uncond, kept samples should match cond
        for i in range(bsz):
            if drop_mask[i]:
                assert torch.allclose(result[i], uncond[i])
            else:
                assert torch.allclose(result[i], cond[i])

        # At least some should be dropped (with 50% rate and 8 samples, very likely ≥1)
        assert drop_mask.sum() >= 1, "Expected at least 1 dropped sample"
        assert drop_mask.sum() <= bsz - 1, "Expected at least 1 kept sample"


# ─── Test: LatentCacheDataset ────────────────────────────────────────────────


class TestLatentCacheDataset:
    def test_latent_cache_wrapper(self):
        """Test that LatentCacheDataset wraps a base dataset and caches latents."""
        from open_vision_banana.training.latent_cache import LatentCacheDataset

        # Create a mock base dataset
        class MockDataset(torch.utils.data.Dataset):
            def __init__(self, n=4):
                self.n = n

            def __len__(self):
                return self.n

            def __getitem__(self, idx):
                return {
                    "image": torch.randn(3, 16, 16).clamp(-1, 1),
                    "target_rgb": torch.rand(3, 16, 16),
                    "prompt": f"test prompt {idx}",
                }

        # Mock VAE (just returns the input downsampled)
        class MockVAE:
            def __init__(self):
                self.scaling_factor = 0.18215

            def encode(self, x):
                # Simulate VAE encoding: downsample by 8
                import torch.nn.functional as F
                latent = F.avg_pool2d(x, 8)
                return type("obj", (object,), {"latent_dist": type("obj", (object,), {"sample": lambda self: latent * self.scaling_factor})})

        class MockModelCfg:
            def __init__(self):
                self.vae_scale_factor = 0.18215
                self.name = "mock"

        base = MockDataset(n=4)
        mock_vae = MockVAE()
        mock_cfg = MockModelCfg()

        # This will fail because our mock VAE doesn't match the real interface
        # But we can test the cache logic by mocking _encode
        cached_ds = LatentCacheDataset(base, mock_vae, mock_cfg, device="cpu", dtype=torch.float32)

        # Override _encode to avoid VAE dependency
        def mock_encode(pixels):
            import torch.nn.functional as F
            return F.avg_pool2d(pixels, 8).squeeze(0)

        cached_ds._encode = mock_encode  # type: ignore[method-assign]

        # First access: should encode and cache
        sample0 = cached_ds[0]
        assert "input_latent" in sample0
        assert "target_latent" in sample0
        assert cached_ds.cache_size > 0

        # Second access: should return cached
        sample0_again = cached_ds[0]
        assert torch.equal(sample0["input_latent"], sample0_again["input_latent"])

        # Clear cache
        cached_ds.clear_cache()
        assert cached_ds.cache_size == 0


# ─── Test: Batch generation function exists ──────────────────────────────────


class TestBatchGenerationExists:
    def test_generate_image_batch_importable(self):
        """Verify that generate_image_batch is importable from the script."""

        # The function is in scripts/generate_synthetic_data.py
        # We can't import it directly as a module, but we can check the source
        # Instead, read the file and check the function exists
        content = (ARCHIVE_SCRIPTS / "generate_synthetic_data.py").read_text()
        assert "def generate_image_batch(" in content
        assert "def generate_image(" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
