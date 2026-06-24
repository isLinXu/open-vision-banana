"""Tests for AdaptiveDepthRGBMapper."""

import torch

from open_vision_banana.core.adaptive_depth_mapper import AdaptiveDepthRGBMapper
from open_vision_banana.core.depth_mapper import DepthRGBMapper


class TestAdaptiveDepthRGBMapper:
    """Test the adaptive depth-to-RGB mapper."""

    def test_inherits_from_base(self):
        """AdaptiveDepthRGBMapper should inherit from DepthRGBMapper."""
        mapper = AdaptiveDepthRGBMapper()
        assert isinstance(mapper, DepthRGBMapper)

    def test_default_params(self):
        """Default parameters should match the paper's values."""
        mapper = AdaptiveDepthRGBMapper()
        assert mapper.lam == -3.0
        assert abs(mapper.c - 10.0 / 3.0) < 1e-6
        assert mapper.near_samples == 512
        assert mapper.far_samples == 128
        assert mapper.d_near == 1.0
        assert mapper.d_far == 10.0

    def test_custom_params(self):
        """Custom parameters should be stored correctly."""
        mapper = AdaptiveDepthRGBMapper(
            lam=-2.0,
            c=5.0,
            near_samples=256,
            far_samples=32,
            d_near=0.5,
            d_far=5.0,
        )
        assert mapper.lam == -2.0
        assert mapper.c == 5.0
        assert mapper.near_samples == 256
        assert mapper.far_samples == 32
        assert mapper.d_near == 0.5
        assert mapper.d_far == 5.0

    def test_depth_to_rgb_shape(self):
        """Output shape should match input shape with channel dim 3."""
        mapper = AdaptiveDepthRGBMapper()
        depth = torch.rand(1, 1, 32, 32) * 10
        rgb = mapper.depth_to_rgb(depth)
        assert rgb.shape == (1, 3, 32, 32)

    def test_depth_to_rgb_range(self):
        """RGB output should be in [0, 1]."""
        mapper = AdaptiveDepthRGBMapper()
        depth = torch.rand(2, 1, 16, 16) * 50
        rgb = mapper.depth_to_rgb(depth)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_roundtrip_accuracy(self):
        """Roundtrip (depth → rgb → depth) should match Paper-level precision."""
        mapper = AdaptiveDepthRGBMapper()
        depth = torch.rand(1, 1, 64, 64) * 8 + 0.5  # [0.5, 8.5] metres
        rgb = mapper.depth_to_rgb(depth)
        recovered = mapper.rgb_to_depth(rgb)
        rel_err = ((recovered - depth).abs() / depth.clamp(min=0.1)).mean()
        assert rel_err < 0.01, f"Mean relative roundtrip error {rel_err:.4f} too large"

    def test_roundtrip_nearby_depths(self):
        """Scalar nearby depths should roundtrip within 0.5 m."""
        mapper = AdaptiveDepthRGBMapper()
        depths = torch.tensor([[[[0.5, 1.0, 2.0, 3.0, 5.0]]]])
        recovered = mapper.rgb_to_depth(mapper.depth_to_rgb(depths))
        assert torch.allclose(depths, recovered, atol=0.5)

    def test_rgb_to_normalized_refine_roundtrip(self):
        """rgb_to_normalized should enable depth roundtrip on mixed near/far field."""
        mapper = AdaptiveDepthRGBMapper()
        depth = torch.rand(1, 1, 32, 32) * 20.0 + 0.1
        recovered = mapper.rgb_to_depth(mapper.depth_to_rgb(depth))
        rel = ((recovered - depth).abs() / depth.clamp(min=0.1)).mean()
        assert rel < 0.02

    def test_far_depth_skips_quantization(self):
        """Depths beyond d_far should not be coarsely quantized on encode."""
        mapper = AdaptiveDepthRGBMapper(d_near=1.0, d_far=10.0)
        depth_far = torch.tensor([[[[50.0]]]])
        depth_near = torch.tensor([[[[2.0]]]])
        rgb_far = mapper.depth_to_rgb(depth_far)
        rgb_near = mapper.depth_to_rgb(depth_near)
        # Far roundtrip should match Paper-like continuous behaviour
        rec_far = mapper.rgb_to_depth(rgb_far)
        assert (rec_far - depth_far).abs() < 1.0
        rec_near = mapper.rgb_to_depth(rgb_near)
        assert (rec_near - depth_near).abs() < 0.5

    def test_logspace_roundtrip(self):
        """1-D log-spaced depths (benchmark protocol) should roundtrip accurately."""
        mapper = AdaptiveDepthRGBMapper()
        depth_values = torch.logspace(-1, 2, 200)
        depth = depth_values.view(1, 1, 1, -1)
        recovered = mapper.rgb_to_depth(mapper.depth_to_rgb(depth))
        rel = ((recovered - depth).abs() / depth.clamp(min=0.1)).squeeze()
        assert rel.mean() < 0.01, f"Mean rel error {rel.mean():.4f} too large"
        assert rel.max() < 0.05, f"Max rel error {rel.max():.4f} too large"

    def test_adaptive_quantize_static(self):
        """_adaptive_quantize should produce per-pixel grid-aligned values."""
        normalized = torch.tensor([0.1, 0.5, 0.9])
        num_levels = torch.tensor([512, 128, 128])
        result = AdaptiveDepthRGBMapper._adaptive_quantize(normalized, num_levels, 512)
        for val, levels in zip(result, num_levels, strict=False):
            step = 1.0 / levels.item()
            remainder = (val / step).round() - (val / step)
            assert abs(remainder) < 1e-5

    def test_monotonic_mapping(self):
        """Deeper depth should produce consistently different RGB (monotonic normalized)."""
        mapper = AdaptiveDepthRGBMapper()
        # Two distinct depth values
        d_near = torch.tensor([[[[2.0]]]])
        d_far = torch.tensor([[[[8.0]]]])
        mapper.depth_to_rgb(d_near)
        mapper.depth_to_rgb(d_far)
        # The normalized values should be different (monotonic)
        n_near = mapper.depth_to_normalized(d_near)
        n_far = mapper.depth_to_normalized(d_far)
        assert n_far > n_near

    def test_batch_consistency(self):
        """Batch processing should produce same result as individual processing."""
        mapper = AdaptiveDepthRGBMapper()
        depth_batch = torch.rand(2, 1, 16, 16) * 10
        rgb_batch = mapper.depth_to_rgb(depth_batch)
        rgb_0 = mapper.depth_to_rgb(depth_batch[0:1])
        rgb_1 = mapper.depth_to_rgb(depth_batch[1:2])
        assert torch.allclose(rgb_batch[0], rgb_0[0], atol=1e-5)
        assert torch.allclose(rgb_batch[1], rgb_1[0], atol=1e-5)

    def test_3d_depth_input(self):
        """Mapper should handle 3D depth input [B, 1, D, H, W]."""
        mapper = AdaptiveDepthRGBMapper()
        depth = torch.rand(1, 1, 4, 16, 16) * 10
        rgb = mapper.depth_to_rgb(depth)
        # Should output [B, 3, D, H, W] or similar
        assert rgb.shape[0] == 1
        assert rgb.shape[1] == 3

    def test_zero_depth(self):
        """Zero depth should map to a valid RGB (near end of spectrum)."""
        mapper = AdaptiveDepthRGBMapper()
        depth = torch.zeros(1, 1, 8, 8)
        rgb = mapper.depth_to_rgb(depth)
        assert rgb.shape == (1, 3, 8, 8)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_gradient_flow(self):
        """Gradients should flow through the mapper."""
        mapper = AdaptiveDepthRGBMapper()
        depth = (torch.rand(1, 1, 8, 8) * 10).requires_grad_(True)
        rgb = mapper.depth_to_rgb(depth)
        loss = rgb.sum()
        loss.backward()
        assert depth.grad is not None

    def test_normalized_to_rgb_spectral(self):
        """_normalized_to_rgb_cube_edge should produce valid RGB from normalized depth."""
        mapper = AdaptiveDepthRGBMapper()
        normalized = torch.rand(1, 1, 16, 16)
        rgb = mapper._normalized_to_rgb_cube_edge(normalized)
        assert rgb.shape == (1, 3, 16, 16)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0
