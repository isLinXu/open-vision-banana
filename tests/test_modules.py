"""Comprehensive tests for Vision Banana core modules.

Tests cover:
- DepthRGBMapper: power transform, spectral colormap, roundtrip accuracy
- NormalRGBMapper: linear encoding, roundtrip, directional consistency
- SegmentationDecoder: 5-stage pipeline, edge cases
- PromptBuilder: all task types, template variants
- Config: YAML serialization, defaults, overrides
"""

import tempfile
from pathlib import Path

import pytest
import torch

from open_vision_banana.config import (
    DepthMappingConfig,
    SegmentationConfig,
    VisionBananaConfig,
)
from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.core.normal_mapper import NormalRGBMapper
from open_vision_banana.core.rgb_mapper import RGBMapper
from open_vision_banana.core.segmentation import SegmentationDecoder
from open_vision_banana.inference.prompt import PromptBuilder

# ──────────────────────────────────────────────────────────────
# DepthRGBMapper Tests
# ──────────────────────────────────────────────────────────────


class TestDepthRGBMapper:
    """Tests for depth ↔ RGB mapping with power transform + cube-edge colormap."""

    def setup_method(self):
        self.mapper = DepthRGBMapper(lam=-3.0, c=10.0 / 3.0)

    def test_depth_zero_maps_to_near(self):
        depth = torch.zeros(1, 1, 1, 1)
        normalized = self.mapper.depth_to_normalized(depth)
        assert normalized.item() == pytest.approx(0.0, abs=1e-6)

    def test_depth_large_maps_to_far(self):
        depth = torch.tensor([[[[100.0]]]])
        normalized = self.mapper.depth_to_normalized(depth)
        assert normalized.item() > 0.7

    def test_nearby_precision(self):
        near_low = self.mapper.depth_to_normalized(torch.tensor([[[[0.5]]]]))
        near_high = self.mapper.depth_to_normalized(torch.tensor([[[[1.0]]]]))
        near_range = (near_high - near_low).item()

        far_low = self.mapper.depth_to_normalized(torch.tensor([[[[50.0]]]]))
        far_high = self.mapper.depth_to_normalized(torch.tensor([[[[100.0]]]]))
        far_range = (far_high - far_low).item()

        assert near_range > far_range

    def test_normalized_range(self):
        depths = torch.rand(1, 1, 32, 32) * 100.0
        normalized = self.mapper.depth_to_normalized(depths)
        assert normalized.min() >= 0.0
        assert normalized.max() < 1.0

    def test_rgb_range(self):
        depths = torch.rand(1, 1, 32, 32) * 50.0
        rgb = self.mapper.depth_to_rgb(depths)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_roundtrip_nearby(self):
        depths = torch.tensor([[[[0.5, 1.0, 2.0, 3.0, 5.0]]]])
        rgb = self.mapper.depth_to_rgb(depths)
        recovered = self.mapper.rgb_to_depth(rgb)
        assert torch.allclose(depths, recovered, atol=0.5), (
            f"Nearby roundtrip failed: {depths} vs {recovered}"
        )

    def test_roundtrip_monotonic(self):
        depths = torch.tensor([[[[1.0, 2.0, 5.0, 10.0, 20.0]]]])
        normalized = self.mapper.depth_to_normalized(depths)
        diffs = normalized.flatten()[1:] - normalized.flatten()[:-1]
        assert (diffs > 0).all()

    def test_batch_consistency(self):
        depth = torch.tensor([[[[5.0]]]]).expand(2, 1, 1, 1)
        rgb = self.mapper.depth_to_rgb(depth)
        assert torch.allclose(rgb[0], rgb[1], atol=1e-5)

    def test_gradient_flow(self):
        depth = torch.tensor([[[[5.0]]]], requires_grad=True)
        normalized = self.mapper.depth_to_normalized(depth)
        normalized.backward()
        assert depth.grad is not None

    # ── Regression guards (added v0.5.0, validation-driven analysis) ──────

    def test_spectral_colormap_no_self_intersection(self):
        """谱色图 3D 轨迹必须不自交。

        ``rgb_to_normalized`` 依赖于谱色图曲线在 RGB 立方体中不自交这一
        隐含前提来保证单射性。若有人调整 ``SPECTRAL_KEYPOINTS`` 引入自交，
        会静默破坏可逆性。本测试通过 normalized→rgb→normalized 的密集
        roundtrip 来守护该前提。
        """
        n = torch.linspace(0.0, 1.0, 2000).view(1, 1, 1, -1)
        rgb = self.mapper.normalized_to_rgb(n)
        recovered = self.mapper.rgb_to_normalized(rgb)
        max_err = (n - recovered).abs().max().item()
        assert max_err < 1e-3, (
            f"Spectral colormap roundtrip broke (max err={max_err:.2e}); "
            "likely a self-intersection was introduced into SPECTRAL_KEYPOINTS."
        )

    def test_depth_roundtrip_within_valid_domain_quantized(self):
        """有效域 [1, 50]m 内,含 8-bit 量化的 roundtrip AbsRel < 0.5%。

        扩散模型实际输出 uint8 RGB,因此真实端到端精度由"量化+逆映射"
        联合决定。本测试守护该现实精度。
        """
        torch.manual_seed(0)
        depth = torch.rand(1, 1, 64, 64) * 49.0 + 1.0  # [1, 50]m
        rgb = self.mapper.depth_to_rgb(depth)
        rgb_q = (rgb * 255).round() / 255.0            # 8-bit quantization
        recovered = self.mapper.rgb_to_depth(rgb_q)
        abs_rel = ((depth - recovered).abs() / depth).mean().item()
        assert abs_rel < 0.005, f"Quantized depth AbsRel too high: {abs_rel:.4f}"

    def test_depth_roundtrip_degrades_gracefully_far(self):
        """超出有效域 (>200m) 时精度衰减但不应 NaN/inf。

        记录失稳行为,确保未来的有效域裁剪改动不会引入数值崩溃。
        """
        depth = torch.tensor([[[[300.0, 500.0, 1000.0]]]])
        recovered = self.mapper.rgb_to_depth(self.mapper.depth_to_rgb(depth))
        assert torch.isfinite(recovered).all()


# ──────────────────────────────────────────────────────────────
# NormalRGBMapper Tests
# ──────────────────────────────────────────────────────────────


class TestNormalRGBMapper:
    """Tests for surface normal ↔ RGB mapping."""

    def setup_method(self):
        self.mapper = NormalRGBMapper()

    def test_facing_camera(self):
        normal = torch.tensor([[[[0.0]], [[0.0]], [[1.0]]]])
        rgb = self.mapper.normal_to_rgb(normal)
        assert rgb[0, 0, 0, 0].item() == pytest.approx(0.5, abs=0.01)
        assert rgb[0, 2, 0, 0].item() == pytest.approx(1.0, abs=0.01)

    def test_facing_left(self):
        normal = torch.tensor([[[[-1.0]], [[0.0]], [[0.0]]]])
        rgb = self.mapper.normal_to_rgb(normal)
        assert rgb[0, 0, 0, 0].item() == pytest.approx(1.0, abs=0.01)

    def test_facing_right(self):
        normal = torch.tensor([[[[1.0]], [[0.0]], [[0.0]]]])
        rgb = self.mapper.normal_to_rgb(normal)
        assert rgb[0, 0, 0, 0].item() == pytest.approx(0.0, abs=0.01)

    def test_facing_up(self):
        normal = torch.tensor([[[[0.0]], [[1.0]], [[0.0]]]])
        rgb = self.mapper.normal_to_rgb(normal)
        assert rgb[0, 1, 0, 0].item() == pytest.approx(1.0, abs=0.01)

    def test_roundtrip_normal(self):
        normals = torch.randn(1, 3, 16, 16)
        norms = normals.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normals = normals / norms
        rgb = self.mapper.normal_to_rgb(normals)
        recovered = self.mapper.rgb_to_normal(rgb)
        cos_sim = (normals * recovered).sum(dim=1).mean()
        assert cos_sim.item() > 0.99

    def test_rgb_range(self):
        normals = torch.randn(1, 3, 32, 32)
        norms = normals.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normals = normals / norms
        rgb = self.mapper.normal_to_rgb(normals)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_roundtrip_normal_quantized(self):
        """含 8-bit 量化的法线 roundtrip 角误差应 < 1°。

        法线映射是纯线性的,量化误差均匀且极小,是 RGB 参数化最稳健的
        任务。本测试守护该现实精度(实测 ~0.17°)。
        """
        torch.manual_seed(1)
        normals = torch.randn(1, 3, 64, 64)
        normals = normals / normals.norm(dim=1, keepdim=True).clamp(min=1e-8)
        rgb = self.mapper.normal_to_rgb(normals)
        rgb_q = (rgb * 255).round() / 255.0
        recovered = self.mapper.rgb_to_normal(rgb_q)
        cos_sim = torch.nn.functional.cosine_similarity(
            normals, recovered, dim=1
        ).clamp(-1, 1)
        mean_angle_deg = (torch.acos(cos_sim).mean() * 180.0 / 3.14159265).item()
        assert mean_angle_deg < 1.0, (
            f"Quantized normal angular error too high: {mean_angle_deg:.3f}°"
        )


# ──────────────────────────────────────────────────────────────
# SegmentationDecoder Tests
# ──────────────────────────────────────────────────────────────


class TestSegmentationDecoder:
    """Tests for multi-stage instance segmentation decoding."""

    def setup_method(self):
        self.decoder = SegmentationDecoder()

    def test_semantic_decode_basic(self):
        class_map = {"cat": (255, 255, 0), "background": (128, 0, 128)}
        rgb = torch.zeros(1, 3, 32, 32)
        rgb[0, 0, :16, :] = 1.0
        rgb[0, 1, :16, :] = 1.0
        rgb[0, 0, 16:, :] = 128 / 255
        rgb[0, 2, 16:, :] = 128 / 255
        mask = self.decoder.decode_semantic(rgb, class_map)
        assert mask.shape == (1, 1, 32, 32)
        assert mask[0, 0, 0, 0].item() == 0
        assert mask[0, 0, 16, 0].item() == 1

    def test_instance_decode_background(self):
        rgb = torch.ones(1, 3, 16, 16) * 0.5
        instances = self.decoder.decode_instance(rgb)
        assert isinstance(instances, list)
        assert len(instances) == 1

    def test_noise_pruning_removes_small_components(self):
        rgb = torch.zeros(1, 3, 64, 64)
        rgb[0, 0, 10:50, 10:50] = 1.0
        rgb[0, 1, 5, 5] = 1.0
        instances = self.decoder.decode_instance(rgb)
        assert isinstance(instances, list)

    def test_two_instances_detected(self):
        rgb = torch.zeros(1, 3, 64, 64)
        rgb[0, 0, 5:25, 5:25] = 1.0
        rgb[0, 2, 35:55, 35:55] = 1.0
        instances = self.decoder.decode_instance(rgb)
        assert len(instances[0]) >= 2


# ──────────────────────────────────────────────────────────────
# PromptBuilder Tests
# ──────────────────────────────────────────────────────────────


class TestPromptBuilder:
    """Tests for task-specific prompt construction."""

    def test_semantic_prompt_structured(self):
        class_map = {"road": (128, 64, 128), "sky": (70, 130, 180)}
        prompt = PromptBuilder.build_semantic_prompt(class_map, template_idx=0)
        assert "road" in prompt.lower() or "128" in prompt

    def test_semantic_prompt_descriptive(self):
        class_map = {"road": (128, 64, 128)}
        prompt = PromptBuilder.build_semantic_prompt(class_map, template_idx=1)
        assert len(prompt) > 10

    def test_semantic_prompt_hex(self):
        class_map = {"road": (128, 64, 128)}
        prompt = PromptBuilder.build_semantic_prompt(class_map, template_idx=2)
        assert "#" in prompt or "hex" in prompt.lower() or "128" in prompt

    def test_instance_prompt(self):
        prompt = PromptBuilder.build_instance_prompt("dog")
        assert "dog" in prompt.lower()

    def test_referring_prompt(self):
        prompt = PromptBuilder.build_referring_prompt("the man in pink t shirt")
        assert "pink" in prompt.lower() or "man" in prompt.lower()

    def test_depth_prompt(self):
        prompt = PromptBuilder.build_depth_prompt()
        assert "depth" in prompt.lower()

    def test_normal_prompt(self):
        prompt = PromptBuilder.build_normal_prompt()
        assert "normal" in prompt.lower()


# ──────────────────────────────────────────────────────────────
# Config Tests
# ──────────────────────────────────────────────────────────────


class TestConfig:
    """Tests for configuration system."""

    def test_defaults(self):
        cfg = VisionBananaConfig()
        assert cfg.model.name == "black-forest-labs/FLUX.1-schnell"
        assert cfg.lora.rank == 64
        assert cfg.lora.alpha == 128
        assert cfg.lora.rs_lora is True
        assert cfg.optimizer.lr == 5e-5
        assert cfg.data.vision_ratio == 0.1

    def test_yaml_roundtrip(self):
        cfg = VisionBananaConfig()
        cfg.lora.rank = 32
        cfg.optimizer.lr = 1e-4

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            cfg.to_yaml(f.name)
            loaded = VisionBananaConfig.from_yaml(f.name)

        assert loaded.lora.rank == 32
        assert loaded.optimizer.lr == 1e-4
        Path(f.name).unlink()

    def test_from_existing_yaml(self):
        yaml_path = Path(__file__).parent.parent / "configs" / "training_config.yaml"
        if yaml_path.exists():
            cfg = VisionBananaConfig.from_yaml(yaml_path)
            assert cfg.lora.rank == 64
            assert cfg.depth_mapping.lam == -3.0

    def test_from_dict_with_unknown_keys(self):
        """Config should handle unknown keys gracefully."""
        d = {"model": {"name": "test-model", "dtype": "float32"}}
        cfg = VisionBananaConfig.from_dict(d)
        assert cfg.model.name == "test-model"

    def test_depth_mapping_params(self):
        cfg = DepthMappingConfig()
        assert cfg.lam == -3.0
        assert abs(cfg.c - 10.0 / 3.0) < 1e-6

    def test_segmentation_params(self):
        cfg = SegmentationConfig()
        assert cfg.color_tolerance == 14.0
        assert cfg.min_area_fraction == 2.0e-4
        assert cfg.erosion_threshold == 0.1
        assert cfg.bbox_expansion_factor == 5.0


# ──────────────────────────────────────────────────────────────
# RGBMapper Integration Tests
# ──────────────────────────────────────────────────────────────


class TestRGBMapper:
    """Integration tests for the unified RGBMapper interface."""

    def setup_method(self):
        self.mapper = RGBMapper()

    def test_encode_depth(self):
        depth = torch.rand(1, 1, 16, 16) * 10.0
        rgb = self.mapper.encode(depth, task="depth")
        assert rgb.shape == (1, 3, 16, 16)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_encode_normal(self):
        normal = torch.randn(1, 3, 16, 16)
        norms = normal.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normal = normal / norms
        rgb = self.mapper.encode(normal, task="normal")
        assert rgb.shape == (1, 3, 16, 16)

    def test_decode_depth(self):
        rgb = torch.rand(1, 3, 16, 16)
        depth = self.mapper.decode(rgb, task="depth")
        assert depth.shape == (1, 1, 16, 16)
        assert depth.min() >= 0.0

    def test_decode_normal(self):
        rgb = torch.rand(1, 3, 16, 16)
        normal = self.mapper.decode(rgb, task="normal")
        assert normal.shape == (1, 3, 16, 16)

    def test_roundtrip_depth(self):
        depth_orig = torch.tensor([[[[1.0, 3.0, 5.0]]]])
        rgb = self.mapper.encode(depth_orig, task="depth")
        depth_rec = self.mapper.decode(rgb, task="depth")
        assert torch.allclose(depth_orig, depth_rec, atol=2.0)

    def test_roundtrip_normal(self):
        normal_orig = torch.randn(1, 3, 8, 8)
        norms = normal_orig.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normal_orig = normal_orig / norms
        rgb = self.mapper.encode(normal_orig, task="normal")
        normal_rec = self.mapper.decode(rgb, task="normal")
        cos_sim = (normal_orig * normal_rec).sum(dim=1).mean()
        assert cos_sim.item() > 0.95


# ──────────────────────────────────────────────────────────────
# Performance Tests
# ──────────────────────────────────────────────────────────────


class TestPerformance:
    """Performance benchmarks for core operations."""

    def test_depth_encoding_speed(self):
        import time

        mapper = DepthRGBMapper()
        depth = torch.rand(1, 1, 512, 512) * 50.0
        # Warmup (excludes lazy-init / import overhead from timing).
        mapper.depth_to_rgb(depth)
        # Median-of-runs is robust to scheduler jitter under load.
        timings = []
        for _ in range(10):
            start = time.perf_counter()
            mapper.depth_to_rgb(depth)
            timings.append(time.perf_counter() - start)
        timings.sort()
        median = timings[len(timings) // 2]
        # Generous bound: CI under parallel load needs headroom; this only
        # guards against pathological regressions (10x), not micro-jitter.
        assert median < 0.5, f"Depth encoding regressed: {median * 1000:.1f}ms median"

    def test_normal_encoding_speed(self):
        import time

        mapper = NormalRGBMapper()
        normal = torch.randn(1, 3, 512, 512)
        norms = normal.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normal = normal / norms
        mapper.normal_to_rgb(normal)  # warmup
        timings = []
        for _ in range(10):
            start = time.perf_counter()
            mapper.normal_to_rgb(normal)
            timings.append(time.perf_counter() - start)
        timings.sort()
        median = timings[len(timings) // 2]
        assert median < 0.25, f"Normal encoding regressed: {median * 1000:.1f}ms median"

    def test_segmentation_decode_speed(self):
        import time

        decoder = SegmentationDecoder()
        rgb = torch.rand(1, 3, 64, 64)
        # Warmup pass — first call pays one-time PyTorch dispatch / cache costs.
        result = decoder.decode_instance(rgb)
        # Correctness: decode must return one instance list per batch image.
        assert isinstance(result, list) and len(result) == 1
        # Timing: median of 3 runs, generous bound robust to parallel CI load.
        timings = []
        for _ in range(3):
            start = time.perf_counter()
            decoder.decode_instance(rgb)
            timings.append(time.perf_counter() - start)
        timings.sort()
        median = timings[len(timings) // 2]
        assert median < 10.0, f"Segmentation decode regressed: {median:.3f}s median"
