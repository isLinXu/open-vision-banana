"""Tests for export/onnx_export.py — ONNX export utilities."""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from open_vision_banana.export.onnx_export import (
    DepthRGBMapperONNX,
    NormalRGBMapperONNX,
    export_flux_transformer_onnx,
    export_full_pipeline_onnx,
    export_mapper_onnx,
)

# Skip all ONNX export tests if onnx is not installed
onnx_available = True
try:
    import onnx  # noqa: F401
except ImportError:
    onnx_available = False

skip_no_onnx = pytest.mark.skipif(not onnx_available, reason="onnx not installed")

# ──────────────────────────────────────────────────────────────────
# DepthRGBMapperONNX tests
# ──────────────────────────────────────────────────────────────────


class TestDepthRGBMapperONNX:
    """Test the ONNX-compatible depth mapper wrapper."""

    def test_forward_output_shape(self):
        """forward() should produce [B, 3, H, W] output."""
        mapper = DepthRGBMapperONNX()
        depth = torch.rand(2, 1, 16, 16) * 10
        rgb = mapper(depth)
        assert rgb.shape == (2, 3, 16, 16)

    def test_forward_output_range(self):
        """Output should be in [0, 1] range."""
        mapper = DepthRGBMapperONNX()
        depth = torch.rand(1, 1, 8, 8) * 100
        rgb = mapper(depth)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_depth_to_normalized(self):
        """depth_to_normalized should map to [0, 1)."""
        mapper = DepthRGBMapperONNX()
        depth = torch.tensor([[[[0.0, 1.0, 5.0, 10.0]]]])
        norm = mapper.depth_to_normalized(depth)
        assert norm.min() >= 0.0
        assert norm.max() < 1.0

    def test_normalized_to_depth_inverse(self):
        """normalized_to_depth should be inverse of depth_to_normalized."""
        mapper = DepthRGBMapperONNX()
        depth = torch.rand(1, 1, 8, 8) * 10
        norm = mapper.depth_to_normalized(depth)
        recovered = mapper.normalized_to_depth(norm)
        assert torch.allclose(depth, recovered, atol=1e-4)

    def test_normalized_to_rgb_shape(self):
        """normalized_to_rgb should produce [B, 3, H, W] output."""
        mapper = DepthRGBMapperONNX()
        norm = torch.rand(1, 1, 8, 8)
        rgb = mapper.normalized_to_rgb(norm)
        assert rgb.shape == (1, 3, 8, 8)

    def test_registered_buffers(self):
        """Should have spectral keypoints as registered buffers."""
        mapper = DepthRGBMapperONNX()
        assert "_positions" in dict(mapper.named_buffers())
        assert "_colors" in dict(mapper.named_buffers())

    def test_eval_mode(self):
        """Should support eval() call."""
        mapper = DepthRGBMapperONNX()
        mapper.eval()
        assert mapper.training is False

    def test_custom_params(self):
        """Should accept custom lam and c parameters."""
        mapper = DepthRGBMapperONNX(lam=-2.0, c=5.0)
        assert mapper.lam == -2.0
        assert mapper.c == 5.0

    def test_zero_depth(self):
        """Zero depth should produce near-zero normalized value."""
        mapper = DepthRGBMapperONNX()
        depth = torch.zeros(1, 1, 4, 4)
        norm = mapper.depth_to_normalized(depth)
        assert norm.max() < 0.01


# ──────────────────────────────────────────────────────────────────
# NormalRGBMapperONNX tests
# ──────────────────────────────────────────────────────────────────


class TestNormalRGBMapperONNX:
    """Test the ONNX-compatible normal mapper wrapper."""

    def test_forward_output_shape(self):
        """forward() should produce [B, 3, H, W] output."""
        mapper = NormalRGBMapperONNX()
        normal = torch.randn(2, 3, 16, 16)
        normal = normal / normal.norm(dim=1, keepdim=True)
        rgb = mapper(normal)
        assert rgb.shape == (2, 3, 16, 16)

    def test_forward_output_range(self):
        """Output should be in [0, 1] range."""
        mapper = NormalRGBMapperONNX()
        normal = torch.randn(1, 3, 8, 8)
        normal = normal / normal.norm(dim=1, keepdim=True)
        rgb = mapper(normal)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_z_normal(self):
        """Z-axis normal should map to specific RGB values."""
        mapper = NormalRGBMapperONNX()
        normal = torch.tensor([[[[0.0]], [[0.0]], [[1.0]]]])
        rgb = mapper(normal)
        # R = (1-0)/2 = 0.5, G = (1+0)/2 = 0.5, B = (1+1)/2 = 1.0
        assert pytest.approx(rgb[0, 0, 0, 0].item(), abs=1e-4) == 0.5
        assert pytest.approx(rgb[0, 1, 0, 0].item(), abs=1e-4) == 0.5
        assert pytest.approx(rgb[0, 2, 0, 0].item(), abs=1e-4) == 1.0


# ──────────────────────────────────────────────────────────────────
# export_mapper_onnx tests
# ──────────────────────────────────────────────────────────────────


@skip_no_onnx
class TestExportMapperOnnx:
    """Test the export_mapper_onnx function."""

    def test_export_depth_to_rgb(self, tmp_path):
        """Should export depth_to_rgb mapper to ONNX."""
        output_path = tmp_path / "depth_to_rgb.onnx"
        export_mapper_onnx(
            "depth_to_rgb",
            str(output_path),
            input_shape=(1, 1, 16, 16),
            opset_version=18,
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_export_normal_to_rgb(self, tmp_path):
        """Should export normal_to_rgb mapper to ONNX."""
        output_path = tmp_path / "normal_to_rgb.onnx"
        export_mapper_onnx(
            "normal_to_rgb",
            str(output_path),
            input_shape=(1, 3, 16, 16),
            opset_version=18,
        )
        assert output_path.exists()

    def test_export_rgb_to_depth(self, tmp_path):
        """Should export rgb_to_depth mapper to ONNX."""
        output_path = tmp_path / "rgb_to_depth.onnx"
        export_mapper_onnx(
            "rgb_to_depth",
            str(output_path),
            input_shape=(1, 3, 16, 16),
            opset_version=18,
        )
        assert output_path.exists()

    def test_export_rgb_to_normal(self, tmp_path):
        """Should export rgb_to_normal mapper to ONNX."""
        output_path = tmp_path / "rgb_to_normal.onnx"
        export_mapper_onnx(
            "rgb_to_normal",
            str(output_path),
            input_shape=(1, 3, 16, 16),
            opset_version=18,
        )
        assert output_path.exists()

    def test_export_unsupported_type_raises(self, tmp_path):
        """Should raise ValueError for unsupported mapper type."""
        with pytest.raises(ValueError, match="Unsupported mapper type"):
            export_mapper_onnx(
                "unsupported",
                str(tmp_path / "out.onnx"),
            )

    def test_export_creates_parent_dirs(self, tmp_path):
        """Should create parent directories if they don't exist."""
        output_path = tmp_path / "nested" / "dir" / "depth.onnx"
        export_mapper_onnx("depth_to_rgb", str(output_path), input_shape=(1, 1, 8, 8))
        assert output_path.exists()

    def test_export_default_input_shape(self, tmp_path):
        """Should work with default input_shape."""
        output_path = tmp_path / "default.onnx"
        export_mapper_onnx("depth_to_rgb", str(output_path))
        assert output_path.exists()


# ──────────────────────────────────────────────────────────────────
# FLUX transformer component export tests
# ──────────────────────────────────────────────────────────────────


@skip_no_onnx
class TestExportFluxTransformerOnnx:
    """Test FLUX transformer component export and compatibility behavior."""

    class MockTransformer(nn.Module):
        config = SimpleNamespace(
            in_channels=8,
            joint_attention_dim=16,
            pooled_projection_dim=4,
            guidance_embeds=False,
        )

        def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            img_ids,
            txt_ids,
            return_dict=True,
        ):
            del encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids
            output = hidden_states * 1.0
            return SimpleNamespace(sample=output) if return_dict else (output,)

    def test_export_with_mock_transformer(self, tmp_path):
        """Should export a FLUX-shaped transformer and component manifest."""

        class MockModel:
            def __init__(self):
                self.transformer = TestExportFluxTransformerOnnx.MockTransformer()

        model = MockModel()
        output_dir = tmp_path / "pipeline"
        result = export_flux_transformer_onnx(
            model,
            output_dir,
            image_sequence_length=4,
            text_sequence_length=3,
        )

        assert result == output_dir / "transformer.onnx"
        assert result.exists()
        manifest = json.loads((output_dir / "export_manifest.json").read_text())
        assert manifest["component"] == "flux_transformer"
        assert manifest["included"] == ["transformer"]
        assert "vae" in manifest["not_included"]

    def test_export_creates_output_dir(self, tmp_path):
        """Should create output directory."""

        class MockModel:
            def __init__(self):
                self.transformer = TestExportFluxTransformerOnnx.MockTransformer()

        model = MockModel()
        output_dir = tmp_path / "nested" / "pipeline"
        export_flux_transformer_onnx(
            model,
            output_dir,
            image_sequence_length=4,
            text_sequence_length=3,
        )
        assert output_dir.exists()
        assert (output_dir / "transformer.onnx").exists()

    def test_export_resolves_nested_pipeline_transformer(self, tmp_path):
        """VisionBanana-style nesting should resolve the transformer module."""

        class MockPipe:
            transformer = TestExportFluxTransformerOnnx.MockTransformer()

        class MockGenerator:
            pipe = MockPipe()

        class MockModel:
            generator = MockGenerator()

        output_dir = tmp_path / "nested-pipeline"
        export_flux_transformer_onnx(
            MockModel(),
            output_dir,
            image_sequence_length=4,
            text_sequence_length=3,
        )
        assert (output_dir / "transformer.onnx").exists()

    def test_rejects_non_flux_forward_signature(self, tmp_path):
        """A generic three-input module must not masquerade as a FLUX export."""

        class NonFluxTransformer(nn.Module):
            def forward(self, hidden, timestep, text_emb):
                return hidden

        with pytest.raises(TypeError, match="FLUX forward contract"):
            export_flux_transformer_onnx(
                SimpleNamespace(transformer=NonFluxTransformer()),
                tmp_path,
                image_sequence_length=4,
                text_sequence_length=3,
            )

    def test_rejects_unet_pipeline_with_actionable_error(self, tmp_path):
        """SDXL/SSD-1B pipelines should report the unsupported architecture."""
        model = SimpleNamespace(
            generator=SimpleNamespace(_t2i_pipe=SimpleNamespace(unet=nn.Identity()))
        )

        with pytest.raises(TypeError, match="supports FLUX transformers only"):
            export_flux_transformer_onnx(model, tmp_path)

    def test_legacy_alias_warns_and_exports_component(self, tmp_path):
        """The inaccurate historical API remains compatible but explicit."""
        model = SimpleNamespace(transformer=self.MockTransformer())
        with pytest.warns(DeprecationWarning, match="only the FLUX transformer"):
            result = export_full_pipeline_onnx(model, tmp_path)
        assert result == tmp_path / "transformer.onnx"
