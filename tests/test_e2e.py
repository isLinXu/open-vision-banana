"""End-to-end tests for ONNX export and import verification."""

import pytest
import torch
import tempfile
from pathlib import Path

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.core.normal_mapper import NormalRGBMapper
from open_vision_banana.export.onnx_export import (
    DepthRGBMapperONNX,
    NormalRGBMapperONNX,
    export_mapper_onnx,
)


class TestONNXExport:
    """Tests for ONNX export and numerical verification."""

    def test_depth_onnx_export(self, tmp_path):
        """Export depth mapper to ONNX and verify file exists."""
        output_path = tmp_path / "depth_to_rgb.onnx"
        export_mapper_onnx("depth_to_rgb", str(output_path))
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_normal_onnx_export(self, tmp_path):
        """Export normal mapper to ONNX and verify file exists."""
        output_path = tmp_path / "normal_to_rgb.onnx"
        export_mapper_onnx("normal_to_rgb", str(output_path))
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_depth_onnx_numerical_consistency(self):
        """Verify ONNX model produces same output as PyTorch."""
        mapper = DepthRGBMapper()
        onnx_mapper = DepthRGBMapperONNX()
        onnx_mapper.eval()

        depth = torch.rand(1, 1, 32, 32) * 50.0
        pt_output = mapper.depth_to_rgb(depth)
        onnx_output = onnx_mapper(depth)

        assert torch.allclose(pt_output, onnx_output, atol=1e-5), \
            "ONNX and PyTorch outputs differ"

    def test_normal_onnx_numerical_consistency(self):
        """Verify ONNX normal mapper matches PyTorch."""
        mapper = NormalRGBMapper()
        onnx_mapper = NormalRGBMapperONNX()
        onnx_mapper.eval()

        normal = torch.randn(1, 3, 32, 32)
        norms = normal.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normal = normal / norms

        pt_output = mapper.normal_to_rgb(normal)
        onnx_output = onnx_mapper(normal)

        assert torch.allclose(pt_output, onnx_output, atol=1e-5), \
            "ONNX and PyTorch normal outputs differ"
