"""Tests for inference optimization utilities."""

from unittest.mock import MagicMock

import torch
import torch.nn as nn

from open_vision_banana.export.onnx_export import NormalRGBMapperONNX
from open_vision_banana.inference.optimization import (
    InferenceOptions,
    apply_pipeline_optimizations,
    estimate_vram_savings,
    quantize_mapper_int8,
)


class TestInferenceOptions:
    def test_vram_notes_for_offload(self):
        opts = InferenceOptions(enable_sequential_cpu_offload=True)
        notes = estimate_vram_savings(opts)
        assert "sequential_cpu_offload" in notes

    def test_apply_vae_slicing(self):
        pipe = MagicMock()
        opts = InferenceOptions(enable_vae_slicing=True)
        applied = apply_pipeline_optimizations(pipe, opts)
        pipe.enable_vae_slicing.assert_called_once()
        assert "vae_slicing" in applied

    def test_apply_attention_slicing(self):
        pipe = MagicMock()
        opts = InferenceOptions(enable_attention_slicing=True, attention_slice_size=1)
        applied = apply_pipeline_optimizations(pipe, opts)
        pipe.enable_attention_slicing.assert_called_once_with(1)
        assert any("attention_slicing" in item for item in applied)


class TestQuantizeMapper:
    def test_int8_mapper_runs(self):
        model = NormalRGBMapperONNX()
        normal = torch.randn(1, 3, 32, 32)
        normal = normal / normal.norm(dim=1, keepdim=True).clamp(min=1e-8)

        out_fp = model(normal)
        qmodel = quantize_mapper_int8(model)
        out_int8 = qmodel(normal)

        assert out_fp.shape == out_int8.shape
        assert torch.allclose(out_fp, out_int8, atol=0.05)
