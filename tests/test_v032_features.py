"""Tests for new features added in v0.3.2 maturity improvement.

Covers:
- FSDP checkpoint save/load (smoke test)
- Inference optimization preset loading
- Data validation pipeline
- Attention visualization (dry-run)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import yaml

from open_vision_banana.inference.optimization import (
    InferenceOptions,
    estimate_vram_savings,
    options_from_dict,
    quantize_mapper_int8,
)
from open_vision_banana.training.fsdp_utils import FSDPWrapper

# ──────────────────────────────────────────────────────────────
# FSDP Wrapper Tests
# ──────────────────────────────────────────────────────────────


class TestFSDPWrapper:
    """Tests for FSDPWrapper checkpoint and state dict management."""

    def test_init_defaults(self):
        wrapper = FSDPWrapper()
        assert wrapper.rank == 0
        assert wrapper.world_size == 1
        assert not wrapper.enabled

    def test_init_with_env(self, monkeypatch):
        monkeypatch.setenv("RANK", "2")
        monkeypatch.setenv("WORLD_SIZE", "8")
        monkeypatch.setenv("LOCAL_RANK", "1")
        wrapper = FSDPWrapper()
        assert wrapper.rank == 2
        assert wrapper.world_size == 8
        assert wrapper.local_rank == 1
        assert wrapper.enabled

    def test_is_main(self):
        wrapper = FSDPWrapper(rank=0, world_size=4)
        assert wrapper.is_main()
        wrapper2 = FSDPWrapper(rank=1, world_size=4)
        assert not wrapper2.is_main()

    def test_single_process_state_dict(self):
        """In single-process mode, state_dict should be a standard dict."""
        wrapper = FSDPWrapper(rank=0, world_size=1)
        model = torch.nn.Linear(10, 10)
        state = wrapper.full_state_dict(model)
        assert isinstance(state, dict)
        assert "weight" in state

    def test_single_process_save_load(self):
        """Save and load in single-process mode."""
        wrapper = FSDPWrapper(rank=0, world_size=1)
        model = torch.nn.Linear(10, 10)
        original_weight = model.weight.clone()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            wrapper.save_checkpoint(model, path)
            # Corrupt the model
            model.weight.data.zero_()
            wrapper.load_checkpoint(model, path)
            assert torch.allclose(model.weight, original_weight)
        finally:
            Path(path).unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────────
# Optimization Preset Tests
# ──────────────────────────────────────────────────────────────


class TestOptimizationPresets:
    """Tests for inference optimization preset system."""

    def test_options_from_dict(self):
        d = {
            "dtype": "float16",
            "enable_model_cpu_offload": True,
            "enable_vae_slicing": True,
        }
        opts = options_from_dict(d)
        assert opts.dtype == "float16"
        assert opts.enable_model_cpu_offload is True
        assert opts.enable_vae_slicing is True
        assert opts.enable_vae_tiling is False

    def test_vram_estimation_notes(self):
        opts = InferenceOptions(
            dtype="float16",
            enable_model_cpu_offload=True,
            enable_vae_slicing=True,
        )
        notes = estimate_vram_savings(opts)
        assert "dtype_float16" in notes
        assert "model_cpu_offload" in notes

    def test_preset_yaml_load(self, tmp_path):
        """Load a preset YAML and verify all profiles are valid."""
        preset_file = tmp_path / "test_presets.yaml"
        data = {
            "profiles": {
                "test_balanced": {
                    "dtype": "float16",
                    "enable_model_cpu_offload": True,
                },
                "test_speed": {
                    "dtype": "float16",
                    "enable_torch_compile": True,
                },
            }
        }
        with open(preset_file, "w") as f:
            yaml.dump(data, f)

        with open(preset_file) as f:
            loaded = yaml.safe_load(f)

        for _name, cfg in loaded["profiles"].items():
            opts = options_from_dict(cfg)
            assert isinstance(opts, InferenceOptions)
            notes = estimate_vram_savings(opts)
            assert isinstance(notes, dict)

    def test_quantize_mapper_int8(self):
        """INT8 quantization on a simple mapper."""
        model = torch.nn.Linear(64, 64)
        q_model = quantize_mapper_int8(model)
        assert q_model is not None
        x = torch.randn(1, 64)
        with torch.no_grad():
            out = q_model(x)
        assert out.shape == (1, 64)


# ──────────────────────────────────────────────────────────────
# Data Validation Tests (script-level smoke)
# ──────────────────────────────────────────────────────────────


class TestDataValidation:
    """Smoke tests for the validate_dataset script interface."""

    def test_validation_report_dataclass(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from validate_dataset import ValidationReport

        report = ValidationReport(task="depth", total_samples=10, valid_samples=8)
        assert report.valid_ratio == 0.8
        d = report.to_dict()
        assert d["task"] == "depth"
        assert d["valid_ratio"] == 0.8

    def test_validate_image_pair_missing(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from validate_dataset import validate_image_pair

        img = tmp_path / "img.png"
        tgt = tmp_path / "tgt.png"
        valid, msg = validate_image_pair(img, tgt, "depth")
        assert not valid
        assert "not found" in (msg or "").lower()

    def test_validate_image_pair_corrupt(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from validate_dataset import validate_image_pair

        img = tmp_path / "img.png"
        img.write_bytes(b"not an image")
        tgt = tmp_path / "tgt.png"
        tgt.write_bytes(b"not an image")
        valid, msg = validate_image_pair(img, tgt, "depth")
        assert not valid
        assert "corrupt" in (msg or "").lower()


# ──────────────────────────────────────────────────────────────
# Attention Visualization Tests
# ──────────────────────────────────────────────────────────────


class TestAttentionVisualization:
    """Tests for attention visualization utilities."""

    @pytest.fixture(autouse=True)
    def setup_imports(self):
        """Conditionally import matplotlib-dependent modules."""
        self._has_mpl = False
        try:
            import matplotlib  # noqa: F401
            import numpy as np  # noqa: F401

            self._has_mpl = True
        except ImportError:
            pass

    def test_attention_extractor_init(self):
        """AttentionExtractor can be initialized on a simple model."""
        if not self._has_mpl:
            pytest.skip("matplotlib not installed")
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from visualize_attention import AttentionExtractor

        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.Linear(10, 10),
        )
        extractor = AttentionExtractor(model)
        assert extractor is not None
        extractor.remove_hooks()

    def test_visualize_attention_grid(self, tmp_path):
        """Grid visualization produces a valid image file."""
        if not self._has_mpl:
            pytest.skip("matplotlib/numpy not installed")
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        import numpy as np
        from visualize_attention import visualize_attention_grid

        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        attn = torch.randn(1, 256, 5).softmax(dim=1)
        tokens = ["a", "b", "c", "d", "e"]
        out = tmp_path / "attn.png"
        visualize_attention_grid(image, attn, list(range(5)), tokens, out)
        assert out.exists()
        assert out.stat().st_size > 0


# ──────────────────────────────────────────────────────────────
# Pipeline Benchmark Tests
# ──────────────────────────────────────────────────────────────


class TestPipelineBenchmark:
    """Tests for the pipeline benchmark data structures."""

    def test_benchmark_result_serialization(self):
        from scripts.benchmark_pipeline import BenchmarkResult

        r = BenchmarkResult(
            task="depth",
            preset="balanced",
            device="cuda",
            batch_size=1,
            resolution=512,
            num_inference_steps=4,
            median_latency_ms=123.4,
            mean_latency_ms=125.0,
            std_latency_ms=5.0,
            p95_latency_ms=130.0,
            throughput_img_per_sec=8.1,
            peak_vram_mb=16000.0,
            preset_notes={"float16": "50% VRAM reduction"},
        )
        d = r.to_dict()
        assert d["task"] == "depth"
        assert d["median_latency_ms"] == 123.4
        assert d["preset_notes"]["float16"] == "50% VRAM reduction"

        # JSON roundtrip
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        assert loaded["task"] == "depth"
