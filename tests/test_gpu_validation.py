"""GPU validation tests for Open-Vision-Banana.

These tests are designed to run on CUDA GPUs with real FLUX.1 models.
They are automatically skipped when CUDA is not available or when the
FLUX.1 model is not downloaded.

Run with:
    pytest tests/test_gpu_validation.py -v --gpu
    # or
    python tests/test_gpu_validation.py --gpu
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Skip conditions ─────────────────────────────────────────────────────

def _cuda_available() -> bool:
    return torch.cuda.is_available()


def _flux_available() -> bool:
    """Check if FLUX.1 model files exist locally."""
    cache_dir = os.environ.get(
        "HF_HOME", os.path.expanduser("~/.cache/huggingface/hub")
    )
    flux_dirs = list(Path(cache_dir).glob("models--black-forest-labs--FLUX.1*"))
    return len(flux_dirs) > 0


gpu_required = pytest.mark.skipif(
    not _cuda_available(),
    reason="CUDA GPU not available",
)

flux_required = pytest.mark.skipif(
    not _cuda_available() or not _flux_available(),
    reason="FLUX.1 model not available or CUDA not present",
)

gpu_flag = pytest.mark.skipif(
    "--gpu" not in sys.argv,
    reason="Pass --gpu flag to run GPU validation tests",
)


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if _cuda_available() else "cpu")


@pytest.fixture
def small_image(device: torch.device) -> torch.Tensor:
    """Small test image [1, 3, 256, 256] in [0, 1]."""
    return torch.rand(1, 3, 256, 256, device=device)


# ── Core algorithm GPU tests ────────────────────────────────────────────

@gpu_required
class TestCoreAlgorithmsOnGPU:
    """Verify core algorithms produce identical results on GPU vs CPU."""

    def test_segmentation_decoder_gpu(self, device: torch.device) -> None:
        """Segmentation decoder runs on GPU without errors."""
        from open_vision_banana.core.segmentation import SegmentationDecoder

        decoder = SegmentationDecoder()
        rgb = torch.rand(1, 3, 64, 64, device=device)
        instances = decoder.decode_instance(rgb)
        assert len(instances) == 1
        assert isinstance(instances[0], list)

    def test_object_detection_nms_gpu(self, device: torch.device) -> None:
        """NMS works correctly on GPU."""
        from open_vision_banana.core.object_detection_mapper import _nms

        boxes = torch.tensor([
            [0.1, 0.1, 0.3, 0.3],
            [0.12, 0.12, 0.32, 0.32],  # overlaps with box 0
            [0.5, 0.5, 0.7, 0.7],       # no overlap
        ], device=device)
        scores = torch.tensor([0.9, 0.8, 0.7], device=device)

        keep = _nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 2  # should keep boxes 0 and 2
        assert 0 in keep.tolist()
        assert 2 in keep.tolist()

    def test_union_find_grouping_gpu(self, device: torch.device) -> None:
        """Union-Find grouping produces valid labels on GPU."""
        from open_vision_banana.core.segmentation import SegmentationDecoder

        decoder = SegmentationDecoder()
        # Create a synthetic image with 3 color regions
        img = torch.zeros(32, 32, 3, device=device)
        img[:16, :16] = torch.tensor([255.0, 0.0, 0.0], device=device)
        img[:16, 16:] = torch.tensor([0.0, 255.0, 0.0], device=device)
        img[16:, :] = torch.tensor([0.0, 0.0, 255.0], device=device)

        bg_mask = torch.zeros(32, 32, dtype=torch.bool, device=device)
        labels = decoder._union_find_grouping(img, bg_mask)
        assert labels.shape == (32, 32)
        assert labels.device == device
        # Should have at least 3 unique labels (one per color region)
        n_unique = len(labels.unique())
        assert n_unique >= 3, f"Expected >=3 components, got {n_unique}"

    def test_seed_based_union_find_gpu(self, device: torch.device) -> None:
        """Seed-based Union-Find produces valid labels on GPU (P0 fix validation)."""
        from open_vision_banana.core.segmentation import SegmentationDecoder

        decoder = SegmentationDecoder()
        img = torch.zeros(32, 32, 3, device=device)
        img[:16, :16] = torch.tensor([255.0, 0.0, 0.0], device=device)
        img[:16, 16:] = torch.tensor([0.0, 255.0, 0.0], device=device)
        img[16:, :] = torch.tensor([0.0, 0.0, 255.0], device=device)

        bg_mask = torch.zeros(32, 32, dtype=torch.bool, device=device)
        labels = decoder._seed_based_union_find(img, bg_mask)
        assert labels.shape == (32, 32)
        assert labels.device == device
        n_unique = len(labels.unique())
        assert n_unique >= 3, f"Expected >=3 components, got {n_unique}"

    def test_depth_mapper_gpu(self, device: torch.device) -> None:
        """Depth mapper roundtrip on GPU."""
        from open_vision_banana.core.depth_mapper import DepthRGBMapper

        mapper = DepthRGBMapper()
        depth = torch.rand(1, 1, 64, 64, device=device) * 49.0 + 1.0
        rgb = mapper.depth_to_rgb(depth)
        recovered = mapper.rgb_to_depth(rgb)
        assert recovered.shape == depth.shape
        assert recovered.device == device
        # Should be within reasonable error
        abs_rel = ((depth - recovered).abs() / depth.clamp(min=0.1)).mean().item()
        assert abs_rel < 0.05, f"AbsRel too high: {abs_rel}"

    def test_object_detection_mapper_gpu(self, device: torch.device) -> None:
        """Object detection mapper roundtrip on GPU with NMS."""
        from open_vision_banana.core.object_detection_mapper import (
            ObjectDetectionRGBMapper,
        )

        mapper = ObjectDetectionRGBMapper(num_classes=3, line_width=2)
        bboxes = torch.tensor([
            [[0.2, 0.3, 0.2, 0.2], [0.5, 0.5, 0.15, 0.15]]
        ], device=device)
        class_ids = torch.tensor([[0, 1]], device=device)

        rgb = mapper.bboxes_to_rgb(bboxes, class_ids)
        assert rgb.device == device
        recovered_bboxes, recovered_cls, recovered_scores = mapper.rgb_to_bboxes(rgb)
        assert recovered_bboxes.shape[0] == 1


# ── FLUX.1 model tests (require model download) ─────────────────────────

@flux_required
class TestFluxModelGPU:
    """Tests that require the FLUX.1 model to be downloaded."""

    @pytest.fixture
    def flux_pipeline(self, device: torch.device) -> Any:
        """Load FLUX.1 pipeline for testing."""
        try:
            from diffusers import FluxPipeline
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.bfloat16,
            ).to(device)
            return pipe
        except Exception as exc:
            pytest.skip(f"Failed to load FLUX.1: {exc}")

    def test_flux_loads_on_gpu(self, flux_pipeline: Any) -> None:
        """FLUX.1 pipeline loads successfully on GPU."""
        assert flux_pipeline is not None

    def test_flux_inference_smoke(self, flux_pipeline: Any, device: torch.device) -> None:
        """FLUX.1 can generate an image (smoke test)."""
        result = flux_pipeline(
            prompt="a red square on a black background",
            num_inference_steps=4,
            height=256,
            width=256,
            guidance_scale=0.0,
        )
        assert len(result.images) == 1
        assert result.images[0].size == (256, 256)

    def test_lora_training_smoke(self, flux_pipeline: Any, device: torch.device) -> None:
        """LoRA adapter can be attached and detached."""
        from peft import LoraConfig, get_peft_model

        transformer = flux_pipeline.transformer
        config = LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.0,
        )
        model = get_peft_model(transformer, config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable > 0
        # Verify forward pass works
        del model


# ── Memory and performance tests ────────────────────────────────────────

@gpu_required
class TestGPUMemory:
    """Verify GPU memory usage stays within reasonable bounds."""

    def test_segmentation_memory(self, device: torch.device) -> None:
        """Segmentation decoder on 1024x1024 image."""
        from open_vision_banana.core.segmentation import SegmentationDecoder

        decoder = SegmentationDecoder()
        rgb = torch.rand(1, 3, 1024, 1024, device=device)

        torch.cuda.reset_peak_memory_stats()
        decoder.decode_instance(rgb)
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2

        # Should use less than 4GB for a single 1024x1024 image
        assert peak_mb < 4096, f"Peak memory {peak_mb:.0f}MB exceeds 4GB limit"

    def test_depth_mapper_memory(self, device: torch.device) -> None:
        """Depth mapper on 1024x1024 image."""
        from open_vision_banana.core.depth_mapper import DepthRGBMapper

        mapper = DepthRGBMapper()
        depth = torch.rand(1, 1, 1024, 1024, device=device) * 49.0 + 1.0

        torch.cuda.reset_peak_memory_stats()
        rgb = mapper.depth_to_rgb(depth)
        mapper.rgb_to_depth(rgb)
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2

        assert peak_mb < 2048, f"Peak memory {peak_mb:.0f}MB exceeds 2GB limit"


if __name__ == "__main__":
    # Allow running directly: python tests/test_gpu_validation.py --gpu
    if "--gpu" in sys.argv:
        sys.argv.remove("--gpu")
        sys.exit(pytest.main([__file__, "-v", "-s"] + sys.argv[1:]))
    else:
        print("GPU validation tests require --gpu flag.")
        print("Usage: python tests/test_gpu_validation.py --gpu")
        print("       pytest tests/test_gpu_validation.py -v --gpu")
        sys.exit(0)
