"""Open Vision Banana — Image Generators are Generalist Vision Learners.

Open-source implementation of Vision Banana: instruction-tuning FLUX.1
on a mixture of its original training data alongside a small amount of
vision task data, achieving SOTA zero-shot transfer on segmentation,
depth estimation, and surface normal estimation.

Subpackages:
- core: Reversible RGB mapping algorithms (no external deps)
- inference: Model loading, generation, and decoding
- training: LoRA instruction tuning and utilities
- evaluation: Metrics for all vision tasks
- export: ONNX and TorchScript model export
- data: Dataset loaders for vision task instruction tuning
"""

__version__ = "0.4.2"

# Lazy imports — only load what's needed
from .config import VisionBananaConfig

__all__ = [
    "VisionBananaConfig",
    # Lazy imports for heavy subpackages:
    # from .core import DepthRGBMapper, NormalRGBMapper, ...
    # from .inference import VisionBanana, ...
    # from .training import InstructionTuner, ...
    # from .evaluation import VisionEvaluator, ...
    # from .export import export_mapper_onnx, ...
]


def __getattr__(name):
    """Lazy import for heavy subpackages."""
    _lazy_map = {
        # Core
        "DepthRGBMapper": ".core.depth_mapper",
        "NormalRGBMapper": ".core.normal_mapper",
        "SegmentationDecoder": ".core.segmentation",
        "RGBMapper": ".core.rgb_mapper",
        "CUBE_EDGE_KEYPOINTS": ".core.depth_mapper",
        # Inference
        "BaseGenerator": ".inference.generator",
        "PromptBuilder": ".inference.prompt",
        "VisualizationDecoder": ".inference.decoder",
        "VisionBanana": ".inference.pipeline",
        # Training
        "InstructionTuner": ".training.tuner",
        "EMAModel": ".training.utils",
        "DDPWrapper": ".training.utils",
        "VisionAugmentation": ".training.utils",
        "CheckpointManager": ".training.checkpoint",
        # Evaluation
        "evaluate_depth": ".evaluation.depth",
        "evaluate_normals": ".evaluation.normal",
        "evaluate_semantic": ".evaluation.semantic",
        "evaluate_instance": ".evaluation.instance",
        "evaluate_referring": ".evaluation.referring",
        "VisionEvaluator": ".evaluation.evaluator",
        # Export
        "DepthRGBMapperONNX": ".export.onnx_export",
        "NormalRGBMapperONNX": ".export.onnx_export",
        "export_mapper_onnx": ".export.onnx_export",
        # Data
        "build_dataset": ".data.datasets",
        "SACoInstanceDataset": ".data.datasets",
        "MixedVisionDataset": ".data.mixed",
    }

    if name in _lazy_map:
        import importlib

        module = importlib.import_module(_lazy_map[name], __package__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
