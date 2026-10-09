"""Open Vision Banana — Image Generators are Generalist Vision Learners.

Unofficial open-source implementation of Vision Banana: instruction-tuning
an open image generator on a mixture of its original training data alongside
a small amount of vision task data, with all task outputs parameterized as
decodable RGB images. The paper reports SOTA zero-shot transfer on
segmentation, depth and surface normal estimation; this package does not
claim to reproduce those numbers.

Subpackages:
- core: Reversible RGB mapping algorithms (no external deps)
- inference: Model loading, generation, and decoding
- training: LoRA instruction tuning and utilities
- evaluation: Metrics for all vision tasks
- export: ONNX and TorchScript model export
- data: Dataset loaders for vision task instruction tuning
"""

__version__ = "0.5.0"

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
        # Tiny standalone model
        "TinyVisionBanana": ".tiny_model",
        "TinyVisionBananaConfig": ".tiny_model",
        # Inference
        "BaseGenerator": ".inference.generator",
        "PromptBuilder": ".inference.prompt",
        "VisualizationDecoder": ".inference.decoder",
        "FrozenDepthEstimator": ".inference.depth_probe",
        "VisionBanana": ".inference.pipeline",
        "InferenceDiagnosticsWriter": ".inference.diagnostics",
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
        "export_flux_transformer_onnx": ".export.onnx_export",
        "export_full_pipeline_onnx": ".export.onnx_export",
        # Serving imports FastAPI lazily inside create_app().
        "create_app": ".serving.fastapi_app",
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
