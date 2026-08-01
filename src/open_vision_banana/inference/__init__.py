"""Inference sub-package — model loading, generation, and decoding.

Provides:
- BaseGenerator: FLUX.1 + LoRA wrapper
- PromptBuilder: Task-specific instruction prompt construction
- VisualizationDecoder: RGB output → structured vision results
- VisionBanana: Unified multi-task model (high-level API)
"""

from .checkpoint import (
    CheckpointNotAvailableError,
    get_registry_entry,
    is_registry_placeholder,
    list_registry_checkpoints,
    resolve_lora_path,
)
from .decoder import VisualizationDecoder
from .depth_probe import FrozenDepthEstimator
from .diagnostics import InferenceDiagnosticsWriter
from .generator import BaseGenerator
from .optimization import InferenceOptions, apply_pipeline_optimizations
from .pipeline import VisionBanana
from .prompt import PromptBuilder

__all__ = [
    "BaseGenerator",
    "PromptBuilder",
    "VisualizationDecoder",
    "FrozenDepthEstimator",
    "InferenceDiagnosticsWriter",
    "VisionBanana",
    "InferenceOptions",
    "apply_pipeline_optimizations",
    "CheckpointNotAvailableError",
    "get_registry_entry",
    "is_registry_placeholder",
    "resolve_lora_path",
    "list_registry_checkpoints",
]
