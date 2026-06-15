"""Inference sub-package — model loading, generation, and decoding.

Provides:
- BaseGenerator: FLUX.1 + LoRA wrapper
- PromptBuilder: Task-specific instruction prompt construction
- VisualizationDecoder: RGB output → structured vision results
- VisionBanana: Unified multi-task model (high-level API)
"""

from .decoder import VisualizationDecoder
from .generator import BaseGenerator
from .pipeline import VisionBanana
from .prompt import PromptBuilder

__all__ = [
    "BaseGenerator",
    "PromptBuilder",
    "VisualizationDecoder",
    "VisionBanana",
]
