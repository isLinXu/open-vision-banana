"""Training sub-package — LoRA instruction tuning and utilities.

Provides:
- InstructionTuner: rsLoRA training pipeline
- VisionTaskDataset: Base dataset for vision tasks
- EMAModel: Exponential Moving Average for stable training
- DDPWrapper: Distributed Data Parallel support
- VisionAugmentation: Task-aware data augmentation
- CheckpointManager: Save/load training checkpoints
"""

from .checkpoint import CheckpointManager
from .tuner import InstructionTuner, VisionTaskDataset
from .utils import DDPWrapper, EMAModel, VisionAugmentation

__all__ = [
    "InstructionTuner",
    "VisionTaskDataset",
    "EMAModel",
    "DDPWrapper",
    "VisionAugmentation",
    "CheckpointManager",
]
