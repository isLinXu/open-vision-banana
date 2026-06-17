"""Training sub-package — LoRA instruction tuning and utilities.

Provides:
- InstructionTuner: rsLoRA training pipeline
- VisionTaskDataset: Base dataset for vision tasks
- EMAModel: Exponential Moving Average for stable training
- DDPWrapper: Distributed Data Parallel support
- VisionAugmentation: Task-aware data augmentation
- CheckpointManager: Save/load training checkpoints
- TrainerInterface: Decoupled trainer for multiple backends
- NativeTrainer: Single GPU/CPU PyTorch native trainer
- DDPTrainer: Multi-GPU DDP trainer
- AccelerateTrainer: HuggingFace Accelerate trainer
- create_trainer: Factory function for trainer creation
"""

from .checkpoint import CheckpointManager
from .trainer_interface import (
    AccelerateTrainer,
    DDPTrainer,
    NativeTrainer,
    TrainerInterface,
    create_trainer,
)
from .tuner import InstructionTuner, VisionTaskDataset
from .utils import DDPWrapper, EMAModel, VisionAugmentation

__all__ = [
    "InstructionTuner",
    "VisionTaskDataset",
    "EMAModel",
    "DDPWrapper",
    "VisionAugmentation",
    "CheckpointManager",
    "TrainerInterface",
    "NativeTrainer",
    "DDPTrainer",
    "AccelerateTrainer",
    "create_trainer",
]
