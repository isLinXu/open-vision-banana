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
- LRFinder: Learning rate range test utility (Smith, 2015)
"""

from .callbacks import TrainingCallbacks
from .checkpoint import CheckpointManager
from .config import TrainingConfig, build_arg_parser, load_config
from .deepspeed_utils import (
    DEFAULT_DEEPSPEED_CONFIG,
    load_deepspeed_config,
    merge_training_into_config,
)
from .early_stopping import EarlyStopping
from .flow_matching import FlowScheduler, pack_latents, unpack_latents
from .fsdp_utils import FSDPWrapper
from .logging import TrainingLogger, create_training_logger_from_config
from .grpo import clipped_surrogate, decoded_metric_reward, group_advantages
from .losses import TrainingLosses
from .opd import TeacherBank, opd_loss, rollout_euler
from .post_training import grpo_sft_gate_status, load_sft_gate, sft_gate_from_eval, write_sft_gate
from .lr_finder import LRFinder
from .model_backend import ModelBackend
from .trainer import Trainer
from .trainer_interface import (
    AccelerateTrainer,
    DDPTrainer,
    DeepSpeedTrainer,
    FSDPTrainer,
    NativeTrainer,
    TrainerInterface,
    create_trainer,
)
from .tuner import InstructionTuner, VisionTaskDataset
from .utils import DDPWrapper, EMAModel, VisionAugmentation

__all__ = [
    # Unified training framework
    "Trainer",
    "TrainingConfig",
    "ModelBackend",
    "FlowScheduler",
    "TrainingLosses",
    "TeacherBank",
    "opd_loss",
    "rollout_euler",
    "clipped_surrogate",
    "decoded_metric_reward",
    "group_advantages",
    "grpo_sft_gate_status",
    "load_sft_gate",
    "sft_gate_from_eval",
    "write_sft_gate",
    "TrainingCallbacks",
    "build_arg_parser",
    "load_config",
    "pack_latents",
    "unpack_latents",
    # Legacy training infrastructure
    "InstructionTuner",
    "VisionTaskDataset",
    "EMAModel",
    "DDPWrapper",
    "FSDPWrapper",
    "DEFAULT_DEEPSPEED_CONFIG",
    "load_deepspeed_config",
    "merge_training_into_config",
    "VisionAugmentation",
    "CheckpointManager",
    "EarlyStopping",
    "TrainingLogger",
    "create_training_logger_from_config",
    "TrainerInterface",
    "NativeTrainer",
    "DDPTrainer",
    "FSDPTrainer",
    "DeepSpeedTrainer",
    "AccelerateTrainer",
    "create_trainer",
    "LRFinder",
]
