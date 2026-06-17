"""Decoupled trainer interface for multiple training backends.

Provides an abstract TrainerInterface that decouples the training loop
from specific backend implementations (PyTorch native, HuggingFace
Accelerate, DeepSpeed, FSDP, etc.).

This allows users to switch between training backends without modifying
the core training logic.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class TrainerInterface(ABC):
    """Abstract trainer interface that decouples training loop from backend.

    Subclasses implement backend-specific logic for:
    - Model wrapping (DDP, FSDP, DeepSpeed, etc.)
    - Device management
    - Gradient scaling (AMP, DeepSpeed FP16, etc.)
    - Checkpoint save/load
    - Distributed coordination

    The core training loop (forward, backward, optimizer step) is shared
    across all backends via the train() method.
    """

    @abstractmethod
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model for the specific backend (DDP, FSDP, etc.)."""
        ...

    @abstractmethod
    def move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to the appropriate device for this backend."""
        ...

    @abstractmethod
    def backward(self, loss: torch.Tensor) -> None:
        """Compute gradients, handling backend-specific scaling."""
        ...

    @abstractmethod
    def optimizer_step(self) -> None:
        """Perform optimizer step with backend-specific handling."""
        ...

    @abstractmethod
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        output_dir: str,
        **kwargs,
    ) -> None:
        """Save checkpoint in backend-specific format."""
        ...

    @abstractmethod
    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: str,
    ) -> dict:
        """Load checkpoint from backend-specific format."""
        ...

    @abstractmethod
    def is_main_process(self) -> bool:
        """Whether this is the main (rank 0) process."""
        ...

    @abstractmethod
    def barrier(self) -> None:
        """Synchronize all processes."""
        ...


class NativeTrainer(TrainerInterface):
    """Native PyTorch trainer — single GPU or CPU.

    Uses standard PyTorch training loop with optional AMP.
    No distributed training support.
    """

    def __init__(
        self,
        device: str = "auto",
        use_amp: bool = True,
    ) -> None:
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.use_amp = use_amp and device != "cpu"
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp
        )

    def wrap_model(self, model: nn.Module) -> nn.Module:
        return model.to(self.device)

    def move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device)

    def backward(self, loss: torch.Tensor) -> None:
        self.scaler.scale(loss).backward()

    def optimizer_step(self) -> None:
        self.scaler.unscale_(self._optimizer)
        torch.nn.utils.clip_grad_norm_(
            self._optimizer.param_groups[0]["params"],
            max_norm=1.0,
        )
        self.scaler.step(self._optimizer)
        self.scaler.update()

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        output_dir: str,
        **kwargs,
    ) -> None:
        ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / "model.pt")
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        logger.info(f"Checkpoint saved to {ckpt_dir}")

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: str,
    ) -> dict:
        ckpt_dir = Path(checkpoint_path)
        model.load_state_dict(
            torch.load(ckpt_dir / "model.pt", weights_only=True)
        )
        optimizer.load_state_dict(
            torch.load(ckpt_dir / "optimizer.pt", weights_only=True)
        )
        logger.info(f"Checkpoint loaded from {ckpt_dir}")
        return {}

    def is_main_process(self) -> bool:
        return True

    def barrier(self) -> None:
        pass  # No-op for single process


class DDPTrainer(NativeTrainer):
    """Distributed Data-Parallel trainer using torch.distributed.

    Wraps model with DistributedDataParallel for multi-GPU training.
    """

    def __init__(
        self,
        world_size: int = 1,
        rank: int = 0,
        local_rank: int = 0,
        backend: str = "nccl",
        use_amp: bool = True,
    ) -> None:
        super().__init__(
            device=f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu",
            use_amp=use_amp,
        )
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self.backend = backend
        self._initialized = False

    def _init_distributed(self) -> None:
        """Initialize distributed process group."""
        if self.world_size <= 1 or self._initialized:
            return
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=self.backend,
                world_size=self.world_size,
                rank=self.rank,
            )
            self._initialized = True
            logger.info(f"DDP initialized: rank={self.rank}/{self.world_size}")

    def wrap_model(self, model: nn.Module) -> nn.Module:
        self._init_distributed()
        if self.world_size <= 1:
            return model.to(self.device)
        model = model.to(self.device)
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
        )

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        output_dir: str,
        **kwargs,
    ) -> None:
        # Only rank 0 saves checkpoint
        if not self.is_main_process():
            return
        # Unwrap DDP model for state_dict
        unwrapped = model.module if hasattr(model, "module") else model
        super().save_checkpoint(unwrapped, optimizer, step, output_dir, **kwargs)

    def is_main_process(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.barrier()


class AccelerateTrainer(TrainerInterface):
    """Trainer using HuggingFace Accelerate backend.

    Delegates device placement, gradient scaling, and distributed
    coordination to the Accelerate library.
    """

    def __init__(self, **accelerate_kwargs) -> None:
        try:
            from accelerate import Accelerator
        except ImportError:
            raise ImportError(
                "HuggingFace Accelerate is required for AccelerateTrainer. "
                "Install with: pip install accelerate"
            )
        self.accelerator = Accelerator(**accelerate_kwargs)

    def wrap_model(self, model: nn.Module) -> nn.Module:
        # Accelerate handles wrapping internally via .prepare()
        return model

    def prepare(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        scheduler=None,
    ) -> tuple:
        """Prepare model, optimizer, dataloader, and scheduler for Accelerate."""
        components = [model, optimizer, dataloader]
        if scheduler is not None:
            components.append(scheduler)
        return self.accelerator.prepare(*components)

    def move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.accelerator.device)

    def backward(self, loss: torch.Tensor) -> None:
        self.accelerator.backward(loss)

    def optimizer_step(self) -> None:
        # Accelerate handles optimizer step internally
        pass

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        output_dir: str,
        **kwargs,
    ) -> None:
        if self.is_main_process():
            unwrapped = self.accelerator.unwrap_model(model)
            ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            unwrapped.save_pretrained(ckpt_dir)
            logger.info(f"Checkpoint saved to {ckpt_dir}")

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: str,
    ) -> dict:
        unwrapped = self.accelerator.unwrap_model(model)
        from peft import PeftModel
        if isinstance(unwrapped, PeftModel):
            loaded = PeftModel.from_pretrained(
                unwrapped.base_model.model, checkpoint_path
            )
            unwrapped.load_state_dict(loaded.state_dict(), strict=False)
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return {}

    def is_main_process(self) -> bool:
        return self.accelerator.is_main_process

    def barrier(self) -> None:
        self.accelerator.wait_for_everyone()


def create_trainer(
    backend: str = "native",
    **kwargs,
) -> TrainerInterface:
    """Factory function to create a trainer for the specified backend.

    Args:
        backend: Training backend name. One of:
            - "native": Single GPU/CPU with PyTorch native
            - "ddp": Multi-GPU with DistributedDataParallel
            - "accelerate": HuggingFace Accelerate
        **kwargs: Backend-specific arguments.

    Returns:
        TrainerInterface instance for the specified backend.
    """
    trainers = {
        "native": NativeTrainer,
        "ddp": DDPTrainer,
        "accelerate": AccelerateTrainer,
    }

    if backend not in trainers:
        raise ValueError(
            f"Unknown training backend: {backend}. "
            f"Available: {list(trainers.keys())}"
        )

    return trainers[backend](**kwargs)


__all__ = [
    "TrainerInterface",
    "NativeTrainer",
    "DDPTrainer",
    "AccelerateTrainer",
    "create_trainer",
]
