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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

    Attributes:
        _optimizer: The optimizer associated with this trainer. Set via
            ``setup_optimizer()`` or directly by the training loop before
            calling ``optimizer_step()``.
    """

    _optimizer: torch.optim.Optimizer | None = None

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

    def setup_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Associate an optimizer with this trainer.

        Must be called before ``optimizer_step()`` for backends that
        require the optimizer reference (e.g. NativeTrainer, DDPTrainer).
        AccelerateTrainer delegates this internally via ``.prepare()``.

        Args:
            optimizer: The optimizer to use for gradient updates.
        """
        self._optimizer = optimizer

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

    Supports gradient accumulation and configurable gradient clipping
    for stable training at large effective batch sizes.
    """

    def __init__(
        self,
        device: str = "auto",
        use_amp: bool = True,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        # P0-4: MPS supports bfloat16 autocast but not GradScaler.
        self.use_autocast = use_amp and device != "cpu"
        self.use_grad_scaler = use_amp and device == "cuda"
        # Backward-compatible alias — callers checking use_amp mean autocast.
        self.use_amp = self.use_autocast
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.use_grad_scaler
        )

    def wrap_model(self, model: nn.Module) -> nn.Module:
        return model.to(self.device)

    def move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device)

    def backward(self, loss: torch.Tensor) -> None:
        self.scaler.scale(loss).backward()

    def optimizer_step(self) -> None:
        if self._optimizer is None:
            raise RuntimeError(
                "NativeTrainer.optimizer_step() called before setup_optimizer(). "
                "Call trainer.setup_optimizer(optimizer) first."
            )
        self.scaler.unscale_(self._optimizer)
        torch.nn.utils.clip_grad_norm_(
            self._optimizer.param_groups[0]["params"],
            max_norm=self.max_grad_norm,
            error_if_nonfinite=True,
        )
        self.scaler.step(self._optimizer)
        self.scaler.update()

    def train_step(
        self,
        model: nn.Module,
        loss: torch.Tensor,
        step_in_accumulation: int,
    ) -> bool:
        """Execute a single training step with gradient accumulation.

        Returns True if an optimizer step was performed (i.e., the
        accumulation boundary was reached), False otherwise.

        Args:
            model: The model being trained.
            loss: The loss tensor for this micro-batch.
            step_in_accumulation: 0-indexed step within the current
                accumulation cycle.  When
                ``(step + 1) % gradient_accumulation_steps == 0``,
                the optimizer step and zero_grad are executed.

        Example::

            for step, batch in enumerate(loader):
                loss = compute_loss(model, batch)
                stepped = trainer.train_step(model, loss, step)
        """
        if not torch.isfinite(loss).all():
            raise FloatingPointError(
                f"Non-finite loss at accumulation step {step_in_accumulation}"
            )
        scaled_loss = loss / self.gradient_accumulation_steps
        self.backward(scaled_loss)

        do_step = (step_in_accumulation + 1) % self.gradient_accumulation_steps == 0
        if do_step:
            self.optimizer_step()
            self._optimizer.zero_grad()
        return do_step

    def flush_accumulation(self, accumulated_steps: int) -> bool:
        """Commit a final partial accumulation window at epoch end."""
        if accumulated_steps <= 0:
            return False
        if self._optimizer is None:
            raise RuntimeError("flush_accumulation() called before setup_optimizer()")
        if accumulated_steps != self.gradient_accumulation_steps:
            scale = self.gradient_accumulation_steps / accumulated_steps
            for group in self._optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)
        self.optimizer_step()
        self._optimizer.zero_grad()
        return True

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
        # P1-15: PEFT checkpoints use adapter layout, not flat model.pt
        try:
            from peft import PeftModel

            if isinstance(model, PeftModel):
                loaded = PeftModel.from_pretrained(
                    model.base_model.model,
                    str(ckpt_dir),
                )
                model.load_state_dict(loaded.state_dict(), strict=False)
                del loaded
                logger.info(f"PEFT checkpoint loaded from {ckpt_dir}")
            elif (ckpt_dir / "model.pt").exists():
                model.load_state_dict(
                    torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=True)
                )
                logger.info(f"Checkpoint loaded from {ckpt_dir / 'model.pt'}")
            elif (ckpt_dir / "adapter_config.json").exists():
                loaded = PeftModel.from_pretrained(model, str(ckpt_dir))
                model.load_state_dict(loaded.state_dict(), strict=False)
                del loaded
                logger.info(f"PEFT adapter loaded from {ckpt_dir}")
        except Exception as e:
            logger.warning(f"Checkpoint load failed: {e}")

        if optimizer is not None and (ckpt_dir / "optimizer.pt").exists():
            optimizer.load_state_dict(
                torch.load(ckpt_dir / "optimizer.pt", weights_only=True)
            )
        return {}

    def is_main_process(self) -> bool:
        return True

    def barrier(self) -> None:
        pass  # No-op for single process


class FSDPTrainer(NativeTrainer):
    """Fully Sharded Data-Parallel trainer using torch.distributed.fsdp.

    Shards model parameters across GPUs for memory-efficient training of
    large models (e.g. FLUX.1 12B + LoRA on 8×A100).
    """

    def __init__(
        self,
        world_size: int = 1,
        rank: int = 0,
        local_rank: int = 0,
        sharding_strategy: str = "full",
        mixed_precision: bool = True,
        min_wrap_params: int = 1_000_000,
        backend: str = "nccl",
        use_amp: bool = True,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        super().__init__(
            device=f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu",
            use_amp=use_amp and torch.cuda.is_available(),
            max_grad_norm=max_grad_norm,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        from .fsdp_utils import FSDPWrapper

        self._fsdp = FSDPWrapper(
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            min_wrap_params=min_wrap_params,
            backend=backend,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        )

    def wrap_model(self, model: nn.Module) -> nn.Module:
        return self._fsdp.wrap(model)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        output_dir: str,
        **kwargs,
    ) -> None:
        if not self.is_main_process():
            return
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if isinstance(model, FSDP):
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                state = model.state_dict()
            ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(state, ckpt_dir / "model.pt")
            torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
            logger.info(f"FSDP checkpoint saved to {ckpt_dir}")
            return
        super().save_checkpoint(model, optimizer, step, output_dir, **kwargs)

    def is_main_process(self) -> bool:
        return self._fsdp.is_main()

    def barrier(self) -> None:
        self._fsdp.barrier()


class DeepSpeedTrainer(NativeTrainer):
    """DeepSpeed ZeRO trainer — delegates to ``initialize_deepspeed``.

    Note:
        ``max_grad_norm`` is used both for DeepSpeed's gradient clipping
        (passed to the DeepSpeed config) and for NativeTrainer's
        ``clip_grad_norm_`` (used when DeepSpeed is not active).
        ``gradient_accumulation_steps`` similarly serves both the DeepSpeed
        config and the NativeTrainer accumulation logic.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        micro_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        lr: float = 5e-5,
        max_grad_norm: float = 1.0,
        use_amp: bool = False,
    ) -> None:
        super().__init__(
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_amp=use_amp,
            max_grad_norm=max_grad_norm,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        self.config_path = config_path
        self.micro_batch_size = micro_batch_size
        self.lr = lr
        self._engine = None

    def wrap_model(self, model: nn.Module) -> nn.Module:
        from .deepspeed_utils import initialize_deepspeed

        engine, _ = initialize_deepspeed(
            model,
            config_path=self.config_path,
            micro_batch_size=self.micro_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            lr=self.lr,
            max_grad_norm=self.max_grad_norm,
        )
        self._engine = engine
        return engine

    def backward(self, loss: torch.Tensor) -> None:
        """Delegate backward to the DeepSpeed engine.

        DeepSpeed handles gradient scaling internally, so we bypass
        NativeTrainer's GradScaler path.
        """
        if self._engine is None:
            super().backward(loss)
        else:
            self._engine.backward(loss)

    def optimizer_step(self) -> None:
        """Delegate optimizer step to the DeepSpeed engine.

        DeepSpeed handles gradient clipping and optimizer step internally.
        """
        if self._engine is None:
            super().optimizer_step()
        else:
            self._engine.step()

    def train_step(
        self,
        model: nn.Module,
        loss: torch.Tensor,
        step_in_accumulation: int,
    ) -> bool:
        """Execute a single training step with DeepSpeed gradient accumulation.

        Returns True if an optimizer step was performed.
        """
        scaled_loss = loss / self.gradient_accumulation_steps
        self.backward(scaled_loss)

        do_step = (step_in_accumulation + 1) % self.gradient_accumulation_steps == 0
        if do_step:
            self.optimizer_step()
            if self._optimizer is not None:
                self._optimizer.zero_grad()
        return do_step

    def is_main_process(self) -> bool:
        if self._engine is None:
            return True
        from .deepspeed_utils import deepspeed_is_main

        return deepspeed_is_main(self._engine)

    def barrier(self) -> None:
        if self._engine is not None and hasattr(self._engine, "barrier"):
            self._engine.barrier()


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
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        super().__init__(
            device=f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu",
            use_amp=use_amp,
            max_grad_norm=max_grad_norm,
            gradient_accumulation_steps=gradient_accumulation_steps,
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
        except ImportError as exc:
            raise ImportError(
                "HuggingFace Accelerate is required for AccelerateTrainer. "
                "Install with: pip install accelerate"
            ) from exc
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
        """Perform optimizer step via Accelerate.

        Delegates to ``self._optimizer.step()`` (prepared by ``prepare()``).
        Accelerate handles gradient scaling internally, so no explicit
        ``unscale_`` is needed.

        Raises:
            RuntimeError: If ``setup_optimizer()`` was not called before
                this method.
        """
        if self._optimizer is None:
            raise RuntimeError(
                "AccelerateTrainer.optimizer_step() called before setup_optimizer(). "
                "Call trainer.setup_optimizer(optimizer) first, or use prepare() "
                "to let Accelerate manage the optimizer."
            )
        self._optimizer.step()

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
            loaded = PeftModel.from_pretrained(unwrapped.base_model.model, checkpoint_path)
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
            - "fsdp": Multi-GPU with FullyShardedDataParallel
            - "deepspeed": Multi-GPU with DeepSpeed ZeRO-3
            - "accelerate": HuggingFace Accelerate
        **kwargs: Backend-specific arguments.

    Returns:
        TrainerInterface instance for the specified backend.
    """
    trainers = {
        "native": NativeTrainer,
        "ddp": DDPTrainer,
        "fsdp": FSDPTrainer,
        "deepspeed": DeepSpeedTrainer,
        "accelerate": AccelerateTrainer,
    }

    if backend not in trainers:
        raise ValueError(f"Unknown training backend: {backend}. Available: {list(trainers.keys())}")

    return trainers[backend](**kwargs)


__all__ = [
    "TrainerInterface",
    "NativeTrainer",
    "DDPTrainer",
    "FSDPTrainer",
    "DeepSpeedTrainer",
    "AccelerateTrainer",
    "create_trainer",
]
