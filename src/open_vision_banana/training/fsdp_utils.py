"""FSDP (Fully Sharded Data Parallel) utilities for large-model training.

Wraps FLUX transformer + LoRA for multi-GPU memory-efficient training.
Intended usage::

    torchrun --nproc_per_node=8 scripts/train.py --fsdp --data-path ...
"""

from __future__ import annotations

import logging
import os
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DistributedSampler

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val is not None else default


class FSDPWrapper:
    """Coordinator for PyTorch FSDP multi-GPU training."""

    def __init__(
        self,
        *,
        sharding_strategy: str = "full",
        mixed_precision: bool = True,
        min_wrap_params: int = 1_000_000,
        backend: str = "nccl",
        rank: int | None = None,
        world_size: int | None = None,
        local_rank: int | None = None,
    ) -> None:
        self.sharding_strategy = sharding_strategy
        self.mixed_precision = mixed_precision
        self.min_wrap_params = min_wrap_params
        self.backend = backend
        self.rank = rank if rank is not None else _env_int("RANK", 0)
        self.world_size = world_size if world_size is not None else _env_int("WORLD_SIZE", 1)
        self.local_rank = local_rank if local_rank is not None else _env_int("LOCAL_RANK", 0)
        self._initialized = False

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    def initialize(self) -> None:
        """Initialize distributed process group when running multi-GPU."""
        if not self.enabled:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("FSDP requires CUDA GPUs")
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=self.backend)
            self._initialized = True
            torch.cuda.set_device(self.local_rank)
            logger.info(
                "FSDP initialized: rank=%d/%d local_rank=%d",
                self.rank,
                self.world_size,
                self.local_rank,
            )

    def wrap(self, model: nn.Module) -> nn.Module:
        """Wrap *model* with FullyShardedDataParallel."""
        self.initialize()
        if not self.enabled:
            device = next(model.parameters()).device
            return model.to(device)

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        strategy_map = {
            "full": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD,
        }
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=self.min_wrap_params,
        )
        mp = None
        if self.mixed_precision:
            mp = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )

        device = torch.device(f"cuda:{self.local_rank}")
        return FSDP(
            model.to(device),
            sharding_strategy=strategy_map.get(self.sharding_strategy, ShardingStrategy.FULL_SHARD),
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp,
            device_id=self.local_rank,
        )

    def get_train_sampler(self, dataset) -> DistributedSampler | None:
        if not self.enabled:
            return None
        return DistributedSampler(dataset, shuffle=True)

    def set_epoch(self, sampler: DistributedSampler | None, epoch: int) -> None:
        if sampler is not None:
            sampler.set_epoch(epoch)

    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def full_state_dict(self, model: nn.Module) -> dict:
        """Return full (non-sharded) state dict for checkpoint saving."""
        if not self.enabled:
            return model.state_dict()
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            return model.state_dict()

    def load_full_state_dict(self, model: nn.Module, state_dict: dict) -> None:
        """Load full state dict into a sharded FSDP model."""
        if not self.enabled:
            model.load_state_dict(state_dict)
            return
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            model.load_state_dict(state_dict)

    def save_checkpoint(self, model: nn.Module, path: str) -> None:
        """Save FSDP-compatible checkpoint (only rank 0 writes)."""
        if self.is_main():
            state_dict = self.full_state_dict(model)
            torch.save(state_dict, path)
        self.barrier()

    def load_checkpoint(self, model: nn.Module, path: str) -> None:
        """Load FSDP-compatible checkpoint."""
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.load_full_state_dict(model, state_dict)
        self.barrier()

    def cleanup(self) -> None:
        if self._initialized:
            torch.distributed.destroy_process_group()


__all__ = ["FSDPWrapper", "_env_int"]
