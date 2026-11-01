"""Data-parallel helpers for the unified Trainer (``scripts/train.py --ddp``).

In LoRA mode only the LoRA parameters train, so instead of wrapping the backbone in
``DistributedDataParallel`` (which would change how ``ModelBackend`` calls it),
every rank runs the plain model on its own shard of the data and the trainable
gradients are averaged across ranks right before each optimizer step. With the
same initial weights and the same averaged gradients, every rank applies the
identical update, so parameters and EMA stay in sync without extra traffic.
Full-parameter mode shards the backbone instead (``full_finetune.py``).
"""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistContext:
    """Where this process sits in a (possibly single-process) job."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistContext:
    """Set up the process group when launched by torchrun (``WORLD_SIZE`` > 1).

    A plain ``python scripts/train.py`` run gets a single-process context.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return DistContext()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    logger.info("DDP: rank %d/%d (local rank %d, %s)", rank, world_size, local_rank, backend)
    return DistContext(rank=rank, world_size=world_size, local_rank=local_rank)


def ensure_process_group() -> None:
    """FSDP needs a process group; a plain single-process run gets a one-rank group."""
    if dist.is_initialized():
        return
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    dist.init_process_group(
        "nccl" if torch.cuda.is_available() else "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=0,
        world_size=1,
    )


def broadcast_float(value: float, src: int = 0) -> float:
    """Rank ``src``'s value on every rank (keeps collective decisions identical)."""
    if not dist.is_initialized():
        return value
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.broadcast(tensor, src=src)
    return float(tensor.item())


def clip_grad_norm(params: list[torch.nn.Parameter], max_norm: float) -> float:
    """Clip the (already synchronized) gradients and return their global pre-clip L2 norm.

    Sharded (FSDP2) gradients give a DTensor norm, which is gathered across ranks.
    """
    norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    if hasattr(norm, "full_tensor"):
        norm = norm.full_tensor()
    return float(norm)


def broadcast_parameters_(params: Iterable[torch.nn.Parameter], ctx: DistContext) -> None:
    """Copy rank 0's values of ``params`` to every rank."""
    if not ctx.enabled:
        return
    for p in params:
        dist.broadcast(p.data, src=0)


def all_reduce_gradients_(params: Iterable[torch.nn.Parameter], ctx: DistContext) -> None:
    """Average the gradients of ``params`` across ranks in place.

    A parameter without a gradient on this rank contributes zeros, so every rank
    reduces the same flat buffer layout.
    """
    if not ctx.enabled:
        return
    params = list(params)
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    grads = [p.grad for p in params]
    flat = _flatten_dense_tensors(grads)
    dist.all_reduce(flat)
    flat.div_(ctx.world_size)
    for grad, synced in zip(grads, _unflatten_dense_tensors(flat, grads), strict=True):
        grad.copy_(synced)


def barrier(ctx: DistContext) -> None:
    if ctx.enabled:
        dist.barrier()


def cleanup(ctx: DistContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()
