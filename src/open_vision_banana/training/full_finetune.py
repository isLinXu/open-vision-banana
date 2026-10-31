"""Full-parameter training of the diffusion backbone with FSDP2 (``train_mode: full``).

The LoRA path keeps one bf16 backbone per rank and averages the adapter gradients
(:mod:`.distributed`). Training every backbone weight does not fit that layout: a
Qwen-Image-2.1 transformer has ~7.1B parameters, and an fp32 master copy plus the
AdamW moments alone exceed one 80 GB card, while bf16 weights updated in place lose
most steps of a 3e-5 learning rate to rounding. Here the backbone is sharded with
``fully_shard``: each rank owns 1/N of the fp32 parameters, gradients (reduced in
fp32) and AdamW state, and the forward / backward run in bf16 on per-block
all-gathers. Frozen parts (VAE, text encoder) stay replicated as before.

Every function that touches the sharded backbone is collective: all ranks call it.
Checkpoints are written by rank 0 as a diffusers ``transformer/`` directory (bf16,
the same layout ``from_pretrained`` reads), so an evaluation loads them like a
base model. The resumable state (sharded fp32 weights, optimizer, EMA) is a
``torch.distributed.checkpoint`` directory.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

BLOCK_ATTRS = ("transformer_blocks", "single_transformer_blocks", "blocks")


def ensure_process_group() -> None:
    """FSDP needs a process group; a plain single-process run gets a one-rank group."""
    if dist.is_initialized():
        return
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    dist.init_process_group(
        "nccl" if torch.cuda.is_available() else "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=0,
        world_size=1,
    )


def shard_backbone(model: torch.nn.Module) -> dict:
    """Make every backbone weight trainable and shard it with FSDP2 (fp32 master, bf16 compute).

    Each transformer block becomes one FSDP unit, the remaining weights (embedders,
    output norm / projection) the root unit. Blocks are cast to fp32 one at a time so
    the peak stays near the bf16 model plus one fp32 block. Returns what a checkpoint
    needs to rebuild the original ``config.json`` (the sharded class is renamed).
    """
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    record = {"class_name": type(model).__name__, "config": dict(getattr(model, "config", {}) or {})}
    blocks = [block for attr in BLOCK_ATTRS for block in getattr(model, attr, None) or []]
    if not blocks:
        raise ValueError(f"{type(model).__name__} has none of {BLOCK_ATTRS}; full training cannot shard it")
    model.requires_grad_(True)
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    # The root's own weights (embedders, output layers) first, while nothing is sharded yet.
    in_blocks = {id(p) for block in blocks for p in block.parameters()}
    for param in model.parameters():
        if id(param) not in in_blocks:
            param.data = param.data.float()
    for block in blocks:
        block.float()
        fully_shard(block, mp_policy=policy)
    fully_shard(model, mp_policy=policy)
    record["fsdp_units"] = len(blocks) + 1
    return record


def reshard(model: torch.nn.Module) -> None:
    """Return every FSDP unit to its sharded parameters.

    The root unit keeps its gathered weights after a forward (it expects the
    backward next), so after a no-grad forward (validation) its registered
    parameters are plain tensors until resharded; EMA swaps work on the shards.
    """
    from torch.distributed.fsdp import FSDPModule

    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


def total_grad_norm(params: list[torch.nn.Parameter]) -> float:
    """Global L2 norm of the sharded gradients (no clipping)."""
    norm = torch.nn.utils.clip_grad_norm_(params, float("inf"))
    if hasattr(norm, "full_tensor"):
        norm = norm.full_tensor()
    return float(norm)


def broadcast_float(value: float, src: int = 0) -> float:
    """Rank ``src``'s value on every rank (keeps collective decisions identical)."""
    if not (dist.is_available() and dist.is_initialized()):
        return value
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.broadcast(tensor, src=src)
    return float(tensor.item())


def full_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The unsharded backbone weights on rank 0's CPU (empty on other ranks)."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    return get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))


def save_transformer(
    model: torch.nn.Module,
    out_dir: Path,
    record: dict,
    *,
    is_main: bool,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Write the backbone as a diffusers ``transformer/`` directory (collective).

    ``record`` comes from :func:`shard_backbone`. Shards are written to a temporary
    directory and renamed into place, so a crash never leaves a half checkpoint
    behind a valid name.
    """
    state = full_state_dict(model)
    if is_main:
        import diffusers
        from huggingface_hub import save_torch_state_dict

        tmp = out_dir.with_name(out_dir.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        save_torch_state_dict(
            {name: tensor.to(dtype) for name, tensor in state.items()},
            tmp,
            filename_pattern="diffusion_pytorch_model{suffix}.safetensors",
            max_shard_size="5GB",
        )
        config = {"_class_name": record["class_name"], "_diffusers_version": diffusers.__version__}
        config.update({k: v for k, v in record["config"].items() if not k.startswith("_")})
        (tmp / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
        shutil.rmtree(out_dir, ignore_errors=True)
        tmp.rename(out_dir)
    del state
    if dist.is_initialized():
        dist.barrier()


def save_train_state(path: Path, model, optimizer, scheduler, ema, step: int, extra: dict, *, is_main: bool) -> None:
    """Sharded resumable state: fp32 weights, AdamW, EMA shadow, schedule, step (collective).

    Written to ``<path>.tmp`` and swapped in, so the previous state survives a crash
    during the save.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    model_state, optim_state = get_state_dict(model, optimizer)
    state = {"model": model_state, "optim": optim_state, "ema": dict(ema.shadow_params) if ema else {}}
    tmp = path.with_name(path.name + ".tmp")
    if is_main:
        shutil.rmtree(tmp, ignore_errors=True)
    if dist.is_initialized():
        dist.barrier()
    dcp.save(state, checkpoint_id=str(tmp))
    if is_main:
        torch.save(
            {"step": step, "scheduler": scheduler.state_dict(), "ema_step": ema.step if ema else None, **extra},
            tmp / "extra.pt",
        )
        old = path.with_name(path.name + ".old")
        shutil.rmtree(old, ignore_errors=True)
        if path.exists():
            path.rename(old)
        tmp.rename(path)
        shutil.rmtree(old, ignore_errors=True)
    if dist.is_initialized():
        dist.barrier()


def load_train_state(path: Path, model, optimizer, scheduler, ema) -> dict:
    """Restore :func:`save_train_state` output into this run (collective); returns its extras."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    if not (path / "extra.pt").is_file():
        raise FileNotFoundError(f"no full-training state at {path}")
    model_state, optim_state = get_state_dict(model, optimizer)
    state = {"model": model_state, "optim": optim_state, "ema": dict(ema.shadow_params) if ema else {}}
    dcp.load(state, checkpoint_id=str(path))
    set_state_dict(model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optim"])
    extra = torch.load(path / "extra.pt", map_location="cpu", weights_only=False)
    scheduler.load_state_dict(extra["scheduler"])
    if ema is not None and extra.get("ema_step") is not None:
        ema.step = int(extra["ema_step"])
    return extra
