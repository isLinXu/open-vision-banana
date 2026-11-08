"""Data-parallel support in the unified Trainer (scripts/train.py under torchrun)."""

from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
from PIL import Image
from torch.utils.data import DistributedSampler, RandomSampler

from open_vision_banana.training.config import TrainingConfig, build_arg_parser
from open_vision_banana.training.distributed import (
    DistContext,
    all_reduce_gradients_,
    barrier,
    broadcast_parameters_,
    cleanup,
    init_distributed,
)
from open_vision_banana.training.trainer import Trainer

GPUS_SH = Path(__file__).resolve().parents[1] / "scripts" / "_gpus.sh"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, world_size: int, port: int, results) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(world_size),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        CUDA_VISIBLE_DEVICES="",
    )
    ctx = init_distributed()

    a = torch.nn.Parameter(torch.full((3,), float(rank)))
    b = torch.nn.Parameter(torch.zeros(2))
    a.grad = torch.full((3,), float(rank + 1))
    if rank == 0:
        b.grad = torch.full((2,), 4.0)  # rank 1 has no grad for b
    all_reduce_gradients_([a, b], ctx)

    w = torch.nn.Parameter(torch.full((2,), float(rank + 10)))
    broadcast_parameters_([w], ctx)
    barrier(ctx)

    results[rank] = (
        ctx.rank,
        ctx.world_size,
        ctx.is_main,
        a.grad.tolist(),
        b.grad.tolist(),
        w.tolist(),
    )
    cleanup(ctx)


def test_two_ranks_average_grads_and_share_rank0_weights() -> None:
    world_size = 2
    results = mp.Manager().dict()
    mp.spawn(_worker, args=(world_size, _free_port(), results), nprocs=world_size, join=True)
    for rank in range(world_size):
        got_rank, got_world, is_main, a_grad, b_grad, w = results[rank]
        assert (got_rank, got_world, is_main) == (rank, world_size, rank == 0)
        assert a_grad == [1.5, 1.5, 1.5]  # mean of 1 and 2
        assert b_grad == [2.0, 2.0]  # mean of 4 and a zero-filled missing grad
        assert w == [10.0, 10.0]  # rank 0's values


def test_single_process_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    ctx = init_distributed()
    assert ctx == DistContext() and not ctx.enabled and ctx.is_main
    p = torch.nn.Parameter(torch.ones(2))
    p.grad = torch.full((2,), 3.0)
    all_reduce_gradients_([p], ctx)
    broadcast_parameters_([p], ctx)
    barrier(ctx)
    cleanup(ctx)
    assert p.grad.tolist() == [3.0, 3.0]


def test_ddp_flag_is_accepted() -> None:
    assert build_arg_parser().parse_args(["--ddp"]).ddp is True


@pytest.mark.parametrize("flag", ["--fsdp", "--deepspeed"])
def test_fsdp_and_deepspeed_flags_fail_clearly(flag: str, capsys) -> None:
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args([flag, "--deepspeed-config", "ds.json"])
    assert f"{flag} is not supported" in capsys.readouterr().err


@pytest.mark.parametrize("dist_ctx", [DistContext(), DistContext(rank=1, world_size=2)])
def test_train_loader_is_sharded_only_under_ddp(tmp_path, dist_ctx: DistContext) -> None:
    rows = []
    Image.new("L", (8, 8), 255).save(tmp_path / "valid_mask.png")
    for i in range(4):
        for kind in ("in", "out"):
            Image.new("RGB", (8, 8), (i * 40, 0, 0)).save(tmp_path / f"{kind}{i}.png")
        rows.append(
            f'{{"task": "depth", "input": "{tmp_path}/in{i}.png", '
            f'"target": "{tmp_path}/out{i}.png", "split": "train", '
            '"valid_mask": "valid_mask.png"}'
        )
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text("\n".join(rows) + "\n")
    config = TrainingConfig(
        pair_manifest=str(manifest), tasks=["depth"], resolution=8, batch_size=1
    )

    train_loader, _ = Trainer._build_datasets(SimpleNamespace(config=config, dist=dist_ctx))

    if dist_ctx.enabled:
        assert isinstance(train_loader.sampler, DistributedSampler)
        assert (train_loader.sampler.num_replicas, train_loader.sampler.rank) == (2, 1)
        assert len(train_loader) == 2  # half of the 4 samples
    else:
        assert isinstance(train_loader.sampler, RandomSampler)
        assert len(train_loader) == 4


@pytest.mark.parametrize(
    ("explicit", "visible", "expected"),
    [("3", "0,1", "3"), ("", "4,5", "2"), ("", "7", "1"), ("", "", "0")],
)
def test_launcher_gpu_count(explicit: str, visible: str, expected: str, tmp_path) -> None:
    """Explicit count wins, then CUDA_VISIBLE_DEVICES; with neither (and a failing
    nvidia-smi, a CPU-only torch) it falls through to 0."""
    for name, body in (("python3", "echo 0"), ("nvidia-smi", "exit 1")):
        stub = tmp_path / name
        stub.write_text(f"#!/bin/sh\n{body}\n")
        stub.chmod(0o755)
    env = {"PATH": f"{tmp_path}:/usr/bin:/bin"}
    if visible:
        env["CUDA_VISIBLE_DEVICES"] = visible
    out = subprocess.run(
        ["bash", "-c", f'source "{GPUS_SH}"; detect_gpus_per_node "{explicit}"'],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == expected
