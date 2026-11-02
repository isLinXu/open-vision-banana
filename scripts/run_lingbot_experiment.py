#!/usr/bin/env python3
"""Validate a paired LingBot run and print its command; train only with --execute.

Two launch families:

* D0/D1, P0/P1: weights-only continuations of the pinned parent adapter (``--parent``).
* Issue #97 arms, fresh starts from an explicit initialization (``--init E|Q --init-dir``):
  E is the Qwen-Image-2.1 base with the pinned E2B parent adapter merged once
  (``merge_metadata.json`` must name that adapter); Q is the unmerged official base.

  =====  ====  =====  ==========================  ======================================
  arm    init  mode   pool                        budget / varied factor
  =====  ====  =====  ==========================  ======================================
  M0/M1  E     LoRA   old / old + LingBot         20k, rank 32, 512
  Q0/Q1  Q     LoRA   old / old + LingBot         20k, rank 32, 512
  F-M1   E     full   old + LingBot               15k, ``--lr`` (DiT only, FSDP2)
  K-Q0/1 Q     LoRA   old / old + LingBot         20k, ``--lora-rank`` 64 or 128 (alpha 64)
  R-M0/1 E     LoRA   old / old + LingBot         20k, resolution 768
  =====  ====  =====  ==========================  ======================================

  Every fresh arm reads ``lingbot_fresh_<pool>.yaml`` (20k, warmup 1000, save every 2500); F-M1
  overrides only the steps and LR, so it keeps that warmup and save cadence.
"""

from __future__ import annotations

# ruff: noqa: E402 -- standalone CLI bootstraps the repository source path.
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from open_vision_banana.res_bundle import sha256_file
from open_vision_banana.training.config import load_config
from scripts.check_eval_checkpoint import check_checkpoint
from scripts.prepare_lingbot_experiments import FINGERPRINT, REVISION

LEGACY_ARMS = ("D0", "D1", "P0", "P1")


class Arm(NamedTuple):
    init: str  # E (merged parent) or Q (official base)
    mode: str  # lora or full
    pool: str  # full_baseline or full_plus_lingbot
    rank_search: bool = False
    resolution: int | None = None  # None: the config's


BASE, PLUS = "full_baseline", "full_plus_lingbot"
FRESH_ARMS = {
    "M0": Arm("E", "lora", BASE),
    "M1": Arm("E", "lora", PLUS),
    "Q0": Arm("Q", "lora", BASE),
    "Q1": Arm("Q", "lora", PLUS),
    "F-M1": Arm("E", "full", PLUS),
    "K-Q0": Arm("Q", "lora", BASE, rank_search=True),
    "K-Q1": Arm("Q", "lora", PLUS, rank_search=True),
    "R-M0": Arm("E", "lora", BASE, resolution=768),
    "R-M1": Arm("E", "lora", PLUS, resolution=768),
}
FULL_STEPS = 15000
RANK_GRID = (64, 128)
# adapter_model.safetensors of khazic/ovb-e2b_cap10k_s42_20261026_1016@036bc72, checkpoint-10000_final.
PARENT_ADAPTER_SHA256 = "ee545bd71347c3929b4bb27d63afb1a12f4754f99933fd5a663fcbd6728dac47"


def _check_output(output: Path) -> None:
    if output.exists():
        raise FileExistsError("Choose a new output directory; existing runs are never replaced")


def _check_revision(model: Path, revision: str) -> None:
    if (
        len(revision) != 40
        or any(c not in "0123456789abcdef" for c in revision)
        or not (model / "model_index.json").is_file()
    ):
        raise ValueError("Supply a locally verified base snapshot and its immutable revision")


def _check_data(data_root: Path) -> dict:
    if (data_root / "BUILD_INCOMPLETE").exists():
        raise ValueError("Incomplete paired data build")
    audit = json.loads((data_root / "experiment-data.json").read_text())
    if (
        audit.get("status") != "complete"
        or audit.get("lingbot_revision") != REVISION
        or audit.get("lingbot_fingerprint") != FINGERPRINT
    ):
        raise ValueError("Unexpected dataset release")
    for name in ("full_baseline", "full_plus_lingbot"):
        if (data_root / name / "BUILD_INCOMPLETE").exists():
            raise ValueError(f"Incomplete paired manifest: {name}")
        if sha256_file(data_root / name / "pairs.jsonl") != audit["manifests"][name]["sha256"]:
            raise ValueError(f"Frozen manifest changed: {name}")
    return audit


def _torchrun(
    world_size: int,
    config_path: Path,
    pool: Path,
    seed: int,
    output: Path,
    *,
    resume: tuple[str, ...] = (),
    extra: tuple[str, ...] = (),
) -> list[str]:
    if world_size not in (1, 2, 4, 8):
        raise ValueError("Choose 1/2/4/8 processes with global batch 32")
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world_size}",
        str(ROOT / "scripts/train.py"),
        "--config",
        str(config_path),
        "--pair-manifest",
        str(pool / "pairs.jsonl"),
        "--data-dir",
        str(pool),
        *resume,
        "--grad-accum",
        str(32 // world_size),
        "--seed",
        str(seed),
        "--output-dir",
        str(output),
        *extra,
    ]


def _provenance(arm, seed, world_size, audit, config_path, command, **extra) -> dict:
    return {
        "arm": arm,
        "seed": seed,
        "world_size": world_size,
        "global_batch": 32,
        **extra,
        "data_audit": audit,
        "config_sha256": sha256_file(config_path),
        "command": command,
    }


def _transformer_shards(model: Path) -> dict[str, int]:
    shards = sorted((model / "transformer").glob("*.safetensors"))
    if not shards:
        raise ValueError(f"No transformer shards under {model}")
    return {p.name: p.stat().st_size for p in shards}


def check_initialization(
    init: str,
    model: Path,
    base_revision: str,
    *,
    verify_weights: bool = False,
    pins: dict[str, str] | None = None,
) -> dict:
    """Identity of an E (merged) or Q (official, unmerged) starting point; refuses a mix-up."""
    _check_revision(model, base_revision)
    metadata_path = model / "merge_metadata.json"
    shards = _transformer_shards(model)
    record = {
        "init_source": init,
        "init_dir": str(model),
        "base_revision": base_revision,
        "transformer_shard_bytes": shards,
        "model_index_sha256": sha256_file(model / "model_index.json"),
    }
    if init == "E":
        if not metadata_path.is_file():
            raise ValueError("E initialization needs a merged snapshot with merge_metadata.json")
        meta = json.loads(metadata_path.read_text())
        if meta.get("adapter_sha256") != PARENT_ADAPTER_SHA256:
            raise ValueError("Merged snapshot does not come from the pinned E2B parent adapter")
        if not str(meta.get("base_revision", "")).startswith(base_revision):
            raise ValueError("Merged snapshot was built from a different base revision")
        merged = meta.get("transformer_shards") or {}
        if set(merged) != set(shards):
            raise ValueError("Merged transformer shards differ from merge_metadata.json")
        record.update(
            merged=True,
            parent_adapter_sha256=PARENT_ADAPTER_SHA256,
            merge_metadata_sha256=sha256_file(metadata_path),
        )
    elif init == "Q":
        if metadata_path.exists():
            raise ValueError(
                "Q initialization must be the unmerged official base, not a merged snapshot"
            )
        merged = {}
        record.update(merged=False, parent_adapter_sha256=None)
    else:
        raise ValueError("init must be E or Q")
    pins = pins or {}
    if any(merged.get(name, digest) != digest for name, digest in pins.items()):
        raise ValueError(
            "Pins disagree with merge_metadata.json; the merge record is authoritative"
        )
    expected = {**merged, **pins}
    if expected and set(expected) != set(shards):
        raise ValueError("Pinned transformer shards differ from the snapshot")
    if verify_weights:
        actual = {name: sha256_file(model / "transformer" / name) for name in shards}
        if expected and actual != expected:
            raise ValueError("Transformer shard hashes differ from the pinned values")
        record["transformer_shard_sha256"] = actual
    if not verify_weights:
        record["weights_check"] = "sizes only (pass --verify-weights to hash)"
    elif expected:
        record["weights_check"] = "sha256 against pins"
    else:
        record["weights_check"] = "sha256 recorded, no reference to compare (pass --weight-pins)"
    return record


def build_fresh_command(
    arm: str,
    data_root: Path,
    output: Path,
    *,
    init_dir: Path,
    base_revision: str,
    lr: float | None = None,
    lora_rank: int | None = None,
    world_size: int = 8,
    seed: int = 42,
    verify_weights: bool = False,
    weight_pins: dict[str, str] | None = None,
) -> tuple[list[str], dict]:
    """Issue #97 arms: a fresh LoRA adapter or full DiT training from the E or Q initialization."""
    if arm not in FRESH_ARMS:
        raise ValueError("Unknown experiment arm")
    spec = FRESH_ARMS[arm]
    if (lr is not None) != (spec.mode == "full"):
        raise ValueError("--lr is required for F-M1 (the LR search) and fixed for every other arm")
    if lr is not None and lr <= 0:
        raise ValueError("--lr must be positive")
    if (lora_rank is not None) != spec.rank_search:
        raise ValueError("--lora-rank is required for K-Q0/K-Q1 and fixed at 32 elsewhere")
    if lora_rank is not None and lora_rank not in RANK_GRID:
        raise ValueError(f"The rank grid is {RANK_GRID}; rank 32 reuses Q0/Q1")
    _check_output(output)
    audit = _check_data(data_root)
    identity = check_initialization(
        spec.init, init_dir, base_revision, verify_weights=verify_weights, pins=weight_pins
    )
    config_path = ROOT / f"configs/experiments/lingbot_fresh_{spec.pool.removeprefix('full_')}.yaml"
    config = load_config(config_path=config_path)
    overrides = []
    if spec.mode == "full":
        overrides += ["--train-mode", "full", "--steps", str(FULL_STEPS), "--lr", repr(lr)]
    if lora_rank is not None:
        overrides += ["--lora-rank", str(lora_rank)]
    if spec.resolution is not None:
        overrides += ["--resolution", str(spec.resolution)]
    command = _torchrun(
        world_size, config_path, data_root / spec.pool, seed, output, extra=tuple(overrides)
    )
    lora = spec.mode == "lora"
    provenance = _provenance(
        arm,
        seed,
        world_size,
        audit,
        config_path,
        command,
        issue=97,
        family="fresh",
        mode=spec.mode,
        pool=spec.pool,
        steps=FULL_STEPS if not lora else config.steps,
        lr=lr if lr is not None else config.lr,
        lora_rank=(lora_rank or config.lora_rank) if lora else None,
        lora_alpha=config.lora_alpha if lora else None,
        resolution=spec.resolution or config.resolution,
        initialization=identity,
    )
    return command, provenance


def build_command(
    arm: str,
    data_root: Path,
    parent: Path,
    parent_sha256: str,
    output: Path,
    *,
    base_model: Path,
    base_revision: str,
    world_size: int = 8,
    seed: int = 42,
) -> tuple[list[str], dict]:
    if arm not in LEGACY_ARMS:
        raise ValueError("Unknown parent-continuation arm")
    _check_output(output)
    _check_revision(base_model, base_revision)
    audit = _check_data(data_root)
    check_checkpoint(parent)
    if sha256_file(parent / "lora.pt") != parent_sha256:
        raise ValueError("Parent training-state hash mismatch")
    adapter = json.loads((parent / "lora_adapter/adapter_config.json").read_text())
    config_path = ROOT / f"configs/experiments/lingbot_{arm.lower()}.yaml"
    config = load_config(config_path=config_path)
    if (
        adapter.get("r") != config.lora_rank
        or adapter.get("lora_alpha") != config.lora_alpha
        or adapter.get("use_rslora") != config.rslora
        or set(adapter.get("target_modules", [])) != set(config.lora_targets)
    ):
        raise ValueError("Parent adapter architecture differs from the experiment")
    pool = data_root / (PLUS if arm.endswith("1") else BASE)
    command = _torchrun(
        world_size,
        config_path,
        pool,
        seed,
        output,
        resume=("--resume", str(parent), "--resume-weights-only"),
    )
    provenance = _provenance(
        arm,
        seed,
        world_size,
        audit,
        config_path,
        command,
        parent_lora_pt_sha256=parent_sha256,
        base_model_revision=base_revision,
        base_model_dir=str(base_model),
        base_model_index_sha256=sha256_file(base_model / "model_index.json"),
        base_integrity_scope="index hashed here; weights must be verified by snapshot preparation",
    )
    return command, provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=[*LEGACY_ARMS, *FRESH_ARMS], required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--base-model-revision", required=True)
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--execute", action="store_true")
    legacy = p.add_argument_group("D/P arms (parent continuation)")
    legacy.add_argument("--parent", type=Path)
    legacy.add_argument("--parent-sha256", help="SHA-256 of pinned parent's lora.pt")
    legacy.add_argument("--base-model-dir", type=Path)
    fresh = p.add_argument_group("issue #97 arms (fresh start)")
    fresh.add_argument("--init-dir", type=Path, help="E: merged snapshot; Q: official base")
    fresh.add_argument("--lr", type=float, help="F-M1 only")
    fresh.add_argument("--lora-rank", type=int, help="K-Q0/K-Q1 only")
    fresh.add_argument("--verify-weights", action="store_true", help="hash every transformer shard")
    fresh.add_argument("--weight-pins", type=Path, help="JSON {shard file name: sha256}")
    a = p.parse_args()
    common = dict(world_size=a.world_size, seed=a.seed)
    data_root, output = a.data_root.resolve(), a.output_dir.resolve()
    if a.arm in LEGACY_ARMS:
        if not (a.parent and a.parent_sha256 and a.base_model_dir):
            p.error("D/P arms need --parent, --parent-sha256 and --base-model-dir")
        model = a.base_model_dir.resolve()
        command, provenance = build_command(
            a.arm,
            data_root,
            a.parent.resolve(),
            a.parent_sha256,
            output,
            base_model=model,
            base_revision=a.base_model_revision,
            **common,
        )
    else:
        if not a.init_dir or a.parent or a.base_model_dir:
            p.error("issue #97 arms take --init-dir and never a parent adapter")
        model = a.init_dir.resolve()
        command, provenance = build_fresh_command(
            a.arm,
            data_root,
            output,
            init_dir=model,
            base_revision=a.base_model_revision,
            lr=a.lr,
            lora_rank=a.lora_rank,
            **common,
            verify_weights=a.verify_weights,
            weight_pins=json.loads(a.weight_pins.read_text()) if a.weight_pins else None,
        )
    print(shlex.join(command), flush=True)
    if a.execute:
        a.output_dir.mkdir(parents=True, exist_ok=False)
        (a.output_dir / "lingbot-launch.json").write_text(json.dumps(provenance, indent=2) + "\n")
        env = dict(
            os.environ,
            OPEN_VISION_BANANA_MODEL_PATH=str(model),
            PYTHONPATH=os.pathsep.join(
                filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")])
            ),
        )
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
