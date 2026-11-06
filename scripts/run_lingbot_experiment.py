#!/usr/bin/env python3
"""Validate a paired LingBot run and print its command; train only with --execute."""

from __future__ import annotations

# ruff: noqa: E402 -- standalone CLI bootstraps the repository source path.
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from open_vision_banana.res_bundle import sha256_file
from open_vision_banana.training.config import load_config
from scripts.check_eval_checkpoint import check_checkpoint
from scripts.prepare_lingbot_experiments import FINGERPRINT, REVISION


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
    if world_size not in (1, 2, 4, 8):
        raise ValueError("Choose 1/2/4/8 processes with global batch 32")
    if arm not in {"D0", "D1", "P0", "P1", "M0", "M1"}:
        raise ValueError("Unknown experiment arm")
    if output.exists():
        raise FileExistsError(
            "Choose a new output directory; this entry point starts fresh continuations"
        )
    if (
        len(base_revision) != 40
        or any(c not in "0123456789abcdef" for c in base_revision)
        or not (base_model / "model_index.json").is_file()
    ):
        raise ValueError("Supply a locally verified base snapshot and its immutable revision")
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
    pool = data_root / ("full_plus_lingbot" if arm.endswith("1") else "full_baseline")
    command = [
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
        "--resume",
        str(parent),
        "--resume-weights-only",
        "--grad-accum",
        str(32 // world_size),
        "--seed",
        str(seed),
        "--output-dir",
        str(output),
    ]
    provenance = {
        "arm": arm,
        "seed": seed,
        "world_size": world_size,
        "global_batch": 32,
        "parent_lora_pt_sha256": parent_sha256,
        "data_audit": audit,
        "config_sha256": sha256_file(config_path),
        "command": command,
        "base_model_revision": base_revision,
        "base_model_dir": str(base_model),
        "base_model_index_sha256": sha256_file(base_model / "model_index.json"),
        "base_integrity_scope": "index hashed here; weights must be verified by snapshot preparation",
    }
    return command, provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=["D0", "D1", "P0", "P1", "M0", "M1"], required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--parent-sha256", required=True, help="SHA-256 of pinned parent's lora.pt")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--base-model-dir", type=Path, required=True)
    p.add_argument("--base-model-revision", required=True)
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    command, provenance = build_command(
        a.arm,
        a.data_root.resolve(),
        a.parent.resolve(),
        a.parent_sha256,
        a.output_dir.resolve(),
        base_model=a.base_model_dir.resolve(),
        base_revision=a.base_model_revision,
        world_size=a.world_size,
        seed=a.seed,
    )
    print(shlex.join(command), flush=True)
    if a.execute:
        a.output_dir.mkdir(parents=True, exist_ok=False)
        (a.output_dir / "lingbot-launch.json").write_text(json.dumps(provenance, indent=2) + "\n")
        env = dict(os.environ, OPEN_VISION_BANANA_MODEL_PATH=str(a.base_model_dir.resolve()))
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
