"""End-to-end SFT → OPD → gated GRPO on Apple GPU with the fake backend.

Skipped automatically when MPS is not available (Linux CI). Run locally with
the sandbox disabled so ``torch.backends.mps.is_available()`` is true.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.training.config import TrainingConfig
from open_vision_banana.training.post_training import SFT_GATE_PROTOCOL, write_sft_gate
from open_vision_banana.training.trainer import Trainer
from scripts.check_experiment_gates import check_run
from test_trainer_system_all_tasks import FakeBackend
import open_vision_banana.training.trainer as trainer_module

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is not available in this process",
)

RES = 32
PROMPT = "Generate a metric depth visualization of this image."


def _write_depth_manifest(root: Path, n_train: int = 4, n_val: int = 2) -> Path:
    images = root / "images"
    targets = root / "targets"
    images.mkdir(parents=True)
    targets.mkdir()
    mapper = DepthRGBMapper(colormap="cube_edge")
    rows = []
    for i in range(n_train + n_val):
        split = "train" if i < n_train else "val"
        rgb = Image.fromarray(np.full((RES, RES, 3), 40 + i * 12, dtype=np.uint8))
        depth = torch.full((1, 1, RES, RES), 1.5 + 0.2 * i)
        depth_rgb = mapper.depth_to_rgb(depth)[0]
        tgt = Image.fromarray((depth_rgb.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
        rgb.save(images / f"{i:03d}.png")
        tgt.save(targets / f"{i:03d}.png")
        rows.append(
            {
                "task": "depth",
                "input": f"images/{i:03d}.png",
                "target": f"targets/{i:03d}.png",
                "split": split,
                "prompt": PROMPT,
            }
        )
    manifest = root / "pairs.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return manifest


def _config(tmp_path: Path, manifest: Path, **kwargs) -> TrainingConfig:
    defaults = dict(
        model_name="sana-1600m",
        device="mps",
        dtype="float32",
        resolution=RES,
        batch_size=1,
        steps=2,
        lr=1e-3,
        output_dir=str(tmp_path / "run"),
        data_dir=str(manifest.parent),
        pair_manifest=str(manifest),
        tasks=["depth"],
        vision_ratio=1.0,
        depth_colormaps=["cube_edge"],
        num_workers=0,
        save_every=2,
        eval_every=2,
        log_every=1,
        warmup_steps=0,
        ema_enabled=False,
        min_snr=False,
        wandb=False,
        skip_preflight=True,
        gradient_checkpointing=False,
    )
    defaults.update(kwargs)
    return TrainingConfig(**defaults)


def test_sft_opd_grpo_train_on_mps(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(trainer_module, "ModelBackend", lambda **kwargs: FakeBackend(**kwargs))
    manifest = _write_depth_manifest(tmp_path)

    sft_dir = tmp_path / "sft"
    Trainer(_config(sft_dir, manifest, output_dir=str(sft_dir), objective="sft")).train()
    effective = json.loads((sft_dir / "effective_config.json").read_text(encoding="utf-8"))
    assert effective["objective"] == "sft"
    assert effective["device"] == "mps"
    assert effective["resolved_device"] == "mps"
    assert effective["preflight"]["skipped"] is True
    sft_report = check_run(run_dir=sft_dir, tasks=["depth"], vision_ratio=1.0)
    statuses = {r.name: r.status for r in sft_report.results}
    assert statuses["run.objective"] == "PASS"
    assert statuses["run.tune_mode"] == "PASS"
    assert statuses["run.checkpoints_present"] == "PASS"

    opd_dir = tmp_path / "opd"
    Trainer(
        _config(
            opd_dir,
            manifest,
            output_dir=str(opd_dir),
            objective="opd",
            opd_teachers={"depth": "self"},
            opd_rollout_steps=4,
            opd_train_timesteps=2,
        )
    ).train()
    assert any(p.name.endswith("_final") for p in opd_dir.glob("checkpoint-*") if p.is_dir())

    gate = write_sft_gate(
        tmp_path / "post_sft_gate.json",
        {
            "protocol": SFT_GATE_PROTOCOL,
            "colormap_ok": True,
            "raw_delta_1.25": 0.14,
            "aligned_delta_1.25": 0.71,
        },
    )
    grpo_dir = tmp_path / "grpo"
    Trainer(
        _config(
            grpo_dir,
            manifest,
            output_dir=str(grpo_dir),
            objective="grpo",
            sft_gate_path=str(gate),
            grpo_group_size=2,
            opd_rollout_steps=2,
            opd_train_timesteps=1,
            eval_every=10,
        )
    ).train()
    finals = [p for p in grpo_dir.glob("checkpoint-*") if p.is_dir() and p.name.endswith("_final")]
    assert finals
    meta = json.loads((finals[0] / "metadata.json").read_text(encoding="utf-8"))
    assert meta.get("objective") == "grpo"
    assert torch.mps.current_allocated_memory() >= 0
