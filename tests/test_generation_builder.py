"""Tests for generation dataset builder and W&B checkpoint logging."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from open_vision_banana.data.generation import build_generation_dataset
from open_vision_banana.data.generation_builder import (
    build_synthetic_generation_dataset,
    load_prompts_file,
)
from open_vision_banana.training.checkpoint import CheckpointManager
from open_vision_banana.training.logging import TrainingLogger


class TestGenerationBuilder:
    def test_synthetic_build_writes_expected_files(self, tmp_path: Path):
        out = build_synthetic_generation_dataset(
            tmp_path / "gen",
            num_samples=5,
            resolution=32,
            seed=0,
        )
        assert (out / "images").is_dir()
        assert len(list((out / "images").glob("*.jpg"))) == 5
        assert (out / "prompts.json").exists()
        assert (out / "metadata.jsonl").exists()

        with (out / "prompts.json").open(encoding="utf-8") as f:
            prompts = json.load(f)
        assert len(prompts) == 5

        ds = build_generation_dataset(out, resolution=32)
        assert len(ds) == 5
        item = ds[0]
        assert item["target_rgb"].shape == (3, 32, 32)
        assert isinstance(item["prompt"], str)

    def test_load_prompts_txt(self, tmp_path: Path):
        path = tmp_path / "prompts.txt"
        path.write_text("first prompt\n\nsecond prompt\n", encoding="utf-8")
        prompts = load_prompts_file(path)
        assert prompts == ["first prompt", "second prompt"]

    def test_load_prompts_json(self, tmp_path: Path):
        path = tmp_path / "prompts.json"
        path.write_text('["alpha", "beta"]', encoding="utf-8")
        assert load_prompts_file(path) == ["alpha", "beta"]


class TestWandbCheckpointLogging:
    def test_log_checkpoint_uploads_when_enabled(self, tmp_path: Path):
        ckpt_dir = tmp_path / "checkpoint-10"
        ckpt_dir.mkdir()
        (ckpt_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

        logger = TrainingLogger.__new__(TrainingLogger)
        logger.wandb_enabled = True
        logger.wandb_log_checkpoints = True
        logger.tensorboard_enabled = False
        logger._writer = None
        logger._wandb = MagicMock()
        artifact = MagicMock(name="vision-banana-lora-step-10")
        logger._wandb.log_artifact.return_value = artifact

        with (
            patch("wandb.Artifact", return_value=artifact, create=True) as mock_artifact,
            patch("wandb.log", create=True) as mock_log,
        ):
            logger.log_checkpoint(ckpt_dir, alias="step-10", metadata={"step": 10}, step=10)

        mock_artifact.assert_called_once()
        logger._wandb.log_artifact.assert_called_once()
        mock_log.assert_called_once()

    def test_log_checkpoint_skipped_when_disabled(self, tmp_path: Path):
        ckpt_dir = tmp_path / "checkpoint-best"
        ckpt_dir.mkdir()
        logger = TrainingLogger(wandb_enabled=False, wandb_log_checkpoints=False)
        logger.log_checkpoint(ckpt_dir, alias="best")  # should not raise
        logger.finish()


class TestCheckpointManagerReturns:
    def test_save_returns_path(self, tmp_path: Path):
        import torch.nn as nn

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.zeros(1))

        import torch

        model = Tiny()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        mgr = CheckpointManager(str(tmp_path))

        # PEFT save_pretrained won't exist on Tiny — test path return via mock
        model.save_pretrained = MagicMock()
        path = mgr.save(model, opt, step=1, epoch=0, loss=0.5, lr=1e-3)
        assert path == tmp_path / "checkpoint-1"
        assert path.exists()

    def test_numbered_checkpoints_exclude_best(self, tmp_path: Path):
        (tmp_path / "checkpoint-best").mkdir()
        (tmp_path / "checkpoint-1").mkdir()
        (tmp_path / "checkpoint-10").mkdir()
        numbered = CheckpointManager._numbered_checkpoints(tmp_path)
        assert len(numbered) == 2
        assert all(p.name != "checkpoint-best" for p in numbered)
