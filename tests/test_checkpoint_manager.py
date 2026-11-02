"""Tests for training/checkpoint.py — CheckpointManager class."""

import json
import random
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from open_vision_banana.training.checkpoint import CheckpointManager

# ──────────────────────────────────────────────────────────────────
# Helper: mock PEFT model
# ──────────────────────────────────────────────────────────────────


class MockPEFTModel:
    """Minimal mock PEFT model for checkpoint testing."""

    def __init__(self):
        self.linear = nn.Linear(4, 2)

    def save_pretrained(self, path):
        """Simulate PEFT save_pretrained by writing a dummy file."""
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.linear.state_dict(), Path(path) / "adapter_model.pt")

    def state_dict(self):
        return self.linear.state_dict()

    def load_state_dict(self, sd, strict=True):
        self.linear.load_state_dict(sd, strict=False)

    @property
    def base_model(self):
        class _Base:
            model = MagicMock()

        return _Base()


# ──────────────────────────────────────────────────────────────────
# Init tests
# ──────────────────────────────────────────────────────────────────


class TestCheckpointManagerInit:
    def test_creates_output_dir(self, tmp_path):
        """Should create output directory if it doesn't exist."""
        out = tmp_path / "checkpoints"
        CheckpointManager(str(out))
        assert out.exists()
        assert out.is_dir()

    def test_default_max_keep(self):
        """Default max_keep should be 5."""
        mgr = CheckpointManager("/tmp/test_ckpt")
        assert mgr.max_keep == 5

    def test_custom_max_keep(self):
        """Custom max_keep should be stored."""
        mgr = CheckpointManager("/tmp/test_ckpt", max_keep=3)
        assert mgr.max_keep == 3


# ──────────────────────────────────────────────────────────────────
# Save tests
# ──────────────────────────────────────────────────────────────────


class TestCheckpointSave:
    def test_save_basic(self, tmp_path):
        """Should save checkpoint with all expected files."""
        mgr = CheckpointManager(str(tmp_path))
        model = MockPEFTModel()
        opt = torch.optim.SGD(model.linear.parameters(), lr=0.01)

        ckpt_dir = mgr.save(
            peft_model=model,
            optimizer=opt,
            step=10,
            epoch=1,
            loss=0.5,
            lr=1e-4,
        )
        assert ckpt_dir.exists()
        assert (ckpt_dir / "adapter_model.pt").exists()
        assert (ckpt_dir / "optimizer.pt").exists()

        meta = json.loads((ckpt_dir / "metadata.json").read_text())
        assert meta["step"] == 10
        assert meta["epoch"] == 1
        assert meta["loss"] == 0.5
        assert meta["lr"] == 1e-4

    def test_save_with_ema(self, tmp_path):
        """Should save EMA weights when provided."""
        mgr = CheckpointManager(str(tmp_path))
        model = MockPEFTModel()
        opt = torch.optim.SGD(model.linear.parameters(), lr=0.01)

        ema = MagicMock()
        ema.shadow_params = {"weight": torch.zeros(2, 4), "bias": torch.zeros(2)}

        ckpt_dir = mgr.save(
            peft_model=model,
            optimizer=opt,
            step=5,
            epoch=0,
            loss=0.3,
            lr=1e-3,
            ema=ema,
        )
        assert (ckpt_dir / "ema_weights.pt").exists()

    def test_save_with_scheduler(self, tmp_path):
        """Should save scheduler state when provided."""
        mgr = CheckpointManager(str(tmp_path))
        model = MockPEFTModel()
        opt = torch.optim.SGD(model.linear.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

        ckpt_dir = mgr.save(
            peft_model=model,
            optimizer=opt,
            step=3,
            epoch=0,
            loss=0.1,
            lr=1e-3,
            scheduler=scheduler,
        )
        assert (ckpt_dir / "scheduler.pt").exists()

    def test_save_best(self, tmp_path):
        """save_best should create checkpoint-best directory."""
        mgr = CheckpointManager(str(tmp_path))
        model = MockPEFTModel()
        opt = torch.optim.SGD(model.linear.parameters(), lr=0.01)

        ckpt_dir = mgr.save_best(
            peft_model=model,
            optimizer=opt,
            step=10,
            epoch=2,
            train_loss=0.3,
            val_loss=0.25,
            lr=1e-4,
        )
        assert ckpt_dir.name == "checkpoint-best"
        meta = json.loads((ckpt_dir / "metadata.json").read_text())
        assert meta["best"] is True
        assert meta["val_loss"] == 0.25
        assert meta["loss"] == 0.3

    def test_save_best_with_ema_and_scheduler(self, tmp_path):
        """save_best should also save EMA and scheduler."""
        mgr = CheckpointManager(str(tmp_path))
        model = MockPEFTModel()
        opt = torch.optim.SGD(model.linear.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
        ema = MagicMock()
        ema.shadow_params = {"weight": torch.zeros(2, 4)}

        ckpt_dir = mgr.save_best(
            peft_model=model,
            optimizer=opt,
            step=10,
            epoch=2,
            train_loss=0.3,
            val_loss=0.25,
            lr=1e-4,
            ema=ema,
            scheduler=scheduler,
        )
        assert (ckpt_dir / "ema_weights.pt").exists()
        assert (ckpt_dir / "scheduler.pt").exists()


# ──────────────────────────────────────────────────────────────────
# Load tests
# ──────────────────────────────────────────────────────────────────


class TestCheckpointLoad:
    def test_rng_checkpoint_is_weights_only_loadable(self, tmp_path):
        """RNG state must not require unsafe pickle deserialization."""
        state = CheckpointManager._capture_rng_state()
        path = tmp_path / "rng_state.pt"
        torch.save(state, path)
        loaded = torch.load(path, weights_only=True)
        assert isinstance(loaded["python"]["state"], torch.Tensor)
        assert isinstance(loaded["numpy"]["state"], torch.Tensor)

    def test_restore_rng_state(self, tmp_path):
        """RNG restoration should reproduce the next random samples."""
        mgr = CheckpointManager(str(tmp_path))
        model = MockPEFTModel()
        opt = torch.optim.SGD(model.linear.parameters(), lr=0.01)

        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        mgr.save(model, opt, step=10, epoch=1, loss=0.5, lr=1e-4)
        expected = (random.random(), float(np.random.rand()), torch.rand(1).item())

        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        mgr.load(
            MockPEFTModel(),
            torch.optim.SGD(MockPEFTModel().linear.parameters(), lr=0.01),
            restore_rng=True,
        )
        actual = (random.random(), float(np.random.rand()), torch.rand(1).item())

        assert actual == pytest.approx(expected)

    def test_legacy_rng_state_is_skipped_without_unsafe_load(self, tmp_path, caplog):
        """Old object-based RNG payloads must not force unsafe deserialization."""
        mgr = CheckpointManager(str(tmp_path))
        model = MockPEFTModel()
        opt = torch.optim.SGD(model.linear.parameters(), lr=0.01)
        ckpt_dir = mgr.save(model, opt, step=10, epoch=1, loss=0.5, lr=1e-4)
        legacy = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
        }
        torch.save(legacy, ckpt_dir / "rng_state.pt")
        mgr.load(model, opt, restore_rng=True)
        assert "not weights-only loadable" in caplog.text

    def test_load_latest(self, tmp_path):
        """Should load the latest checkpoint when step is None."""
        mgr = CheckpointManager(str(tmp_path))
        model = MockPEFTModel()
        opt = torch.optim.SGD(model.linear.parameters(), lr=0.01)

        # Save two checkpoints
        mgr.save(model, opt, step=10, epoch=1, loss=0.5, lr=1e-4)
        mgr.save(model, opt, step=20, epoch=2, loss=0.3, lr=1e-5)

        model2 = MockPEFTModel()
        opt2 = torch.optim.SGD(model2.linear.parameters(), lr=0.01)
        meta = mgr.load(model2, opt2)

        assert meta["step"] == 20
        assert meta["epoch"] == 2

    def test_load_specific_step(self, tmp_path):
        """Should load a specific checkpoint by step number."""
        mgr = CheckpointManager(str(tmp_path))
        model = MockPEFTModel()
        opt = torch.optim.SGD(model.linear.parameters(), lr=0.01)

        mgr.save(model, opt, step=10, epoch=1, loss=0.5, lr=1e-4)
        mgr.save(model, opt, step=20, epoch=2, loss=0.3, lr=1e-5)

        model2 = MockPEFTModel()
        opt2 = torch.optim.SGD(model2.linear.parameters(), lr=0.01)
        meta = mgr.load(model2, opt2, step=10)

        assert meta["step"] == 10

    def test_load_not_found_raises(self, tmp_path):
        """Should raise FileNotFoundError when no checkpoint exists."""
        mgr = CheckpointManager(str(tmp_path))
        model = MockPEFTModel()
        with pytest.raises(FileNotFoundError, match="No checkpoint found"):
            mgr.load(model)

    def test_load_with_ema(self, tmp_path):
        """Should load EMA weights when provided."""
        mgr = CheckpointManager(str(tmp_path))
        model = MockPEFTModel()
        opt = torch.optim.SGD(model.linear.parameters(), lr=0.01)

        ema_save = MagicMock()
        ema_save.shadow_params = {"weight": torch.ones(2, 4), "bias": torch.ones(2)}

        mgr.save(model, opt, step=10, epoch=1, loss=0.5, lr=1e-4, ema=ema_save)

        model2 = MockPEFTModel()
        opt2 = torch.optim.SGD(model2.linear.parameters(), lr=0.01)
        ema_load = MagicMock()
        ema_load.shadow_params = {}

        mgr.load(model2, opt2, ema=ema_load, step=10)
        assert "weight" in ema_load.shadow_params

    def test_load_with_scheduler(self, tmp_path):
        """Should load scheduler state when provided."""
        mgr = CheckpointManager(str(tmp_path))
        model = MockPEFTModel()
        opt = torch.optim.SGD(model.linear.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

        mgr.save(model, opt, step=10, epoch=1, loss=0.5, lr=1e-4, scheduler=scheduler)

        model2 = MockPEFTModel()
        opt2 = torch.optim.SGD(model2.linear.parameters(), lr=0.01)
        scheduler2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=10, gamma=0.1)

        mgr.load(model2, opt2, scheduler=scheduler2, step=10)
        # Scheduler state should be loaded without error

    def test_load_no_metadata(self, tmp_path):
        """Should return default metadata when metadata.json is missing."""
        mgr = CheckpointManager(str(tmp_path))
        model = MockPEFTModel()
        opt = torch.optim.SGD(model.linear.parameters(), lr=0.01)

        ckpt_dir = mgr.save(model, opt, step=10, epoch=1, loss=0.5, lr=1e-4)
        # Remove metadata to simulate missing file
        (ckpt_dir / "metadata.json").unlink()

        model2 = MockPEFTModel()
        opt2 = torch.optim.SGD(model2.linear.parameters(), lr=0.01)
        meta = mgr.load(model2, opt2, step=10)
        assert meta["step"] == 0
        assert meta["loss"] == 0.0


# ──────────────────────────────────────────────────────────────────
# Pruning tests
# ──────────────────────────────────────────────────────────────────


class TestCheckpointPruning:
    def test_prune_old_checkpoints(self, tmp_path):
        """Should remove oldest checkpoints beyond max_keep."""
        mgr = CheckpointManager(str(tmp_path), max_keep=2)
        model = MockPEFTModel()
        opt = torch.optim.SGD(model.linear.parameters(), lr=0.01)

        # Save 4 checkpoints
        for step in [10, 20, 30, 40]:
            mgr.save(model, opt, step=step, epoch=1, loss=0.5, lr=1e-4)

        # Only the latest 2 should remain (step 30, 40)
        remaining = sorted(mgr._numbered_checkpoints(mgr.output_dir))
        assert len(remaining) == 2
        assert mgr._ckpt_step(remaining[0]) == 30
        assert mgr._ckpt_step(remaining[1]) == 40

    def test_prune_does_not_remove_best(self, tmp_path):
        """checkpoint-best should not be pruned."""
        mgr = CheckpointManager(str(tmp_path), max_keep=1)
        model = MockPEFTModel()
        opt = torch.optim.SGD(model.linear.parameters(), lr=0.01)

        mgr.save(model, opt, step=10, epoch=1, loss=0.5, lr=1e-4)
        mgr.save_best(model, opt, step=10, epoch=1, train_loss=0.5, val_loss=0.3, lr=1e-4)
        mgr.save(model, opt, step=20, epoch=2, loss=0.3, lr=1e-5)

        # checkpoint-best should still exist
        assert (tmp_path / "checkpoint-best").exists()


# ──────────────────────────────────────────────────────────────────
# Utility method tests
# ──────────────────────────────────────────────────────────────────


class TestCheckpointUtilityMethods:
    def test_ckpt_step_extraction(self, tmp_path):
        """_ckpt_step should extract numeric step from dir name."""
        assert CheckpointManager._ckpt_step(Path("/tmp/checkpoint-100")) == 100
        assert CheckpointManager._ckpt_step(Path("/tmp/checkpoint-0")) == 0

    def test_ckpt_step_non_numeric(self, tmp_path):
        """_ckpt_step should return -1 for non-numeric suffix."""
        assert CheckpointManager._ckpt_step(Path("/tmp/checkpoint-best")) == -1
        assert CheckpointManager._ckpt_step(Path("/tmp/checkpoint-abc")) == -1

    def test_numbered_checkpoints_excludes_best(self, tmp_path):
        """_numbered_checkpoints should exclude checkpoint-best."""
        (tmp_path / "checkpoint-10").mkdir()
        (tmp_path / "checkpoint-20").mkdir()
        (tmp_path / "checkpoint-best").mkdir()

        checkpoints = CheckpointManager._numbered_checkpoints(tmp_path)
        assert len(checkpoints) == 2
        assert all("best" not in c.name for c in checkpoints)

    def test_numbered_checkpoints_sorted(self, tmp_path):
        """_numbered_checkpoints should return sorted list."""
        for step in [30, 10, 20]:
            (tmp_path / f"checkpoint-{step}").mkdir()

        checkpoints = CheckpointManager._numbered_checkpoints(tmp_path)
        steps = [CheckpointManager._ckpt_step(c) for c in checkpoints]
        assert steps == [10, 20, 30]

    def test_latest_checkpoint_none(self, tmp_path):
        """_latest_checkpoint should return None when no checkpoints exist."""
        mgr = CheckpointManager(str(tmp_path))
        assert mgr._latest_checkpoint() is None

    def test_latest_checkpoint_returns_newest(self, tmp_path):
        """_latest_checkpoint should return the newest checkpoint."""
        mgr = CheckpointManager(str(tmp_path))
        model = MockPEFTModel()
        opt = torch.optim.SGD(model.linear.parameters(), lr=0.01)

        mgr.save(model, opt, step=10, epoch=1, loss=0.5, lr=1e-4)
        mgr.save(model, opt, step=20, epoch=2, loss=0.3, lr=1e-5)

        latest = mgr._latest_checkpoint()
        assert latest is not None
        assert mgr._ckpt_step(latest) == 20
