"""Mock-based tests for distributed training components.

These tests use unittest.mock to simulate multi-process distributed training
without requiring multiple GPUs or actual process-group initialisation.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from open_vision_banana.training.fsdp_utils import FSDPWrapper
from open_vision_banana.training.trainer_interface import (
    AccelerateTrainer,
    DDPTrainer,
    DeepSpeedTrainer,
    FSDPTrainer,
    NativeTrainer,
    create_trainer,
)
from open_vision_banana.training.utils import DDPWrapper

# ──────────────────────────────────────────────────────────────
# DDPWrapper Mock Tests
# ──────────────────────────────────────────────────────────────


class TestDDPWrapperMock:
    """Mock tests for DDPWrapper in multi-process mode."""

    def test_multi_process_initializes_dist(self):
        with (
            patch(
                "open_vision_banana.training.utils.torch.distributed.is_initialized",
                return_value=False,
            ),
            patch(
                "open_vision_banana.training.utils.torch.distributed.init_process_group"
            ) as mock_init,
            patch("open_vision_banana.training.utils.torch.distributed.destroy_process_group"),
        ):
            ddp = DDPWrapper(world_size=4, rank=0, local_rank=0)
            ddp.initialize()
            mock_init.assert_called_once_with(backend="nccl", world_size=4, rank=0)
            assert ddp._initialized is True

    def test_multi_process_wrap_ddp(self):
        model = MagicMock()
        mock_ddp_instance = MagicMock()

        with (
            patch(
                "open_vision_banana.training.utils.torch.distributed.is_initialized",
                return_value=False,
            ),
            patch("open_vision_banana.training.utils.torch.distributed.init_process_group"),
            patch("open_vision_banana.training.utils.torch.distributed.destroy_process_group"),
            patch(
                "open_vision_banana.training.utils.torch.nn.parallel.DistributedDataParallel",
                return_value=mock_ddp_instance,
            ) as mock_ddp_cls,
        ):
            ddp = DDPWrapper(world_size=4, rank=0, local_rank=0)
            wrapped = ddp.wrap(model)
            mock_ddp_cls.assert_called_once()
            assert wrapped is mock_ddp_instance

    def test_barrier_multi_process(self):
        with (
            patch(
                "open_vision_banana.training.utils.torch.distributed.is_initialized",
                return_value=True,
            ),
            patch("open_vision_banana.training.utils.torch.distributed.barrier") as mock_barrier,
        ):
            ddp = DDPWrapper(world_size=4, rank=0)
            ddp.barrier()
            mock_barrier.assert_called_once()

    def test_barrier_single_process(self):
        ddp = DDPWrapper(world_size=1)
        # Should not raise even when dist is not initialised
        ddp.barrier()

    def test_cleanup(self):
        with (
            patch(
                "open_vision_banana.training.utils.torch.distributed.is_initialized",
                return_value=False,
            ),
            patch("open_vision_banana.training.utils.torch.distributed.init_process_group"),
            patch(
                "open_vision_banana.training.utils.torch.distributed.destroy_process_group"
            ) as mock_destroy,
        ):
            ddp = DDPWrapper(world_size=4, rank=0)
            ddp.initialize()
            ddp.cleanup()
            mock_destroy.assert_called_once()

    def test_already_initialized_skips_init(self):
        with (
            patch(
                "open_vision_banana.training.utils.torch.distributed.is_initialized",
                return_value=True,
            ),
            patch(
                "open_vision_banana.training.utils.torch.distributed.init_process_group"
            ) as mock_init,
        ):
            ddp = DDPWrapper(world_size=4, rank=0)
            ddp.initialize()
            mock_init.assert_not_called()


# ──────────────────────────────────────────────────────────────
# FSDPWrapper Mock Tests
# ──────────────────────────────────────────────────────────────


class TestFSDPWrapperMock:
    """Mock tests for FSDPWrapper in multi-process mode."""

    def test_multi_process_initializes_dist(self):
        with (
            patch(
                "open_vision_banana.training.fsdp_utils.torch.cuda.is_available",
                return_value=True,
            ),
            patch(
                "open_vision_banana.training.fsdp_utils.torch.distributed.is_initialized",
                return_value=False,
            ),
            patch(
                "open_vision_banana.training.fsdp_utils.torch.distributed.init_process_group"
            ) as mock_init,
            patch("open_vision_banana.training.fsdp_utils.torch.distributed.destroy_process_group"),
            patch("open_vision_banana.training.fsdp_utils.torch.cuda.set_device"),
        ):
            fsdp = FSDPWrapper(world_size=4, rank=0, local_rank=0)
            fsdp.initialize()
            mock_init.assert_called_once_with(backend="nccl")
            assert fsdp._initialized is True

    def test_multi_process_wrap_fsdp(self):
        model = MagicMock()
        mock_fsdp_instance = MagicMock()

        with (
            patch(
                "open_vision_banana.training.fsdp_utils.torch.cuda.is_available",
                return_value=True,
            ),
            patch(
                "open_vision_banana.training.fsdp_utils.torch.distributed.is_initialized",
                return_value=False,
            ),
            patch("open_vision_banana.training.fsdp_utils.torch.distributed.init_process_group"),
            patch("open_vision_banana.training.fsdp_utils.torch.distributed.destroy_process_group"),
            patch("open_vision_banana.training.fsdp_utils.torch.cuda.set_device"),
            patch(
                "torch.distributed.fsdp.FullyShardedDataParallel",
                return_value=mock_fsdp_instance,
            ),
            patch(
                "torch.distributed.fsdp.MixedPrecision",
            ),
            patch(
                "torch.distributed.fsdp.ShardingStrategy",
            ),
            patch(
                "torch.distributed.fsdp.wrap.size_based_auto_wrap_policy",
            ),
        ):
            fsdp = FSDPWrapper(world_size=4, rank=0, local_rank=0)
            wrapped = fsdp.wrap(model)
            assert wrapped is mock_fsdp_instance

    def test_get_train_sampler_enabled(self):
        from torch.utils.data import TensorDataset

        dataset = TensorDataset(torch.randn(8, 2))
        mock_sampler = MagicMock()

        with patch(
            "open_vision_banana.training.fsdp_utils.DistributedSampler",
            return_value=mock_sampler,
        ):
            fsdp = FSDPWrapper(world_size=4, rank=0)
            sampler = fsdp.get_train_sampler(dataset)
            assert sampler is mock_sampler

    def test_set_epoch(self):
        mock_sampler = MagicMock()
        fsdp = FSDPWrapper(world_size=1)
        fsdp.set_epoch(mock_sampler, epoch=3)
        mock_sampler.set_epoch.assert_called_once_with(3)

    def test_set_epoch_none(self):
        fsdp = FSDPWrapper(world_size=1)
        # Should not raise when sampler is None
        fsdp.set_epoch(None, epoch=3)

    def test_full_state_dict_single_process(self):
        model = nn.Linear(4, 2)
        fsdp = FSDPWrapper(world_size=1)
        state = fsdp.full_state_dict(model)
        assert isinstance(state, dict)
        assert "weight" in state

    def test_load_full_state_dict_single_process(self):
        model = nn.Linear(4, 2)
        fsdp = FSDPWrapper(world_size=1)
        state = model.state_dict()
        fsdp.load_full_state_dict(model, state)
        # If it didn't raise, loading succeeded

    def test_save_and_load_checkpoint_single_process(self, tmp_path: Path):
        model = nn.Linear(4, 2)
        fsdp = FSDPWrapper(world_size=1)
        ckpt_path = tmp_path / "ckpt.pt"

        fsdp.save_checkpoint(model, str(ckpt_path))
        assert ckpt_path.exists()

        fsdp.load_checkpoint(model, str(ckpt_path))
        # Should not raise

    def test_barrier_multi_process(self):
        with (
            patch(
                "open_vision_banana.training.fsdp_utils.torch.distributed.is_initialized",
                return_value=True,
            ),
            patch(
                "open_vision_banana.training.fsdp_utils.torch.distributed.barrier"
            ) as mock_barrier,
        ):
            fsdp = FSDPWrapper(world_size=4, rank=0)
            fsdp.barrier()
            mock_barrier.assert_called_once()

    def test_cleanup(self):
        with (
            patch(
                "open_vision_banana.training.fsdp_utils.torch.cuda.is_available",
                return_value=True,
            ),
            patch(
                "open_vision_banana.training.fsdp_utils.torch.distributed.is_initialized",
                return_value=False,
            ),
            patch("open_vision_banana.training.fsdp_utils.torch.distributed.init_process_group"),
            patch(
                "open_vision_banana.training.fsdp_utils.torch.distributed.destroy_process_group"
            ) as mock_destroy,
            patch("open_vision_banana.training.fsdp_utils.torch.cuda.set_device"),
        ):
            fsdp = FSDPWrapper(world_size=4, rank=0)
            fsdp.initialize()
            fsdp.cleanup()
            mock_destroy.assert_called_once()

    def test_wrap_no_cuda_raises(self):
        model = nn.Linear(4, 2)
        with patch(
            "open_vision_banana.training.fsdp_utils.torch.cuda.is_available",
            return_value=False,
        ):
            fsdp = FSDPWrapper(world_size=4, rank=0)
            with pytest.raises(RuntimeError, match="FSDP requires CUDA GPUs"):
                fsdp.wrap(model)

    def test_is_main(self):
        fsdp_main = FSDPWrapper(world_size=4, rank=0)
        assert fsdp_main.is_main() is True

        fsdp_worker = FSDPWrapper(world_size=4, rank=2)
        assert fsdp_worker.is_main() is False


# ──────────────────────────────────────────────────────────────
# DDPTrainer Mock Tests
# ──────────────────────────────────────────────────────────────


class TestDDPTrainerMock:
    """Mock tests for DDPTrainer."""

    def test_create_trainer(self):
        trainer = create_trainer(backend="ddp", world_size=1, rank=0, local_rank=0)
        assert isinstance(trainer, DDPTrainer)
        assert trainer.is_main_process()

    def test_multi_process_wrap_ddp(self):
        model = MagicMock()
        mock_ddp_instance = MagicMock()

        with (
            patch(
                "open_vision_banana.training.trainer_interface.torch.distributed.is_initialized",
                return_value=False,
            ),
            patch(
                "open_vision_banana.training.trainer_interface.torch.distributed.init_process_group"
            ),
            patch(
                "open_vision_banana.training.trainer_interface.torch.distributed.destroy_process_group"
            ),
            patch(
                "open_vision_banana.training.trainer_interface.torch.nn.parallel.DistributedDataParallel",
                return_value=mock_ddp_instance,
            ) as mock_ddp_cls,
        ):
            trainer = DDPTrainer(world_size=4, rank=0, local_rank=0)
            wrapped = trainer.wrap_model(model)
            mock_ddp_cls.assert_called_once()
            assert wrapped is mock_ddp_instance

    def test_barrier_multi_process(self):
        with (
            patch(
                "open_vision_banana.training.trainer_interface.torch.distributed.is_initialized",
                return_value=True,
            ),
            patch(
                "open_vision_banana.training.trainer_interface.torch.distributed.barrier"
            ) as mock_barrier,
        ):
            trainer = DDPTrainer(world_size=4, rank=0)
            trainer.barrier()
            mock_barrier.assert_called_once()

    def test_save_checkpoint_unwraps_ddp(self, tmp_path: Path):
        model = nn.Linear(4, 2)
        mock_ddp = MagicMock()
        mock_ddp.module = model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        trainer = DDPTrainer(world_size=1, rank=0)
        trainer.save_checkpoint(mock_ddp, optimizer, step=10, output_dir=str(tmp_path))
        ckpt_dir = tmp_path / "checkpoint-10"
        assert (ckpt_dir / "model.pt").exists()

    def test_save_checkpoint_skips_non_main(self):
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = DDPTrainer(world_size=4, rank=1)
        # With single-process model (no DDP), it should still write because
        # DDPTrainer.save_checkpoint checks is_main_process() which is False
        # for rank=1, so it should return early without writing.
        with patch.object(NativeTrainer, "save_checkpoint") as mock_native_save:
            trainer.save_checkpoint(model, optimizer, step=10, output_dir="/tmp")
            mock_native_save.assert_not_called()

    def test_is_main_process(self):
        assert DDPTrainer(world_size=4, rank=0).is_main_process() is True
        assert DDPTrainer(world_size=4, rank=1).is_main_process() is False


# ──────────────────────────────────────────────────────────────
# FSDPTrainer Mock Tests
# ──────────────────────────────────────────────────────────────


class TestFSDPTrainerMock:
    """Mock tests for FSDPTrainer."""

    def test_create_trainer(self):
        trainer = create_trainer(backend="fsdp", world_size=1, rank=0, local_rank=0)
        assert isinstance(trainer, FSDPTrainer)
        assert trainer.is_main_process()

    def test_multi_process_wrap_fsdp(self):
        model = MagicMock()
        mock_fsdp_instance = MagicMock()

        with (
            patch(
                "open_vision_banana.training.fsdp_utils.torch.cuda.is_available",
                return_value=True,
            ),
            patch(
                "open_vision_banana.training.fsdp_utils.torch.distributed.is_initialized",
                return_value=False,
            ),
            patch("open_vision_banana.training.fsdp_utils.torch.distributed.init_process_group"),
            patch("open_vision_banana.training.fsdp_utils.torch.distributed.destroy_process_group"),
            patch("open_vision_banana.training.fsdp_utils.torch.cuda.set_device"),
            patch(
                "torch.distributed.fsdp.FullyShardedDataParallel",
                return_value=mock_fsdp_instance,
            ),
            patch(
                "torch.distributed.fsdp.MixedPrecision",
            ),
            patch(
                "torch.distributed.fsdp.ShardingStrategy",
            ),
            patch(
                "torch.distributed.fsdp.wrap.size_based_auto_wrap_policy",
            ),
        ):
            trainer = FSDPTrainer(world_size=4, rank=0, local_rank=0)
            wrapped = trainer.wrap_model(model)
            assert wrapped is mock_fsdp_instance

    def test_barrier_delegates(self):
        with patch.object(FSDPWrapper, "barrier", return_value=None):
            trainer = FSDPTrainer(world_size=1, rank=0)
            # FSDPTrainer doesn't have a direct FSDPWrapper in multi-process that
            # uses the same mock path, but we can test via patching the internal _fsdp
            trainer._fsdp = FSDPWrapper(world_size=1, rank=0)
            with patch.object(trainer._fsdp, "barrier") as inner:
                trainer.barrier()
                inner.assert_called_once()

    def test_save_checkpoint_main_only(self, tmp_path: Path):
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = FSDPTrainer(world_size=4, rank=1, local_rank=0)
        with patch.object(FSDPWrapper, "save_checkpoint") as mock_save:
            trainer.save_checkpoint(model, optimizer, step=10, output_dir=str(tmp_path))
            mock_save.assert_not_called()

    def test_is_main_process(self):
        assert FSDPTrainer(world_size=4, rank=0).is_main_process() is True
        assert FSDPTrainer(world_size=4, rank=2).is_main_process() is False


# ──────────────────────────────────────────────────────────────
# AccelerateTrainer Mock Tests
# ──────────────────────────────────────────────────────────────


class TestAccelerateTrainerMock:
    """Mock tests for AccelerateTrainer using mocked Accelerator."""

    def _patch_accelerate(self):
        return patch("accelerate.Accelerator", autospec=True)

    def test_create_trainer(self):
        with self._patch_accelerate() as mock_cls:
            trainer = create_trainer(backend="accelerate")
            assert isinstance(trainer, AccelerateTrainer)
            mock_cls.assert_called_once()

    def test_wrap_model(self):
        with self._patch_accelerate():
            trainer = AccelerateTrainer()
            model = nn.Linear(4, 2)
            assert trainer.wrap_model(model) is model

    def test_backward_delegates(self):
        with self._patch_accelerate() as mock_cls:
            mock_inst = mock_cls.return_value
            trainer = AccelerateTrainer()
            loss = torch.tensor(1.0)
            trainer.backward(loss)
            mock_inst.backward.assert_called_once_with(loss)

    def test_optimizer_step_delegates(self):
        with self._patch_accelerate():
            trainer = AccelerateTrainer()
            mock_optimizer = MagicMock()
            trainer.setup_optimizer(mock_optimizer)
            trainer.optimizer_step()
            mock_optimizer.step.assert_called_once()

    def test_optimizer_step_without_setup(self):
        with self._patch_accelerate():
            trainer = AccelerateTrainer()
            with pytest.raises(RuntimeError, match="setup_optimizer"):
                trainer.optimizer_step()

    def test_is_main_process(self):
        with self._patch_accelerate() as mock_cls:
            mock_inst = mock_cls.return_value
            mock_inst.is_main_process = True
            trainer = AccelerateTrainer()
            assert trainer.is_main_process() is True

            mock_inst.is_main_process = False
            assert trainer.is_main_process() is False

    def test_barrier_delegates(self):
        with self._patch_accelerate() as mock_cls:
            mock_inst = mock_cls.return_value
            trainer = AccelerateTrainer()
            trainer.barrier()
            mock_inst.wait_for_everyone.assert_called_once()

    def test_prepare_returns_tuple(self):
        with self._patch_accelerate() as mock_cls:
            mock_inst = mock_cls.return_value
            trainer = AccelerateTrainer()
            model = nn.Linear(4, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            dataloader = MagicMock()
            mock_inst.prepare.return_value = (model, optimizer, dataloader)
            result = trainer.prepare(model, optimizer, dataloader)
            assert result == (model, optimizer, dataloader)


# ──────────────────────────────────────────────────────────────
# DeepSpeedTrainer Mock Tests
# ──────────────────────────────────────────────────────────────


class TestDeepSpeedTrainerMock:
    """Mock tests for DeepSpeedTrainer."""

    def _patch_initialize(self):
        return patch(
            "open_vision_banana.training.deepspeed_utils.initialize_deepspeed",
            autospec=True,
        )

    def test_create_trainer(self):
        trainer = create_trainer(backend="deepspeed")
        assert isinstance(trainer, DeepSpeedTrainer)

    def test_wrap_model_delegates(self):
        mock_engine = MagicMock()
        with self._patch_initialize() as mock_init:
            mock_init.return_value = (mock_engine, MagicMock())
            trainer = DeepSpeedTrainer()
            model = nn.Linear(4, 2)
            wrapped = trainer.wrap_model(model)
            mock_init.assert_called_once()
            assert wrapped is mock_engine

    def test_backward_delegates(self):
        mock_engine = MagicMock()
        trainer = DeepSpeedTrainer()
        trainer._engine = mock_engine  # type: ignore[assignment]
        loss = torch.tensor(1.0)
        trainer.backward(loss)
        mock_engine.backward.assert_called_once_with(loss)

    def test_optimizer_step_delegates(self):
        mock_engine = MagicMock()
        trainer = DeepSpeedTrainer()
        trainer._engine = mock_engine  # type: ignore[assignment]
        mock_optimizer = MagicMock()
        trainer.setup_optimizer(mock_optimizer)
        trainer.optimizer_step()
        mock_engine.step.assert_called_once()

    def test_is_main_process_no_engine(self):
        trainer = DeepSpeedTrainer()
        assert trainer.is_main_process() is True

    def test_is_main_process_delegates(self):
        mock_engine = MagicMock()
        trainer = DeepSpeedTrainer()
        trainer._engine = mock_engine  # type: ignore[assignment]
        mock_engine.global_rank = 0
        with patch(
            "open_vision_banana.training.deepspeed_utils.deepspeed_is_main",
            return_value=True,
        ) as mock_is_main:
            assert trainer.is_main_process() is True
            mock_is_main.assert_called_once_with(mock_engine)

    def test_train_step_with_engine(self):
        mock_engine = MagicMock()
        trainer = DeepSpeedTrainer(gradient_accumulation_steps=1)
        trainer._engine = mock_engine  # type: ignore[assignment]
        model = nn.Linear(4, 2)
        loss = torch.tensor(1.0)
        do_step = trainer.train_step(model, loss, step_in_accumulation=0)
        mock_engine.backward.assert_called_once()
        mock_engine.step.assert_called_once()
        assert do_step is True

    def test_train_step_with_accumulation(self):
        mock_engine = MagicMock()
        trainer = DeepSpeedTrainer(gradient_accumulation_steps=2)
        trainer._engine = mock_engine  # type: ignore[assignment]
        model = nn.Linear(4, 2)
        loss = torch.tensor(1.0)
        do_step = trainer.train_step(model, loss, step_in_accumulation=0)
        mock_engine.backward.assert_called_once()
        mock_engine.step.assert_not_called()
        assert do_step is False


# ──────────────────────────────────────────────────────────────
# create_trainer Factory Tests
# ──────────────────────────────────────────────────────────────


class TestCreateTrainerFactory:
    """Tests for the create_trainer factory function."""

    def test_create_native(self):
        trainer = create_trainer("native")
        assert isinstance(trainer, NativeTrainer)

    def test_create_ddp(self):
        trainer = create_trainer("ddp", world_size=1, rank=0, local_rank=0)
        assert isinstance(trainer, DDPTrainer)

    def test_create_fsdp(self):
        trainer = create_trainer("fsdp", world_size=1, rank=0, local_rank=0)
        assert isinstance(trainer, FSDPTrainer)

    def test_create_deepspeed(self):
        trainer = create_trainer("deepspeed")
        assert isinstance(trainer, DeepSpeedTrainer)

    def test_create_accelerate(self):
        with patch("accelerate.Accelerator", autospec=True):
            trainer = create_trainer("accelerate")
            assert isinstance(trainer, AccelerateTrainer)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown training backend"):
            create_trainer("unknown")


# ──────────────────────────────────────────────────────────────
# NativeTrainer Extended Tests
# ──────────────────────────────────────────────────────────────


class TestNativeTrainerExtended:
    """Additional tests for NativeTrainer."""

    def test_train_step_no_accumulation(self):
        trainer = NativeTrainer(device="cpu", gradient_accumulation_steps=1)
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer.setup_optimizer(optimizer)
        loss = torch.tensor(1.0, requires_grad=True)

        do_step = trainer.train_step(model, loss, step_in_accumulation=0)
        assert do_step is True

    def test_train_step_with_accumulation(self):
        trainer = NativeTrainer(device="cpu", gradient_accumulation_steps=2)
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer.setup_optimizer(optimizer)
        loss = torch.tensor(1.0, requires_grad=True)

        do_step_0 = trainer.train_step(model, loss, step_in_accumulation=0)
        assert do_step_0 is False

        do_step_1 = trainer.train_step(model, loss, step_in_accumulation=1)
        assert do_step_1 is True

    def test_optimizer_step_without_setup(self):
        trainer = NativeTrainer(device="cpu")
        with pytest.raises(RuntimeError, match="setup_optimizer"):
            trainer.optimizer_step()

    def test_save_and_load_checkpoint(self, tmp_path: Path):
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = NativeTrainer(device="cpu")

        trainer.save_checkpoint(model, optimizer, step=5, output_dir=str(tmp_path))
        ckpt_dir = tmp_path / "checkpoint-5"
        assert (ckpt_dir / "model.pt").exists()
        assert (ckpt_dir / "optimizer.pt").exists()

        # Load into a fresh model/optimizer
        model2 = nn.Linear(4, 2)
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        trainer.load_checkpoint(model2, optimizer2, str(ckpt_dir))
        assert torch.allclose(model.weight, model2.weight)

    def test_device_selection_cpu(self):
        with (
            patch(
                "open_vision_banana.training.trainer_interface.torch.cuda.is_available",
                return_value=False,
            ),
            patch(
                "open_vision_banana.training.trainer_interface.torch.backends.mps.is_available",
                return_value=False,
            ),
        ):
            trainer = NativeTrainer(device="auto")
            assert str(trainer.device) == "cpu"

    def test_amp_disabled_on_cpu(self):
        trainer = NativeTrainer(device="cpu", use_amp=True)
        assert trainer.use_amp is False
        assert trainer.use_autocast is False
        assert trainer.use_grad_scaler is False
