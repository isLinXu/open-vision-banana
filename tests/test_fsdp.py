"""Tests for FSDP training utilities."""

import torch
import torch.nn as nn

from open_vision_banana.training.fsdp_utils import FSDPWrapper
from open_vision_banana.training.trainer_interface import FSDPTrainer, create_trainer


class TestFSDPWrapper:
    def test_single_process_disabled(self):
        fsdp = FSDPWrapper(world_size=1, rank=0, local_rank=0)
        assert not fsdp.enabled
        assert fsdp.is_main()

    def test_wrap_noop_single_gpu(self):
        fsdp = FSDPWrapper(world_size=1)
        model = nn.Linear(4, 2)
        wrapped = fsdp.wrap(model)
        assert wrapped is model

    def test_sampler_none_when_disabled(self):
        fsdp = FSDPWrapper(world_size=1)
        assert fsdp.get_train_sampler([]) is None


class TestFSDPTrainer:
    def test_create_trainer(self):
        trainer = create_trainer(backend="fsdp", world_size=1, rank=0, local_rank=0)
        assert isinstance(trainer, FSDPTrainer)
        assert trainer.is_main_process()

    def test_wrap_single_process(self):
        trainer = FSDPTrainer(world_size=1)
        model = nn.Linear(4, 2)
        wrapped = trainer.wrap_model(model)
        assert wrapped is model
