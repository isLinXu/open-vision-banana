"""Tests for v0.3 P1 features: gradient accumulation, evaluate_batch, LR finder."""

import torch
import torch.nn as nn

from open_vision_banana.evaluation.evaluator import VisionEvaluator
from open_vision_banana.training.lr_finder import LRFinder
from open_vision_banana.training.trainer_interface import DDPTrainer, NativeTrainer

# ──────────────────────────────────────────────────────────────────────
# Gradient accumulation & gradient clipping tests
# ──────────────────────────────────────────────────────────────────────


class TestGradientAccumulation:
    def test_default_accumulation_steps(self):
        trainer = NativeTrainer(device="cpu", use_amp=False)
        assert trainer.gradient_accumulation_steps == 1

    def test_custom_accumulation_steps(self):
        trainer = NativeTrainer(
            device="cpu",
            use_amp=False,
            gradient_accumulation_steps=4,
        )
        assert trainer.gradient_accumulation_steps == 4

    def test_custom_grad_norm(self):
        trainer = NativeTrainer(
            device="cpu",
            use_amp=False,
            max_grad_norm=5.0,
        )
        assert trainer.max_grad_norm == 5.0

    def test_train_step_accumulation(self):
        """Test that train_step returns True only at accumulation boundary."""
        trainer = NativeTrainer(
            device="cpu",
            use_amp=False,
            gradient_accumulation_steps=3,
        )
        model = nn.Linear(4, 2)
        trainer.wrap_model(model)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer.setup_optimizer(opt)

        # Step 0: should NOT trigger optimizer step
        loss = model(torch.randn(2, 4)).sum()
        stepped = trainer.train_step(model, loss, 0)
        assert not stepped

        # Step 1: should NOT trigger optimizer step
        loss = model(torch.randn(2, 4)).sum()
        stepped = trainer.train_step(model, loss, 1)
        assert not stepped

        # Step 2: should trigger optimizer step (3rd step)
        loss = model(torch.randn(2, 4)).sum()
        stepped = trainer.train_step(model, loss, 2)
        assert stepped

    def test_train_step_no_accumulation(self):
        """With accumulation=1, every step should trigger optimizer step."""
        trainer = NativeTrainer(
            device="cpu",
            use_amp=False,
            gradient_accumulation_steps=1,
        )
        model = nn.Linear(4, 2)
        trainer.wrap_model(model)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer.setup_optimizer(opt)

        loss = model(torch.randn(2, 4)).sum()
        stepped = trainer.train_step(model, loss, 0)
        assert stepped

    def test_ddp_inherits_accumulation(self):
        trainer = DDPTrainer(
            world_size=1,
            gradient_accumulation_steps=2,
            max_grad_norm=2.0,
        )
        assert trainer.gradient_accumulation_steps == 2
        assert trainer.max_grad_norm == 2.0


# ──────────────────────────────────────────────────────────────────────
# VisionEvaluator.evaluate_batch tests
# ──────────────────────────────────────────────────────────────────────


class TestEvaluateBatch:
    def test_depth_batch_eval(self):
        """Test evaluate_batch for depth estimation."""

        # Create a mock model that returns its input as rgb_output
        class MockModel:
            def eval(self):
                pass

            def predict(self, **kwargs):
                return {"rgb_output": kwargs["image"] * 0.5 + 0.5}

        model = MockModel()
        evaluator = VisionEvaluator(model, device="cpu")

        image = torch.randn(2, 3, 8, 8)  # [-1, 1]
        target_rgb = torch.rand(2, 3, 8, 8)  # [0, 1]

        metrics = evaluator.evaluate_batch(
            image=image,
            target_rgb=target_rgb,
            task="depth",
        )
        assert len(metrics) > 0
        assert all(isinstance(v, float) for v in metrics.values())

    def test_normal_batch_eval(self):
        class MockModel:
            def eval(self):
                pass

            def predict(self, **kwargs):
                return {"rgb_output": kwargs["image"] * 0.5 + 0.5}

        evaluator = VisionEvaluator(MockModel(), device="cpu")
        image = torch.randn(1, 3, 8, 8)
        target_rgb = torch.rand(1, 3, 8, 8)

        metrics = evaluator.evaluate_batch(
            image=image,
            target_rgb=target_rgb,
            task="normal",
        )
        assert len(metrics) > 0

    def test_semantic_batch_eval(self):
        class MockModel:
            def eval(self):
                pass

            def predict(self, **kwargs):
                return {"rgb_output": kwargs["image"] * 0.5 + 0.5}

        cmap = {"cat": (255, 0, 0), "dog": (0, 255, 0)}
        evaluator = VisionEvaluator(MockModel(), device="cpu")
        image = torch.randn(1, 3, 8, 8)
        target_rgb = torch.rand(1, 3, 8, 8)

        metrics = evaluator.evaluate_batch(
            image=image,
            target_rgb=target_rgb,
            task="semantic",
            class_color_map=cmap,
        )
        assert len(metrics) > 0


# ──────────────────────────────────────────────────────────────────────
# LRFinder tests
# ──────────────────────────────────────────────────────────────────────


class TestLRFinder:
    def test_basic_range_test(self):
        """Test that LR finder runs and produces LR/loss lists."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
        criterion = nn.MSELoss()

        # Simple dataset
        x = torch.randn(50, 10)
        y = torch.randn(50, 1)
        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        finder = LRFinder(model, optimizer, criterion, device="cpu")
        finder.range_test(loader, start_lr=1e-7, end_lr=1e-1, num_iter=20)

        assert len(finder.lrs) > 0
        assert len(finder.losses) > 0
        assert len(finder.lrs) == len(finder.losses)
        # LRs should be monotonically increasing
        for i in range(1, len(finder.lrs)):
            assert finder.lrs[i] > finder.lrs[i - 1]

    def test_suggest_lr(self):
        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
        criterion = nn.MSELoss()

        x = torch.randn(50, 10)
        y = torch.randn(50, 1)
        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        finder = LRFinder(model, optimizer, criterion, device="cpu")
        finder.range_test(loader, start_lr=1e-7, end_lr=1e-1, num_iter=20)

        lr = finder.suggest_lr()
        assert 1e-7 <= lr <= 1e-1

    def test_restore(self):
        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
        criterion = nn.MSELoss()

        original_weight = model.weight.data.clone()

        x = torch.randn(20, 10)
        y = torch.randn(20, 1)
        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        finder = LRFinder(model, optimizer, criterion, device="cpu")
        finder.range_test(loader, num_iter=10)

        # Weights should have changed
        assert not torch.allclose(model.weight.data, original_weight)

        # Restore
        finder.restore()
        assert torch.allclose(model.weight.data, original_weight)

    def test_early_stop_on_divergence(self):
        """Test that LR finder stops when loss diverges."""
        model = nn.Linear(10, 1)
        # Very high initial LR to force divergence
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
        criterion = nn.MSELoss()

        x = torch.randn(20, 10)
        y = torch.randn(20, 1)
        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        finder = LRFinder(model, optimizer, criterion, device="cpu")
        # Use a very high end_lr to force divergence
        finder.range_test(loader, start_lr=1e-7, end_lr=1e10, num_iter=100)

        # Should stop before 100 iterations
        assert len(finder.lrs) < 100

    def test_dict_batch(self):
        """Test LR finder with dict-style batches."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
        criterion = nn.MSELoss()

        class DictDataset(torch.utils.data.Dataset):
            def __init__(self, n):
                self.x = torch.randn(n, 10)
                self.y = torch.randn(n, 1)

            def __len__(self):
                return len(self.x)

            def __getitem__(self, i):
                return {"image": self.x[i], "target_rgb": self.y[i]}

        loader = torch.utils.data.DataLoader(DictDataset(20), batch_size=4)

        finder = LRFinder(model, optimizer, criterion, device="cpu")
        finder.range_test(loader, num_iter=10)

        assert len(finder.lrs) > 0
