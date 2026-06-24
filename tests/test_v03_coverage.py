"""Tests for v0.3 coverage improvement: instance eval, split, union_find, trainer_interface."""

import pytest
import torch
import torch.nn as nn

from open_vision_banana.core.union_find import UnionFind
from open_vision_banana.data.split import split_dataset, split_vision_datasets
from open_vision_banana.evaluation.instance import (
    compute_iou_box,
    compute_iou_mask,
    compute_iou_matrix_batched,
    evaluate_instance,
)
from open_vision_banana.training.trainer_interface import (
    AccelerateTrainer,
    DDPTrainer,
    FSDPTrainer,
    NativeTrainer,
    create_trainer,
)

# ──────────────────────────────────────────────────────────────────────
# Instance evaluation tests
# ──────────────────────────────────────────────────────────────────────


class TestComputeIoUBox:
    def test_overlap(self):
        box1 = (0, 0, 4, 4)
        box2 = (2, 2, 6, 6)
        iou = compute_iou_box(box1, box2)
        # intersection = 2*2=4, area1=16, area2=16, union=28
        assert pytest.approx(iou, abs=0.001) == 4.0 / 28.0

    def test_no_overlap(self):
        box1 = (0, 0, 2, 2)
        box2 = (3, 3, 5, 5)
        assert compute_iou_box(box1, box2) == 0.0

    def test_identical(self):
        box = (1, 1, 3, 3)
        assert compute_iou_box(box, box) == 1.0

    def test_zero_area(self):
        box = (0, 0, 0, 0)
        assert compute_iou_box(box, box) == 0.0


class TestComputeIoUMask:
    def test_identical_masks(self):
        mask1 = torch.zeros(4, 4, dtype=torch.bool)
        mask2 = torch.zeros(4, 4, dtype=torch.bool)
        mask1[0, 0] = True
        mask1[0, 1] = True
        mask2 = mask1.clone()
        assert compute_iou_mask(mask1, mask2) == 1.0

    def test_disjoint_masks(self):
        mask1 = torch.zeros(4, 4, dtype=torch.bool)
        mask2 = torch.zeros(4, 4, dtype=torch.bool)
        mask1[0, 0] = True
        mask2[3, 3] = True
        assert compute_iou_mask(mask1, mask2) == 0.0

    def test_partial_overlap(self):
        mask1 = torch.zeros(4, 4, dtype=torch.bool)
        mask2 = torch.zeros(4, 4, dtype=torch.bool)
        mask1[0, 0] = True
        mask1[0, 1] = True
        mask2[0, 1] = True
        mask2[0, 2] = True
        # intersection=1, union=3
        assert pytest.approx(compute_iou_mask(mask1, mask2), abs=0.001) == 1.0 / 3.0


class TestComputeIoUMatrixBatched:
    def test_matrix_shape(self):
        pred = [{"mask": torch.zeros(4, 4, dtype=torch.bool)} for _ in range(3)]
        gt = [{"mask": torch.zeros(4, 4, dtype=torch.bool)} for _ in range(2)]
        matrix = compute_iou_matrix_batched(pred, gt)
        assert matrix.shape == (3, 2)

    def test_empty_pred(self):
        gt = [{"mask": torch.zeros(4, 4, dtype=torch.bool)} for _ in range(2)]
        matrix = compute_iou_matrix_batched([], gt)
        assert matrix.shape == (0, 2)

    def test_empty_gt(self):
        pred = [{"mask": torch.zeros(4, 4, dtype=torch.bool)} for _ in range(3)]
        matrix = compute_iou_matrix_batched(pred, [])
        assert matrix.shape == (3, 0)

    def test_diagonal_iou(self):
        """Identical masks should give IoU=1 on the diagonal."""
        mask = torch.zeros(4, 4, dtype=torch.bool)
        mask[1:3, 1:3] = True
        pred = [{"mask": mask.clone()} for _ in range(3)]
        gt = [{"mask": mask.clone()} for _ in range(3)]
        matrix = compute_iou_matrix_batched(pred, gt)
        for i in range(3):
            assert pytest.approx(matrix[i, i].item(), abs=0.001) == 1.0


class TestEvaluateInstance:
    def test_perfect_match(self):
        mask = torch.zeros(8, 8, dtype=torch.bool)
        mask[2:5, 2:5] = True
        instances = [{"mask": mask.clone()}]
        result = evaluate_instance(instances, instances)
        assert result["AP@0.5"] == 1.0
        assert result["AP@0.75"] == 1.0
        assert result["mAP@[0.5:0.95]"] == 1.0

    def test_no_predictions(self):
        gt_mask = torch.zeros(8, 8, dtype=torch.bool)
        gt_mask[0, 0] = True
        gt_instances = [{"mask": gt_mask}]
        result = evaluate_instance([], gt_instances)
        assert result["AP@0.5"] == 0.0
        assert result["mAP@[0.5:0.95]"] == 0.0

    def test_empty_gt(self):
        mask = torch.zeros(8, 8, dtype=torch.bool)
        mask[0, 0] = True
        pred_instances = [{"mask": mask}]
        result = evaluate_instance(pred_instances, [])
        assert result["AP@0.5"] == 0.0

    def test_custom_thresholds(self):
        mask = torch.zeros(8, 8, dtype=torch.bool)
        mask[2:5, 2:5] = True
        instances = [{"mask": mask}]
        result = evaluate_instance(instances, instances, iou_thresholds=[0.5])
        assert "AP@0.50" in result or "AP@0.5" in result


# ──────────────────────────────────────────────────────────────────────
# Dataset split tests
# ──────────────────────────────────────────────────────────────────────


class TestSplitDataset:
    def test_basic_split(self):
        from torch.utils.data import TensorDataset

        ds = TensorDataset(torch.arange(100).float().unsqueeze(1))
        train, val = split_dataset(ds, val_ratio=0.2)
        assert len(train) == 80
        assert len(val) == 20

    def test_zero_ratio(self):
        from torch.utils.data import TensorDataset

        ds = TensorDataset(torch.arange(10).float().unsqueeze(1))
        train, val = split_dataset(ds, val_ratio=0.0)
        assert train is ds
        assert val is None

    def test_too_small(self):
        from torch.utils.data import TensorDataset

        ds = TensorDataset(torch.arange(1).float().unsqueeze(1))
        train, val = split_dataset(ds, val_ratio=0.5)
        assert train is ds
        assert val is None

    def test_reproducible(self):
        from torch.utils.data import TensorDataset

        ds1 = TensorDataset(torch.arange(50).float().unsqueeze(1))
        ds2 = TensorDataset(torch.arange(50).float().unsqueeze(1))
        train1, val1 = split_dataset(ds1, val_ratio=0.3, seed=42)
        train2, val2 = split_dataset(ds2, val_ratio=0.3, seed=42)
        # Same indices should be selected
        val1_indices = sorted(val1.indices)
        val2_indices = sorted(val2.indices)
        assert val1_indices == val2_indices


class TestSplitVisionDatasets:
    def test_multiple_datasets(self):
        from torch.utils.data import TensorDataset

        ds1 = TensorDataset(torch.arange(50).float().unsqueeze(1))
        ds2 = TensorDataset(torch.arange(50).float().unsqueeze(1))
        train_list, val = split_vision_datasets([ds1, ds2], val_ratio=0.2)
        assert len(train_list) == 1
        assert val is not None
        assert len(val) > 0

    def test_empty_list(self):
        train_list, val = split_vision_datasets([], val_ratio=0.2)
        assert train_list == []
        assert val is None

    def test_zero_ratio(self):
        from torch.utils.data import TensorDataset

        ds = TensorDataset(torch.arange(10).float().unsqueeze(1))
        train_list, val = split_vision_datasets([ds], val_ratio=0.0)
        assert train_list == [ds]
        assert val is None


# ──────────────────────────────────────────────────────────────────────
# UnionFind tests
# ──────────────────────────────────────────────────────────────────────


class TestUnionFind:
    def test_initial_state(self):
        uf = UnionFind(5)
        for i in range(5):
            assert uf.find(i) == i

    def test_basic_union(self):
        uf = UnionFind(5)
        uf.union(0, 1)
        assert uf.connected(0, 1)
        assert not uf.connected(0, 2)

    def test_transitive_union(self):
        uf = UnionFind(10)
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(2, 3)
        assert uf.connected(0, 3)
        assert uf.find(0) == uf.find(3)

    def test_union_by_rank(self):
        uf = UnionFind(10)
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(0, 2)  # Now 0-1-2-3 all connected
        assert uf.connected(1, 3)

    def test_path_compression(self):
        uf = UnionFind(10)
        # Create a chain: 0->1->2->3->4
        for i in range(4):
            uf.union(i, i + 1)
        # After find, path should be compressed
        root = uf.find(4)
        # find should return the same root for all
        for i in range(5):
            assert uf.find(i) == root

    def test_self_connected(self):
        uf = UnionFind(5)
        assert uf.connected(2, 2)

    def test_disjoint_sets(self):
        uf = UnionFind(6)
        uf.union(0, 1)
        uf.union(3, 4)
        assert uf.connected(0, 1)
        assert uf.connected(3, 4)
        assert not uf.connected(0, 3)
        assert not uf.connected(1, 4)
        assert not uf.connected(0, 5)


# ──────────────────────────────────────────────────────────────────────
# TrainerInterface tests
# ──────────────────────────────────────────────────────────────────────


class TestNativeTrainer:
    def test_auto_device(self):
        trainer = NativeTrainer(device="auto")
        assert trainer.device.type in ("cuda", "mps", "cpu")

    def test_explicit_cpu(self):
        trainer = NativeTrainer(device="cpu", use_amp=False)
        assert trainer.device.type == "cpu"
        assert not trainer.use_amp

    def test_wrap_model(self):
        trainer = NativeTrainer(device="cpu", use_amp=False)
        model = nn.Linear(10, 5)
        wrapped = trainer.wrap_model(model)
        assert wrapped is model  # On CPU, .to() returns self

    def test_move_to_device(self):
        trainer = NativeTrainer(device="cpu")
        t = torch.randn(3, 3)
        moved = trainer.move_to_device(t)
        assert moved.device.type == "cpu"

    def test_backward_and_step(self):
        trainer = NativeTrainer(device="cpu", use_amp=False)
        model = nn.Linear(4, 2)
        trainer.wrap_model(model)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer.setup_optimizer(opt)

        x = torch.randn(2, 4)
        y = model(x).sum()
        trainer.backward(y)
        trainer.optimizer_step()
        # Verify gradients were applied
        assert model.weight.grad is not None

    def test_optimizer_step_without_setup(self):
        trainer = NativeTrainer(device="cpu", use_amp=False)
        model = nn.Linear(4, 2)
        trainer.wrap_model(model)
        with pytest.raises(RuntimeError, match="setup_optimizer"):
            trainer.optimizer_step()

    def test_save_load_checkpoint(self, tmp_path):
        trainer = NativeTrainer(device="cpu", use_amp=False)
        model = nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer.save_checkpoint(model, opt, step=10, output_dir=str(tmp_path))
        # Verify files exist
        assert (tmp_path / "checkpoint-10" / "model.pt").exists()
        assert (tmp_path / "checkpoint-10" / "optimizer.pt").exists()

        # Load into new model
        model2 = nn.Linear(4, 2)
        opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        meta = trainer.load_checkpoint(model2, opt2, str(tmp_path / "checkpoint-10"))
        assert meta == {}
        assert torch.allclose(model.weight, model2.weight)

    def test_is_main_process(self):
        trainer = NativeTrainer(device="cpu", use_amp=False)
        assert trainer.is_main_process()

    def test_barrier(self):
        trainer = NativeTrainer(device="cpu", use_amp=False)
        trainer.barrier()  # Should be no-op


class TestDDPTrainer:
    def test_single_process(self):
        trainer = DDPTrainer(world_size=1, rank=0, local_rank=0)
        assert trainer.world_size == 1
        assert trainer.is_main_process()

    def test_barrier_no_dist(self):
        trainer = DDPTrainer(world_size=1)
        trainer.barrier()  # Should be no-op with world_size=1

    def test_save_checkpoint_single_process(self, tmp_path):
        trainer = DDPTrainer(world_size=1)
        model = nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer.save_checkpoint(model, opt, step=5, output_dir=str(tmp_path))
        assert (tmp_path / "checkpoint-5" / "model.pt").exists()


class TestCreateTrainer:
    def test_create_native(self):
        trainer = create_trainer(backend="native", device="cpu", use_amp=False)
        assert isinstance(trainer, NativeTrainer)

    def test_create_ddp(self):
        trainer = create_trainer(backend="ddp", world_size=1)
        assert isinstance(trainer, DDPTrainer)

    def test_create_fsdp(self):
        trainer = create_trainer(backend="fsdp", world_size=1)
        assert isinstance(trainer, FSDPTrainer)

    def test_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown training backend"):
            create_trainer(backend="unknown")

    def test_create_accelerate_import_error(self):
        # Accelerate may or may not be installed; if it fails, verify ImportError
        try:
            trainer = create_trainer(backend="accelerate")
            assert isinstance(trainer, AccelerateTrainer)
        except ImportError:
            pass  # Accelerate not installed — acceptable


# ──────────────────────────────────────────────────────────────────────
# EMAModel v0.3 optimization tests
# ──────────────────────────────────────────────────────────────────────


class TestEMAModelV03:
    def test_param_refs_captured(self):
        from open_vision_banana.training.utils import EMAModel

        model = nn.Linear(10, 5)
        ema = EMAModel(model, decay=0.99)
        assert hasattr(ema, "_param_refs")
        assert len(ema._param_refs) > 0
        # Each ref should be (name, param) tuple
        for name, p in ema._param_refs:
            assert isinstance(name, str)
            assert isinstance(p, torch.Tensor)

    def test_update_with_refs(self):
        from open_vision_banana.training.utils import EMAModel

        model = nn.Linear(10, 5)
        ema = EMAModel(model, decay=0.99)

        # Modify weights
        with torch.no_grad():
            model.weight.add_(1.0)

        ema.update(model)
        # Shadow params should have moved toward new weights
        assert not torch.allclose(ema.shadow_params["weight"], model.weight.data)

    def test_warmup_skip(self):
        from open_vision_banana.training.utils import EMAModel

        model = nn.Linear(10, 5)
        original_shadow = {k: v.clone() for k, v in model.state_dict().items()}
        ema = EMAModel(model, decay=0.99, update_after_step=10)

        with torch.no_grad():
            model.weight.add_(10.0)

        ema.update(model)
        # Should not update during warmup
        assert torch.allclose(ema.shadow_params["weight"], original_shadow["weight"])

    def test_apply_and_restore_shadow(self):
        from open_vision_banana.training.utils import EMAModel

        model = nn.Linear(10, 5)
        model.weight.data.clone()
        ema = EMAModel(model, decay=0.99)

        # Modify model weights
        with torch.no_grad():
            model.weight.add_(5.0)

        # Apply shadow (should restore to original)
        backup = ema.apply_shadow(model)
        assert torch.allclose(model.weight.data, ema.shadow_params["weight"])

        # Restore (should put back modified weights)
        ema.restore_shadow(model, backup)
        assert not torch.allclose(model.weight.data, ema.shadow_params["weight"])
