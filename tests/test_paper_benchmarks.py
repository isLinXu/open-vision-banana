"""Tests for decoded paper benchmark data and pixel-level aggregation."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from open_vision_banana.data import build_dataset
from open_vision_banana.evaluation.benchmark import (
    DepthAccumulator,
    NormalAccumulator,
    SemanticAccumulator,
    apply_depth_benchmark_protocol,
)
from scripts.evaluate_paper_benchmarks import (
    evaluate_compatible,
    main,
    preflight_jobs,
    resolve_inference_steps,
    resolve_runtime_dtype,
    write_results,
)
from scripts.prepare_cityscapes_19class import (
    discover_flat_pairs,
    write_paper_val_layout,
    write_training_split,
)


def test_depth_accumulator_accepts_cpu_mask_with_device_pred() -> None:
    accumulator = DepthAccumulator()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    pred = torch.ones(1, 1, 2, 2, device=device)
    gt = torch.ones(1, 1, 2, 2, device=device)
    mask = torch.ones(2, 2, dtype=torch.bool)
    accumulator.update(pred, gt, mask)
    assert accumulator.compute()["valid_pixels"] == 4
    assert accumulator.compute()["delta_1.25"] == 1.0


def test_nyu_protocol_preserves_batch_shape_for_channelless_mask() -> None:
    pred = torch.ones(2, 1, 2, 2)
    gt = torch.ones(2, 1, 2, 2)
    channelless_mask = torch.ones(2, 2, 2, dtype=torch.bool)
    _, _, mask = apply_depth_benchmark_protocol("nyu", pred, gt, channelless_mask)
    assert mask.shape == gt.shape
    accumulator = DepthAccumulator()
    accumulator.update(pred, gt, mask)
    assert accumulator.compute()["valid_pixels"] == 8


def test_mps_evaluation_requires_float32() -> None:
    assert resolve_runtime_dtype("float32", "mps") == torch.float32
    with pytest.raises(ValueError, match="requires --dtype float32"):
        resolve_runtime_dtype("float16", "mps")
    assert resolve_runtime_dtype("float16", "mps", allow_mps_reduced_dtype=True) == torch.float16


def test_depth_accumulator_reports_median_aligned_delta() -> None:
    accumulator = DepthAccumulator()
    pred = torch.tensor([[[[2.0, 4.0], [6.0, 8.0]]]])
    gt = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    accumulator.update(pred, gt)
    metrics = accumulator.compute()
    assert metrics["AbsRel"] == 1.0
    assert metrics["delta_1.25"] == 0.0
    assert metrics["AbsRel_aligned"] == pytest.approx(0.0)
    assert metrics["delta_1.25_aligned"] == 1.0


def test_depth_accumulator_skips_degenerate_alignment() -> None:
    accumulator = DepthAccumulator()
    pred = torch.tensor([[[[1e-8, 1e-8], [1e-8, 1e-8]]]])
    gt = torch.tensor([[[[4.0, 8.0], [12.0, 16.0]]]])
    accumulator.update(pred, gt)
    metrics = accumulator.compute()
    assert metrics["AbsRel"] > 0.9
    assert metrics["aligned_valid_pixels"] == 0
    assert metrics["AbsRel_aligned"] == 0.0


def test_nyu_protocol_clips_exploding_predictions() -> None:
    pred = torch.tensor([[[[10000.0, 2.0]]]])
    gt = torch.tensor([[[[2.0, 2.0]]]])
    capped_pred, capped_gt, mask = apply_depth_benchmark_protocol("nyu_depth", pred, gt)
    assert float(capped_pred.max()) == 10.0
    assert mask.all()
    acc = DepthAccumulator()
    acc.update(capped_pred, capped_gt, mask)
    metrics = acc.compute()
    assert metrics["RMSE"] < 10.0
    assert metrics["AbsRel"] == pytest.approx((8.0 / 2.0 + 0.0) / 2.0)


def test_nyu_protocol_masks_gt_beyond_10m() -> None:
    pred = torch.tensor([[[[2.0, 2.0]]]])
    gt = torch.tensor([[[[2.0, 12.0]]]])
    _, _, mask = apply_depth_benchmark_protocol("nyu", pred, gt)
    assert bool(mask[..., 0])
    assert not bool(mask[..., 1])


def test_kitti_protocol_clips_to_80m() -> None:
    pred = torch.tensor([[[[200.0, 20.0]]]])
    gt = torch.tensor([[[[20.0, 20.0]]]])
    capped_pred, _, mask = apply_depth_benchmark_protocol("kitti", pred, gt)
    assert float(capped_pred.max()) == 80.0
    assert mask.all()


def test_eth3d_protocol_clips_to_80m() -> None:
    pred = torch.tensor([[[[10000.0, 12.0]]]])
    gt = torch.tensor([[[[12.0, 12.0]]]])
    capped_pred, _, mask = apply_depth_benchmark_protocol("eth3d", pred, gt)
    assert float(capped_pred.max()) == 80.0
    assert mask.all()


def test_unnamed_dataset_does_not_apply_depth_cap() -> None:
    pred = torch.tensor([[[[10000.0]]]])
    gt = torch.tensor([[[[2.0]]]])
    capped_pred, _, _ = apply_depth_benchmark_protocol("", pred, gt)
    assert float(capped_pred.max()) == 10000.0


def test_depth_accumulator_is_pixel_weighted() -> None:
    accumulator = DepthAccumulator()
    accumulator.update(torch.ones(1, 1, 1, 3), torch.ones(1, 1, 1, 3))
    accumulator.update(
        torch.tensor([[[[2.0]]]]),
        torch.tensor([[[[1.0]]]]),
    )
    metrics = accumulator.compute()
    assert metrics["valid_pixels"] == 4
    assert metrics["AbsRel"] == 0.25
    assert metrics["delta_1.25"] == 0.75


def test_normal_accumulator_reports_decoded_angles() -> None:
    gt = torch.tensor([[[[0.0]], [[0.0]], [[1.0]]]])
    pred = torch.tensor([[[[0.0]], [[1.0]], [[0.0]]]])
    accumulator = NormalAccumulator()
    accumulator.update(pred, gt)
    metrics = accumulator.compute()
    assert metrics["valid_pixels"] == 1
    assert metrics["mean_angle_deg"] == 90.0


def test_semantic_accumulator_uses_global_confusion() -> None:
    accumulator = SemanticAccumulator(num_classes=2)
    accumulator.update(torch.tensor([[[0, 1]]]), torch.tensor([[[0, 0]]]))
    accumulator.update(torch.tensor([[[1, 1]]]), torch.tensor([[[1, 1]]]))
    metrics = accumulator.compute()
    assert metrics["valid_pixels"] == 4
    assert metrics["mIoU"] == pytest.approx(7 / 12)


def test_semantic_accumulator_accepts_channel_dim_pred() -> None:
    accumulator = SemanticAccumulator(num_classes=2)
    pred = torch.tensor([[[[0, 1], [1, 0]]]])  # [B, 1, H, W]
    gt = torch.tensor([[[0, 1], [1, 0]]])  # [B, H, W]
    accumulator.update(pred, gt)
    metrics = accumulator.compute()
    assert metrics["valid_pixels"] == 4
    assert metrics["mIoU"] == pytest.approx(1.0)


def test_cityscapes_label_ids_map_to_train_ids(tmp_path: Path) -> None:
    image_dir = tmp_path / "leftImg8bit" / "val" / "demo"
    label_dir = tmp_path / "gtFine" / "val" / "demo"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (2, 2), (10, 20, 30)).save(image_dir / "demo_000000_000000_leftImg8bit.png")
    labels = np.array([[7, 8], [33, 255]], dtype=np.uint8)
    Image.fromarray(labels).save(label_dir / "demo_000000_000000_gtFine_labelIds.png")

    dataset = build_dataset("semantic", str(tmp_path), split="val", resolution=2)
    item = dataset[0]
    assert item["gt_labels"].tolist() == [[0, 1], [18, 255]]


def test_benchmark_preflight_reports_empty_adapter_before_model_load(tmp_path: Path) -> None:
    result = preflight_jobs(
        [("depth", "nyu", tmp_path, "nyu_depth")],
        split="val",
        resolution=2,
    )

    assert result["ok"] is False
    record = result["jobs"][0]
    assert record["samples"] == 0
    assert "no usable pairs" in record["error"]


def test_nyu_flat_test_export_is_not_reused_as_train(tmp_path: Path) -> None:
    from open_vision_banana.data.datasets import NYUDepthDataset

    (tmp_path / "images").mkdir()
    (tmp_path / "depths").mkdir()
    Image.new("RGB", (2, 2), (10, 20, 30)).save(tmp_path / "images" / "00001.jpg")
    Image.fromarray(np.ones((2, 2), dtype=np.uint16)).save(tmp_path / "depths" / "00001.png")

    assert len(NYUDepthDataset(tmp_path, split="train", resolution=2)) == 0
    assert len(NYUDepthDataset(tmp_path, split="test", resolution=2)) == 1
    # ``val`` is an evaluation alias for a flat extracted test export.
    assert len(NYUDepthDataset(tmp_path, split="val", resolution=2)) == 1


def test_nyu_split_directories_are_scanned_independently(tmp_path: Path) -> None:
    from open_vision_banana.data.datasets import NYUDepthDataset

    for split, name in (("train", "00001"), ("test", "00002")):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "depths").mkdir(parents=True)
        Image.new("RGB", (2, 2), (10, 20, 30)).save(tmp_path / split / "images" / f"{name}.jpg")
        Image.fromarray(np.ones((2, 2), dtype=np.uint16)).save(
            tmp_path / split / "depths" / f"{name}.png"
        )

    assert len(NYUDepthDataset(tmp_path, split="train", resolution=2)) == 1
    assert len(NYUDepthDataset(tmp_path, split="test", resolution=2)) == 1


def test_nyu_official_h5_split_is_loaded_without_cross_split_leakage(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    from open_vision_banana.data.datasets import NYUDepthDataset

    for split, name in (("train", "00001"), ("test", "00002")):
        split_dir = tmp_path / split / "official"
        split_dir.mkdir(parents=True)
        with h5py.File(split_dir / f"{name}.h5", "w") as handle:
            handle.create_dataset("rgb", data=np.zeros((3, 2, 2), dtype=np.uint8))
            handle.create_dataset("depth", data=np.ones((2, 2), dtype=np.float32))

    train = NYUDepthDataset(tmp_path, split="train", resolution=2)
    test = NYUDepthDataset(tmp_path, split="test", resolution=2)
    assert len(train) == len(test) == 1
    assert train.samples[0]["image"].name == "00001.h5"
    assert test.samples[0]["image"].name == "00002.h5"
    assert test[0]["gt_depth"].shape == (1, 2, 2)


def test_benchmark_preflight_accepts_cityscapes_val_layout(tmp_path: Path) -> None:
    image_dir = tmp_path / "leftImg8bit" / "val" / "demo"
    label_dir = tmp_path / "gtFine" / "val" / "demo"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (2, 2), (10, 20, 30)).save(image_dir / "demo_000000_000000_leftImg8bit.png")
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(
        label_dir / "demo_000000_000000_gtFine_labelIds.png"
    )

    result = preflight_jobs(
        [("semantic", "cityscapes", tmp_path, "semantic")],
        split="val",
        resolution=2,
    )

    assert result["ok"] is True
    assert result["jobs"][0]["samples"] == 1
    assert len(result["jobs"][0]["dataset_fingerprint"]) == 64


def test_preflight_cli_does_not_require_model_name(tmp_path: Path, monkeypatch) -> None:
    image_dir = tmp_path / "leftImg8bit" / "val" / "demo"
    label_dir = tmp_path / "gtFine" / "val" / "demo"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (2, 2), (10, 20, 30)).save(
        image_dir / "demo_000000_000000_leftImg8bit.png"
    )
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(
        label_dir / "demo_000000_000000_gtFine_labelIds.png"
    )
    output = tmp_path / "preflight.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_paper_benchmarks.py",
            "--benchmark",
            "semantic",
            "--cityscapes-root",
            str(tmp_path),
            "--preflight-only",
            "--resolution",
            "2",
            "--output-json",
            str(output),
        ],
    )

    assert main() == 0
    assert output.is_file()


def test_evaluate_compatible_drops_unknown_kwargs() -> None:
    class OldEvaluator:
        def evaluate(self, dataloader, task: str, seed: int = 0):
            return {"task": task, "seed": seed, "n": dataloader}

    metrics = evaluate_compatible(
        OldEvaluator(),
        7,
        task="semantic",
        seed=42,
        depth_curve="metric_power",
        depth_log_alpha=0.4,
    )
    assert metrics == {"task": "semantic", "seed": 42, "n": 7}


def test_evaluate_compatible_keeps_var_keyword() -> None:
    class NewEvaluator:
        def evaluate(self, dataloader, task: str, **kwargs):
            return {"task": task, "kwargs": kwargs}

    metrics = evaluate_compatible(
        NewEvaluator(),
        None,
        task="depth",
        depth_curve="log_relative",
    )
    assert metrics["kwargs"]["depth_curve"] == "log_relative"


def test_write_results_serializes_non_finite_metrics_as_null(tmp_path: Path) -> None:
    output = tmp_path / "metrics.json"
    write_results(
        output,
        {"per_class_IoU": [1.0, math.nan, math.inf, -math.inf]},
    )
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed["per_class_IoU"] == [1.0, None, None, None]


def test_prepare_cityscapes_19class_writes_train_and_paper_val(tmp_path: Path) -> None:
    image_dir = tmp_path / "leftImg8bit" / "train"
    label_dir = tmp_path / "gtFine" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4), (10, 20, 30)).save(image_dir / "cityscapes_000000.png")
    Image.fromarray(np.full((4, 4, 3), 26, dtype=np.uint8)).save(
        label_dir / "cityscapes_000000.png"
    )

    pairs = discover_flat_pairs(tmp_path)
    train_out = tmp_path / "train_split"
    paper_root = tmp_path / "paper"
    assert write_training_split(pairs, train_out, resolution=2) == 1
    assert write_paper_val_layout(pairs, paper_root) == 1

    target = np.array(Image.open(train_out / "targets" / "cityscapes_000000.png"))
    assert tuple(target[0, 0]) == (0, 0, 142)
    dataset = build_dataset("semantic", str(paper_root), split="val", resolution=2)
    assert len(dataset) == 1
    assert int(dataset[0]["gt_labels"][0, 0]) == 13


def test_resolve_inference_steps_uses_sana_training_schedule() -> None:
    assert resolve_inference_steps("sana-600m-512px", 1) == 20
    assert resolve_inference_steps("sana-600m-512px", 20) == 20
    assert resolve_inference_steps("black-forest-labs/FLUX.1-schnell", 1) == 1


def test_nyu_protocol_uses_eigen_crop_on_native_frame() -> None:
    gt = torch.ones(1, 1, 480, 640)
    _, _, mask = apply_depth_benchmark_protocol("nyu", gt.clone(), gt)
    # Eigen crop: rows 45:471, cols 41:601.
    assert int(mask.sum()) == (471 - 45) * (601 - 41)
    assert bool(mask[0, 0, 45, 41]) and not bool(mask[0, 0, 44, 41])
    assert bool(mask[0, 0, 470, 600]) and not bool(mask[0, 0, 471, 600])


def test_nyu_eigen_crop_is_fractional_on_resized_canvas() -> None:
    gt = torch.ones(1, 1, 512, 512)
    _, _, mask = apply_depth_benchmark_protocol("nyu_depth", gt.clone(), gt)
    rows = mask[0, 0].any(dim=1).nonzero().flatten()
    # Top of the image is kept (Eigen crop starts at ~9% of the height), unlike
    # the KITTI Garg crop which discards the top ~41%.
    assert int(rows.min()) < 0.15 * 512
    assert int(rows.max()) > 0.95 * 512


def test_kitti_garg_crop_applies_on_resized_canvas() -> None:
    gt = torch.ones(1, 1, 512, 512) * 10.0
    _, _, mask = apply_depth_benchmark_protocol("kitti", gt.clone(), gt)
    rows = mask[0, 0].any(dim=1).nonzero().flatten()
    assert int(rows.min()) >= int(0.40 * 512)


def test_instance_decoding_cli_override_defaults_to_paper_and_rejects_unknown_fields() -> None:
    import pytest

    from open_vision_banana.core.segmentation import PAPER_INSTANCE_DECODING_CONFIG
    from scripts.evaluate_paper_benchmarks import parse_instance_decoding

    assert parse_instance_decoding(None) == PAPER_INSTANCE_DECODING_CONFIG
    ablation = parse_instance_decoding("seed_merge_rule=pixel_and_seed, spatial_merge_area_basis=pixel")
    assert ablation.seed_merge_rule == "pixel_and_seed"
    assert ablation.spatial_merge_area_basis == "pixel"
    assert ablation.color_tolerance == 14.0
    with pytest.raises(ValueError, match="Unknown instance decoding field"):
        parse_instance_decoding("tau=14")
