"""Native-resolution depth evaluation and per-image aggregation."""

import numpy as np
import pytest
import torch
from PIL import Image

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.data.datasets import NYUDepthDataset
from open_vision_banana.evaluation.benchmark import DepthAccumulator
from open_vision_banana.evaluation.evaluator import VisionEvaluator


class _ConstDepthModel:
    """Direct-call model that always 'predicts' a constant depth at 8x8."""

    def __init__(self, depth_m: float):
        self._rgb = DepthRGBMapper().depth_to_rgb(torch.full((1, 1, 8, 8), depth_m))

    def eval(self):
        return self

    def __call__(self, image):
        return self._rgb.expand(image.shape[0], -1, -1, -1).clone()


def _batch(gt_native: float, native_hw=(48, 64)):
    h, w = native_hw
    return {
        "image": torch.zeros(1, 3, 8, 8),
        "target_rgb": torch.zeros(1, 3, 8, 8),
        "gt_depth": torch.full((1, 1, 8, 8), gt_native),
        "valid_mask": torch.ones(1, 8, 8, dtype=torch.bool),
        "gt_depth_native": torch.full((1, 1, h, w), gt_native),
        "valid_mask_native": torch.ones(1, h, w, dtype=torch.bool),
    }


def test_native_resolution_scores_at_gt_size():
    ev = VisionEvaluator(_ConstDepthModel(3.0), device="cpu")
    m = ev.evaluate([_batch(3.0)], task="depth", dataset_name="other", native_resolution=True)
    assert m["valid_pixels"] == 48 * 64
    assert m["delta_1.25"] == 1.0


def test_default_resolution_scores_at_model_size():
    ev = VisionEvaluator(_ConstDepthModel(3.0), device="cpu")
    m = ev.evaluate([_batch(3.0)], task="depth", dataset_name="other")
    assert m["valid_pixels"] == 8 * 8


def test_native_resolution_requires_native_gt():
    ev = VisionEvaluator(_ConstDepthModel(3.0), device="cpu")
    batch = _batch(3.0)
    del batch["gt_depth_native"]
    with pytest.raises(ValueError, match="gt_depth_native"):
        ev.evaluate([batch], task="depth", dataset_name="other", native_resolution=True)


def test_per_image_aggregation_differs_from_pixel_weighted():
    acc = DepthAccumulator()
    # Image A: 100 pixels, perfect. Image B: 4 pixels, completely wrong.
    acc.update(torch.ones(1, 1, 10, 10), torch.ones(1, 1, 10, 10))
    acc.update(torch.full((1, 1, 2, 2), 4.0), torch.ones(1, 1, 2, 2))
    m = acc.compute()
    assert m["images"] == 2
    assert m["delta_1.25_per_image"] == pytest.approx(0.5)
    assert m["delta_1.25"] == pytest.approx(100 / 104)
    assert m["AbsRel_per_image"] == pytest.approx((0.0 + 3.0) / 2)


def test_nyu_native_gt_returns_unresized_depth(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "depths").mkdir()
    Image.fromarray(np.zeros((48, 64, 3), dtype=np.uint8)).save(tmp_path / "images" / "a.png")
    np.save(tmp_path / "depths" / "a.npy", np.full((48, 64), 2.5, dtype=np.float32))

    plain = NYUDepthDataset(tmp_path, split="val", resolution=16)
    assert len(plain) == 1
    assert "gt_depth_native" not in plain[0]

    native = NYUDepthDataset(tmp_path, split="val", resolution=16, native_gt=True)[0]
    assert native["gt_depth"].shape[-2:] == (16, 16)
    assert native["gt_depth_native"].shape == (1, 48, 64)
    assert native["valid_mask_native"].shape == (48, 64)
    assert float(native["gt_depth_native"].max()) == pytest.approx(2.5)


def test_nyu_accepts_depth_directory_name(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "depth").mkdir()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(tmp_path / "images" / "a.png")
    np.save(tmp_path / "depth" / "a.npy", np.full((8, 8), 2.0, dtype=np.float32))
    assert len(NYUDepthDataset(tmp_path, split="val", resolution=4)) == 1


# ── Other benchmark adapters ─────────────────────────────────────────────────

from open_vision_banana.data.datasets import (  # noqa: E402
    DIODEDataset,
    ETH3DDataset,
    IBimsDepthDataset,
    KITTIDataset,
)


def _rgb(path, hw=(48, 64)):
    Image.fromarray(np.zeros((*hw, 3), dtype=np.uint8)).save(path)


def test_kitti_native_gt(tmp_path):
    (tmp_path / "val" / "rgb").mkdir(parents=True)
    (tmp_path / "val" / "depth").mkdir(parents=True)
    _rgb(tmp_path / "val" / "rgb" / "a.png", (40, 120))
    depth = np.zeros((40, 120), dtype=np.float32)
    depth[10:20, 10:30] = 7.5  # sparse LiDAR: most pixels invalid
    np.save(tmp_path / "val" / "depth" / "a.npy", depth)

    item = KITTIDataset(tmp_path, split="val", resolution=16, native_gt=True)[0]
    assert item["gt_depth_native"].shape == (1, 40, 120)
    assert int(item["valid_mask_native"].sum()) == 10 * 20
    assert "gt_depth_native" not in KITTIDataset(tmp_path, split="val", resolution=16)[0]


def test_eth3d_native_gt_masks_inf_and_far(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "depths").mkdir()
    _rgb(tmp_path / "images" / "a.png")
    depth = np.full((48, 64), 3.0, dtype=np.float32)
    depth[0, :] = np.inf
    depth[1, :] = 500.0
    np.save(tmp_path / "depths" / "a.npy", depth)
    item = ETH3DDataset(tmp_path, split="val", resolution=16, native_gt=True)[0]
    assert item["gt_depth_native"].shape == (1, 48, 64)
    assert int(item["valid_mask_native"].sum()) == 46 * 64
    assert torch.isfinite(item["gt_depth_native"]).all()


def test_ibims_native_gt(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "depths").mkdir()
    _rgb(tmp_path / "images" / "a.png")
    np.save(tmp_path / "depths" / "a.npy", np.full((48, 64), 2.0, dtype=np.float32))
    item = IBimsDepthDataset(tmp_path, split="val", resolution=16, native_gt=True)[0]
    assert item["gt_depth_native"].shape == (1, 48, 64)
    assert item["valid_mask_native"].all()


def _make_diode(root, scene, name, depth_value=4.0, hw=(48, 64)):
    d = root / "val" / scene / "scene_00019" / "scan_00183"
    d.mkdir(parents=True, exist_ok=True)
    _rgb(d / f"{name}.png", hw)
    # official storage: depth [H, W, 1] float32, mask [H, W]
    np.save(d / f"{name}_depth.npy", np.full((*hw, 1), depth_value, dtype=np.float32))
    mask = np.ones(hw, dtype=np.float32)
    mask[:8] = 0
    np.save(d / f"{name}_depth_mask.npy", mask)


def test_diode_reads_official_layout_and_filters_scene_type(tmp_path):
    _make_diode(tmp_path, "indoors", "00019_00183_indoors_000_010")
    _make_diode(tmp_path, "outdoor", "00020_00001_outdoor_000_000", depth_value=30.0)

    everything = DIODEDataset(tmp_path, split="val", resolution=16)
    indoor = DIODEDataset(tmp_path, split="val", resolution=16, scene_type="indoors", native_gt=True)
    outdoor = DIODEDataset(tmp_path, split="val", resolution=16, scene_type="outdoor")
    assert len(everything) == 2 and len(indoor) == 1 and len(outdoor) == 1
    assert "outdoor" in str(outdoor.samples[0]["image"])

    item = indoor[0]
    assert item["gt_depth_native"].shape == (1, 48, 64)  # [H,W,1] handled
    assert int(item["valid_mask_native"].sum()) == 40 * 64  # mask applied at native size
    assert item["gt_depth"].shape == (1, 16, 16)


def test_diode_still_reads_legacy_rgb_naming(tmp_path):
    d = tmp_path / "val" / "indoor"
    d.mkdir(parents=True)
    _rgb(d / "s_rgb.png")
    np.save(d / "s_depth.npy", np.full((48, 64), 2.0, dtype=np.float32))
    assert len(DIODEDataset(tmp_path, split="val", resolution=16, scene_type="indoors")) == 1


def test_diode_rejects_unknown_scene_type(tmp_path):
    with pytest.raises(ValueError):
        DIODEDataset(tmp_path, split="val", scene_type="underwater")


def test_benchmark_script_wires_native_support_and_diode_indoor():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "evaluate_paper_benchmarks.py"
    spec = importlib.util.spec_from_file_location("evaluate_paper_benchmarks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert {"nyu_depth", "ibims", "diode", "kitti", "eth3d"} <= module.NATIVE_GT_DATASETS
    assert module.BENCHMARK_DATASET_KWARGS["diode"] == {"scene_type": "indoors"}
