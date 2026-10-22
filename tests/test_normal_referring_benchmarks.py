"""Surface-normal adapters, trivial-normal gates, and RefCOCOg cIoU wiring."""

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from open_vision_banana.core.normal_mapper import NormalRGBMapper
from open_vision_banana.data.benchmark_io import load_normal_map
from open_vision_banana.data.datasets import (
    DIODENormalDataset,
    NYUNormalDataset,
    RefCOCOgDataset,
    ScanNetNormalDataset,
    VirtualKITTINormalDataset,
)
from open_vision_banana.evaluation.benchmark import NormalAccumulator
from open_vision_banana.evaluation.evaluator import VisionEvaluator
from open_vision_banana.evaluation.gates import evaluate_normal_gate
from open_vision_banana.evaluation.normal_baselines import evaluate_normal_baselines

ROOT = Path(__file__).resolve().parent.parent


def _rgb(path: Path, size=(8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size[1], size[0]), color=(20, 40, 60)).save(path)


def test_load_normal_map_treats_zero_vectors_as_invalid(tmp_path):
    array = np.zeros((4, 6, 3), dtype=np.float32)
    array[:, :3] = (0.0, 0.0, 1.0)
    path = tmp_path / "n.npy"
    np.save(path, array)
    normal, valid = load_normal_map(path)
    assert normal.shape == (3, 4, 6)
    assert int(valid.sum()) == 4 * 3
    assert torch.allclose(normal[:, :, :3], torch.tensor([0.0, 0.0, 1.0]).view(3, 1, 1), atol=1e-5)
    assert normal[:, :, 3:].abs().sum() == 0


def test_nyu_normal_native_keeps_unresized_map_and_refuses_h5_only(tmp_path):
    images = tmp_path / "images"
    normals = tmp_path / "normals"
    images.mkdir()
    normals.mkdir()
    _rgb(images / "a.png", (6, 8))
    array = np.zeros((6, 8, 3), dtype=np.float32)
    array[:, :4] = (0.0, 0.0, 1.0)
    np.save(normals / "a.npy", array)

    item = NYUNormalDataset(tmp_path, split="val", resolution=4, native_gt=True)[0]
    assert item["gt_normal_native"].shape == (3, 6, 8)
    assert int(item["valid_mask_native"].sum()) == 6 * 4
    assert item["gt_normal"].shape == (3, 4, 4)
    assert int(item["valid_mask"].sum()) < 16

    h5_root = tmp_path / "h5_only"
    (h5_root / "nyu_depth_v2_labeled.mat").parent.mkdir()
    assert len(NYUNormalDataset(h5_root, split="val")) == 0


def test_diode_normal_indoor_mask_and_native(tmp_path):
    indoor = tmp_path / "val" / "indoors" / "scene"
    outdoor = tmp_path / "val" / "outdoor" / "scene"
    _rgb(indoor / "00019_00183_indoors_000_010.png", (4, 6))
    _rgb(outdoor / "00019_00183_outdoor_000_010.png", (4, 6))
    normal = np.zeros((4, 6, 3), dtype=np.float32)
    normal[..., 2] = 1.0
    np.save(indoor / "00019_00183_indoors_000_010_normal.npy", normal)
    np.save(outdoor / "00019_00183_outdoor_000_010_normal.npy", normal)
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[:, :2] = 1
    np.save(indoor / "00019_00183_indoors_000_010_depth_mask.npy", mask)

    ds = DIODENormalDataset(tmp_path, split="val", resolution=4, scene_type="indoors", native_gt=True)
    assert len(ds) == 1
    item = ds[0]
    assert item["gt_normal_native"].shape == (3, 4, 6)
    assert int(item["valid_mask_native"].sum()) == 4 * 2
    with pytest.raises(ValueError):
        DIODENormalDataset(tmp_path, split="val", scene_type="underwater")


def test_scannet_prefers_image_normal_dirs_over_color_heuristic(tmp_path):
    _rgb(tmp_path / "images" / "frame.png", (8, 8))
    (tmp_path / "normals").mkdir()
    np.save(tmp_path / "normals" / "frame.npy", np.zeros((8, 8, 3), dtype=np.float32) + (0.0, 0.0, 1.0))
    _rgb(tmp_path / "scene_color.png", (8, 8))
    _rgb(tmp_path / "scene_normal.png", (8, 8))
    ds = ScanNetNormalDataset(tmp_path, split="train", resolution=4, native_gt=True)
    assert len(ds) == 1
    assert ds.samples[0]["image"].name == "frame.png"
    assert ds[0]["gt_normal_native"].shape == (3, 8, 8)


def test_scannet_color_path_still_pairs(tmp_path):
    _rgb(tmp_path / "scene_0_color.png")
    _rgb(tmp_path / "scene_0_normal.png")
    assert len(ScanNetNormalDataset(tmp_path, resolution=8)) == 1


def test_vkitti_prefers_sibling_normal_over_depth_derivation(tmp_path):
    rgb = tmp_path / "Scene01" / "clone" / "frames" / "rgb" / "Camera_0"
    normal = tmp_path / "Scene01" / "clone" / "frames" / "normal" / "Camera_0"
    depth = tmp_path / "Scene01" / "clone" / "frames" / "depth" / "Camera_0"
    _rgb(rgb / "00000.jpg", (5, 7))
    _rgb(depth / "00000.png", (5, 7))
    # VKITTI2 ships no normals: with only depth present the dataset derives them (see
    # tests/test_vkitti_normals_from_depth.py); a sibling normal/ map takes precedence.
    derived = VirtualKITTINormalDataset(tmp_path, split="val", resolution=4)
    assert len(derived) == 1 and derived.samples[0]["target"] is None
    normal.mkdir(parents=True)
    array = np.zeros((5, 7, 3), dtype=np.float32)
    array[..., 2] = 1
    np.save(normal / "00000.npy", array)
    ds = VirtualKITTINormalDataset(tmp_path, split="val", resolution=4, native_gt=True)
    assert len(ds) == 1
    assert ds[0]["gt_normal_native"].shape == (3, 5, 7)


def test_normal_per_image_mean_differs_from_pixel_weighted():
    acc = NormalAccumulator()
    z = torch.tensor([0.0, 0.0, 1.0]).view(1, 3, 1, 1)
    y = torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1, 1)
    acc.update(z.expand(1, 3, 10, 10), z.expand(1, 3, 10, 10))
    acc.update(y, z)
    metrics = acc.compute()
    assert metrics["images"] == 2
    assert metrics["mean_angle_deg"] == pytest.approx(90.0 / 101)
    assert metrics["mean_angle_deg_per_image"] == pytest.approx(45.0)


def test_trivial_normals_and_gate():
    gt = torch.zeros(1, 3, 2, 2)
    gt[:, 2] = 1.0
    baselines = evaluate_normal_baselines([{"gt_normal": gt, "valid_mask": torch.ones(1, 2, 2, dtype=torch.bool)}])
    assert baselines["facing_camera"]["Mean Angular Error (°)"] == pytest.approx(0.0, abs=1e-4)
    assert baselines["axis_up"]["Mean Angular Error (°)"] == pytest.approx(90.0, abs=1e-3)
    assert baselines["oracle_image_mean"]["Mean Angular Error (°)"] == pytest.approx(0.0, abs=1e-4)

    numbers = {
        "facing_camera": {"Mean Angular Error (°)": 30.0},
        "axis_up": {"Mean Angular Error (°)": 40.0},
        "oracle_image_mean": {"Mean Angular Error (°)": 12.0},
    }
    passed = evaluate_normal_gate({"Mean Angular Error (°)": 8.0}, numbers, margin_deg=5.0)
    assert passed.metric_gate and passed.structure_gate and passed.passed
    short = evaluate_normal_gate({"Mean Angular Error (°)": 20.0}, numbers, margin_deg=5.0)
    assert short.metric_gate and not short.structure_gate
    weak = evaluate_normal_gate({"Mean Angular Error (°)": 27.0}, numbers, margin_deg=5.0)
    assert not weak.metric_gate


class _FacingCamera:
    def __init__(self):
        normal = torch.tensor([0.0, 0.0, 1.0]).view(1, 3, 1, 1).expand(1, 3, 8, 8)
        self._rgb = NormalRGBMapper().normal_to_rgb(normal)

    def eval(self):
        return self

    def __call__(self, image):
        return self._rgb.expand(image.shape[0], -1, -1, -1).clone()


def test_evaluator_scores_normals_at_native_resolution():
    ev = VisionEvaluator(_FacingCamera(), device="cpu")
    native = torch.zeros(1, 3, 6, 10)
    native[:, 2] = 1.0
    batch = {
        "image": torch.zeros(1, 3, 8, 8),
        "target_rgb": torch.zeros(1, 3, 8, 8),
        "gt_normal": torch.zeros(1, 3, 8, 8),
        "valid_mask": torch.ones(1, 8, 8, dtype=torch.bool),
        "gt_normal_native": native,
        "valid_mask_native": torch.ones(1, 6, 10, dtype=torch.bool),
    }
    metrics = ev.evaluate([batch], task="normal", native_resolution=True)
    assert metrics["valid_pixels"] == 60
    assert metrics["Mean Angular Error (°)"] == pytest.approx(0.0, abs=1e-3)
    del batch["gt_normal_native"]
    with pytest.raises(ValueError, match="gt_normal_native"):
        ev.evaluate([batch], task="normal", native_resolution=True)


class _WhiteMask:
    def eval(self):
        return self

    def __call__(self, image):
        return torch.ones(image.shape[0], 3, 8, 8)


def test_evaluator_referring_native_ciou():
    ev = VisionEvaluator(_WhiteMask(), device="cpu")
    gt = torch.zeros(1, 16, 16, dtype=torch.bool)
    gt[:, :8, :8] = True
    batch = {
        "image": torch.zeros(1, 3, 8, 8),
        "target_rgb": torch.zeros(1, 3, 8, 8),
        "gt_mask_native": gt,
    }
    metrics = ev.evaluate([batch], task="referring", expression="the box", native_resolution=True)
    assert metrics["cIoU"] == pytest.approx(0.25)
    assert metrics["samples"] == 1


def test_refcocog_umd_split_all_sentences_and_native_mask(tmp_path):
    images = tmp_path / "images" / "val2014"
    images.mkdir(parents=True)
    Image.new("RGB", (20, 12), color=(1, 2, 3)).save(images / "COCO_val2014_000000000123.jpg")
    refs = [
        {
            "image_id": 123,
            "ann_id": 7,
            "split": "val",
            "file_name": "COCO_val2014_000000000123.jpg",
            "sentences": [{"sent": "the left chair"}, {"raw": "a chair on the left"}],
        },
        {"image_id": 123, "ann_id": 7, "split": "train", "sentences": ["not this split"]},
        {"image_id": 123, "ann_id": 8, "split": "val", "sentences": ["bbox only"]},
    ]
    with open(tmp_path / "refs(umd).p", "wb") as handle:
        pickle.dump(refs, handle)
    instances = {
        "images": [{"id": 123, "height": 12, "width": 20}],
        "annotations": [
            {"id": 7, "image_id": 123, "segmentation": [[1, 1, 8, 1, 8, 6, 1, 6]], "bbox": [1, 1, 7, 5]},
            {"id": 8, "image_id": 123, "bbox": [0, 0, 4, 4]},
        ],
    }
    (tmp_path / "instances.json").write_text(json.dumps(instances))
    ds = RefCOCOgDataset(tmp_path, split="val", resolution=8, native_gt=True, protocol="umd")
    assert len(ds) == 2
    assert {ds[i]["expression"] for i in range(2)} == {"the left chair", "a chair on the left"}
    assert ds[0]["gt_mask_native"].shape == (12, 20)
    assert int(ds[0]["gt_mask_native"].sum()) > 0
    assert ds[0]["target_rgb"].shape == (3, 8, 8)


def test_benchmark_script_wires_normal_and_referring():
    import importlib.util

    path = ROOT / "scripts" / "evaluate_paper_benchmarks.py"
    spec = importlib.util.spec_from_file_location("evaluate_paper_benchmarks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert {"nyu_normal", "diode_normal", "normal", "vkitti_normal", "referring"} <= module.NATIVE_GT_DATASETS
    assert module.BENCHMARK_DATASET_KWARGS["diode_normal"] == {"scene_type": "indoors"}
    assert module.BENCHMARK_DATASET_KWARGS["referring"] == {"protocol": "umd", "strict": True}
    parser_help = subprocess.run(
        [sys.executable, str(path), "--help"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "referring" in parser_help
    assert "--refcocog-root" in parser_help


def test_normal_gate_cli(tmp_path):
    model = tmp_path / "model.json"
    base = tmp_path / "base.json"
    model.write_text(
        json.dumps({"benchmarks": {"normal/nyu": {"decoded_metrics": {"Mean Angular Error (°)": 8.0}}}})
    )
    base.write_text(
        json.dumps(
            {
                "baselines": {
                    "facing_camera": {"Mean Angular Error (°)": 30.0},
                    "oracle_image_mean": {"Mean Angular Error (°)": 12.0},
                }
            }
        )
    )
    passed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_normal_gate.py"),
            "--model-json",
            str(model),
            "--baseline-json",
            str(base),
            "--benchmark",
            "normal/nyu",
        ],
        capture_output=True,
        text=True,
    )
    assert passed.returncode == 0, passed.stderr
    model.write_text(
        json.dumps({"benchmarks": {"normal/nyu": {"decoded_metrics": {"Mean Angular Error (°)": 20.0}}}})
    )
    failed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_normal_gate.py"),
            "--model-json",
            str(model),
            "--baseline-json",
            str(base),
        ],
        capture_output=True,
        text=True,
    )
    assert failed.returncode == 1
