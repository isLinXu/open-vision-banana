"""Inference protocol pins: guidance_scale threading, semantic native-resolution scoring, metadata."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from open_vision_banana.data.datasets import CityscapesSegDataset
from open_vision_banana.evaluation.evaluator import VisionEvaluator

ROOT = Path(__file__).resolve().parents[1]


class _RecordingModel:
    """predict() echoes an RGB from the GT and records the kwargs it received."""

    def __init__(self, rgb: torch.Tensor) -> None:
        self.rgb = rgb
        self.calls: list[dict] = []

    def eval(self):
        return self

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return {"rgb_output": self.rgb}


def test_guidance_scale_reaches_predict_only_when_set():
    model = _RecordingModel(torch.rand(1, 3, 8, 8))
    ev = VisionEvaluator(model, device="cpu")
    batch = {"image": torch.rand(1, 3, 8, 8), "target_rgb": torch.rand(1, 3, 8, 8)}
    ev.evaluate([batch], task="depth", num_inference_steps=20, guidance_scale=4.0, seed=7)
    assert model.calls[0]["num_inference_steps"] == 20 and model.calls[0]["guidance_scale"] == 4.0 and model.calls[0]["seed"] == 7
    model.calls.clear()
    ev.evaluate([batch], task="depth")  # default 0.0 -> not forwarded (third-party predict APIs)
    assert "guidance_scale" not in model.calls[0]


def _cityscapes_fixture(root: Path) -> None:
    img_dir = root / "leftImg8bit" / "val" / "city"
    gt_dir = root / "gtFine" / "val" / "city"
    img_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    Image.new("RGB", (128, 64), (90, 90, 90)).save(img_dir / "city_000000_000019_leftImg8bit.png")
    labels = np.full((64, 128), 7, np.uint8)  # road (labelId 7 -> trainId 0)
    labels[:, 64:] = 23  # sky (labelId 23 -> trainId 10)
    Image.fromarray(labels).save(gt_dir / "city_000000_000019_gtFine_labelIds.png")


def test_semantic_native_resolution_scores_at_gt_size(tmp_path: Path):
    _cityscapes_fixture(tmp_path)
    ds = CityscapesSegDataset(tmp_path, split="val", resolution=32, native_gt=True)
    assert len(ds) == 1
    sample = ds[0]
    assert sample["gt_labels"].shape == (32, 32) and sample["gt_labels_native"].shape == (64, 128)
    assert sample["gt_labels_native"][0, 0] == 0 and sample["gt_labels_native"][0, 127] == 10

    # a perfect 32x32 prediction scored at native 64x128 -> mIoU 1 for the two present classes
    pred_rgb = sample["target_rgb"].unsqueeze(0)
    model = _RecordingModel(pred_rgb)
    ev = VisionEvaluator(model, device="cpu")
    batch = {
        "image": sample["image"].unsqueeze(0), "target_rgb": pred_rgb,
        "gt_labels": sample["gt_labels"].unsqueeze(0), "gt_labels_native": sample["gt_labels_native"].unsqueeze(0),
    }
    metrics = ev.evaluate([batch], task="semantic", num_classes=19, class_color_map=CityscapesSegDataset.CITYSCAPES_COLORS, native_resolution=True)
    assert metrics["mIoU"] > 0.99, metrics
    # without the native GT the flag must fail loudly, not silently fall back to 32x32
    ds_plain = CityscapesSegDataset(tmp_path, split="val", resolution=32)
    assert "gt_labels_native" not in ds_plain[0]
    try:
        ev.evaluate([{k: v for k, v in batch.items() if k != "gt_labels_native"}], task="semantic", num_classes=19,
                    class_color_map=CityscapesSegDataset.CITYSCAPES_COLORS, native_resolution=True)
    except ValueError as exc:
        assert "gt_labels_native" in str(exc)
    else:
        raise AssertionError("native_resolution without native GT should raise")


def test_paper_benchmark_script_exposes_inference_protocol_flags():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate_paper_benchmarks.py"), "--help"],
                         capture_output=True, text=True, cwd=ROOT, env={"PYTHONPATH": "src"})
    assert out.returncode == 0, out.stderr
    for flag in ("--num-inference-steps", "--guidance-scale", "--strength", "--seed", "--native-resolution"):
        assert flag in out.stdout
    src = (ROOT / "scripts" / "evaluate_paper_benchmarks.py").read_text()
    assert '"inference_protocol"' in src and "num_inference_steps=1 is a smoke setting" in src
    assert json.loads('{"ok": true}')["ok"]  # keep json import honest
