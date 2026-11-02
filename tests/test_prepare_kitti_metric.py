"""Tests for KITTI metric-depth preparation."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.prepare_kitti_metric import (
    KITTI_VAL_DRIVES,
    find_kitti_metric_pairs,
    load_kitti_depth_meters,
    prepare_kitti_metric,
)


def _write_kitti_pair(tmp: Path, drive: str, frame: str, *, split: str = "train") -> None:
    depth_dir = tmp / "depth" / split / drive / "proj_depth" / "groundtruth" / "image_02"
    rgb_dir = tmp / "rgb" / "2011_09_26" / drive / "image_02" / "data"
    if drive.startswith("2011_09_28"):
        rgb_dir = tmp / "rgb" / "2011_09_28" / drive / "image_02" / "data"
    depth_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 16), color=(10, 20, 30)).save(rgb_dir / frame)
    depth = np.zeros((16, 32), dtype=np.uint16)
    depth[:, :16] = 256 * 12  # 12 metres
    Image.fromarray(depth).save(depth_dir / frame)


def test_finds_train_pairs_and_skips_val_drives(tmp_path: Path) -> None:
    train_drive = "2011_09_26_drive_0001_sync"
    val_drive = next(iter(KITTI_VAL_DRIVES))
    _write_kitti_pair(tmp_path, train_drive, "0000000005.png")
    _write_kitti_pair(tmp_path, val_drive, "0000000005.png", split="val")
    pairs = find_kitti_metric_pairs(tmp_path / "depth", tmp_path / "rgb", split="train")
    assert len(pairs) == 1
    assert pairs[0][2] == train_drive


def test_prepare_encodes_metric_metres(tmp_path: Path) -> None:
    drive = "2011_09_26_drive_0001_sync"
    _write_kitti_pair(tmp_path, drive, "0000000005.png")
    out = tmp_path / "prepared"
    count = prepare_kitti_metric(tmp_path / "depth", tmp_path / "rgb", out, resolution=8)
    assert count == 1
    metric = np.load(next((out / "metric_depth").glob("*.npy")))
    assert metric.shape == (8, 8)
    assert np.nanmax(metric) == 12.0
    row = json.loads((out / "pairs.jsonl").read_text().splitlines()[0])
    assert row["task"] == "metric_depth"
    assert row["prompt"] == "Generate a metric depth visualization of this image."
    loaded = load_kitti_depth_meters(
        tmp_path
        / "depth"
        / "train"
        / drive
        / "proj_depth"
        / "groundtruth"
        / "image_02"
        / "0000000005.png"
    )
    assert float(loaded.max()) == 12.0
