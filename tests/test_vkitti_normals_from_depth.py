"""VKITTI2: normals derived from centimetre depth + intrinsics, in the paper frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from open_vision_banana.data.datasets import VirtualKITTIDataset, VirtualKITTINormalDataset
from open_vision_banana.data.depth_geometry import (
    VKITTI2_DEFAULT_INTRINSICS,
    depth_to_normals,
    load_vkitti_depth_png,
    read_vkitti_intrinsics,
)

FX, FY, CX, CY = VKITTI2_DEFAULT_INTRINSICS
H, W = 96, 128


def _ground_plane_depth(
    height_m: float = 1.5, cy: float = CY * H / 375, fy: float = FY * H / 375
) -> np.ndarray:
    v = np.arange(H, dtype=float)[:, None]
    z = np.where(v > cy + 2, fy * height_m / np.maximum(v - cy, 1e-6), np.nan)
    return np.broadcast_to(z, (H, W)).copy()


def test_analytic_planes_in_paper_frame():
    fx, fy, cx, cy = FX * W / 1242, FY * H / 375, CX * W / 1242, CY * H / 375
    wall, ok = depth_to_normals(np.full((H, W), 8.0), fx, fy, cx, cy)
    assert ok.all() and np.allclose(wall[H // 2, W // 2], (0, 0, 1), atol=1e-6)
    ground, ok = depth_to_normals(_ground_plane_depth(), fx, fy, cx, cy)
    assert np.allclose(ground[H - 10, W // 2], (0, 1, 0), atol=1e-4)  # floor faces up (+y up)
    assert not ok[: int(cy)].any()  # sky rows invalid
    u = np.arange(W, dtype=float)[None, :]
    right = np.broadcast_to(
        np.where(u > cx + 2, fx * 3.0 / np.maximum(u - cx, 1e-6), np.nan), (H, W)
    )
    normal, _ = depth_to_normals(right, fx, fy, cx, cy)
    assert np.allclose(
        normal[H // 2, W - 10], (-1, 0, 0), atol=1e-4
    )  # wall on the right faces left
    with pytest.raises(ValueError):
        depth_to_normals(np.zeros((2, H, W)), fx, fy, cx, cy)


def _vkitti_scene(root: Path) -> None:
    frames = root / "Scene01" / "clone" / "frames"
    (frames / "rgb" / "Camera_0").mkdir(parents=True)
    (frames / "depth" / "Camera_0").mkdir(parents=True)
    for frame in range(2):
        Image.new("RGB", (W, H), (120, 120, 120)).save(
            frames / "rgb" / "Camera_0" / f"rgb_{frame:05d}.jpg"
        )
        depth_cm = np.full((H, W), 65535, np.uint16)  # sky by default
        depth_cm[H // 2 :, :] = 800  # 8.00 m fronto-parallel wall in centimetres
        Image.fromarray(depth_cm, "I;16").save(
            frames / "depth" / "Camera_0" / f"depth_{frame:05d}.png"
        )
    fx, fy, cx, cy = FX * W / 1242, FY * H / 375, CX * W / 1242, CY * H / 375
    (root / "Scene01" / "clone" / "intrinsic.txt").write_text(
        "frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]\n"
        + "".join(f"{f} 0 {fx} {fy} {cx} {cy}\n" for f in range(2))
    )


def test_vkitti_layout_derives_normals_and_reads_intrinsics(tmp_path: Path):
    _vkitti_scene(tmp_path)
    table = read_vkitti_intrinsics(tmp_path / "Scene01" / "clone" / "intrinsic.txt")
    assert set(table) == {(0, 0), (1, 0)} and table[(0, 0)][0] == pytest.approx(FX * W / 1242)
    metres, valid = load_vkitti_depth_png(
        tmp_path / "Scene01" / "clone" / "frames" / "depth" / "Camera_0" / "depth_00000.png"
    )
    assert metres[H - 1, 0] == pytest.approx(8.0) and not valid[0, 0] and valid[H - 1, 0]

    ds = VirtualKITTINormalDataset(tmp_path, split="test", resolution=32)
    assert len(ds) == 2 and ds.samples[0]["frame"] == 0 and ds.samples[0]["camera"] == 0
    sample = ds[0]
    assert sample["image"].shape == (3, 32, 32)
    normal, valid = sample["gt_normal"], sample["valid_mask"]
    assert normal.shape[0] == 3
    centre = normal[:, -4, normal.shape[-1] // 2]
    assert torch.allclose(centre, torch.tensor([0.0, 0.0, 1.0]), atol=1e-3)  # wall faces the camera
    assert not bool(valid[0, 0]) and bool(
        valid[-4, valid.shape[-1] // 2]
    )  # sky invalid, wall valid

    # the depth adapter uses centimetres and masks the sky
    depth_ds = VirtualKITTIDataset(tmp_path, split="test", resolution=32)
    from open_vision_banana.core.depth_mapper import DepthRGBMapper

    target = depth_ds[0]["target_rgb"]
    decoded = DepthRGBMapper().rgb_to_depth(target.unsqueeze(0))[0, 0]
    assert abs(float(decoded[-2, 16]) - 8.0) < 0.1  # 800 cm -> 8 m, not 0.8 m


def test_precomputed_normals_take_precedence(tmp_path: Path):
    _vkitti_scene(tmp_path)
    normals_dir = tmp_path / "Scene01" / "clone" / "frames" / "normal" / "Camera_0"
    normals_dir.mkdir(parents=True)
    pre = np.zeros((H, W, 3), np.float32)
    pre[..., 1] = 1.0
    np.save(normals_dir / "rgb_00000.npy", pre)
    ds = VirtualKITTINormalDataset(tmp_path, split="test", resolution=32)
    assert (
        ds.samples[0]["target"] is not None and ds.samples[1].get("target") is None
    )  # one precomputed, one derived
    assert torch.allclose(ds[0]["gt_normal"][:, 5, 5], torch.tensor([0.0, 1.0, 0.0]), atol=1e-3)
