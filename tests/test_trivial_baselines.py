"""Tests for trivial depth baselines under the benchmark protocol."""

import pytest
import torch

from open_vision_banana.evaluation.trivial_baselines import (
    evaluate_trivial_baselines,
    mean_depth_map,
)


def _batches(depths):
    out = []
    for d in depths:
        d = torch.as_tensor(d, dtype=torch.float32)
        out.append({"gt_depth": d.unsqueeze(0), "valid_mask": d > 0})
    return out


def test_constant_scene_is_solved_by_oracle_constant():
    batches = _batches([torch.full((1, 8, 8), 3.0)] * 2)
    res = evaluate_trivial_baselines(batches, dataset_name="other", constants=(1.0, 3.0))
    assert res["oracle_constant"]["delta_1.25"] == 1.0
    assert res["oracle_constant"]["constant_value_m"] == pytest.approx(3.0)
    assert res["constant@3m"]["delta_1.25"] == 1.0
    assert res["constant@1m"]["delta_1.25"] == 0.0


def test_oracle_image_median_beats_dataset_constant_on_mixed_scales():
    near = torch.full((1, 8, 8), 1.0)
    far = torch.full((1, 8, 8), 8.0)
    res = evaluate_trivial_baselines(_batches([near, far]), dataset_name="other", constants=())
    assert res["oracle_image_median"]["delta_1.25"] == 1.0
    assert res["oracle_constant"]["delta_1.25"] < 1.0


def test_invalid_pixels_are_ignored():
    d = torch.full((1, 4, 4), 2.0)
    d[..., :2] = 0.0
    res = evaluate_trivial_baselines(_batches([d]), dataset_name="other", constants=(2.0,))
    assert res["constant@2m"]["valid_pixels"] == 8
    assert res["constant@2m"]["delta_1.25"] == 1.0


def test_nyu_protocol_caps_gt_at_10m():
    d = torch.full((1, 8, 8), 2.0)
    d[..., 0] = 50.0  # beyond the 10 m NYU cap -> excluded
    res = evaluate_trivial_baselines(_batches([d]), dataset_name="nyu", constants=(2.0,))
    assert res["constant@2m"]["valid_pixels"] == 56


def test_prior_mean_map_recovers_layout():
    # Depth increases with row index; a prior built from the same layout
    # should score perfectly, a global constant should not.
    layout = torch.linspace(1.0, 5.0, 16).view(1, 16, 1).expand(1, 16, 16).contiguous()
    batches = _batches([layout, layout])
    prior = mean_depth_map(batches, (16, 16))
    res = evaluate_trivial_baselines(batches, dataset_name="other", constants=(), prior_map=prior)
    assert res["prior_mean_map"]["delta_1.25"] == 1.0
    assert res["oracle_constant"]["delta_1.25"] < 1.0


def test_mean_depth_map_requires_valid_pixels():
    with pytest.raises(ValueError):
        mean_depth_map(_batches([torch.zeros(1, 4, 4)]), (4, 4))


def test_script_native_resolution_end_to_end(tmp_path):
    """depth_trivial_baselines.py --native-resolution on a tiny NYU-flat layout."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    import numpy as np
    from PIL import Image

    (tmp_path / "images").mkdir()
    (tmp_path / "depths").mkdir()
    for i, hw in enumerate([(48, 64), (60, 80)]):  # different native sizes
        Image.fromarray(np.zeros((*hw, 3), dtype=np.uint8)).save(tmp_path / "images" / f"{i}.png")
        np.save(tmp_path / "depths" / f"{i}.npy", np.full(hw, 3.0, dtype=np.float32))
    out = tmp_path / "out.json"
    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "depth_trivial_baselines.py"),
            "--dataset-key",
            "nyu_depth",
            "--dataset-name",
            "other",
            "--root",
            str(tmp_path),
            "--split",
            "val",
            "--resolution",
            "16",
            "--native-resolution",
            "--constants",
            "3",
            "--output-json",
            str(out),
        ],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(root / "src")},
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text())
    assert payload["native_resolution"] is True and payload["samples"] == 2
    assert payload["baselines"]["constant@3m"]["valid_pixels"] == 48 * 64 + 60 * 80
    assert payload["baselines"]["oracle_constant"]["delta_1.25"] == 1.0
