"""Issue #139 recipe v2: depth_rel targets, random semantic legends, masked flow loss."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.core.relative_depth_mapper import DEPTH_REL_PROMPT, RelativeDepthMapper
from open_vision_banana.data.openvocab_semantic import parse_prompt_mapping
from open_vision_banana.data.validation import validate_manifest_contract
from open_vision_banana.training.config import build_arg_parser, load_config
from open_vision_banana.training.losses import TrainingLosses, latent_loss_mask
from open_vision_banana.training.trainer import ManifestVisionDataset, Trainer, _collate_fn

EXEMPT = {"depth_admission": "legacy_baseline_exempt"}


def _write_depth(tmp_path, holes=True):
    h, w = 16, 24
    depth = torch.linspace(0.5, 8.0, w).repeat(h, 1)
    if holes:
        depth[:2, :4] = 0.0
    rgb = (DepthRGBMapper("cube_edge").depth_to_rgb(depth[None, None])[0] * 255).round().to(torch.uint8)
    Image.fromarray(rgb.permute(1, 2, 0).numpy()).save(tmp_path / "depth.png")
    Image.fromarray(np.full((h, w, 3), 90, np.uint8)).save(tmp_path / "rgb.png")
    mask = np.full((h, w), 255, np.uint8)
    mask[10, 10] = 0
    Image.fromarray(mask).save(tmp_path / "mask.png")
    return depth


def _semantic_rows(tmp_path):
    target = np.zeros((16, 16, 3), np.uint8)
    target[:] = (40, 40, 40)
    target[:8] = (255, 0, 0)
    target[8:, :8] = (0, 255, 0)
    Image.fromarray(target).save(tmp_path / "seg.png")
    mapping = {"wall": [255, 0, 0], "floor": [0, 255, 0], "background": [40, 40, 40]}
    other = {"sofa": [1, 2, 3], "background": [40, 40, 40]}
    base = {"task": "semantic", "input": "rgb.png", "target": "seg.png", "source": "ade20k",
            "prompt": "legacy"}
    return [dict(base, split="train", mapping=mapping), dict(base, split="val", mapping=mapping),
            dict(base, split="train", mapping=other)]


def _manifest(tmp_path, rows):
    path = tmp_path / "pairs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_depth_rows_become_depth_rel_with_mask(tmp_path):
    depth = _write_depth(tmp_path)
    rows = [
        {"task": "depth", "split": "train", "input": "rgb.png", "target": "depth.png",
         "valid_mask": "mask.png", "prompt": "metric", **EXEMPT},
        {"task": "metric_depth", "split": "val", "input": "rgb.png", "target": "depth.png"},
    ]
    manifest = _manifest(tmp_path, rows)
    train = ManifestVisionDataset(manifest, 16, "train", {"depth_rel"}, aspect_preserving_crop=False)
    assert [s["task"] for s in train.samples] == ["depth_rel"]
    assert train.samples[0]["prompt"] == DEPTH_REL_PROMPT
    item = train[0]
    mask = item["loss_mask"][0] > 0.5
    rgb = item["target_rgb"]
    assert item["image"].shape == rgb.shape == (3, 16, 16)
    assert not mask.all() and mask.float().mean() > 0.7
    mapper = RelativeDepthMapper()
    stored = (rgb * 255).round().to(torch.uint8)
    assert torch.equal(mapper.sentinel_mask(stored), ~mask)
    grey = mapper.decode(rgb)[mask]
    assert grey.max() == pytest.approx(1.0) and grey.min() == pytest.approx(0.0)
    # Near (left, small depth) is brighter than far (right).
    assert mapper.decode(rgb)[8, 1] > mapper.decode(rgb)[8, 14]
    assert ManifestVisionDataset(manifest, 16, "train", {"depth"}).samples[0]["task"] == "depth"
    val = ManifestVisionDataset(manifest, 16, "val", {"depth_rel"}, aspect_preserving_crop=True)
    assert "valid_mask_path" not in val.samples[0]
    assert val[0]["loss_mask"].shape == (1, 16, 16)
    batch = _collate_fn([item, {"target_rgb": torch.zeros(3, 16, 16), "image": torch.zeros(3, 16, 16),
                                "prompt": "x", "task": "semantic"}])
    assert batch["loss_mask"].shape == (2, 1, 16, 16) and batch["loss_mask"][1].all()
    del depth


def test_aspect_crop_keeps_mask_aligned(tmp_path):
    _write_depth(tmp_path, holes=False)
    rows = [{"task": "depth", "split": "train", "input": "rgb.png", "target": "depth.png", **EXEMPT}]
    ds = ManifestVisionDataset(_manifest(tmp_path, rows), 8, "train", {"depth_rel"},
                               aspect_preserving_crop=True)
    item = ds[0]
    assert item["target_rgb"].shape == (3, 8, 8) and item["loss_mask"].all()


def test_random_v2_semantic_legend(tmp_path):
    Image.fromarray(np.full((16, 16, 3), 90, np.uint8)).save(tmp_path / "rgb.png")
    manifest = _manifest(tmp_path, _semantic_rows(tmp_path))
    ds = ManifestVisionDataset(manifest, 16, "train", {"semantic"}, semantic_legend_mode="random_v2",
                               seed=3)
    assert ds.semantic_vocabulary == {"ade20k": ("floor", "sofa", "wall")}
    seen_modes = set()
    for _ in range(30):
        item = ds[0]
        legend = parse_prompt_mapping(item["prompt"])
        assert legend and "background" not in legend
        target = (item["target_rgb"] * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
        assert (target[8:, 8:] == 0).all()
        if "sofa" in legend:
            seen_modes.add("negatives")
        for name, region in (("wall", target[:8]), ("floor", target[8:, :8])):
            expected = legend.get(name, (0, 0, 0))
            assert (region == expected).all()
    assert "negatives" in seen_modes
    # The absent-only row falls back to negatives only when nothing is visible.
    assert (ds[1]["target_rgb"] >= 0).all()
    val = ManifestVisionDataset(manifest, 16, "val", {"semantic"}, semantic_legend_mode="random_v2")
    assert val[0]["prompt"] == "legacy"
    no_mapping = ManifestVisionDataset(manifest, 16, "train", {"semantic"}, semantic_legend_mode="random_v2")
    no_mapping.samples[0].pop("mapping")
    assert no_mapping._random_legend(torch.zeros(3, 4, 4), "plain prompt", no_mapping.samples[0])[1] == "plain prompt"
    empty = torch.full((3, 4, 4), 40 / 255)
    assert ds._random_legend(empty, "kept", {"mapping": {"background": (40, 40, 40)}})[1] == "kept"


def test_latent_mask_and_masked_flow_loss():
    pixel = torch.ones(2, 1, 16, 16)
    pixel[0, :, :8] = 0
    cells = latent_loss_mask(pixel, torch.Size([2, 4, 2, 2]))
    assert cells.shape == (2, 1, 2, 2)
    assert cells[0].flatten().tolist() == [0, 0, 1, 1] and cells[1].all()
    assert latent_loss_mask(pixel, torch.Size([2, 4, 1, 2, 2])).shape == (2, 1, 1, 2, 2)
    with pytest.raises(ValueError):
        latent_loss_mask(pixel[:, 0], torch.Size([2, 4, 2, 2]))
    with pytest.raises(ValueError):
        latent_loss_mask(pixel, torch.Size([3, 4, 2, 2]))
    losses = TrainingLosses(min_snr=False)
    pred = torch.zeros(2, 4, 2, 2)
    target = torch.ones(2, 4, 2, 2)
    target[0, :, 0] = 5.0  # only inside the masked half
    t = torch.full((2,), 0.5)
    _, plain, _ = losses.compute_flow_loss(pred, target, t, ["depth_rel"] * 2, "cpu")
    _, masked, _ = losses.compute_flow_loss(pred, target, t, ["depth_rel"] * 2, "cpu", pixel_mask=pixel)
    assert plain[0] > 1.0 and masked[0] == pytest.approx(1.0)
    assert masked[1] == pytest.approx(plain[1])


def test_trainer_pixel_mask_respects_flag():
    batch = {"loss_mask": torch.ones(1, 1, 4, 4)}
    on = SimpleNamespace(config=SimpleNamespace(geometry_loss_mask=True), device="cpu")
    off = SimpleNamespace(config=SimpleNamespace(geometry_loss_mask=False), device="cpu")
    assert Trainer._pixel_loss_mask(on, batch).shape == (1, 1, 4, 4)
    assert Trainer._pixel_loss_mask(off, batch) is None
    assert Trainer._pixel_loss_mask(on, {}) is None


def test_contract_audit_counts_invalid_pixels(tmp_path):
    _write_depth(tmp_path)
    rows = [{"task": "depth", "split": "train", "input": "rgb.png", "target": "depth.png",
             "valid_mask": "mask.png", "source_dataset": "arkitscenes"},
            {"task": "depth", "split": "train", "input": "rgb.png", "target": "depth.png"},
            {"task": "depth", "split": "train", "input": "rgb.png", "target": "missing.png",
             "source_dataset": "gone"}]
    report = validate_manifest_contract(_manifest(tmp_path, rows), model_name="qwen-image-2.1",
                                        relative_depth=True)
    audit = report.stats["depth_rel_audit"]
    assert audit["arkitscenes"]["invalid_fraction_mean"] == pytest.approx(9 / (16 * 24))
    assert audit["arkitscenes"]["sentinel_mismatch_pixels"] == 0
    assert audit["arkitscenes"]["roundtrip_max_abs"] <= 0.5 / 255 + 1e-6
    assert audit["unknown"]["invalid_fraction_mean"] == pytest.approx(8 / (16 * 24))
    assert any("missing file" in issue.message for issue in report.errors)
    plain = validate_manifest_contract(_manifest(tmp_path, rows[:2]), model_name="qwen-image-2.1")
    assert "depth_rel_audit" not in plain.stats


def test_recipe_v2_config_loads(tmp_path):
    args = build_arg_parser().parse_args(["--config", "configs/experiments/i139_recipe_v2_r64.yaml"])
    config = load_config(args)
    assert config.tasks[0] == "depth_rel"
    assert config.semantic_legend_mode == "random_v2" and config.geometry_loss_mask
    assert config.task_sampling_weights["depth_rel"] == pytest.approx(0.35)
    assert config.lr == pytest.approx(1e-5)
    bad = build_arg_parser().parse_args(["--config", "configs/experiments/i139_recipe_v2_r64.yaml",
                                         "--semantic-legend-mode", "nope"])
    with pytest.raises(ValueError):
        load_config(bad)
