"""CPU-only depth admission, geometry and legacy bypass regression tests."""

import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from open_vision_banana.data.depth_training_quality import (
    admit_depth_training_row,
    check_training_dataset,
    require_depth_manifest,
)
from open_vision_banana.training.trainer import ManifestVisionDataset


def make_row(root, name, mask, task="depth", split="train"):
    depth = np.where(mask, 2.0, 0.0).astype(np.float32)
    np.save(root / f"{name}.npy", depth)
    rgb = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    rgb[~mask] = 0
    Image.fromarray(rgb).save(root / f"{name}.png")
    return dict(
        input=f"{name}.png",
        target=f"{name}.png",
        metric_depth=f"{name}.npy",
        task=task,
        split=split,
    )


def test_threshold_precedes_sample_cap_and_preserves_validation(tmp_path):
    mask = np.ones((100, 100), bool)
    mask.flat[:201] = False
    bad = make_row(tmp_path, "bad", mask)
    mask.flat[200] = True
    exact = make_row(tmp_path, "exact", mask)
    val = dict(bad, split="val")
    path = tmp_path / "pairs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [bad, exact, val]))
    ds = ManifestVisionDataset(path, 100, "train", {"depth"}, max_samples=1)
    assert len(ds) == 1
    assert ds.samples[0]["image_path"].endswith("exact.png")
    assert len(ManifestVisionDataset(path, 100, "val", {"depth"})) == 1
    assert ds.depth_quality_rejections == {"native": 1, "geometry": 0}


@pytest.mark.parametrize(
    "module",
    ["open_vision_banana.training.trainer", "scripts.train_sana_512", "scripts.train_flux"],
)
def test_geometry_rejection_is_counted_and_skipped_before_cap(tmp_path, module):
    loader = importlib.import_module(module).ManifestVisionDataset
    mask = np.ones((100, 100), bool)
    mask[:2] = False
    geometry = make_row(tmp_path, "rounded_holes", mask)
    mask[2, 0] = False
    native = make_row(tmp_path, "native_bad", mask)
    mask[:] = True
    good = make_row(tmp_path, "clean", mask)
    path = tmp_path / "pairs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [geometry, native, good]))
    dataset = loader(path, 128, "train", {"depth"}, max_samples=1)
    assert len(dataset) == 1
    assert dataset.samples[0]["image_path"].endswith("clean.png")
    assert dataset.depth_quality_rejections == {"native": 1, "geometry": 1}


def test_unknown_depth_fails_but_unrelated_task_does_not(tmp_path):
    row = make_row(tmp_path, "a", np.ones((10, 10), bool))
    del row["metric_depth"]
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="depth quality"):
        ManifestVisionDataset(path, 10, "train", {"depth"})
    row["task"] = "normal"
    path.write_text(json.dumps(row))
    assert len(ManifestVisionDataset(path, 10, "train", {"normal"})) == 1


@pytest.mark.parametrize("module", ["scripts.train_sana_512", "scripts.train_flux"])
def test_legacy_loaders_filter_before_cap_and_fail_closed(tmp_path, module):
    loader = importlib.import_module(module).ManifestVisionDataset
    mask = np.ones((100, 100), bool)
    mask.flat[:201] = False
    bad = make_row(tmp_path, "legacy_bad", mask)
    mask[:] = True
    good = make_row(tmp_path, "legacy_good", mask)
    path = tmp_path / "pairs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [bad, good]))
    ds = loader(path, 100, "train", {"depth"}, max_samples=1)
    assert len(ds) == 1
    assert ds.samples[0]["image_path"].endswith("legacy_good.png")
    del good["metric_depth"]
    path.write_text(json.dumps(good))
    with pytest.raises(ValueError, match="depth quality"):
        loader(path, 100, "train", {"depth"})


@pytest.mark.parametrize("transpose", [False, True])
def test_only_clean_crop_offsets_are_used(tmp_path, transpose):
    mask = np.ones((100, 200), bool)
    mask[:4, :100] = False  # exactly 2% full-frame, 4% in the leftmost crop
    if transpose:
        mask = mask.T
    row = make_row(tmp_path, "crop", mask)
    offsets = admit_depth_training_row(row, tmp_path, 100, aspect_preserving_crop=True)
    assert offsets and 0 not in offsets
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(row))
    ds = ManifestVisionDataset(path, 100, "train", {"depth"}, aspect_preserving_crop=True)
    for _ in range(40):
        target = ds[0]["target_rgb"]
        holes = (target == 0).all(dim=0)
        assert int(holes.sum()) <= 200


def test_no_clean_crop_uses_verified_full_frame(tmp_path):
    # Every 100px crop includes 250 invalid pixels; full-frame square
    # resizing reduces the strip's width to keep fewer than 200 holes.
    mask = np.ones((100, 150), bool)
    mask[:5, 50:100] = False
    row = make_row(tmp_path, "fallback", mask)
    result = admit_depth_training_row(row, tmp_path, 100, aspect_preserving_crop=True)
    assert result == ()


def test_raw_wrappers_cannot_bypass_admission():
    raw = SimpleNamespace(TASK_TYPE="depth", samples=[])
    with pytest.raises(ValueError, match="pair-manifest"):
        check_training_dataset(SimpleNamespace(datasets=[SimpleNamespace(dataset=raw)]))
    check_training_dataset(SimpleNamespace(task="normal", samples=[]))
    with pytest.raises(ValueError, match="pair-manifest"):
        require_depth_manifest(["depth"], "tar")


@pytest.mark.parametrize("attribute", ["datasets", "vision_datasets", "task_datasets"])
def test_mapping_wrappers_cannot_bypass_admission(attribute):
    raw = SimpleNamespace(TASK_TYPE="depth", samples=[])
    wrapper = SimpleNamespace(**{attribute: {"depth": raw}})
    with pytest.raises(ValueError, match="pair-manifest"):
        check_training_dataset(wrapper)


@pytest.mark.parametrize("source_split", ["official_train", "raw_train_derived", " Training "])
def test_source_training_aliases_are_accepted(tmp_path, source_split):
    row = make_row(tmp_path, "official", np.ones((10, 10), bool))
    row["source_split"] = source_split
    assert admit_depth_training_row(row, tmp_path, 10) == (0,)


@pytest.mark.parametrize("source_split", ["val", "test", "validation"])
def test_heldout_source_cannot_be_relabelled_train(tmp_path, source_split):
    row = make_row(tmp_path, "heldout", np.ones((10, 10), bool))
    row["source_split"] = source_split
    with pytest.raises(ValueError, match="source_split"):
        admit_depth_training_row(row, tmp_path, 10)


def test_depth_augmentation_does_not_add_unverified_crop():
    import torch

    from open_vision_banana.training.utils import VisionAugmentation

    image = torch.rand(3, 100, 100)
    aug = VisionAugmentation(resolution=100, hflip_prob=0, crop_scale=(0.5, 0.5))
    actual, target = aug(image, image, task="depth")
    assert torch.equal(actual, image)
    assert torch.equal(target, image)
