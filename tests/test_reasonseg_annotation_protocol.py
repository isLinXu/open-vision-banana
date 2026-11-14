"""Official annotation semantics, ignored predictions, and additive shard scoring."""
import json

import numpy as np
import pytest
import torch
from PIL import Image

from open_vision_banana.data.paper_eval_datasets import ReasonSegDataset, reasonseg_lisa_labels
from open_vision_banana.evaluation.referring import ReferringAccumulator
from open_vision_banana.evaluation.shards import accumulator_state, merge_accumulator_states


def polygon(label, lo, hi):
    return {"label": label, "points": [[lo, lo], [hi, lo], [hi, hi], [lo, hi]]}


def test_lisa_overlap_flag_truncation_and_native_validity(tmp_path):
    shapes = [polygon("target", 0, 9), polygon("ignore", 2, 7),
              polygon("target", 3.9, 4.9), polygon("FLAG", 0, 11)]
    labels = reasonseg_lisa_labels(shapes, 12, 12)
    assert labels[0, 0] == 1 and labels[11, 11] == 0  # flag never paints
    assert labels[2, 2] == 255 and labels[3, 3] == 1  # small target overrides ignore
    assert labels[4, 4] == 1 and labels[5, 5] == 255  # truncation, not rounding
    Image.new("RGB", (12, 12)).save(tmp_path / "a.jpg")
    (tmp_path / "a.json").write_text(json.dumps({"text": ["one", "two"], "shapes": shapes}))
    ds = ReasonSegDataset(tmp_path, resolution=6, native_gt=True, annotation_mode="lisa_v1")
    assert len(ds) == 2
    item = ds[0]
    assert np.array_equal(item["gt_mask_native"].numpy(), labels == 1)
    assert np.array_equal(item["valid_mask_native"].numpy(), labels != 255)
    assert item["valid_mask"].shape == (6, 6)
    legacy = ReasonSegDataset(tmp_path, native_gt=True)[0]
    assert "valid_mask_native" not in legacy
    assert legacy["gt_mask_native"][10, 10]  # historical union includes flag


def test_ignored_predictions_do_not_change_metrics_and_shards_merge():
    gt = torch.tensor([[True, False], [False, False]])
    valid = torch.tensor([[True, True], [False, False]])
    pred = torch.tensor([[True, True], [True, True]])
    whole, a, b = ReferringAccumulator(), ReferringAccumulator(), ReferringAccumulator()
    for accumulator, prediction in [(a, pred), (b, gt)]:
        accumulator.update(prediction, gt, valid_mask=valid)
        whole.update(prediction, gt, valid_mask=valid)
    assert a.compute()["cIoU"] == .5
    assert whole.compute()["cIoU"] == pytest.approx(2 / 3)
    merged = merge_accumulator_states([accumulator_state(a), accumulator_state(b)])
    assert merged.compute() == pytest.approx(whole.compute())
    with pytest.raises(ValueError, match="no valid"):
        whole.update(gt, gt, valid_mask=torch.zeros_like(gt))


def test_invalid_annotation_mode_fails(tmp_path):
    with pytest.raises(ValueError, match="annotation_mode"):
        ReasonSegDataset(tmp_path, annotation_mode="typo")
