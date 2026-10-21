"""Rea2Seg rows become white-on-black referring pairs."""

import json

import numpy as np
import pytest

from scripts.build_rea2seg_manifest import (
    assign_splits,
    decode_rle,
    mask_to_rgb,
    referring_prompt,
)


def test_prompt_uses_question_not_chain_of_thought():
    text = referring_prompt("the cesspit")
    assert "cesspit" in text.casefold()
    assert "white" in text
    assert "black" in text


def test_mask_is_white_on_black():
    mask_utils = pytest.importorskip("pycocotools.mask")
    encoded = mask_utils.encode(np.array([[0, 1], [1, 0]], dtype=np.uint8, order="F"))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    nested = {"rle": {"size": encoded["size"], "counts": encoded["counts"]}}
    decoded = decode_rle(json.dumps(nested))
    rgb = np.asarray(mask_to_rgb(decoded))
    assert rgb[0, 1].tolist() == [255, 255, 255]
    assert rgb[0, 0].tolist() == [0, 0, 0]


def test_each_category_keeps_a_train_and_val_row():
    categories = ["gqa-seg"] * 10 + ["reasonseg"] * 5
    splits = assign_splits(categories, val_fraction=0.2, seed=42)
    for name in ("gqa-seg", "reasonseg"):
        group = [split for split, category in zip(splits, categories) if category == name]
        assert "train" in group
        assert "val" in group
