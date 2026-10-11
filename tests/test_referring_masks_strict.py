"""Prevent RES benchmark protocol drift, mask loss and invalid-GT scoring."""

from __future__ import annotations

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from open_vision_banana.data.datasets import RefCOCOgDataset
from open_vision_banana.data.res_dataset import RESManifestDataset
from open_vision_banana.evaluation.evaluator import VisionEvaluator
from open_vision_banana.inference.decoder import VisualizationDecoder
from open_vision_banana.inference.generator import _tensor_to_pil
from open_vision_banana.inference.pipeline import VisionBanana
from open_vision_banana.inference.prompt import PromptBuilder
from open_vision_banana.referring_contract import (
    BINARY_REFERRING_ENCODING,
    build_binary_referring_prompt,
)
from open_vision_banana.referring_masks import decode_coco_segmentation

coco_mask = pytest.importorskip("pycocotools.mask")

POLYGONS = [[1.2, 1.4, 6.6, 1.4, 6.6, 5.7, 1.2, 5.7], [12, 8, 18, 8, 18, 11, 12, 11]]


def _fixture(root, *, layout="images/train2014", protocol="umd"):
    folder = root / layout
    folder.mkdir(parents=True, exist_ok=True)
    image = folder / "COCO_train2014_000000000123.jpg"
    Image.new("RGB", (20, 12), (100, 120, 140)).save(image)
    refs = [
        {
            "ref_id": 3,
            "image_id": 123,
            "ann_id": 7,
            "split": "val",
            "sentences": [{"raw": "the two chairs"}, {"sent": "both chairs"}],
        }
    ]
    instances = {
        "images": [{"id": 123, "height": 12, "width": 20, "file_name": image.name}],
        "annotations": [{"id": 7, "image_id": 123, "segmentation": POLYGONS}],
    }
    _write(root, refs, instances, protocol=protocol)
    return refs, instances, image


def _write(root, refs, instances, *, protocol="umd"):
    with (root / f"refs({protocol}).p").open("wb") as handle:
        pickle.dump(refs, handle)
    (root / "instances.json").write_text(json.dumps(instances), encoding="utf-8")


def _load(root, **kwargs):
    return RefCOCOgDataset(root, split="val", resolution=8, native_gt=True, strict=True, **kwargs)


@pytest.mark.parametrize("layout", ["", "images", "images/train2014", "images/val2014"])
def test_strict_native_masks_match_coco_and_preserve_components(tmp_path, layout):
    refs, instances, _ = _fixture(tmp_path, layout=layout)
    # Unselected records must not alter the selected evaluation set.
    refs.append({"split": "train", "image_id": 999, "ann_id": 999, "sentences": ["train only"]})
    _write(tmp_path, refs, instances)
    dataset = _load(tmp_path)
    expected = coco_mask.decode(coco_mask.merge(coco_mask.frPyObjects(POLYGONS, 12, 20))).astype(
        bool
    )
    assert len(dataset) == 2
    assert [dataset[i]["expression"] for i in range(2)] == ["the two chairs", "both chairs"]
    actual = dataset[0]["gt_mask_native"].numpy()
    np.testing.assert_array_equal(actual, expected)
    assert actual[:7, :8].any() and actual[8:, 12:].any()
    assert set(dataset[0]["target_rgb"].unique().tolist()) == {0.0, 1.0}


@pytest.mark.parametrize("available,requested", [("g", "umd"), ("umd", "google")])
def test_strict_forbids_fallback_to_another_protocol(tmp_path, available, requested):
    _fixture(tmp_path, protocol=available)
    with pytest.raises(FileNotFoundError, match="fallback.*forbidden"):
        _load(tmp_path, protocol=requested)
    # Backwards-compatible defaults still support legacy fixtures.
    assert len(RefCOCOgDataset(tmp_path, split="val", protocol=requested)) == 2


@pytest.mark.parametrize("filename", ["g", "google"])
def test_strict_google_accepts_only_google_aliases(tmp_path, filename):
    _fixture(tmp_path, protocol=filename)
    assert len(_load(tmp_path, protocol="google")) == 2


def test_strict_requires_instances(tmp_path):
    _fixture(tmp_path)
    (tmp_path / "instances.json").unlink()
    with pytest.raises(FileNotFoundError, match="instances.json"):
        _load(tmp_path)


@pytest.mark.parametrize(
    "segmentation",
    [None, [], {}, [[1, 1, 1, 1, 1, 1]], [[1, 1, 2]], [[1, 1, float("nan"), 2, 3, 4]]],
)
def test_strict_rejects_missing_malformed_and_empty_gt(tmp_path, segmentation):
    refs, instances, _ = _fixture(tmp_path)
    instances["annotations"][0]["segmentation"] = segmentation
    _write(tmp_path, refs, instances)
    with pytest.raises(ValueError, match="ann_id=7"):
        _load(tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        "missing_annotation",
        "wrong_image",
        "missing_image_metadata",
        "missing_split",
        "empty_split",
        "no_selected_split",
        "empty_expression",
        "one_empty_expression",
        "boolean_annotation_id",
        "malformed_image_id",
    ],
)
def test_strict_ref_integrity(tmp_path, change):
    refs, instances, _ = _fixture(tmp_path)
    if change == "missing_annotation":
        refs[0]["ann_id"] = 99
    elif change == "wrong_image":
        instances["annotations"][0]["image_id"] = 99
    elif change == "missing_image_metadata":
        instances["images"][0]["id"] = 99
    elif change == "missing_split":
        del refs[0]["split"]
    elif change == "empty_split":
        refs[0]["split"] = ""
    elif change == "no_selected_split":
        refs[0]["split"] = "train"
    elif change == "empty_expression":
        refs[0]["sentences"] = []
    elif change == "boolean_annotation_id":
        refs[0]["ann_id"] = True
    elif change == "malformed_image_id":
        refs[0]["image_id"] = []
    else:
        refs[0]["sentences"].append({"sent": " "})
    _write(tmp_path, refs, instances)
    with pytest.raises(ValueError):
        _load(tmp_path)


@pytest.mark.parametrize("change", ["missing", "corrupt", "dimension_mismatch"])
def test_strict_checks_images_before_evaluation(tmp_path, change):
    refs, instances, image = _fixture(tmp_path)
    if change == "missing":
        image.unlink()
    elif change == "corrupt":
        image.write_bytes(b"invalid JPEG")
    else:
        instances["images"][0]["width"] = 19
        _write(tmp_path, refs, instances)
    with pytest.raises((ValueError, FileNotFoundError), match="image_id=123"):
        _load(tmp_path)


def _uncompressed_rle(mask):
    flat = mask.ravel(order="F")
    counts = []
    start = 0
    value = False
    for index, pixel in enumerate(flat):
        if pixel != value:
            counts.append(index - start)
            start = index
            value = bool(pixel)
    counts.append(len(flat) - start)
    return {"size": list(mask.shape), "counts": counts}


@pytest.mark.parametrize("encoding", ["uncompressed", "compressed_string", "compressed_bytes"])
def test_rle_native_pixels_match_coco(encoding):
    rng = np.random.default_rng(18)
    for height, width in [(1, 3), (12, 20), (71, 91)]:
        expected = rng.integers(0, 2, (height, width), dtype=np.uint8)
        expected[0, 0] = 1
        if encoding == "uncompressed":
            rle = _uncompressed_rle(expected)
        else:
            rle = coco_mask.encode(np.asfortranarray(expected))
            if encoding == "compressed_string":
                rle["counts"] = rle["counts"].decode("ascii")
        actual = decode_coco_segmentation(rle, height, width)
        assert actual.dtype == np.bool_
        np.testing.assert_array_equal(actual, expected.astype(bool))


def test_strict_dataset_loads_uncompressed_rle(tmp_path):
    refs, instances, _ = _fixture(tmp_path)
    expected = np.zeros((12, 20), dtype=np.uint8)
    expected[2:5, 1:4] = 1
    expected[8:11, 13:18] = 1
    instances["annotations"][0]["segmentation"] = _uncompressed_rle(expected)
    _write(tmp_path, refs, instances)
    np.testing.assert_array_equal(
        _load(tmp_path)[0]["gt_mask_native"].numpy(), expected.astype(bool)
    )


@pytest.mark.parametrize(
    "rle",
    [
        {"size": [2, 3], "counts": [1, 1, 4]},
        {"size": [3, 2], "counts": [1, 1]},
        {"size": [3, 2], "counts": [0, 7]},
        {"size": [3, 2], "counts": [-1, 7]},
        {"size": [3, 2], "counts": [1.0, 5.0]},
        {"size": [3, 2], "counts": [6]},
        {"size": [3, 2], "counts": "11"},
        {"size": [3, 2], "counts": "0"},
        {"size": [3, 2], "counts": "P"},
        {"size": [3, 2], "counts": "\x00"},
    ],
)
def test_rle_rejects_bad_dimensions_runs_and_empty_foreground(rle):
    with pytest.raises(ValueError):
        decode_coco_segmentation(rle, 3, 2)


def test_polygon_clipping_remains_coco_compatible():
    polygons = [[-2, -1, 5.8, -1, 5.8, 4.3, -2, 4.3]]
    expected = coco_mask.decode(coco_mask.merge(coco_mask.frPyObjects(polygons, 3, 4)))
    np.testing.assert_array_equal(decode_coco_segmentation(polygons, 3, 4), expected.astype(bool))


def test_strict_and_portable_evaluation_preserve_identical_bright_input(tmp_path):
    _, _, image_path = _fixture(tmp_path)
    Image.new("RGB", (20, 12), (192, 192, 192)).save(image_path)
    native = decode_coco_segmentation(POLYGONS, 12, 20)
    Image.fromarray(native.astype(np.uint8) * 255).save(tmp_path / "mask.png")
    expression = "the two chairs"
    row = {
        "sample_id": "ref3",
        "task": "referring",
        "split": "val",
        "source_dataset": "refcocog",
        "source_protocol": "umd",
        "source_split": "val",
        "input": image_path.relative_to(tmp_path).as_posix(),
        "target": "mask.png",
        "width": 20,
        "height": 12,
        "expression": expression,
        "prompt": build_binary_referring_prompt(expression),
        "encoding": BINARY_REFERRING_ENCODING,
    }
    (tmp_path / "pairs.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    strict = _load(tmp_path)
    portable = RESManifestDataset(tmp_path, split="val", resolution=8)
    observed = []
    model = object.__new__(VisionBanana)
    model.training_prompts = {}
    model.prompt_builder = PromptBuilder()
    model.decoder = VisualizationDecoder()

    def generate(**kwargs):
        observed.append(_tensor_to_pil(kwargs["image"][0]).getpixel((0, 0)))
        return torch.ones_like(kwargs["image"])

    model.generator = SimpleNamespace(generate=generate)
    for dataset in (strict, portable):
        assert dataset[0]["image"].min() > 0
        assert dataset[0]["image_range"] == "minus_one_one"
        VisionEvaluator(model, device="cpu").evaluate(
            DataLoader(dataset, batch_size=1),
            task="referring",
            native_resolution=True,
            max_batches=1,
        )
    assert observed == [(192, 192, 192), (192, 192, 192)]
    assert "image_range" not in RefCOCOgDataset(tmp_path, split="val", resolution=8)[0]
