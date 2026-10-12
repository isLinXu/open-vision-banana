"""Mixed RES sources reconcile to existing supervision without duplicating it."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("ijson")
pytest.importorskip("pycocotools")

from pycocotools import mask as coco_mask

from open_vision_banana.res_manifest import (
    SOURCE_SPECS,
    build_res_manifest,
    validate_res_manifest,
)
from open_vision_banana.res_mixed import MIXED_SPLITS, audit_mixed_res


@pytest.fixture
def corpus(tmp_path):
    source, images, release = (tmp_path / name for name in ("sources", "images", "release"))
    images.mkdir()
    metadata, annotations = [], []
    for image_id in range(1, 10):
        name = f"COCO_train2014_{image_id:012d}.jpg"
        Image.new("RGB", (12, 10), (image_id * 20, 5, 2)).save(images / name)
        metadata.append({"id": image_id, "file_name": name, "width": 12, "height": 10})
        mask = [[1, 1, 4, 1, 4, 4, 1, 4], [7, 5, 10, 5, 10, 8, 7, 8]]
        if image_id == 3:
            pixels = np.zeros((10, 12), dtype=np.uint8, order="F")
            pixels[1:4, 1:4] = 1
            mask = coco_mask.encode(pixels)
            mask["counts"] = mask["counts"].decode("ascii")
        annotations.append(
            {
                "id": image_id * 10,
                "image_id": image_id,
                "category_id": 18,
                "bbox": [1, 1, 9, 7],
                "segmentation": mask,
            }
        )
    mixed = {split: [] for split in MIXED_SPLITS}
    partitions = {
        "refcoco": [("train", 1), ("train", 5), ("val", 2), ("testA", 3), ("testB", 4)],
        "refcoco+": [("train", 1), ("val", 5), ("testA", 6), ("testB", 7)],
        "refcocog": [("train", 1), ("val", 8), ("test", 9)],
    }
    for dataset, protocol in SOURCE_SPECS:
        folder = source / dataset
        folder.mkdir(parents=True)
        refs = []
        for ref_id, (split, image_id) in enumerate(partitions[dataset]):
            sentences = [{"sent_id": ref_id * 10, "raw": "The OBJECT!", "sent": "the object"}]
            if split == "train" and image_id == 1:
                sentences.append(
                    {"sent_id": ref_id * 10 + 1, "raw": "Second phrase", "sent": "second phrase"}
                )
            refs.append(
                {
                    "ref_id": ref_id,
                    "image_id": image_id,
                    "ann_id": image_id * 10,
                    "split": split,
                    "sentences": sentences,
                }
            )
            name = (
                "train"
                if split == "train"
                else f"{split}_{dataset.replace('+', 'plus')}_{protocol}"
            )
            ann = annotations[image_id - 1]
            mixed[name].append(
                {
                    "image_id": image_id,
                    "width": 12,
                    "height": 10,
                    "category_id": 1,
                    "bbox": ann["bbox"],
                    "mask": ann["segmentation"],
                    "expressions": [s["sent"] for s in sentences],
                }
            )
        (folder / "instances.json").write_text(
            json.dumps(
                {
                    "images": metadata,
                    "annotations": annotations,
                    "categories": [{"id": 1}, {"id": 18}],
                }
            ),
            encoding="utf-8",
        )
        (folder / f"refs({protocol}).p").write_bytes(pickle.dumps(refs, protocol=2))
    build_res_manifest(source, images, release)
    validate_res_manifest(release)
    path = tmp_path / "instances_nogoogle.json"
    path.write_text(json.dumps(mixed), encoding="utf-8")
    return path, source, images, release


def test_complete_reconciliation_preserves_raw_text_and_shared_images(corpus):
    path, _, _, release = corpus
    before = (release / "pairs.jsonl").read_bytes()
    report = audit_mixed_res(*corpus)
    assert report["status"] == "passed"
    assert report["counts"]["train"] == {"records": 4, "expressions": 7}
    assert report["expressions_matched_to_official_sent"] == 15
    assert report["novel_expressions"] == 0
    assert report["train_expressions_on_official_held_out_images"] == 1
    assert report["canonical_released_counts"] == {"train": 2, "val": 3, "test": 5}
    assert report["unique_images_decoded_and_sha256_matched"] == 9
    assert report["released_masks_pixel_identical"] == 9
    assert report["mask_types"] == {"polygon": 8, "rle": 1}
    assert (release / "pairs.jsonl").read_bytes() == before
    assert '"expression":"The OBJECT!"' in before.decode()
    assert str(path.parent) not in json.dumps(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("expressions", []),
        ("expressions", [""]),
        ("expressions", ["unknown expression"]),
        ("expressions", ["the object", "second phrase", "the object"]),
        ("image_id", True),
        ("width", 0),
        ("height", 11),
        ("category_id", True),
        ("category_id", 18),
        ("bbox", [0, 0, 1, 1]),
        ("mask", [[0, 0, 1, 0, 1, 1]]),
    ],
)
def test_rejects_corrupted_or_unmatched_annotation(corpus, field, value):
    path = corpus[0]
    data = json.loads(path.read_text())
    data["train"][0][field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Mixed RES"):
        audit_mixed_res(*corpus)


@pytest.mark.parametrize("operation", ["missing", "unknown", "duplicate", "empty", "nonobject"])
def test_split_and_multiplicity_validation(corpus, operation):
    path = corpus[0]
    data = json.loads(path.read_text())
    if operation == "missing":
        del data["test_refcocog_umd"]
    elif operation == "unknown":
        data["new_split"] = data["train"]
    elif operation == "duplicate":
        data["val_refcoco_unc"].append(data["val_refcoco_unc"][0])
    elif operation == "empty":
        data["train"] = []
    else:
        data["train"][0] = 7
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        audit_mixed_res(*corpus)


@pytest.mark.parametrize(
    "size,match", [((12, 10), "image differs"), ((13, 10), "native image dimensions")]
)
def test_requires_same_native_image_bytes_and_dimensions(corpus, size, match):
    Image.new("RGB", size, "white").save(corpus[2] / "COCO_train2014_000000000001.jpg")
    with pytest.raises(ValueError, match=match):
        audit_mixed_res(*corpus)


@pytest.mark.parametrize("name", ["pairs.jsonl", "assets.jsonl"])
def test_rejects_changed_release_metadata(corpus, name):
    with (corpus[3] / name).open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="metadata changed"):
        audit_mixed_res(*corpus)


def test_rejects_changed_released_mask(corpus):
    target = next((corpus[3] / "targets").rglob("*.png"))
    Image.new("RGB", (12, 10), "white").save(target)
    with pytest.raises(ValueError, match="target checksum"):
        audit_mixed_res(*corpus)


def test_rejects_incomplete_release(corpus):
    (corpus[3] / "BUILD_INCOMPLETE").touch()
    with pytest.raises(ValueError, match="incomplete"):
        audit_mixed_res(*corpus)


def test_rejects_mismatched_official_source_hash(corpus):
    with (corpus[1] / "refcoco" / "instances.json").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="source hashes differ"):
        audit_mixed_res(*corpus)
