"""Synthetic source fixtures exercise full RES builds, with no model imports."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("ijson")
pytest.importorskip("pycocotools")

from open_vision_banana.res_manifest import (
    SOURCE_SPECS,
    PrimitiveUnpickler,
    build_res_manifest,
    portable_path,
    validate_res_manifest,
    write_image_attribution,
)


def _ref(ref_id: int, image: int, split: str, *expressions: str) -> dict:
    return {
        "ref_id": ref_id, "image_id": image, "ann_id": image * 10, "split": split,
        "sentences": [{"sent_id": ref_id * 10 + index, "raw": text, "sent": text.lower()}
                      for index, text in enumerate(expressions)],
    }


@pytest.fixture
def sources(tmp_path: Path) -> tuple[Path, Path]:
    annotation_root = tmp_path / "source"
    image_root = tmp_path / "images"
    image_root.mkdir()
    images, annotations = [], []
    for number in range(1, 6):
        name = f"COCO_train2014_{number:012d}.jpg"
        Image.new("RGB", (12, 10), (number * 40, number * 10, 2)).save(image_root / name)
        images.append({"id": number, "file_name": name, "width": 12, "height": 10})
        annotations.append({"id": number * 10, "image_id": number, "bbox": [1, 1, 9, 7],
                            "segmentation": [[1, 1, 4, 1, 4, 4, 1, 4], [7, 5, 10, 5, 10, 8, 7, 8]]})
    refs_by_source = {
        "refcoco": [_ref(1, 1, "train", "The object", "another phrase"),
                    _ref(2, 2, "train", "leaks into plus val"),
                    _ref(3, 3, "testA", "test target"), _ref(4, 5, "val", "duplicate", "duplicate")],
        "refcoco+": [_ref(1, 1, "train", "The object"), _ref(2, 2, "val", "plus validation"),
                     _ref(3, 4, "train", "training only")],
        "refcocog": [_ref(1, 3, "train", "leaks into unc testA"),
                     _ref(2, 1, "train", "third expression"), _ref(3, 5, "test", "g test")],
    }
    for source, protocol in SOURCE_SPECS:
        folder = annotation_root / source
        folder.mkdir(parents=True)
        (folder / "instances.json").write_text(json.dumps({"images": images, "annotations": annotations}))
        (folder / f"refs({protocol}).p").write_bytes(pickle.dumps(refs_by_source[source], protocol=2))
    return annotation_root, image_root


def test_build_global_exclusion_dedup_and_native_mask_reuse(sources, tmp_path):
    output = tmp_path / "output"
    report = build_res_manifest(*sources, output)
    assert report["counts"] == {"train": 4, "val": 3, "test": 2}
    assert report["excluded_train_by_source"] == {"refcoco/unc": 1, "refcocog/umd": 1}
    assert report["duplicate_train_by_source"] == {"refcoco+/unc": 1}
    assert report["evaluation_blacklist_images"] == 3
    assert report["unique_targets"] == 5
    rows = [json.loads(line) for line in (output / "pairs.jsonl").read_text().splitlines()]
    image_one = [row for row in rows if row["image_id"] == 1]
    assert len(image_one) == 3
    assert len({row["target"] for row in image_one}) == 1
    assert image_one[0]["expression"] == "The object"
    assert sum(row["expression"] == "duplicate" for row in rows) == 2
    pixels = np.asarray(Image.open(output / image_one[0]["target"]))
    assert pixels.shape == (10, 12, 3)
    assert pixels[2, 2].tolist() == [255, 255, 255]
    assert pixels[6, 8].tolist() == [255, 255, 255]
    assert pixels[0, 0].tolist() == [0, 0, 0]
    audit = validate_res_manifest(output)
    assert audit["status"] == "passed"
    assert audit["official_eval_multiplicity_preserved"] is True
    assert audit["verified_assets"] == 10
    assert not (output / ".build.sqlite").exists()
    assert not (output / "BUILD_INCOMPLETE").exists()
    # Reports must be publishable without disclosing source machine locations.
    assert str(tmp_path) not in (output / "build_report.json").read_text()


def test_pilot_preserves_global_blacklist_and_eval_multiplicity(sources, tmp_path):
    output = tmp_path / "pilot"
    report = build_res_manifest(*sources, output, max_train_samples=1)
    assert report["counts"] == {"train": 1, "val": 3, "test": 2}
    assert report["evaluation_blacklist_images"] == 3
    assert report["pilot_omitted_unique_train"] == 3
    assert validate_res_manifest(output)["status"] == "passed"


def test_build_is_reproducible(sources, tmp_path):
    first = build_res_manifest(*sources, tmp_path / "first")
    second = build_res_manifest(*sources, tmp_path / "second")
    assert first == second


def test_attribution_preserves_available_source_metadata_and_nulls(sources, tmp_path):
    annotation_root, image_root = sources
    for source, _ in SOURCE_SPECS:
        path = annotation_root / source / "instances.json"
        payload = json.loads(path.read_text())
        payload["licenses"] = [{"id": 4, "name": "source license name", "url": "https://example.org/license"}]
        payload["images"][0].update({"license": 4, "coco_url": "https://example.org/coco.jpg",
                                     "flickr_url": "https://example.org/flickr.jpg"})
        path.write_text(json.dumps(payload))
    output = tmp_path / "output"
    report = build_res_manifest(annotation_root, image_root, output)
    rows = [json.loads(line) for line in (output / "image_attribution.jsonl").read_text().splitlines()]
    assert len(rows) == report["unique_images"] == 5
    assert rows[0] == {
        "image_id": 1, "input": "images/coco2014/COCO_train2014_000000000001.jpg", "license": 4,
        "license_name": "source license name", "license_url": "https://example.org/license",
        "coco_url": "https://example.org/coco.jpg", "flickr_url": "https://example.org/flickr.jpg",
        "file_name": "COCO_train2014_000000000001.jpg", "width": 12, "height": 10,
    }
    assert rows[1]["license"] is None and rows[1]["license_url"] is None
    assert rows[1]["coco_url"] is None and rows[1]["flickr_url"] is None
    assert "image_attribution.jsonl" in report["files"]
    assert validate_res_manifest(output)["status"] == "passed"


@pytest.mark.parametrize("failure", ["conflict", "missing"])
def test_attribution_rejects_conflicting_or_missing_sources(sources, tmp_path, failure):
    annotation_root, image_root = sources
    output = tmp_path / "output"
    build_res_manifest(annotation_root, image_root, output)
    for source, _ in SOURCE_SPECS:
        path = annotation_root / source / "instances.json"
        payload = json.loads(path.read_text())
        if failure == "missing":
            payload["images"] = payload["images"][1:]
        else:
            payload["images"][0]["coco_url"] = f"https://example.org/{source}.jpg"
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Conflicting image attribution|Missing COCO attribution"):
        write_image_attribution(annotation_root, output)


def test_missing_protocol_does_not_fallback(sources, tmp_path):
    annotations, images = sources
    (annotations / "refcocog" / "refs(umd).p").unlink()
    with pytest.raises(FileNotFoundError, match="Required official"):
        build_res_manifest(annotations, images, tmp_path / "out")
    assert (tmp_path / "out" / "BUILD_INCOMPLETE").exists()


def test_image_dimension_mismatch_fails(sources, tmp_path):
    annotations, images = sources
    Image.new("RGB", (1, 1)).save(images / "COCO_train2014_000000000001.jpg")
    with pytest.raises(ValueError, match="dimensions disagree"):
        build_res_manifest(annotations, images, tmp_path / "out")


@pytest.mark.parametrize("value", ["../escape.png", "images/../../escape.png", "/tmp/a.png",
                                  "C:/data/a.png", "C:\\data\\a.png", "images//a.png",
                                  "images/NUL.jpg", "images/a?.png", "images/trailing. "])
def test_portable_path_rejects_unsafe_paths(tmp_path, value):
    with pytest.raises(ValueError):
        portable_path(tmp_path, value)


def test_validator_rejects_manifest_traversal(sources, tmp_path):
    output = tmp_path / "output"
    build_res_manifest(*sources, output)
    manifest = output / "pairs.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]["target"] = "../escape.png"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="escapes dataset"):
        validate_res_manifest(output)


def test_validator_detects_modified_mask(sources, tmp_path):
    output = tmp_path / "output"
    build_res_manifest(*sources, output)
    target = next((output / "targets" / "coco2014").glob("*.png"))
    Image.new("RGB", (12, 10), (100, 100, 100)).save(target)
    with pytest.raises(ValueError, match="Asset integrity mismatch"):
        validate_res_manifest(output)


def test_ref_pickle_cannot_load_globals(tmp_path):
    path = tmp_path / "unsafe.p"
    path.write_bytes(pickle.dumps(Path("never_constructed")))
    with path.open("rb") as stream, pytest.raises(pickle.UnpicklingError, match="Globals are forbidden"):
        PrimitiveUnpickler(stream).load()
