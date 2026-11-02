"""Parquet mirror -> REFER refs(*).p + instances.json (replaces the dead UNC host)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

from scripts.convert_refer_parquet import convert_refer_parquet  # noqa: E402


def _row(ref_id: int, ann_id: int, image_id: int, split: str, sents: list[str]) -> dict:
    return {
        "image_id": image_id,
        "split": split,
        "sentences": json.dumps([{"sent_id": 10 * ref_id + i, "sent": s, "raw": s, "tokens": s.split()} for i, s in enumerate(sents)]),
        "file_name": f"COCO_train2014_{image_id:012d}_{ann_id}.jpg",
        "category_id": 1,
        "ann_id": ann_id,
        "sent_ids": [10 * ref_id + i for i in range(len(sents))],
        "ref_id": ref_id,
        "raw_anns": json.dumps({"segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]], "area": 100.0, "bbox": [0, 0, 10, 10], "iscrowd": 0}),
        "raw_image_info": json.dumps({"file_name": f"COCO_train2014_{image_id:012d}.jpg", "width": 64, "height": 48, "license": 1}),
    }


def _write(dir_: Path, shard: str, rows: list[dict]) -> None:
    (dir_ / "data").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), dir_ / "data" / f"{shard}-00000-of-00001-abc.parquet")


def test_refcocog_umd_round_trip(tmp_path: Path):
    src = tmp_path / "refcocog"
    _write(src, "train", [_row(1, 100, 7, "train", ["the left dog"]), _row(2, 101, 7, "train", ["a red ball", "the ball"])])
    _write(src, "validation", [_row(3, 102, 8, "val", ["woman in blue"])])
    _write(src, "test", [_row(4, 103, 9, "test", ["the lamp"])])
    out = tmp_path / "refer" / "refcocog"
    summary = convert_refer_parquet(src, out, split_by="umd")
    assert summary == {"refs": 4, "images": 3, "annotations": 4, "train": 2, "val": 1, "test": 1}

    refs = pickle.load((out / "refs(umd).p").open("rb"))
    assert [r["ref_id"] for r in refs] == [1, 2, 3, 4]
    assert refs[1]["sentences"][1]["raw"] == "the ball" and refs[1]["sent_ids"] == [20, 21]
    assert refs[0]["file_name"] == "COCO_train2014_000000000007_100.jpg"  # REFER keeps the ann suffix
    inst = json.loads((out / "instances.json").read_text())
    assert {i["id"] for i in inst["images"]} == {7, 8, 9}
    assert inst["images"][0]["file_name"] == "COCO_train2014_000000000007.jpg"
    assert inst["annotations"][0]["segmentation"] == [[0, 0, 10, 0, 10, 10, 0, 10]]
    assert inst["categories"] == [{"id": 1, "name": "1", "supercategory": "object"}]

    # The rebuilt pickle feeds the zero-shot exclusion list directly
    from scripts.build_qwen_baseline_manifest import load_excluded_image_ids

    assert load_excluded_image_ids([out / "refs(umd).p"]) == {8, 9}  # val + test only


def test_refcoco_unc_maps_test_shard_to_test_a(tmp_path: Path):
    src = tmp_path / "refcoco"
    _write(src, "test", [_row(1, 100, 7, "test", ["x"])])
    _write(src, "testB", [_row(2, 101, 8, "testB", ["y"])])
    summary = convert_refer_parquet(src, tmp_path / "out", split_by="unc")
    assert summary["testA"] == 1 and summary["testB"] == 1
    with pytest.raises(ValueError, match="unexpected split"):
        convert_refer_parquet(src, tmp_path / "out2", split_by="umd")
    with pytest.raises(FileNotFoundError):
        convert_refer_parquet(tmp_path / "empty", tmp_path / "out3", split_by="unc")
