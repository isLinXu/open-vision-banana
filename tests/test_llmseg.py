"""LLM-Seg source selection and shared RES loading, on tiny native-mask fixtures."""

import json

import numpy as np
import pytest
from PIL import Image

from open_vision_banana.llmseg import build_llmseg
from open_vision_banana.referring_contract import build_binary_referring_prompt


def fixture(tmp_path, *, count=8):
    coco = pytest.importorskip("pycocotools.mask")
    root = tmp_path / "images"
    root.mkdir()
    pixels = np.zeros((6, 8), dtype=np.uint8)
    pixels[1:4, 2:5] = 1
    rle = coco.encode(np.asfortranarray(pixels))
    rle["counts"] = rle["counts"].decode()
    data = {}
    for i in range(1, count + 1):
        name = f"{i:012d}.jpg"
        Image.new("RGB", (8, 6), (i * 20, 30, 10)).save(root / name)
        data[name] = {
            "from_dataset": "coco",
            "qa_pairs": [
                {
                    "question_id": f"q{j}",
                    "question": f"Which object has attribute {j}?",
                    "answer": "SECRET ANSWER",
                    "rle_seg": rle,
                }
                for j in range(3)
            ],
        }
    data["ego.jpg"] = {
        "from_dataset": "ego_objects",
        "qa_pairs": data["000000000001.jpg"]["qa_pairs"],
    }
    train = tmp_path / "train.json"
    val = tmp_path / "validation.json"
    train.write_text(json.dumps(data))
    val.write_text(json.dumps({"000000000001.jpg": data["000000000001.jpg"]}))
    return train, val, root, data


def records(root):
    return [json.loads(line) for line in (root / "training.jsonl").read_text().splitlines()]


def test_selection_provenance_native_masks_and_existing_loader(tmp_path):
    from open_vision_banana.training.trainer import ManifestVisionDataset, _collate_fn

    train, val, images, _ = fixture(tmp_path)
    out = tmp_path / "out"
    report = build_llmseg(
        train, val, images, out, max_samples=4, exclude_image_ids=[2], verify_release=False
    )
    rows = records(out)
    assert len(rows) == len({r["image_id"] for r in rows}) == 4
    assert not {1, 2} & {r["image_id"] for r in rows}
    assert report["selection"]["eligible_images_before_asset_checks"] == 6
    assert report["selection"]["upstream_ego_objects_images"] == 1
    for row in rows:
        assert row["source_dataset"] == "wangjunchi/LLMSeg"
        assert row["source_protocol"] == "official_train_coco_one_question_v1"
        assert row["split"] == row["source_split"] == "train"
        assert row["prompt"] == build_binary_referring_prompt(row["expression"])
        assert row["answer"] == "SECRET ANSWER" and "SECRET ANSWER" not in row["prompt"]
        assert row["width"] == 8 and row["height"] == 6
        with Image.open(out / row["target"]) as mask:
            assert mask.size == (8, 6) and mask.mode == "RGB"
            assert set(np.unique(mask)) == {0, 255}
    ds = ManifestVisionDataset(out / "training.jsonl", 16, "train", {"referring"}, data_root=out)
    batch = _collate_fn([ds[0], ds[1]])
    assert batch["image"].shape == batch["target_rgb"].shape == (2, 3, 16, 16)
    assert batch["expression"] == [rows[0]["expression"], rows[1]["expression"]]
    assert set(batch["target_rgb"].unique().tolist()) == {0.0, 1.0}


def test_seed_is_independent_of_json_key_order(tmp_path):
    train, val, images, data = fixture(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    build_llmseg(train, val, images, a, max_samples=3, verify_release=False)
    train.write_text(json.dumps(dict(reversed(list(data.items())))))
    build_llmseg(train, val, images, b, max_samples=3, verify_release=False)
    assert [(r["image_id"], r["question_id"]) for r in records(a)] == [
        (r["image_id"], r["question_id"]) for r in records(b)
    ]


def test_official_train_and_validation_hashes_are_required(tmp_path):
    train, val, images, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="SHA256"):
        build_llmseg(train, val, images, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_missing_image_never_becomes_a_silent_cleaning_rejection(tmp_path):
    train, val, images, _ = fixture(tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing COCO"):
        build_llmseg(
            train,
            val,
            images / "absent",
            tmp_path / "out",
            verify_release=False,
            reject_invalid=True,
        )
    assert (tmp_path / "out" / "BUILD_INCOMPLETE").exists()


def test_quality_filter_and_rejection_accounting(tmp_path):
    train, val, images, data = fixture(tmp_path)
    for qa in data["000000000002.jpg"]["qa_pairs"]:
        qa["rle_seg"] = {"size": [6, 8], "counts": [0, 48]}
    for qa in data["000000000003.jpg"]["qa_pairs"]:
        qa["rle_seg"] = {"size": [6, 8], "counts": [100]}
    train.write_text(json.dumps(data))
    out = tmp_path / "out"
    report = build_llmseg(
        train, val, images, out, max_samples=100, verify_release=False, reject_invalid=True
    )
    assert report["llmseg_rows"] == 5 and report["rejected_rows"] == 2
    assert report["scanned_rows"] == report["llmseg_rows"] + report["rejected_rows"]
    assert len((out / "rejected.jsonl").read_text().splitlines()) == 2
    assert not {1, 2, 3} & {r["image_id"] for r in records(out)}


@pytest.mark.parametrize("invalid", ["../000000000002.jpg", "https://example.com/image.jpg"])
def test_reject_noncanonical_image_paths(tmp_path, invalid):
    train, val, images, data = fixture(tmp_path)
    data[invalid] = data.pop("000000000002.jpg")
    train.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid COCO image name"):
        build_llmseg(train, val, images, tmp_path / "out", verify_release=False)


def test_2014_image_names_and_exact_heldout_exclusion(tmp_path):
    train, val, images, _ = fixture(tmp_path)
    for path in images.glob("*.jpg"):
        path.rename(path.with_name("COCO_train2014_" + path.name))
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text(
        json.dumps({"split": "test", "input": "COCO_train2014_000000000002.jpg"}) + "\n"
    )
    report = build_llmseg(
        train,
        val,
        images,
        tmp_path / "out",
        max_samples=100,
        exclude=[(heldout, images)],
        verify_release=False,
        reject_invalid=True,
    )
    assert report["rejected_by_reason"] == {"heldout_image_overlap": 1}
    assert 2 not in {r["image_id"] for r in records(tmp_path / "out")}


def test_existing_bundle_and_restore_support_source_adapter(tmp_path):
    from open_vision_banana.res_bundle import package_dataset, restore_bundle
    from open_vision_banana.res_preparation import validate_prepared_pairs
    from open_vision_banana.training.trainer import ManifestVisionDataset

    train, val, images, _ = fixture(tmp_path)
    out = tmp_path / "out"
    build_llmseg(train, val, images, out, max_samples=3, verify_release=False)
    checked = validate_prepared_pairs(out)
    assert checked["counts"] == {"train": 3}
    assert checked["assets_decoded"] == 4  # three images, shared native mask
    bundle, restored = tmp_path / "bundle", tmp_path / "restored"
    index = package_dataset(out, bundle, shard_bytes=200)
    assert index["format"] == "ovb_res_bundle_v1"
    restore_bundle(bundle, restored)
    assert (restored / "training.jsonl").read_bytes() == (out / "training.jsonl").read_bytes()
    ds = ManifestVisionDataset(restored / "training.jsonl", 16, "train", {"referring"})
    assert len(ds) == 3 and ds[0]["target_rgb"].shape == (3, 16, 16)
    target = restored / records(restored)[0]["target"]
    changed = np.zeros((6, 8, 3), dtype=np.uint8)
    changed[0, 0] = 255
    Image.fromarray(changed).save(target)
    with pytest.raises(ValueError, match="Target pixels changed"):
        validate_prepared_pairs(restored)


def test_merge_keeps_existing_official_validation_and_all_three_sources(tmp_path):
    from open_vision_banana.rea2seg import build_rea2seg
    from open_vision_banana.training.trainer import ManifestVisionDataset, _collate_fn
    from scripts.merge_multitask_manifest import merge_manifests

    train, val, images, data = fixture(tmp_path)
    llm = tmp_path / "llm"
    build_llmseg(train, val, images, llm, max_samples=3, verify_release=False)
    # Reuse an unselected image for a different query, exercising mixed-source collation.
    qa = data["000000000008.jpg"]["qa_pairs"][0]
    rea = tmp_path / "rea"
    build_rea2seg(
        [
            (
                {
                    "image": {"path": str(images / "000000000008.jpg")},
                    "question": "Which object is closest?",
                    "cot_answer": "answer",
                    "category": "spatial",
                    "mask": {"rle": qa["rle_seg"]},
                },
                {"source_shard": "fixture", "source_row": 0},
                tmp_path,
            )
        ],
        rea,
    )
    base = tmp_path / "base"
    base.mkdir()
    Image.new("RGB", (8, 6), "white").save(base / "input.png")
    Image.new("RGB", (8, 6), "white").save(base / "target.png")
    original = {
        "task": "referring",
        "encoding": "binary_white_black_v1",
        "expression": "white square",
        "prompt": build_binary_referring_prompt("white square"),
        "input": "input.png",
        "target": "target.png",
        "split": "val",
        "source_split": "val",
        "source_dataset": "refcoco",
        "source_protocol": "unc",
    }
    (base / "training.jsonl").write_text(json.dumps(original) + "\n")
    combined = tmp_path / "combined" / "training.jsonl"
    audit = merge_manifests(
        [base / "training.jsonl", rea / "training.jsonl", llm / "training.jsonl"],
        combined,
        val_percent=0,
    )
    assert audit["per_task_dropped"] == {}
    assert audit["per_task_split"]["referring"] == {"train": 4, "val": 1}
    train_ds = ManifestVisionDataset(combined, 16, "train", {"referring"})
    val_ds = ManifestVisionDataset(combined, 16, "val", {"referring"})
    assert len(train_ds) == 4 and len(val_ds) == 1
    assert val_ds[0]["expression"] == original["expression"]
    assert _collate_fn([train_ds[0], val_ds[0]])["target_rgb"].shape == (2, 3, 16, 16)


def test_full_train_all_questions_and_both_sources(tmp_path):
    train, val, images, _ = fixture(tmp_path, count=3)
    ego = tmp_path / "ego"
    ego.mkdir()
    Image.new("RGB", (8, 6), (7, 80, 160)).save(ego / "ego.jpg")
    out = tmp_path / "full"
    report = build_llmseg(
        train,
        val,
        images,
        out,
        full_train=True,
        max_samples=None,
        ego_image_root=ego,
        verify_release=False,
    )
    rows = records(out)
    assert len(rows) == 9  # COCO 2 images x 3 questions + EgoObjects 1 x 3.
    assert report["selection"]["eligible_images_before_asset_checks"] == 3
    assert report["selection"]["eligible_questions_before_asset_checks"] == 9
    assert {r["source_protocol"] for r in rows} == {"official_train_all_questions_v1"}
    assert {r["split"] for r in rows} == {"train"}
    assert len([r for r in rows if r["category"] == "ego_objects"]) == 3
    assert all("image_id" not in r for r in rows if r["category"] == "ego_objects")
    assert all("SECRET ANSWER" not in r["prompt"] for r in rows)
    from open_vision_banana.res_bundle import package_dataset, restore_bundle
    from open_vision_banana.res_preparation import validate_prepared_pairs

    assert validate_prepared_pairs(out)["counts"] == {"train": 9}
    bundle, restored = tmp_path / "bundle", tmp_path / "restored"
    package_dataset(out, bundle, shard_bytes=1000)
    restore_bundle(bundle, restored)
    assert (out / "training.jsonl").read_bytes() == (restored / "training.jsonl").read_bytes()


def test_full_train_excludes_ego_validation_and_rejects_missing_assets(tmp_path):
    train, val, images, data = fixture(tmp_path, count=3)
    val.write_text(json.dumps({"ego.jpg": data["ego.jpg"]}))
    out = tmp_path / "full"
    build_llmseg(
        train,
        val,
        images,
        out,
        full_train=True,
        max_samples=None,
        ego_image_root=tmp_path / "missing",
        verify_release=False,
    )
    assert len(records(out)) == 9
    assert {r["category"] for r in records(out)} == {"coco"}
    val.write_text(json.dumps({"000000000001.jpg": data["000000000001.jpg"]}))
    with pytest.raises(FileNotFoundError, match="Missing EgoObjects"):
        build_llmseg(
            train,
            val,
            images,
            tmp_path / "missing-out",
            full_train=True,
            max_samples=None,
            ego_image_root=tmp_path / "missing",
            verify_release=False,
            reject_invalid=True,
        )


def test_full_train_requires_explicit_assets_and_no_cap(tmp_path):
    train, val, images, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="max_samples=None"):
        build_llmseg(
            train,
            val,
            images,
            tmp_path / "out",
            full_train=True,
            ego_image_root=images,
            verify_release=False,
        )
    with pytest.raises(ValueError, match="ego_image_root"):
        build_llmseg(
            train,
            val,
            images,
            tmp_path / "out",
            full_train=True,
            max_samples=None,
            verify_release=False,
        )


@pytest.mark.parametrize("name", ["../ego.jpg", "/ego.jpg", "C:ego.jpg", "a\\ego.jpg"])
def test_full_train_rejects_unsafe_ego_names(tmp_path, name):
    train, val, images, data = fixture(tmp_path)
    data[name] = data.pop("ego.jpg")
    train.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid EgoObjects image name"):
        build_llmseg(
            train,
            val,
            images,
            tmp_path / "out",
            full_train=True,
            max_samples=None,
            ego_image_root=images,
            verify_release=False,
        )


def test_explicit_coco_stage_does_not_claim_both_sources(tmp_path):
    train, val, images, _ = fixture(tmp_path, count=3)
    out = tmp_path / "coco-stage"
    report = build_llmseg(
        train,
        val,
        images,
        out,
        full_train=True,
        max_samples=None,
        source_categories=("coco",),
        verify_release=False,
    )
    assert len(records(out)) == 6
    assert report["selection"]["source_categories"] == ["coco"]
    assert not report["selection"]["all_source_categories_selected"]
    assert {r["category"] for r in records(out)} == {"coco"}
