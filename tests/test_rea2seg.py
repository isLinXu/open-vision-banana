"""Synthetic, CPU-only conversion and existing-loader integration tests."""

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from open_vision_banana.rea2seg import build_rea2seg, cached_hf_shards, parquet_rows
from open_vision_banana.referring_contract import build_binary_referring_prompt


def sample(category="gqa-seg", question="  Which object is left?  ", color="red", compressed=True):
    coco = pytest.importorskip("pycocotools.mask")
    pixels = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
    rle = coco.encode(np.asfortranarray(pixels))
    rle["counts"] = rle["counts"].decode("ascii") if compressed else [1, 2, 2, 1]
    data = io.BytesIO()
    Image.new("RGB", (3, 2), color).save(data, format="PNG")
    return {
        "image": {"bytes": data.getvalue(), "path": None},
        "question": question,
        "cot_answer": "ANSWER MUST NOT ENTER THE PROMPT",
        "mask": json.dumps({"rle": rle}),
        "category": category,
    }


def rows(*samples):
    return [
        (
            row,
            {"source_shard": "train.parquet", "source_row": i, "source_shard_sha256": "a" * 64},
            Path("."),
        )
        for i, row in enumerate(samples)
    ]


def read(root):
    return [json.loads(line) for line in (root / "training.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("compressed", [True, False])
def test_native_rle_prompt_and_existing_loader(tmp_path, compressed):
    from open_vision_banana.training.trainer import ManifestVisionDataset, _collate_fn

    original = sample(compressed=compressed)
    original["id"] = "source-id-7"
    root = tmp_path / "export"
    report = build_rea2seg(rows(original), root, revision="fixture-revision")
    row = read(root)[0]
    assert row["expression"] == original["question"]
    assert row["prompt"] == build_binary_referring_prompt(original["question"])
    assert original["cot_answer"] not in row["prompt"]
    assert row["cot_answer"] == original["cot_answer"]
    assert row["source_revision"] == "fixture-revision"
    assert row["source_sample_id"] == "source-id-7"
    assert row["source_row"] == 0 and row["source_split"] == "train"
    assert (row["width"], row["height"]) == (3, 2)
    assert row["encoding"] == "binary_white_black_v1"
    for key in ("input", "target"):
        assert not Path(row[key]).is_absolute() and ".." not in Path(row[key]).parts
        with Image.open(root / row[key]) as image:
            assert image.size == (3, 2) and image.format == "PNG"
    with Image.open(root / row["target"]) as image:
        pixels = np.asarray(image)
        assert image.mode == "RGB"
        np.testing.assert_array_equal(pixels[..., 0], [[0, 255, 0], [255, 0, 255]])
        assert np.all(pixels == pixels[..., :1])
    assert report["counts"] == {"train": 1}
    assert not (root / "BUILD_INCOMPLETE").exists()
    dataset = ManifestVisionDataset(
        root / "training.jsonl", 7, "train", {"referring"}, data_root=root
    )
    assert len(dataset) == 1
    item = dataset[0]
    assert item["prompt"] == row["prompt"]
    assert set(item["target_rgb"].unique().tolist()) == {0.0, 1.0}
    assert _collate_fn([item])["expression"] == [original["question"]]
    assert len(ManifestVisionDataset(root / "training.jsonl", 7, "val", {"referring"})) == 0


def test_filter_cap_and_stable_id(tmp_path):
    source = rows(sample(), sample("spatial", "query B"), sample("spatial", "query C"))
    report = build_rea2seg(source, tmp_path / "a", categories=["spatial"], max_samples=1)
    build_rea2seg(source[1:2], tmp_path / "b")
    assert read(tmp_path / "a")[0]["sample_id"] == read(tmp_path / "b")[0]["sample_id"]
    assert report["scanned_rows"] == 2 and report["category_filtered_rows"] == 1
    assert report["rea2seg_category_counts"] == {"spatial": 1}
    with pytest.raises(ValueError, match="No Rea2Seg"):
        build_rea2seg(rows(sample()), tmp_path / "empty", categories=["spatial"])


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"question": " "}, "nonempty expression"),
        ({"cot_answer": None}, "cot_answer"),
        ({"category": "unknown"}, "Unknown category"),
        ({"split": "test"}, "Evaluation rows"),
        ({"width": 100}, "width differs"),
        ({"mask": "not json"}, "Expecting value"),
        ({"mask": "{}"}, "rle object"),
        ({"mask": '{"rle":{"size":[3,2],"counts":[1,5]}}'}, "native image dimensions"),
        ({"mask": '{"rle":{"size":[2,3],"counts":[1,6]}}'}, "cover exactly"),
        ({"mask": '{"rle":{"size":[2,3],"counts":"P"}}'}, "Truncated"),
        ({"mask": '{"rle":{"size":[2,3],"counts":[6]}}'}, "empty foreground"),
        ({"image": {"bytes": b"invalid"}}, "cannot identify image"),
        ({"image": {"path": "https://example.invalid/image.png"}}, "local asset"),
    ],
)
def test_invalid_inputs_fail_with_source_location(tmp_path, updates, match):
    root = tmp_path / "failed"
    with pytest.raises(ValueError, match=match) as exc:
        build_rea2seg(rows({**sample(), **updates}), root)
    assert "train.parquet:0" in str(exc.value)
    assert (root / "BUILD_INCOMPLETE").exists()
    assert not (root / "training.jsonl").exists()


def test_duplicates_paths_and_multiple_queries(tmp_path):
    with pytest.raises(ValueError, match="Duplicate image/query"):
        build_rea2seg(rows(sample(), sample("reasonseg")), tmp_path / "duplicate")
    build_rea2seg(rows(sample(), sample(question="other query")), tmp_path / "good")
    a, b = read(tmp_path / "good")
    assert a["input"] == b["input"] and a["sample_id"] != b["sample_id"]
    before = (tmp_path / "good" / "training.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        build_rea2seg(rows(sample()), tmp_path / "good")
    assert (tmp_path / "good" / "training.jsonl").read_bytes() == before
    repeated = rows(sample(), sample(question="other query"))
    repeated[1][1]["source_row"] = 0
    with pytest.raises(ValueError, match="duplicate sample_id"):
        build_rea2seg(repeated, tmp_path / "same-id")


def existing(root, split="val", color="blue", task="referring", question="existing query"):
    root.mkdir()
    Image.new("RGB", (3, 2), color).save(root / "image.bmp")
    Image.new("RGB", (3, 2), "white").save(root / "target.png")
    row = {
        "input": "image.bmp",
        "target": "target.png",
        "task": task,
        "split": split,
        "expression": question,
        "prompt": build_binary_referring_prompt(question),
        "sample_id": "existing",
        "extra_metadata": {"keep": True},
    }
    manifest = root / "pairs.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return manifest, root


def test_merge_roots_validation_and_other_task_preserved(tmp_path):
    from open_vision_banana.training.trainer import ManifestVisionDataset

    manifest, base = existing(tmp_path / "base", task="normal")
    moved = tmp_path / "elsewhere.jsonl"
    manifest.rename(moved)
    root = tmp_path / "mixed"
    report = build_rea2seg(rows(sample()), root, merge=(moved, base))
    old, new = read(root)
    assert old["split"] == "val" and new["split"] == "train"
    assert old["task"] == "normal" and old["extra_metadata"] == {"keep": True}
    assert (root / old["target"]).read_bytes() == (base / "target.png").read_bytes()
    assert report["counts"] == {"val": 1, "train": 1}
    assert len(ManifestVisionDataset(root / "training.jsonl", 5, "val", {"normal"})) == 1
    assert len(ManifestVisionDataset(root / "training.jsonl", 5, "train", {"referring"})) == 1


@pytest.mark.parametrize("split", ["val", "test"])
def test_reencoded_heldout_overlap(tmp_path, split):
    exclusion = existing(tmp_path / "eval", split=split, color="red")
    with pytest.raises(ValueError, match="Held-out image overlap"):
        build_rea2seg(rows(sample()), tmp_path / "out", exclude=[exclusion])


def test_merge_cross_split_reasonseg_and_test_rejection(tmp_path):
    base = existing(tmp_path / "val", color="red")
    with pytest.raises(ValueError, match="Cross-split image"):
        build_rea2seg(rows(sample()), tmp_path / "cross", merge=base)
    base = existing(tmp_path / "train", split="train", color="red")
    with pytest.raises(ValueError, match="ReasonSeg image already"):
        build_rea2seg(rows(sample("reasonseg")), tmp_path / "dup", merge=base)
    test = existing(tmp_path / "test", split="test")
    with pytest.raises(ValueError, match="only explicit train/val"):
        build_rea2seg(rows(sample()), tmp_path / "test-out", merge=test)
    with pytest.raises(ValueError, match="Held-out"):
        build_rea2seg(
            rows(sample(color="green")),
            tmp_path / "val-test",
            merge=base,
            exclude=[existing(tmp_path / "test-red", split="test", color="red")],
        )


def test_parquet_cli_and_local_image_path(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    row = sample()
    (tmp_path / "source.png").write_bytes(row["image"]["bytes"])
    row["image"] = {"bytes": None, "path": "source.png"}
    parquet = tmp_path / "train-00000-of-00001.parquet"
    pq.write_table(pa.Table.from_pylist([row]), parquet)
    root = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_rea2seg.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--parquet",
            str(parquet),
            "--output-dir",
            str(root),
            "--max-samples",
            "1",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout)["rea2seg_rows"] == 1
    assert read(root)[0]["source_shard"] == parquet.name
    assert len(read(root)[0]["source_shard_sha256"]) == 64
    with pytest.raises(ValueError, match="distinct"):
        list(parquet_rows([parquet, parquet]))
    bad = tmp_path / "bad.parquet"
    pq.write_table(pa.Table.from_pylist([{"question": "only"}]), bad)
    with pytest.raises(ValueError, match="required fields"):
        list(parquet_rows([bad]))


def test_hf_cache_is_offline_and_complete(tmp_path, monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    snapshot = tmp_path / ("a" * 40)
    (snapshot / "data").mkdir(parents=True)
    shard = snapshot / "data" / "train-00000-of-00001.parquet"
    shard.touch()
    calls = []

    def cached(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(hub, "snapshot_download", cached)
    paths, revision = cached_hf_shards("source/dataset", "pinned")
    assert paths == [shard] and revision == "a" * 40
    assert calls == [
        {
            "repo_id": "source/dataset",
            "repo_type": "dataset",
            "revision": "pinned",
            "local_files_only": True,
        }
    ]
    shard.rename(snapshot / "data" / "train-00000-of-00002.parquet")
    with pytest.raises(ValueError, match="Incomplete"):
        cached_hf_shards("source/dataset", "pinned")
    with pytest.raises(ValueError, match="revision"):
        cached_hf_shards("source/dataset", "")


def test_import_does_not_load_optional_readers():
    module = Path(__file__).resolve().parents[1] / "src/open_vision_banana/rea2seg.py"
    assert importlib.util.find_spec("open_vision_banana.rea2seg") is not None
    code = (
        "import sys; from open_vision_banana import rea2seg; "
        "assert not {'pyarrow', 'datasets', 'huggingface_hub', 'pycocotools'} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=module.parents[2])


@pytest.mark.parametrize(
    "problem,match",
    [
        ("source_split", "cannot be relabeled"),
        ("size", "dimensions differ"),
        ("width", "width differs"),
        ("binary", "not white/black"),
        ("missing", "Missing asset"),
        ("incomplete", "incomplete dataset"),
    ],
)
def test_invalid_existing_data_is_not_silently_changed(tmp_path, problem, match):
    manifest, base = existing(tmp_path / "base")
    row = json.loads(manifest.read_text())
    if problem == "source_split":
        row["source_split"] = "test"
    elif problem == "size":
        Image.new("RGB", (4, 4)).save(base / "target.png")
    elif problem == "width":
        row["width"] = 4
    elif problem == "binary":
        row["encoding"] = "binary_white_black_v1"
        Image.new("RGB", (3, 2), "red").save(base / "target.png")
    elif problem == "missing":
        row["input"] = "missing.png"
    else:
        (base / "BUILD_INCOMPLETE").touch()
    manifest.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match=match):
        build_rea2seg(rows(sample()), tmp_path / "out", merge=(manifest, base))


def test_explicit_exclusion_keeps_existing_validation(tmp_path):
    merge = existing(tmp_path / "base")
    exclusion = existing(tmp_path / "eval", split="val")
    # Train entries in an evaluation manifest are deliberately not exclusions.
    with exclusion[0].open("a") as stream:
        stream.write(json.dumps({"input": "not-needed.png", "split": "train"}) + "\n")
    report = build_rea2seg(rows(sample()), tmp_path / "out", merge=merge, exclude=[exclusion])
    assert report["counts"] == {"val": 1, "train": 1}
    assert report["exclusion_rows"] == report["exclusion_images"] == 1


def test_same_query_with_conflicting_mask_fails(tmp_path):
    other = sample()
    other["mask"] = json.dumps({"rle": {"size": [2, 3], "counts": [0, 6]}})
    with pytest.raises(ValueError, match="Duplicate image/query"):
        build_rea2seg(rows(sample(), other), tmp_path / "out")


@pytest.mark.parametrize(
    "filename,coco_id",
    [("2408208.jpg", None), ("000000000042.jpg", 42), ("COCO_train2014_000000000042.jpg", 42)],
)
def test_parquet_coco_identity_does_not_confuse_visual_genome_ids(tmp_path, filename, coco_id):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    original = sample()
    original["image"]["path"] = filename
    shard = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist([original]), shard)
    converted = list(parquet_rows([shard]))
    assert converted[0][0].get("image_id") == coco_id
    assert converted[0][0]["source_image"] == filename
    if coco_id is not None:
        with pytest.raises(ValueError, match="Held-out COCO image id overlap"):
            build_rea2seg(converted, tmp_path / "out", exclude_image_ids=[coco_id])
    else:
        assert (
            build_rea2seg(converted, tmp_path / "out", exclude_image_ids=[2408208])["rea2seg_rows"]
            == 1
        )
