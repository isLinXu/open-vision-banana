"""Pinned supplement accounting and cross-task held-out protection."""

import json

import pytest
from PIL import Image

from open_vision_banana.data.hypersim_supplement import file_sha256
from open_vision_banana.data.manifest import fingerprint_records, read_jsonl, resolve_manifest_path
from scripts import merge_main_depth_reasoning as merger


def fixture_inputs(tmp_path, monkeypatch):
    sources = {}
    release = {}
    monkeypatch.setattr(
        merger,
        "SOURCES",
        {name: (1, task, source) for name, (_, task, source) in merger.SOURCES.items()},
    )
    for index, (name, (_, task, source)) in enumerate(merger.SOURCES.items()):
        folder = tmp_path / name
        folder.mkdir()
        Image.new("RGB", (16, 16), (index + 1, 20, 40)).save(folder / "input.png")
        Image.new("RGB", (16, 16), (255, 0, 0)).save(folder / "target.png")
        row = dict(
            task=task,
            split="train",
            sample_id=name,
            source_dataset=source,
            input="input.png",
            target="target.png",
            prompt="prompt",
            source_scene="ai_011_001",
        )
        path = folder / "pairs.jsonl"
        path.write_text(json.dumps(row) + "\n")
        sources[name] = path
        release[name] = {
            "manifest_sha256": file_sha256(path),
            "dataset_fingerprint": fingerprint_records([row], folder),
        }
    release_path = tmp_path / "release.json"
    release_path.write_text(json.dumps(release))
    exclude = tmp_path / "excluded.json"
    exclude.write_text("[999]")
    return sources, release_path, exclude


def test_supplement_merge_keeps_exact_counts_and_rebases(tmp_path, monkeypatch):
    sources, release, exclude = fixture_inputs(tmp_path, monkeypatch)
    out = tmp_path / "combined" / "pairs.jsonl"
    report = merger.merge(sources, out, release, exclude)
    assert report["task_split_counts"] == {"depth/train": 1, "referring/train": 2}
    assert report["per_task_dropped"] == {}
    assert report["zero_shot_claim_allowed"] is False
    for row in read_jsonl(out):
        assert resolve_manifest_path(row["input"], out).is_file()
    sources["rea2seg"].write_text("{}\n")
    with pytest.raises(ValueError, match="pinned release"):
        merger.merge(sources, out, release, exclude)


def test_missing_assets_fail_without_publishing_output(tmp_path, monkeypatch):
    sources, release, exclude = fixture_inputs(tmp_path, monkeypatch)
    (sources["hypersim"].parent / "target.png").unlink()
    out = tmp_path / "combined.jsonl"
    with pytest.raises(ValueError, match="Missing assets"):
        merger.merge(sources, out, release, exclude)
    assert not out.exists()


def test_main_join_preserves_heldout_order_and_rejects_cross_task_overlap(tmp_path, monkeypatch):
    sources, release, exclude = fixture_inputs(tmp_path, monkeypatch)
    rows = []
    for index, task in enumerate(sorted(merger.MAIN_TASKS)):
        name = f"train_{task}.png"
        Image.new("RGB", (16, 16), (index + 50, 50, 50)).save(tmp_path / name)
        rows.append(dict(task=task, split="train", input=name, target=name, prompt=task))
    for name, colour in [("z", 100), ("a", 150)]:
        Image.new("RGB", (16, 16), (colour, 30, 20)).save(tmp_path / f"{name}.png")
        rows.append(
            dict(
                task="depth", split="val", input=f"{name}.png", target=f"{name}.png", sample_id=name
            )
        )
    base = tmp_path / "base.jsonl"
    base.write_text("".join(json.dumps(r) + "\n" for r in rows))
    base.with_suffix(".audit.json").write_text(
        json.dumps(
            {
                "manifest_sha256": file_sha256(base),
                "rejected_non_metric_depth_rows": 0,
            }
        )
    )
    out = tmp_path / "main" / "pairs.jsonl"
    report = merger.merge(sources, out, release, exclude, base=base)
    assert report["train_heldout_overlap"] == 0
    assert [r["sample_id"] for r in read_jsonl(out) if r["split"] == "val"] == ["z", "a"]
    # A differently named image in a different task still leaks by decoded pixels.
    Image.open(tmp_path / "z.png").save(tmp_path / "train_semantic.png")
    original = out.read_bytes()
    with pytest.raises(ValueError, match="occur in both"):
        merger.merge(sources, out, release, exclude, base=base)
    assert out.read_bytes() == original


def test_coco_identity_overlap_is_detected_even_with_different_pixels(tmp_path):
    for index, name in enumerate(["000000000012.jpg", "COCO_val2014_000000000012.jpg"]):
        Image.new("RGB", (16, 16), (index * 100, 0, 0)).save(tmp_path / name)
    rows = [
        dict(input="000000000012.jpg", split="train"),
        dict(input="COCO_val2014_000000000012.jpg", split="val"),
    ]
    with pytest.raises(ValueError, match="COCO IDs"):
        merger.assert_no_heldout_overlap(rows, tmp_path / "pairs.jsonl")


@pytest.mark.parametrize(
    "extra,match",
    [
        ({"image_id": 999}, "Held-out COCO image in supplement"),  # Visual Genome mapped to COCO
        (
            {"source_image": "cityscapes_frankfurt_000001_035864_leftImg8bit.png"},
            "Benchmark-source row in supplement",
        ),
    ],
)
def test_rea2seg_source_identity_reaches_the_heldout_gates(tmp_path, monkeypatch, extra, match):
    sources, release, exclude = fixture_inputs(tmp_path, monkeypatch)
    path = sources["rea2seg"]
    row = dict(json.loads(path.read_text()), **extra)
    path.write_text(json.dumps(row) + "\n")
    pinned = json.loads(release.read_text())
    pinned["rea2seg"] = {
        "manifest_sha256": file_sha256(path),
        "dataset_fingerprint": fingerprint_records([row], path.parent),
    }
    release.write_text(json.dumps(pinned))
    with pytest.raises(ValueError, match=match):
        merger.merge(sources, tmp_path / "combined.jsonl", release, exclude)
