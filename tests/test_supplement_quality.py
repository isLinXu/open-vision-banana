"""Exhaustive crop safety, image-level propagation and pinned merge exclusions."""

import io
import json

import numpy as np
import pytest
from PIL import Image

from open_vision_banana.data.hypersim_supplement import file_sha256
from open_vision_banana.data.manifest import fingerprint_records, read_jsonl
from open_vision_banana.data.supplement_quality import excluded_images, minimum_crop_foreground


@pytest.mark.parametrize("size", [(19, 7), (7, 19), (9, 9), (40, 13)])
def test_all_crop_minimum_matches_brute_force_existing_resize(size):
    rng = np.random.default_rng(42)
    width, height = size
    mask = Image.fromarray((rng.random((height, width)) < 0.04).astype(np.uint8) * 255)
    resolution = 8
    scale = resolution / min(size)
    resized = mask.resize(
        (max(resolution, round(width * scale)), max(resolution, round(height * scale))),
        Image.Resampling.NEAREST,
    )
    expected = min(
        int(np.count_nonzero(np.asarray(resized.crop((x, y, x + resolution, y + resolution)))))
        for x in range(resized.width - resolution + 1)
        for y in range(resized.height - resolution + 1)
    )
    assert minimum_crop_foreground(mask, resolution) == expected


def test_boundary_disconnected_and_vanishing_targets():
    mask = np.zeros((8, 16), dtype=np.uint8)
    mask[:, 0] = 255
    assert minimum_crop_foreground(Image.fromarray(mask), 8) == 0
    mask[:, 7:9] = 255
    assert minimum_crop_foreground(Image.fromarray(mask), 8) == 8
    tiny = np.zeros((16, 32), dtype=np.uint8)
    tiny[0, 0] = 255
    assert minimum_crop_foreground(Image.fromarray(tiny), 4) == 0
    with pytest.raises(ValueError, match="binary"):
        minimum_crop_foreground(Image.new("L", (8, 8), 128), 8)


def test_image_exclusion_propagates_across_sources_and_coco_aliases():
    records = [
        dict(rgb_pixels_sha256="a", coco_image_ids=[1], minimum_crop_foreground_pixels=0),
        dict(rgb_pixels_sha256="b", coco_image_ids=[1], minimum_crop_foreground_pixels=1),
        dict(rgb_pixels_sha256="b", coco_image_ids=[2], minimum_crop_foreground_pixels=1),
        dict(rgb_pixels_sha256="c", coco_image_ids=[2], minimum_crop_foreground_pixels=1),
        dict(rgb_pixels_sha256="d", coco_image_ids=[3], minimum_crop_foreground_pixels=1),
    ]
    assert excluded_images(list(reversed(records))) == ({"a", "b", "c"}, {1, 2})


def test_merger_applies_pinned_filter_before_counting(tmp_path, monkeypatch):
    from scripts import merge_main_depth_reasoning as merger
    from tests.test_main_depth_reasoning_merge import fixture_inputs

    sources, release_path, exclude = fixture_inputs(tmp_path, monkeypatch)
    exclude.write_text("[999, 1000]")
    rules = {"resolution": 512, "sources": {}}
    for name in ("rea2seg", "llmseg"):
        kept = [] if name == "rea2seg" else read_jsonl(sources[name])
        rules["sources"][name] = {
            "input_manifest_sha256": file_sha256(sources[name]),
            "retained_rows": len(kept),
            "retained_fingerprint": fingerprint_records(kept, sources[name].parent),
            "excluded_sample_ids": [name] if name == "rea2seg" else [],
        }
    filter_path = tmp_path / "crop-filter.json"
    filter_path.write_text(json.dumps(rules))
    release = json.loads(release_path.read_text())
    release["referring_filter"] = {"path": str(filter_path), "sha256": file_sha256(filter_path)}
    release_path.write_text(json.dumps(release))
    output = tmp_path / "combined" / "pairs.jsonl"
    report = merger.merge(sources, output, release_path, exclude)
    assert report["supplement_rows"] == 2
    assert report["excluded_coco_image_ids"] == 2
    assert report["source_rows_after_ref_filter"]["rea2seg"] == 0
    assert {r["sample_id"] for r in read_jsonl(output)} == {"hypersim", "llmseg"}
    filter_path.write_text("{}")
    with pytest.raises(ValueError, match="quality filter differs"):
        merger.merge(sources, output, release_path, exclude)


def test_depth_filter_removes_a_large_pixel_error_hidden_by_the_mean(tmp_path):
    import h5py
    import torch

    from open_vision_banana.data.hypersim_supplement import prepare_frame, write_json
    from scripts.filter_depth_reasoning_supplement import filter_depth
    from scripts.package_hypersim_supplement import validate_native

    torch.set_num_threads(1)
    source, output = tmp_path / "initial", tmp_path / "filtered"
    records = []
    for frame in (0, 2):
        rgb, hdf = io.BytesIO(), io.BytesIO()
        Image.new("RGB", (64, 48), (100 + frame, 120, 140)).save(rgb, format="JPEG")
        distance = np.full((48, 64), 4, dtype=np.float32)
        if frame:
            distance[20, 20] = 60
        with h5py.File(hdf, "w") as handle:
            handle.create_dataset("dataset", data=distance)
        records.append(
            prepare_frame(
                dict(
                    scene_name="ai_011_001",
                    camera_name="cam_00",
                    frame_id=frame,
                    source_split="train",
                ),
                rgb.getvalue(),
                hdf.getvalue(),
                source,
            )
        )
    assert records[1]["roundtrip_error_sum"] / records[1]["valid_pixels"] < 0.01
    assert records[1]["roundtrip_error_max"] > 0.05
    for name, rows in (("pairs.jsonl", [r["row"] for r in records]), ("provenance.jsonl", records)):
        (source / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    (source / "rejected.jsonl").write_text("")
    write_json(
        source / "manifest.json",
        {
            "pairs_written": 2,
            "manifest_sha256": file_sha256(source / "pairs.jsonl"),
            "provenance_sha256": file_sha256(source / "provenance.jsonl"),
            "source_fingerprint": "test-initial",
            "quality_policy": {},
            "selection": {"count": 2},
        },
    )
    report = filter_depth(source, output, expected_initial_count=2)
    assert report["pairs_written"] == 1 and report["pre_filter"]["removed_max_error"] == 1
    assert report["data_contract"]["encoded_roundtrip_abs_error"]["max"] <= 0.05
    assert not (output / records[1]["row"]["input"]).exists()
    assert (source / records[1]["row"]["input"]).is_file()
    assert validate_native(output, workers=1)["assets_decoded"] == 3
