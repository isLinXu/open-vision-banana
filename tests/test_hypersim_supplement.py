"""Supplement selection, metric geometry and the existing training contract."""

import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.data.hypersim_supplement import (
    decode_quantized_rgb,
    file_sha256,
    frame_id,
    prepare_frame,
    select_candidates,
)
from open_vision_banana.data.manifest import resolve_manifest_path
from open_vision_banana.training.trainer import ManifestVisionDataset, _collate_fn
from scripts.merge_multitask_manifest import merge_manifests


def test_palette_decode_equals_full_pixel_decoder():
    torch.set_num_threads(1)
    mapper = DepthRGBMapper()
    depth = torch.linspace(0, 500, 16384).reshape(1, 1, 128, 128)
    rgb = (mapper.depth_to_rgb(depth)[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    expected = mapper.rgb_to_depth(
        torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1)[None]
    )[0, 0].numpy()
    np.testing.assert_array_equal(decode_quantized_rgb(rgb, mapper), expected)


def metadata(rows):
    return (
        "scene_name,camera_name,frame_id,included_in_public_release,exclude_reason,split_partition_name\n"
        + "\n".join(rows)
        + "\n"
    )


def test_selection_excludes_original_volumes_nonpublic_heldout_and_adjacent_frames():
    rows = [
        "ai_010_099,cam_00,0,True,,train",
        "ai_011_001,cam_00,0,True,,train",
        "ai_011_001,cam_00,1,True,,train",
        "ai_011_001,cam_00,2,True,,train",
        "ai_011_002,cam_00,0,True,,train",
        "ai_011_003,cam_00,0,False,person,train",
        "ai_011_004,cam_00,0,True,,val",
        "ai_011_005,cam_00,0,True,,test",
    ]
    selected = select_candidates(metadata(rows))
    assert selected == select_candidates(metadata(list(reversed(rows))))
    assert len(selected) == 3
    assert len({r["scene_name"] for r in selected[:2]}) == 2
    assert all(r["frame_id"] % 2 == 0 for r in selected)
    assert not any(r["scene_name"].startswith("ai_010") for r in selected)


def test_duplicate_or_unsafe_source_identity_fails():
    line = "ai_011_001,cam_00,0,True,,train"
    with pytest.raises(ValueError, match="Duplicate"):
        select_candidates(metadata([line, line]))
    with pytest.raises(ValueError, match="identity"):
        frame_id({"scene_name": "../escape", "camera_name": "cam_00", "frame_id": 0})


def make_frame(tmp_path: Path, *, shape=(48, 64), bad=False):
    import h5py

    rgb = io.BytesIO()
    Image.new("RGB", (64, 48), (100, 120, 140)).save(rgb, format="JPEG")
    raw = np.full(shape, 4.0, dtype=np.float32)
    raw[0, 0] = np.nan
    if bad:
        raw[:] = np.nan
    depth = io.BytesIO()
    with h5py.File(depth, "w") as handle:
        handle.create_dataset("dataset", data=raw)
    row = {
        "scene_name": "ai_011_001",
        "camera_name": "cam_00",
        "frame_id": 0,
        "source_split": "train",
    }
    return prepare_frame(row, rgb.getvalue(), depth.getvalue(), tmp_path)


def test_native_metric_conversion_and_existing_loader(tmp_path):
    torch.set_num_threads(1)
    record = make_frame(tmp_path)
    row = record["row"]
    planar = np.load(tmp_path / row["metric_depth"])
    assert planar.dtype == np.float32 and planar.shape == (48, 64)
    assert np.isnan(planar[0, 0])
    assert planar[24, 32] == pytest.approx(4, abs=0.002)
    assert planar[-1, -1] < 3.5
    assert record["invalid_pixels"] == 1
    assert record["roundtrip_error_sum"] / record["valid_pixels"] < 0.01
    for asset in record["assets"]:
        assert file_sha256(tmp_path / asset["path"]) == asset["sha256"]
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(row) + "\n")
    dataset = ManifestVisionDataset(path, 32, "train", {"depth"}, aspect_preserving_crop=True)
    batch = _collate_fn([dataset[0]])
    assert batch["image"].shape == (1, 3, 32, 32)
    assert batch["task"] == ["depth"]
    assert torch.isfinite(batch["target_rgb"]).all()
    # Native metric references survive the same merger used by the six-task run.
    merged = tmp_path / "nested" / "pairs.jsonl"
    report = merge_manifests([path], merged, val_percent=0)
    assert report["per_task_dropped"] == {}
    merged_row = json.loads(merged.read_text())
    assert (
        resolve_manifest_path(merged_row["metric_depth"], merged).resolve()
        == (tmp_path / row["metric_depth"]).resolve()
    )


def test_bad_depth_does_not_produce_training_assets(tmp_path):
    with pytest.raises(ValueError, match="50%"):
        make_frame(tmp_path, bad=True)
    assert not (tmp_path / "images").exists()
    with pytest.raises(ValueError, match="size mismatch"):
        make_frame(tmp_path, shape=(24, 32))


def test_main_extension_keeps_six_tasks_and_half_generation():
    from open_vision_banana.training.config import load_config

    root = Path(__file__).resolve().parents[1]
    path = root / "configs/experiments/qwen_image_21_main_depth_reasoning.yaml"
    config = load_config(config_path=path)
    assert set(config.tasks) == {
        "depth",
        "normal",
        "semantic",
        "instance",
        "referring",
        "generation",
    }
    assert config.vision_ratio == 0.5 and config.lr == 3e-5
    assert config.resume_weights_only and config.resume is None
    assert config.lora_rank == 32 and config.lora_alpha == 64
    raw = yaml.safe_load(path.read_text())
    assert raw["data"]["pair_manifest"] == "data/qwen_main_depth_reasoning/pairs.jsonl"


def test_build_tops_up_quality_and_duplicate_rejections(tmp_path, monkeypatch):
    pytest.importorskip("remotezip")
    from open_vision_banana.data.hypersim_supplement import sha256, write_json
    from scripts import prepare_hypersim_supplement as builder

    candidates = [
        dict(scene_name="ai_011_001", camera_name="cam_00", frame_id=i * 2, source_split="train")
        for i in range(4)
    ]
    meta = tmp_path / "metadata.csv"
    meta.write_bytes(b"test metadata")
    monkeypatch.setattr(builder, "METADATA_SHA256", sha256(meta.read_bytes()))
    monkeypatch.setattr(builder, "select_candidates", lambda _: candidates)

    def prepare_scene(scene, rows, output):
        for row in rows:
            record = make_frame(output)
            identity = frame_id(row)
            record["source"] = row
            record["row"].update(sample_id=identity, source_frame=row["frame_id"])
            record["rgb_pixels_sha256"] = "same" if row["frame_id"] in (2, 4) else identity
            if row["frame_id"] == 0:
                record["roundtrip_error_sum"] = record["valid_pixels"] * 0.1
            write_json(output / "records" / f"{identity}.json", record)

    monkeypatch.setattr(builder, "process_scene", prepare_scene)
    output = tmp_path / "prepared"
    report = builder.build(output, count=2, workers=1, metadata=meta)
    assert report["pairs_written"] == 2 and report["rejected_frames"] == 2
    for name in ("pairs.jsonl", "manifest.json", "provenance.jsonl", "rejected.jsonl"):
        assert b"\r\n" not in (output / name).read_bytes()
    assert len(report["source_fingerprint"]) == 64
    assert report["data_contract"]["encoded_roundtrip_abs_error"]["mean"] < 0.01
    assert not (output / "BUILD_INCOMPLETE").exists()
    with pytest.raises(ValueError, match="specification changed"):
        builder.build(output, count=1, workers=1, metadata=meta)


def test_native_package_and_compressed_restore_keep_metric_assets(tmp_path):
    from open_vision_banana.data.hypersim_supplement import write_json
    from open_vision_banana.res_bundle import restore_bundle
    from scripts.package_hypersim_supplement import package

    root = tmp_path / "data"
    record = make_frame(root)
    for name, values in [("pairs.jsonl", [record["row"]]), ("provenance.jsonl", [record])]:
        (root / name).write_text("".join(json.dumps(v) + "\n" for v in values))
    (root / "rejected.jsonl").write_text("")
    for name in ("selection_spec.json", "quality_policy.json"):
        write_json(root / name, {})
    write_json(
        root / "manifest.json",
        {
            "pairs_written": 1,
            "manifest_sha256": file_sha256(root / "pairs.jsonl"),
            "provenance_sha256": file_sha256(root / "provenance.jsonl"),
        },
    )
    bundle, restored = tmp_path / "bundle", tmp_path / "restored"
    index = package(root, bundle, workers=1, shard_mb=1)
    assert index["encoding"] == "metric_power_cube_edge_v1"
    assert all(e["path"].endswith(".tar.gz") for e in index["files"] if e["kind"] == "tar")
    assert restore_bundle(bundle, restored)["asset_files"] == 3
    for asset in record["assets"]:
        assert file_sha256(restored / asset["path"]) == asset["sha256"]
    assert package(root, bundle, workers=1, shard_mb=1) == index
    (root / record["row"]["metric_depth"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Changed asset"):
        package(root, tmp_path / "corrupt", workers=1)
