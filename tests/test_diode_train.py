"""Synthetic native conversion/split/integrity tests; no dataset download or GPU."""

import io
import json
import tarfile

import numpy as np
import pytest
from PIL import Image

from open_vision_banana.data.diode_train import (
    DiodeSource,
    array_bytes,
    convert_pair,
    diode_ids,
    excluded_ids,
    finalize,
    immutable_write,
    native_plane,
    prepare_arrays,
)


def arrays():
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    depth = np.array([[0.1, 2, 80], [np.nan, -1, 7]], np.float32)
    mask = np.array([[1, 1, 1], [1, 1, 0]], np.uint8)
    return rgb, depth, mask


def test_all_native_measurements_no_observed98_gates():
    metric, valid, stats = prepare_arrays(*arrays())
    np.testing.assert_array_equal(metric, np.array([[0.1, 2, 80], [0, 0, 0]], np.float32))
    assert metric.dtype == np.float32
    assert stats["valid_fraction"] == 0.5
    assert stats["depth_max_m"] == 80
    assert not valid[1].any()


@pytest.mark.parametrize("change", ["shape", "rgb", "nonbinary", "nanmask"])
def test_reject_invalid_arrays(change):
    rgb, depth, mask = arrays()
    if change == "shape":
        depth = depth.T
    elif change == "rgb":
        rgb = rgb.astype(float)
    elif change == "nonbinary":
        mask[0, 0] = 2
    else:
        mask = mask.astype(float)
        mask[0, 0] = np.nan
    with pytest.raises(ValueError):
        prepare_arrays(rgb, depth, mask)


def test_all_invalid_still_retained():
    rgb, depth, mask = arrays()
    metric, valid, stats = prepare_arrays(rgb, depth, mask * 0)
    assert not metric.any() and not valid.any()
    assert stats["depth_min_m"] is None


def test_conversion_integrity_resume_and_corruption(tmp_path):
    rgb, depth, mask = arrays()
    provenance = dict(
        source_dataset="diode_indoor",
        source_id="sample",
        depth_provenance="source_valid_measurements",
    )
    record = convert_pair(tmp_path, "diode_sample", rgb, depth, mask, provenance)
    row = record["row"]
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / row["input"])), rgb)
    metric = np.load(tmp_path / row["metric_depth"], allow_pickle=False)
    assert metric[0, 2] == 80 and metric.dtype == np.float32
    target = np.asarray(Image.open(tmp_path / row["target"]))
    assert not target[1].any()
    assert row["loss_mask_applied"] is False
    assert row["encoding_max_error_m"] is not None
    assert convert_pair(tmp_path, "diode_sample", rgb, depth, mask, provenance, True) == record
    with pytest.raises(ValueError):
        convert_pair(tmp_path, "diode_sample", rgb, depth, mask, provenance)
    report = finalize(tmp_path, [record], {})
    assert report["output_assets"] == 4
    assert finalize(tmp_path, [record], {}, True) == report
    (tmp_path / row["valid_mask"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        finalize(tmp_path, [record], {}, True)
    with pytest.raises(ValueError, match="Existing output"):
        convert_pair(tmp_path, "diode_sample", rgb, depth, mask, provenance, True)


def test_immutable_does_not_overwrite(tmp_path):
    path = tmp_path / "file"
    immutable_write(path, b"first")
    with pytest.raises(ValueError):
        immutable_write(path, b"second", True)
    assert path.read_bytes() == b"first"


def metadata():
    return dict(
        train=dict(indoors=dict(scene=dict(scan=["a", "b"]))),
        val=dict(indoors=dict(other=dict(scan=["v"]))),
    )


def test_diode_metadata_duplicates_and_leakage():
    allowed, heldout = diode_ids(metadata())
    assert allowed["a"] == ("scene", "scan", "a") and heldout == {"v"}
    m = metadata()
    m["train"]["indoors"]["scene"]["scan"].append("a")
    with pytest.raises(ValueError):
        diode_ids(m)
    m = metadata()
    m["val"]["indoors"]["other"]["scan"] = ["a"]
    with pytest.raises(ValueError):
        diode_ids(m)


def add_triplet(root, stem="a", prefix="train/indoors/scene/scan"):
    path = root / prefix
    path.mkdir(parents=True, exist_ok=True)
    rgb, depth, mask = arrays()
    (path / f"{stem}.png").write_bytes(array_bytes(rgb, True))
    (path / f"{stem}_depth.npy").write_bytes(array_bytes(depth[..., None]))
    (path / f"{stem}_depth_mask.npy").write_bytes(array_bytes(mask))


def test_diode_directory_triplets(tmp_path):
    add_triplet(tmp_path)
    add_triplet(tmp_path, "v", "val/indoors/other/scan")
    allowed, heldout = diode_ids(metadata())
    source = DiodeSource(tmp_path)
    jobs = source.jobs(allowed, heldout)
    assert list(jobs) == ["a"]
    # Wrong dimensions must not silently resize into the official native format.
    with pytest.raises(ValueError, match="geometry"):
        list(source.samples(jobs, "metadatahash"))
    add_triplet(tmp_path, "v")
    with pytest.raises(ValueError, match="Non-train"):
        DiodeSource(tmp_path).jobs(allowed, heldout)


def test_diode_duplicate_and_incomplete(tmp_path):
    add_triplet(tmp_path)
    add_triplet(tmp_path, prefix="wrapper/train/indoors/scene/scan")
    allowed, heldout = diode_ids(metadata())
    with pytest.raises(ValueError, match="Duplicate"):
        DiodeSource(tmp_path).jobs(allowed, heldout)
    (tmp_path / "train/indoors/scene/scan/a_depth.npy").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        DiodeSource(tmp_path).jobs(allowed, heldout)


@pytest.mark.parametrize("name", ["../escape.png", "/absolute.png"])
def test_unsafe_tar_rejected(tmp_path, name):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as handle:
        item = tarfile.TarInfo(name)
        item.size = 1
        handle.addfile(item, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe"):
        DiodeSource(archive)


def test_exclude_pr104_and_new_source_ids(tmp_path):
    path = tmp_path / "pairs.jsonl"
    rows = [
        dict(sample_id="diode_a"),
        dict(source_dataset="diode_indoor", source_rgb="train/indoors/scene/scan/b.png"),
        dict(source_dataset="other", source_id="a"),
    ]
    path.write_text("\n".join(map(json.dumps, rows)))
    assert excluded_ids([path]) == {"diode_a", "diode_b"}


def test_native_plane_preserves_singleton_spatial_axis():
    assert native_plane(np.zeros((1, 3, 1))).shape == (1, 3)
    with pytest.raises(ValueError):
        native_plane(np.zeros((3,)))


def test_tar_and_directory_inventory_agree(tmp_path):
    root = tmp_path / "raw"
    add_triplet(root)
    archive = tmp_path / "train.tar"
    with tarfile.open(archive, "w") as handle:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                handle.add(path, arcname=path.relative_to(root).as_posix())
    allowed, heldout = diode_ids(metadata())
    source = DiodeSource(archive)
    try:
        jobs = source.jobs(allowed, heldout)
        assert jobs == DiodeSource(root).jobs(allowed, heldout)
        for name in jobs["a"]:
            assert source.read(name) == (root / name).read_bytes()
    finally:
        source.close()


def test_encoding_gate_rejects_before_writing_and_accepts_boundary(tmp_path):
    from open_vision_banana.data.diode_train import EncodingQualityError

    rgb, depth, mask = arrays()
    baseline = convert_pair(tmp_path / "baseline", "diode_sample", rgb, depth, mask, {})
    error = baseline["row"]["encoding_max_error_m"]
    assert error > 0
    rejected_root = tmp_path / "rejected"
    with pytest.raises(EncodingQualityError) as caught:
        convert_pair(
            rejected_root, "diode_sample", rgb, depth, mask, {}, max_encoding_error_m=error / 2
        )
    assert not rejected_root.exists()
    assert caught.value.stats["encoding_max_error_m"] == error
    assert caught.value.stats["valid_fraction"] == 0.5
    accepted = convert_pair(
        tmp_path / "accepted", "diode_sample", rgb, depth, mask, {}, max_encoding_error_m=error
    )
    assert accepted == baseline


@pytest.mark.parametrize("cap", [-1, float("nan"), float("inf")])
def test_encoding_gate_invalid_threshold(tmp_path, cap):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        convert_pair(tmp_path / "out", "diode_sample", *arrays(), {}, max_encoding_error_m=cap)
    assert not (tmp_path / "out").exists()
