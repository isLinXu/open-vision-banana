"""A Hub release restores the exact portable assets and refuses corrupt input."""

import hashlib
import io
import json
import tarfile

import pytest

from open_vision_banana.res_bundle import package_dataset, restore_bundle, safe_path, sha256_file


def fixture_dataset(root):
    root.mkdir()
    (root / "images").mkdir()
    (root / "targets").mkdir()
    (root / "images/a.jpg").write_bytes(b"image bytes")
    (root / "targets/a.png").write_bytes(b"mask bytes")
    (root / "pairs.jsonl").write_text(
        json.dumps({"input": "images/a.jpg", "target": "targets/a.png"}) + "\n"
    )
    assets = [
        {"path": name, "bytes": (root / name).stat().st_size, "sha256": sha256_file(root / name)}
        for name in ("images/a.jpg", "targets/a.png")
    ]
    (root / "assets.jsonl").write_text("".join(json.dumps(a) + "\n" for a in assets))
    files = {
        name: {"bytes": (root / name).stat().st_size, "sha256": sha256_file(root / name)}
        for name in ("pairs.jsonl", "assets.jsonl")
    }
    fp = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (root / "build_report.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "files": files,
                "dataset_fingerprint": fp,
                "counts": {"train": 1},
            }
        )
    )
    (root / "validation_report.json").write_text(
        json.dumps({"status": "passed", "dataset_fingerprint": fp, "counts": {"train": 1}})
    )


def test_roundtrip_and_idempotent_restore(tmp_path):
    data, bundle, restored = (tmp_path / n for n in ("data", "bundle", "restored"))
    fixture_dataset(data)
    index = package_dataset(data, bundle, 4)
    assert index["asset_files"] == 2
    assert restore_bundle(bundle, restored)["asset_files"] == 2
    restore_bundle(bundle, restored)
    for path in data.rglob("*"):
        if path.is_file():
            assert (restored / path.relative_to(data)).read_bytes() == path.read_bytes()
    (bundle / "pairs.jsonl").write_text("corrupt")
    with pytest.raises(ValueError, match="Checksum"):
        restore_bundle(bundle, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "path", ["../escape", "/absolute", "C:/drive", "a\\..\\b", "a/../b", "a//b"]
)
def test_unsafe_path(tmp_path, path):
    with pytest.raises(ValueError, match="Unsafe"):
        safe_path(tmp_path, path)


def test_tar_traversal_rejected_even_if_index_matches(tmp_path):
    data, bundle = tmp_path / "data", tmp_path / "bundle"
    fixture_dataset(data)
    index = package_dataset(data, bundle, 1024)
    entry = next(x for x in index["files"] if x["kind"] == "tar")
    archive = bundle / entry["path"]
    with tarfile.open(archive, "w") as tar:
        member = tarfile.TarInfo("../escaped.txt")
        member.size = 1
        tar.addfile(member, io.BytesIO(b"x"))
    entry["sha256"], entry["size"] = sha256_file(archive), archive.stat().st_size
    (bundle / "bundle.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="Unsafe"):
        restore_bundle(bundle, tmp_path / "restored")
    assert not (tmp_path / "escaped.txt").exists()
    assert (tmp_path / "restored/BUILD_INCOMPLETE").is_file()


def test_package_excludes_unrelated_json_and_rejects_stale_validation(tmp_path):
    data = tmp_path / "data"
    fixture_dataset(data)
    (data / "unrelated.json").write_text('{"private": true}')
    bundle = tmp_path / "bundle"
    package_dataset(data, bundle, 1024)
    assert not (bundle / "unrelated.json").exists()
    (data / "images/a.jpg").write_bytes(b"changed after validation")
    with pytest.raises(ValueError, match="Validated asset changed"):
        package_dataset(data, tmp_path / "changed", 1024)


def test_package_rejects_incomplete_build(tmp_path):
    data = tmp_path / "data"
    fixture_dataset(data)
    (data / "BUILD_INCOMPLETE").touch()
    with pytest.raises(ValueError, match="incomplete"):
        package_dataset(data, tmp_path / "bundle", 1024)
