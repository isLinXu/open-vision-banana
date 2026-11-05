import json

import numpy as np
import pytest
from PIL import Image

from open_vision_banana.data.indoor_depth import clean, orient, save_pair
from open_vision_banana.training.trainer import ManifestVisionDataset


def frame():
    rng = np.random.default_rng(42)
    rgb = rng.integers(25, 220, (100, 100, 3), dtype=np.uint8)
    depth = np.broadcast_to(np.linspace(1, 4, 100, dtype=np.float32), (100, 100)).copy()
    valid = np.ones(depth.shape, bool)
    valid[:2] = False
    depth[~valid] = 0
    return rgb, depth, valid


def test_exact_98_percent_boundary():
    rgb, depth, valid = frame()
    _, observed, stats = clean(rgb, depth, valid)
    assert observed.sum() == 9800
    assert stats["valid_fraction"] == 0.98
    valid[2, 0] = False
    with pytest.raises(ValueError, match="insufficient_observed"):
        clean(rgb, depth, valid)


def test_missing_measurements_are_never_filled(tmp_path):
    rgb, depth, valid = frame()
    record = save_pair(
        tmp_path, "test", rgb, depth, valid, dict(split="train", source_dataset="test")
    )
    row = record["row"]
    stored = np.array(Image.open(tmp_path / row["metric_depth"]))
    target = np.array(Image.open(tmp_path / row["target"]))
    assert np.all(stored[~valid] == 0)
    assert np.all(target[~valid] == 0)
    assert np.max(np.abs(stored[valid] / 1000 - depth[valid])) < 0.00051
    assert np.array_equal(np.array(Image.open(tmp_path / row["valid_mask"])) > 0, valid)


def test_nonfinite_and_out_of_range_are_invalid():
    rgb, depth, _ = frame()
    valid = np.ones(depth.shape, bool)
    depth[:2] = np.nan
    _, observed, _ = clean(rgb, depth, valid)
    assert not observed[:2].any()
    depth[2, 0] = 21
    with pytest.raises(ValueError, match="insufficient_observed"):
        clean(rgb, depth, valid)


def test_flat_blurred_and_dark_are_rejected():
    rgb, depth, valid = frame()
    with pytest.raises(ValueError, match="flat_or_distant"):
        clean(rgb, np.ones_like(depth), np.ones_like(valid))
    for value in (0, 100, 255):
        with pytest.raises(ValueError, match="blurred"):
            clean(np.full_like(rgb, value), depth, valid)


def test_orientation_matches_apples_reference():
    a = np.arange(12, dtype=np.uint16).reshape(3, 4)
    assert np.array_equal(orient(a, "Left"), np.rot90(a, -1))
    assert np.array_equal(orient(a, "Right"), np.rot90(a, 1))
    assert np.array_equal(orient(a, "Down"), np.rot90(a, 2))
    with pytest.raises(ValueError):
        orient(a, "unknown")


def test_unchanged_native_loader_accepts_the_pairs(tmp_path):
    rgb, depth, valid = frame()
    row = save_pair(
        tmp_path, "test", rgb, depth, valid, dict(split="train", source_dataset="arkitscenes")
    )["row"]
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    ds = ManifestVisionDataset(path, 128, "train", {"depth"}, data_root=tmp_path)
    item = ds[0]
    assert item["image"].shape == item["target_rgb"].shape == (3, 128, 128)
    # Validity masks are provenance; the existing trainer has no masked-loss contract.
    assert "valid_mask" not in item


def test_selection_validation_and_existing_bundle_restore(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    from open_vision_banana.data.indoor_depth import POLICY
    from open_vision_banana.res_bundle import restore_bundle

    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    (data / "records").mkdir(parents=True)
    rgb, depth, valid = frame()
    for i, split in enumerate(("train", "val")):
        image = rgb if i == 0 else np.flip(rgb, axis=1).copy()
        record = save_pair(
            data,
            f"test{i}",
            image,
            depth,
            valid,
            dict(
                split=split,
                source_dataset="arkitscenes",
                source_scene=str(i),
                source_video=str(i),
                source_split="Training" if split == "train" else "Validation",
                timestamp_error_seconds=0.0,
                policy=POLICY,
            ),
        )
        video = dict(
            video=str(i),
            scene=str(i),
            fold=record["row"]["source_split"],
            accepted=[record],
            rejected=[],
            candidate_count=1,
        )
        (data / "records" / f"{i}.json").write_text(json.dumps(video), encoding="utf-8")
    (data / "acquisition_plan.json").write_text(
        json.dumps(dict(policy=POLICY, metadata_sha256="test", partial=True)), encoding="utf-8"
    )
    (data / "ACQUISITION_COMPLETE").touch()
    incomplete = subprocess.run(
        [sys.executable, str(root / "scripts/finalize_indoor_depth.py"), "--source", str(data)],
        capture_output=True,
        text=True,
    )
    assert incomplete.returncode != 0
    assert "partial acquisition plan" in incomplete.stderr
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/finalize_indoor_depth.py"),
            "--source",
            str(data),
            "--development-partial",
        ],
        check=True,
        capture_output=True,
    )
    bundle = tmp_path / "bundle"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/validate_package_indoor_depth.py"),
            str(data),
            "--bundle-dir",
            str(bundle),
            "--workers",
            "1",
        ],
        check=True,
        capture_output=True,
    )
    restored = tmp_path / "restored"
    result = restore_bundle(bundle, restored)
    assert result["asset_files"] == 8
    rows = [
        json.loads(x) for x in (restored / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        for key in ("input", "target", "metric_depth", "valid_mask"):
            assert (restored / row[key]).read_bytes() == (data / row[key]).read_bytes()


def test_zip_range_reader_checks_content_and_crc():
    import io
    import zipfile

    from scripts.prepare_arkitscenes_indoor import read_member

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("confidence/test.png", b"observations")
    blob = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        info = z.getinfo("confidence/test.png")

    class Response:
        status_code = 206

        def __init__(self, part, start):
            self.headers = {"Content-Range": f"bytes {start}-{start + len(part) - 1}/{len(blob)}"}
            self.raw = io.BytesIO(part)

        def raise_for_status(self):
            pass

        def close(self):
            pass

    class Client:
        def __init__(self, corrupt=False):
            self.corrupt = corrupt

        def get(self, url, headers, **kwargs):
            assert headers["If-Match"] == '"fixed"' and kwargs["stream"]
            start, end = map(int, headers["Range"][6:].split("-"))
            payload = bytearray(blob)
            if self.corrupt:
                payload[30 + len(info.filename)] += 1
            return Response(bytes(payload[start : end + 1]), start)

    archive = dict(url="https://example.test/asset.zip", etag='"fixed"', bytes=len(blob))
    assert read_member(Client(), archive, info) == b"observations"
    with pytest.raises(ValueError, match="CRC"):
        read_member(Client(True), archive, info)


def test_sun_sensor_bit_rotation_roundtrip():
    from scripts.prepare_sunrgbd_indoor import decode_sensor_mm

    mm = np.array([0, 1, 250, 1000, 8191, 8192, 20000, 65535], dtype=np.uint32)
    encoded = (((mm << 3) | (mm >> 13)) & 65535).astype(np.uint16)
    assert np.array_equal(decode_sensor_mm(encoded), mm)


def test_structure_gate_suppresses_noise_on_blank_surfaces(tmp_path):
    from scripts.finalize_indoor_depth import content_statistics

    rng = np.random.default_rng(42)
    noisy_wall = np.clip(128 + rng.normal(0, 4, (512, 512)), 0, 255).astype(np.uint8)
    yy, xx = np.indices((512, 512))
    structured = (40 + 160 * ((xx // 64 + yy // 64) % 2)).astype(np.uint8)
    for name, pixels in (("wall", noisy_wall), ("structured", structured)):
        Image.fromarray(pixels).save(tmp_path / f"{name}.png")
    wall = content_statistics(tmp_path / "wall.png")
    room = content_statistics(tmp_path / "structured.png")
    assert wall["edge_density_256"] < 0.01
    assert room["edge_density_256"] >= 0.01
    assert room["edge_tile_fraction"] >= 0.2


def test_small_archive_transport_checks_etag_and_size():
    import io
    import zipfile

    from scripts.prepare_arkitscenes_indoor import load_small_archive

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("confidence.png", b"native")
    payload = buffer.getvalue()

    class Response:
        status_code = 200

        def __init__(self, etag):
            self.headers = {"ETag": etag}
            self.raw = io.BytesIO(payload)

        def raise_for_status(self):
            pass

        def close(self):
            pass

    class Client:
        def __init__(self, etag='"fixed"'):
            self.etag = etag

        def get(self, url, **kwargs):
            assert kwargs["headers"]["If-Match"] == '"fixed"'
            return Response(self.etag)

    archive = dict(url="https://example.test/confidence.zip", etag='"fixed"', bytes=len(payload))
    with load_small_archive(Client(), archive) as z:
        assert z.read("confidence.png") == b"native"
    with pytest.raises(ValueError, match="source changed"):
        load_small_archive(Client('"changed"'), archive)
    with pytest.raises(ValueError, match="size mismatch"):
        load_small_archive(Client(), {**archive, "bytes": len(payload) + 1})
    with pytest.raises(ValueError, match="bounded"):
        load_small_archive(Client(), {**archive, "bytes": 33 * 1024**2})


def test_acquisition_distributes_sessions_across_configured_routes(monkeypatch):
    from scripts.prepare_arkitscenes_indoor import session

    routes = {"http://127.0.0.1:18001", "http://127.0.0.1:18002"}
    monkeypatch.setenv("INDOOR_DATA_PROXIES", ",".join(sorted(routes)))
    with session() as first, session() as second:
        assert {first.proxies["https"], second.proxies["https"]} == routes
        assert not first.trust_env and not second.trust_env
