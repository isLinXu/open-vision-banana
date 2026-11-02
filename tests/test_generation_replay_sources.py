"""build_generation_replay_manifest: parquet (embedded bytes), WebDataset tar, image dir + captions."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest
from PIL import Image

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from open_vision_banana.training.trainer import ManifestVisionDataset  # noqa: E402
from scripts.build_generation_replay_manifest import build_generation_replay_manifest  # noqa: E402


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buf, format="PNG")
    return buf.getvalue()


def test_omniedit_parquet_shards_with_embedded_images(tmp_path: Path):
    shards = tmp_path / "omniedit" / "data"
    shards.mkdir(parents=True)
    rows = [
        {
            "omni_edit_id": f"oe_{i}",
            "task": "color",
            "src_img": {"bytes": _png_bytes((10, 20, 30)), "path": f"{i}_src.png"},
            "edited_img": {"bytes": _png_bytes((200, 20, 30)), "path": f"{i}_edit.png"},
            "edited_prompt_list": ["make the sky purple", "turn the sky violet"],
            "width": 16,
            "height": 16,
        }
        for i in range(3)
    ]
    rows.append({**rows[0], "omni_edit_id": "oe_empty", "edited_prompt_list": []})  # no instruction
    pq.write_table(pa.Table.from_pylist(rows[:2]), shards / "train-00000-of-00002.parquet")
    pq.write_table(pa.Table.from_pylist(rows[2:]), shards / "train-00001-of-00002.parquet")

    out = tmp_path / "replay" / "omniedit.jsonl"
    audit = build_generation_replay_manifest(
        shards,
        out,
        kind="edit",
        preset="omniedit",
        data_root=tmp_path,
        images_dir=tmp_path / "replay" / "omniedit_images",
        val_fraction=0.0,
    )
    assert audit["rows_written"] == 3 and audit["skipped_missing_text"] == 1
    written = [json.loads(line) for line in out.read_text().splitlines()]
    assert all(r["task"] == "generation" and r["prompt"] == "make the sky purple" for r in written)
    assert all(
        r["input"].startswith("replay/omniedit_images/") and r["input"] != r["target"]
        for r in written
    )
    assert (
        sorted((tmp_path / "replay" / "omniedit_images").glob("*.png"))[0].name == "oe_0_edit.png"
    )

    ds = ManifestVisionDataset(out, 16, "train", {"generation"}, data_root=tmp_path)
    sample = ds[0]
    assert sample["task"] == "generation" and sample["target_rgb"].shape == (3, 16, 16)
    assert sample["image"].shape == (3, 16, 16)
    # editing replay: source differs from target
    assert float((sample["image"] * 0.5 + 0.5 - sample["target_rgb"]).abs().mean()) > 0.1

    with pytest.raises(ValueError, match="images-dir"):
        build_generation_replay_manifest(
            shards, tmp_path / "x.jsonl", kind="edit", preset="omniedit"
        )


def test_cwd_relative_images_dir_inside_the_data_root(tmp_path: Path, monkeypatch):
    # run_e0_smoke.sh passes relative paths: --images-dir data/prep/replay/images --data-root data/prep/replay
    monkeypatch.chdir(tmp_path)
    shards = Path("omniedit") / "data"
    shards.mkdir(parents=True)
    row = {
        "omni_edit_id": "oe_0",
        "task": "color",
        "src_img": {"bytes": _png_bytes((10, 20, 30)), "path": "0_src.png"},
        "edited_img": {"bytes": _png_bytes((200, 20, 30)), "path": "0_edit.png"},
        "edited_prompt_list": ["make the sky purple"],
        "width": 16,
        "height": 16,
    }
    pq.write_table(pa.Table.from_pylist([row]), shards / "train-00000-of-00001.parquet")
    replay = Path("prep") / "replay"
    build_generation_replay_manifest(
        shards,
        replay / "omniedit.jsonl",
        kind="edit",
        preset="omniedit",
        data_root=replay,
        images_dir=replay / "images",
        val_fraction=0.0,
    )
    [written] = [json.loads(line) for line in (replay / "omniedit.jsonl").read_text().splitlines()]
    assert written["input"].startswith("images/") and written["target"].startswith("images/")
    assert (replay / written["input"]).is_file() and (replay / written["target"]).is_file()


def test_record_paths_relative_to_the_data_root_are_kept(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    (data / "imgs").mkdir(parents=True)
    for name in ("a.png", "b.png"):
        Image.new("RGB", (8, 8)).save(data / "imgs" / name)
    records = data / "records.jsonl"
    records.write_text(
        json.dumps({"source": "imgs/a.png", "target": "imgs/b.png", "instruction": "x"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)  # cwd is not the data root
    audit = build_generation_replay_manifest(
        records, data / "replay.jsonl", kind="edit", data_root=data, val_fraction=0.0
    )
    [written] = [json.loads(line) for line in (data / "replay.jsonl").read_text().splitlines()]
    assert audit["rows_written"] == 1
    assert (written["input"], written["target"]) == ("imgs/a.png", "imgs/b.png")


def test_webdataset_tar_and_captions_sidecar(tmp_path: Path):
    tar_path = tmp_path / "shard.tar"
    with tarfile.open(tar_path, "w") as tar:
        for key, caption in (("000", "a red cube"), ("001", "a blue sphere")):
            img = _png_bytes((255, 0, 0))
            info = tarfile.TarInfo(f"{key}.png")
            info.size = len(img)
            tar.addfile(info, io.BytesIO(img))
            txt = caption.encode()
            info = tarfile.TarInfo(f"{key}.txt")
            info.size = len(txt)
            tar.addfile(info, io.BytesIO(txt))
    out = tmp_path / "wds.jsonl"
    audit = build_generation_replay_manifest(
        tar_path, out, kind="t2i", data_root=tmp_path, images_dir=tmp_path / "wds_images"
    )
    assert audit["rows_written"] == 2
    rows = {json.loads(line)["prompt"] for line in out.read_text().splitlines()}
    assert rows == {"a red cube", "a blue sphere"}

    # BLIP3o-60k layout: numbered images in a tar + one caption per line in a sidecar file
    blip_tar = tmp_path / "journeyDB.tar"
    with tarfile.open(blip_tar, "w") as tar:
        for idx in (170, 171):
            img = _png_bytes((0, 255, 0))
            info = tarfile.TarInfo(f"{idx}.jpg")
            info.size = len(img)
            tar.addfile(info, io.BytesIO(img))
    captions = tmp_path / "journeyDB.txt"
    captions.write_text("\n".join(f"caption {i}" for i in range(200)) + "\n", encoding="utf-8")
    out2 = tmp_path / "blip.jsonl"
    audit2 = build_generation_replay_manifest(
        blip_tar,
        out2,
        kind="t2i",
        data_root=tmp_path,
        images_dir=tmp_path / "blip_images",
        captions_file=captions,
        caption_index_base=0,
    )
    assert audit2["rows_written"] == 2
    prompts = {}
    for line in out2.read_text().splitlines():
        row = json.loads(line)
        prompts[Path(row["target"]).stem] = row["prompt"]
    assert prompts == {"170": "caption 170", "171": "caption 171"}

    # plain directory of images with a captions file
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for idx in (5, 6):
        (img_dir / f"{idx}.png").write_bytes(_png_bytes((1, 2, 3)))
    out3 = tmp_path / "dir.jsonl"
    audit3 = build_generation_replay_manifest(
        img_dir, out3, kind="t2i", data_root=tmp_path, captions_file=captions
    )
    assert audit3["rows_written"] == 2
