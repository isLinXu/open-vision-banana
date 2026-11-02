"""Official 19-color Cityscapes targets must not use the ADE-150 palette."""

from pathlib import Path

from PIL import Image

from open_vision_banana.data.datasets import CityscapesSegDataset
from open_vision_banana.task_prompts import SEMANTIC_SEGMENTATION_PROMPT
from scripts.build_cityscapes19_manifest import labelids_to_trainid_rgb
from scripts.setup_cityscapes_official import VAL_CITIES


def _write_official_tree(root: Path, *, train_pairs: int = 1, val_per_city: int = 1) -> None:
    index = 0
    for city in ("aachen",):
        for _ in range(train_pairs):
            stem = f"{city}_{index:06d}_000019"
            image_dir = root / "leftImg8bit" / "train" / city
            label_dir = root / "gtFine" / "train" / city
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 8), (10, 20, 30)).save(image_dir / f"{stem}_leftImg8bit.png")
            Image.new("L", (16, 8), 7).save(label_dir / f"{stem}_gtFine_labelIds.png")
            index += 1
    for city in VAL_CITIES:
        for _ in range(val_per_city):
            stem = f"{city}_{index:06d}_000019"
            image_dir = root / "leftImg8bit" / "val" / city
            label_dir = root / "gtFine" / "val" / city
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 8), (10, 20, 30)).save(image_dir / f"{stem}_leftImg8bit.png")
            Image.new("L", (16, 8), 7).save(label_dir / f"{stem}_gtFine_labelIds.png")
            index += 1


def test_labelids_paint_official_road_purple(tmp_path: Path) -> None:
    label = tmp_path / "label.png"
    Image.new("L", (8, 8), 7).save(label)
    rgb = labelids_to_trainid_rgb(label, resolution=8)
    assert rgb.getpixel((0, 0)) == CityscapesSegDataset.CITYSCAPES_COLORS["road"]
    assert SEMANTIC_SEGMENTATION_PROMPT.startswith(
        "Generate a semantic segmentation visualization image"
    )


def test_manifest_writer_uses_cityscapes_prompt(tmp_path: Path, monkeypatch) -> None:
    import json

    from scripts import build_cityscapes19_manifest as build

    cityscapes = tmp_path / "cs"
    _write_official_tree(cityscapes)
    monkeypatch.setattr(build, "OFFICIAL_TRAIN_PAIRS", 1)
    monkeypatch.setattr(build, "OFFICIAL_VAL_PAIRS", 3)
    monkeypatch.setattr(build, "verify_official_train", lambda root: None)
    monkeypatch.setattr(build, "verify_official_val", lambda root: None)

    out = tmp_path / "cityscapes19"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_cityscapes19_manifest.py",
            "--cityscapes-root",
            str(cityscapes),
            "--output",
            str(out),
            "--resolution",
            "8",
        ],
    )
    assert build.main() == 0
    rows = [line for line in (out / "pairs.jsonl").read_text().splitlines() if line]
    assert len(rows) == 4
    row = json.loads(rows[0])
    assert row["prompt"] == SEMANTIC_SEGMENTATION_PROMPT
    assert row["palette"] == "cityscapes_19_official"
    assert (
        json.loads((out / "metadata.json").read_text())["palette_source"]
        == "cityscapes_19_official"
    )
