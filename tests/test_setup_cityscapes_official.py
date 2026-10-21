"""Official Cityscapes val layout checks. cityscapes8 must not pass."""

from pathlib import Path

import pytest
from PIL import Image

from scripts.setup_cityscapes_official import (
    OFFICIAL_TRAIN_PAIRS,
    OFFICIAL_VAL_PAIRS,
    VAL_CITIES,
    official_train_pairs,
    official_val_pairs,
    verify_official_train,
    verify_official_val,
)


def _write_pair(root: Path, city: str, index: int) -> None:
    stem = f"{city}_{index:06d}_000019"
    image_dir = root / "leftImg8bit" / "val" / city
    label_dir = root / "gtFine" / "val" / city
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image_dir / f"{stem}_leftImg8bit.png")
    Image.new("L", (8, 8), 7).save(label_dir / f"{stem}_gtFine_labelIds.png")


def test_verify_rejects_missing_official_cities(tmp_path: Path) -> None:
    _write_pair(tmp_path, "aachen", 0)
    with pytest.raises(FileNotFoundError, match="missing cities"):
        verify_official_val(tmp_path)


def test_verify_counts_official_pairs(tmp_path: Path) -> None:
    index = 0
    for city in VAL_CITIES:
        for _ in range(2):
            _write_pair(tmp_path, city, index)
            index += 1
    pairs = official_val_pairs(tmp_path)
    assert len(pairs) == 6
    with pytest.raises(FileNotFoundError, match=str(OFFICIAL_VAL_PAIRS)):
        verify_official_val(tmp_path)


def test_verify_train_counts_official_pairs(tmp_path: Path) -> None:
    image_dir = tmp_path / "leftImg8bit" / "train" / "aachen"
    label_dir = tmp_path / "gtFine" / "train" / "aachen"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image_dir / "aachen_000000_000019_leftImg8bit.png")
    Image.new("L", (8, 8), 7).save(label_dir / "aachen_000000_000019_gtFine_labelIds.png")
    assert len(official_train_pairs(tmp_path)) == 1
    with pytest.raises(FileNotFoundError, match=str(OFFICIAL_TRAIN_PAIRS)):
        verify_official_train(tmp_path)
