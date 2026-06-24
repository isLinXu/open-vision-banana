"""Tests for dataset validation utilities."""

from pathlib import Path

from PIL import Image

from open_vision_banana.data.validation import validate_vision_dataset


def _write_rgb(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


class TestValidateVisionDataset:
    def test_valid_pair_passes(self, tmp_path: Path):
        _write_rgb(tmp_path / "images" / "a.jpg", (128, 128), (100, 120, 140))
        _write_rgb(tmp_path / "targets" / "a.png", (128, 128), (200, 50, 50))

        report = validate_vision_dataset(tmp_path)
        assert report.ok
        assert report.pair_count == 1
        assert len(report.errors) == 0

    def test_missing_targets_dir(self, tmp_path: Path):
        (tmp_path / "images").mkdir()
        report = validate_vision_dataset(tmp_path)
        assert not report.ok
        assert any("targets" in i.message for i in report.errors)

    def test_unpaired_image_is_error(self, tmp_path: Path):
        _write_rgb(tmp_path / "images" / "lonely.jpg", (64, 64), (0, 0, 0))
        (tmp_path / "targets").mkdir()

        report = validate_vision_dataset(tmp_path)
        assert not report.ok
        assert report.pair_count == 0

    def test_size_mismatch_is_error(self, tmp_path: Path):
        _write_rgb(tmp_path / "images" / "a.jpg", (128, 128), (10, 10, 10))
        _write_rgb(tmp_path / "targets" / "a.jpg", (256, 256), (10, 10, 10))

        report = validate_vision_dataset(tmp_path)
        assert not report.ok
        assert any("Size mismatch" in i.message for i in report.errors)

    def test_orphan_target_is_warning(self, tmp_path: Path):
        _write_rgb(tmp_path / "images" / "a.jpg", (64, 64), (0, 0, 0))
        _write_rgb(tmp_path / "targets" / "a.jpg", (64, 64), (0, 0, 0))
        _write_rgb(tmp_path / "targets" / "orphan.jpg", (64, 64), (0, 0, 0))

        report = validate_vision_dataset(tmp_path)
        assert report.ok
        assert any("without matching image" in i.message for i in report.warnings)

    def test_min_pairs_enforced(self, tmp_path: Path):
        (tmp_path / "images").mkdir()
        (tmp_path / "targets").mkdir()

        report = validate_vision_dataset(tmp_path, min_pairs=1)
        assert not report.ok

    def test_inconsistent_resolutions_warning(self, tmp_path: Path):
        _write_rgb(tmp_path / "images" / "a.jpg", (64, 64), (0, 0, 0))
        _write_rgb(tmp_path / "targets" / "a.jpg", (64, 64), (0, 0, 0))
        _write_rgb(tmp_path / "images" / "b.jpg", (128, 128), (0, 0, 0))
        _write_rgb(tmp_path / "targets" / "b.jpg", (128, 128), (0, 0, 0))

        report = validate_vision_dataset(tmp_path)
        assert report.ok
        assert any("Inconsistent resolutions" in i.message for i in report.warnings)
