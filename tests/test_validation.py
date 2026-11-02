"""Tests for dataset validation utilities."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.data.validation import (
    validate_manifest_contract,
    validate_vision_dataset,
)
from open_vision_banana.model_registry import get_model_config
from open_vision_banana.task_prompts import SEMANTIC_SEGMENTATION_PROMPT


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

    def test_depth_roundtrip_validation_passes_encoded_target(self, tmp_path: Path):
        _write_rgb(tmp_path / "images" / "a.png", (64, 64), (0, 0, 0))
        depth = torch.full((1, 1, 64, 64), 3.0)
        rgb = DepthRGBMapper().depth_to_rgb(depth)[0].permute(1, 2, 0)
        (tmp_path / "targets").mkdir()
        Image.fromarray((rgb.numpy() * 255).round().astype(np.uint8)).save(
            tmp_path / "targets" / "a.png"
        )

        report = validate_vision_dataset(tmp_path, task="depth")
        assert report.ok
        assert report.stats["depth_rgb_roundtrip_mae"]["max"] <= 2.0 / 255.0

    def test_depth_off_colormap_path_is_error(self, tmp_path: Path):
        _write_rgb(tmp_path / "images" / "a.png", (64, 64), (0, 0, 0))
        _write_rgb(tmp_path / "targets" / "a.png", (64, 64), (80, 120, 40))
        report = validate_vision_dataset(tmp_path, task="depth")
        assert not report.ok
        assert any("off the cube_edge path" in issue.message for issue in report.errors)

    def test_nonunit_normal_target_is_error(self, tmp_path: Path):
        _write_rgb(tmp_path / "images" / "a.png", (64, 64), (0, 0, 0))
        _write_rgb(tmp_path / "targets" / "a.png", (64, 64), (128, 128, 128))
        report = validate_vision_dataset(tmp_path, task="normal")
        assert not report.ok
        assert any("not unit length" in issue.message for issue in report.errors)

    def test_nonbinary_referring_target_is_error(self, tmp_path: Path):
        _write_rgb(tmp_path / "images" / "a.png", (64, 64), (0, 0, 0))
        _write_rgb(tmp_path / "targets" / "a.png", (64, 64), (128, 128, 128))
        report = validate_vision_dataset(tmp_path, task="referring")
        assert not report.ok
        assert any("not a binary" in issue.message for issue in report.errors)

    def test_structured_target_requires_png(self, tmp_path: Path):
        _write_rgb(tmp_path / "images" / "a.jpg", (64, 64), (0, 0, 0))
        _write_rgb(tmp_path / "targets" / "a.jpg", (64, 64), (255, 255, 255))
        report = validate_vision_dataset(tmp_path, task="semantic")
        assert not report.ok
        assert any("lossless PNG" in issue.message for issue in report.errors)


class TestValidateManifestContract:
    def test_sana_registry_includes_attention_lora_and_300_tokens(self):
        for name in ("sana-600m", "sana-600m-512px", "sana-1600m", "sana-sprint"):
            cfg = get_model_config(name)
            assert cfg.text_encoder_max_length == 300
            for attn_proj in ("to_q", "to_k", "to_v", "to_out.0"):
                assert attn_proj in cfg.lora_target_modules

    def test_catches_ffn_only_lora_and_grayscale_depth_and_relative_prompt(self, tmp_path: Path):
        _write_rgb(tmp_path / "images" / "d.png", (32, 32), (10, 20, 30))
        _write_rgb(tmp_path / "targets" / "d.png", (32, 32), (180, 180, 180))
        manifest = tmp_path / "pairs.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "task": "depth",
                    "input": "images/d.png",
                    "target": "targets/d.png",
                    "split": "train",
                    "prompt": "Generate a relative depth map.",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        report = validate_manifest_contract(
            manifest,
            data_root=tmp_path,
            model_name="sana-1600m",
            lora_targets=["linear", "proj_out", "conv_inverted", "conv_point"],
            depth_curve="metric_power",
        )
        assert not report.ok
        messages = [issue.message for issue in report.errors]
        assert any("attention Q/K/V" in m for m in messages)
        assert any("relative depth" in m for m in messages)
        assert any("grayscale" in m for m in messages)

    def test_catches_77_token_truncation_for_cityscapes_legend(self, tmp_path: Path):
        _write_rgb(tmp_path / "images" / "s.png", (32, 32), (10, 20, 30))
        _write_rgb(tmp_path / "targets" / "s.png", (32, 32), (128, 64, 128))
        manifest = tmp_path / "pairs.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "task": "semantic",
                    "input": "images/s.png",
                    "target": "targets/s.png",
                    "split": "train",
                    "prompt": SEMANTIC_SEGMENTATION_PROMPT,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        # flux-dev has default text_encoder_max_length=77 (CLIP truncation)
        report_flux = validate_manifest_contract(
            manifest, data_root=tmp_path, model_name="flux-dev"
        )
        assert not report_flux.ok
        assert any(
            "exceed flux-dev text_encoder_max_length=77" in i.message for i in report_flux.errors
        )

        # Fixed sana-1600m has text_encoder_max_length=300 and attention LoRA targets
        report_sana = validate_manifest_contract(
            manifest, data_root=tmp_path, model_name="sana-1600m"
        )
        assert report_sana.ok
        assert report_sana.stats["cityscapes_semantic_rows"] == 1
        assert report_sana.stats["has_attention_lora"] is True
