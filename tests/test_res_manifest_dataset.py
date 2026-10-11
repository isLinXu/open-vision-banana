"""Portable RES bundles preserve native evaluation pixels and official splits."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch
from PIL import Image

from open_vision_banana.data.res_dataset import RESManifestDataset
from open_vision_banana.referring_contract import (
    BINARY_REFERRING_ENCODING,
    build_binary_referring_prompt,
)
from scripts.evaluate_res_manifest import main


def _bundle(root):
    (root / "images").mkdir()
    (root / "targets").mkdir()
    Image.new("RGB", (5, 3), (255, 255, 255)).save(root / "images/image.png")
    mask = np.zeros((3, 5, 3), dtype=np.uint8)
    mask[0, 0] = 255
    mask[2, 4] = 255
    Image.fromarray(mask).save(root / "targets/mask.png")
    rows = []
    for index, source_split in enumerate(["train", "val", "testA", "testB"]):
        expression = f"object {index}"
        rows.append(
            {
                "sample_id": f"ref{index}",
                "task": "referring",
                "split": source_split if source_split in {"train", "val"} else "test",
                "source_dataset": "refcoco",
                "source_protocol": "unc",
                "source_split": source_split,
                "input": "images/image.png",
                "target": "targets/mask.png",
                "width": 5,
                "height": 3,
                "expression": expression,
                "prompt": build_binary_referring_prompt(expression),
                "encoding": BINARY_REFERRING_ENCODING,
            }
        )
    _write(root, rows)
    return rows, mask[..., 0] == 255


def _write(root, rows):
    (root / "pairs.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_native_mask_and_expression_rows_are_preserved(tmp_path):
    rows, native = _bundle(tmp_path)
    dataset = RESManifestDataset(tmp_path, split="test", resolution=2)
    assert len(dataset) == 2
    assert dataset[0]["source_split"] == "testA"
    assert dataset[1]["source_split"] == "testB"
    item = dataset[0]
    np.testing.assert_array_equal(item["gt_mask_native"].numpy(), native)
    assert item["prompt"] == rows[2]["prompt"]
    assert item["expression"] == rows[2]["expression"]
    assert item["image_range"] == "minus_one_one"
    assert item["image"].shape == (3, 2, 2)
    assert item["image"].eq(1).all()
    assert set(item["target_rgb"].unique().tolist()) <= {0.0, 1.0}
    assert dataset.validate_assets() == {
        "samples": 2,
        "unique_images": 1,
        "unique_masks": 1,
        "source_counts": {"refcoco/unc/testA": 1, "refcoco/unc/testB": 1},
    }
    assert (
        dataset.manifest_sha256
        == hashlib.sha256((tmp_path / "pairs.jsonl").read_bytes()).hexdigest()
    )


def test_source_filters_preserve_official_testa(tmp_path):
    _bundle(tmp_path)
    dataset = RESManifestDataset(
        tmp_path,
        split="test",
        source_dataset="refcoco",
        source_protocol="unc",
        source_split="testA",
    )
    assert len(dataset) == 1
    assert dataset[0]["sample_id"] == "ref2"
    with pytest.raises(ValueError, match="No RES"):
        RESManifestDataset(tmp_path, split="test", source_protocol="umd")


def test_evaluation_requires_explicit_held_out_split(tmp_path):
    _bundle(tmp_path)
    with pytest.raises(TypeError, match="split"):
        RESManifestDataset(tmp_path)
    with pytest.raises(ValueError, match="evaluation split"):
        RESManifestDataset(tmp_path, split="train")
    with pytest.raises(ValueError, match="incompatible"):
        RESManifestDataset(tmp_path, split="val", source_split="testA")


@pytest.mark.parametrize(
    "change",
    [
        "traversal",
        "absolute",
        "wrong_encoding",
        "prompt",
        "duplicate",
        "missing_asset",
        "source_split",
    ],
)
def test_manifest_contract_rejects_invalid_rows(tmp_path, change):
    rows, _ = _bundle(tmp_path)
    row = rows[1]
    if change == "traversal":
        row["input"] = "../image.png"
    elif change == "absolute":
        row["target"] = "C:/mask.png"
    elif change == "wrong_encoding":
        row["encoding"] = "legacy_color_distance_v1"
    elif change == "prompt":
        row["prompt"] = "Segment all objects"
    elif change == "duplicate":
        rows.append(dict(row))
    elif change == "missing_asset":
        row["target"] = "targets/absent.png"
    else:
        row["source_split"] = "train"
    _write(tmp_path, rows)
    with pytest.raises((ValueError, FileNotFoundError)):
        RESManifestDataset(tmp_path, split="val")


@pytest.mark.parametrize("change", ["grey", "colored", "empty", "size", "metadata", "corrupt"])
def test_native_asset_validation_catches_invalid_masks(tmp_path, change):
    rows, _ = _bundle(tmp_path)
    if change == "metadata":
        rows[1]["width"] = 4
        _write(tmp_path, rows)
    elif change == "corrupt":
        (tmp_path / "targets/mask.png").write_bytes(b"broken PNG")
    else:
        mask = np.zeros((3, 5, 3), dtype=np.uint8)
        mask[0, 0] = 255
        if change == "grey":
            mask[0, 0] = 127
        elif change == "colored":
            mask[0, 0] = [255, 0, 0]
        elif change == "empty":
            mask.fill(0)
        else:
            mask = mask[:, :4]
        Image.fromarray(mask).save(tmp_path / "targets/mask.png")
    dataset = RESManifestDataset(tmp_path, split="val")
    with pytest.raises((ValueError, OSError)):
        dataset.validate_assets()


def test_preflight_checks_metadata_for_shared_assets(tmp_path):
    rows, _ = _bundle(tmp_path)
    rows[3]["width"] = 4
    _write(tmp_path, rows)
    dataset = RESManifestDataset(tmp_path, split="test")
    with pytest.raises(ValueError, match="dimensions"):
        dataset.validate_assets()


def test_cli_preflight_outputs_fingerprint_without_loading_model(tmp_path, monkeypatch):
    from open_vision_banana.inference.pipeline import VisionBanana

    _bundle(tmp_path)

    def forbidden_load(*args, **kwargs):
        raise AssertionError("preflight must not load a model")

    monkeypatch.setattr(VisionBanana, "from_pretrained", forbidden_load)
    output = tmp_path / "preflight.json"
    assert (
        main(
            [
                "--dataset-root",
                str(tmp_path),
                "--split",
                "test",
                "--source-split",
                "testB",
                "--preflight",
                "--output-json",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(output.read_text())
    assert result["status"] == "preflight_passed"
    assert result["counts"]["samples"] == 1
    assert result["filters"]["source_split"] == "testB"
    assert result["protocol"]["native_resolution"] is True
    assert len(result["manifest_sha256"]) == 64


def test_cli_scores_native_pixels_with_expression_and_encoding(tmp_path, monkeypatch):
    from open_vision_banana.inference.pipeline import VisionBanana

    _bundle(tmp_path)
    calls = []

    class WhiteMask:
        def eval(self):
            return self

        def predict(self, **kwargs):
            calls.append(kwargs)
            return {"rgb_output": torch.ones(1, 3, 2, 2)}

    monkeypatch.setattr(VisionBanana, "from_pretrained", lambda **kwargs: WhiteMask())
    output = tmp_path / "metrics.json"
    assert (
        main(
            [
                "--dataset-root",
                str(tmp_path),
                "--split",
                "test",
                "--model-name",
                "test/mock",
                "--resolution",
                "2",
                "--device",
                "cpu",
                "--output-json",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(output.read_text())
    assert result["status"] == "evaluated"
    assert result["metrics"]["samples"] == 2
    assert result["metrics"]["cIoU"] == pytest.approx(2 / 15)
    assert [call["expression"] for call in calls] == ["object 2", "object 3"]
    assert all(call["referring_encoding"] == BINARY_REFERRING_ENCODING for call in calls)
    assert all(call["image"].shape == (1, 3, 2, 2) for call in calls)


def test_public_dataset_exports():
    from open_vision_banana import RESManifestDataset as PublicDataset
    from open_vision_banana.data import RESManifestDataset as DataDataset

    assert PublicDataset is RESManifestDataset
    assert DataDataset is RESManifestDataset


@pytest.mark.parametrize("requested, effective", [(1, 20), (12, 12)])
def test_sana_report_records_resolved_steps_and_unused_strength(requested, effective):
    from types import SimpleNamespace

    from scripts.evaluate_res_manifest import inference_settings

    model = SimpleNamespace(
        generator=SimpleNamespace(
            model_config=SimpleNamespace(family="sana", default_num_inference_steps=20)
        )
    )
    settings = inference_settings(model, requested, 0.85)
    assert settings["requested"]["num_inference_steps"] == requested
    assert settings["effective"]["num_inference_steps"] == effective
    assert settings["effective"]["guidance_scale"] == 1.0
    assert settings["effective"]["strength"] is None


def test_unknown_backend_does_not_claim_requested_settings_were_effective():
    from scripts.evaluate_res_manifest import inference_settings

    settings = inference_settings(object(), 1, 0.85)
    assert settings["effective"] is None
    assert "not recorded" in settings["settings_status"]
