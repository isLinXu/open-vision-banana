"""Synthetic full-size native fixtures for the independent build auditor."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def load_script(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_script("validate_diode_train")
build = load_script("prepare_diode_train")


@pytest.fixture
def native_build(tmp_path):
    source = tmp_path / "source"
    folder = source / "train" / "indoors" / "scene" / "scan"
    folder.mkdir(parents=True)
    rgb = np.full((768, 1024, 3), 128, dtype=np.uint8)
    depth = np.full((768, 1024, 1), 2.5, dtype=np.float32)
    mask = np.ones((768, 1024), dtype=np.uint8)
    mask[:2] = 0
    depth[2, 0, 0] = np.nan
    depth[3, 0, 0] = -1
    Image.fromarray(rgb).save(folder / "id0000.png")
    np.save(folder / "id0000_depth.npy", depth)
    np.save(folder / "id0000_depth_mask.npy", mask)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "train": {"indoors": {"scene": {"scan": [f"id{i:04d}" for i in range(8574)]}}},
                "val": {"indoors": ["heldout"]},
            }
        )
    )
    root = tmp_path / "build"
    build.main(
        [
            "--source",
            "diode",
            "--input",
            str(source),
            "--official-metadata",
            str(metadata),
            "--output-dir",
            str(root),
            "--allow-partial",
        ]
    )
    return root, metadata


def test_partial_source_pass_does_not_claim_complete(native_build):
    root, metadata = native_build
    result = audit.audit(root, official_metadata=metadata)
    assert result["status"] == "passed", result
    assert result["source_complete"] is False
    assert result["scope"] == "partial_source_only"
    assert result["build_complete_marker_verified"] is True
    assert result["checks"]["asset_hashes"] == 4
    assert result["checks"]["source_rows_verified"] == 1
    assert result["checks"]["native_asset_audit_passed"] is True
    assert "loader_smoke" not in result


@pytest.mark.parametrize(
    "name", ["../outside", "/absolute", "C:/file", "a\\b", "a/../b", "a//b", "./x"]
)
def test_unsafe_paths_rejected(tmp_path, name):
    with pytest.raises(ValueError, match="Unsafe"):
        audit.safe_path(tmp_path, name)


def test_asset_corruption_fails(native_build):
    root, metadata = native_build
    target = root / "targets" / "diode_id0000.png"
    target.write_bytes(target.read_bytes() + b"corruption")
    result = audit.audit(root, official_metadata=metadata)
    assert result["status"] == "failed"
    assert any("hash/size mismatch" in error for error in result["failures"])


def test_missing_metadata_fails_explicitly(native_build):
    root, _ = native_build
    result = audit.audit(root)
    assert result["status"] == "failed"
    assert any("--official-metadata is required" in error for error in result["failures"])


def test_extra_asset_fails(native_build):
    root, metadata = native_build
    (root / "images" / "orphan.png").write_bytes(b"orphan")
    result = audit.audit(root, official_metadata=metadata)
    assert result["status"] == "failed"
    assert any("Unexpected/missing files" in error for error in result["failures"])


@pytest.mark.parametrize("cap, expected_pairs", [(0.0, 0), (0.05, 1)])
def test_encoding_gate_and_rejections_verified(native_build, cap, expected_pairs):
    root, metadata = native_build
    source = json.loads((root / "build_config.json").read_text())["input"]
    output = root.parent / f"gated-{cap}"
    build.main(
        [
            "--source",
            "diode",
            "--input",
            source,
            "--official-metadata",
            str(metadata),
            "--output-dir",
            str(output),
            "--allow-partial",
            "--max-encoding-error-m",
            str(cap),
        ]
    )
    result = audit.audit(output, official_metadata=metadata)
    assert result["status"] == "passed", result
    assert result["statistics"]["output_pairs"] == expected_pairs
    assert result["statistics"]["quality_rejected_pairs"] == 1 - expected_pairs
    if not expected_pairs:
        rejection = json.loads((output / "quality_rejections.jsonl").read_text())
        assert rejection["reason"] == "explicit_max_encoding_error"
        assert rejection["encoding_max_error_m"] > cap
        rejection["max_encoding_error_m"] = cap + 1
        (output / "quality_rejections.jsonl").write_text(json.dumps(rejection) + "\n")
        bad = audit.audit(output, official_metadata=metadata)
        assert bad["status"] == "failed"
        assert any("Rejection threshold mismatch" in failure for failure in bad["failures"])


@pytest.mark.parametrize(
    "option,value", [("--min-valid-fraction", "0.97"), ("--max-encoding-error-m", "0.051")]
)
def test_clean_gates_cannot_be_weakened(tmp_path, option, value):
    with pytest.raises(SystemExit):
        build.main(
            [
                "--source",
                "diode",
                "--input",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "out"),
                option,
                value,
            ]
        )
    assert not (tmp_path / "out").exists()


def test_native_resume_is_identical(native_build):
    root, metadata = native_build
    config = json.loads((root / "build_config.json").read_text())
    assert config["min_valid_fraction"] == 0.98
    assert config["max_encoding_error_m"] == 0.05
    before = (root / "report.json").read_bytes()
    args = [
        "--source",
        "diode",
        "--input",
        config["input"],
        "--official-metadata",
        str(metadata),
        "--output-dir",
        str(root),
        "--allow-partial",
    ]
    build.main(args + ["--resume"])
    assert (root / "report.json").read_bytes() == before
    with pytest.raises(ValueError, match="nonempty"):
        build.main(args)
    with pytest.raises(ValueError, match="Existing output"):
        build.main(args + ["--resume", "--min-valid-fraction", "0.99"])


def test_auditor_has_no_training_dependency_or_loader_option():
    import ast
    import inspect

    source = Path(audit.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("open_vision_banana.training")
        elif isinstance(node, ast.Import):
            assert all(
                not alias.name.startswith("open_vision_banana.training") for alias in node.names
            )
    assert "ManifestVisionDataset" not in source
    assert "loader_accounting" not in source
    assert "--skip-loader-smoke" not in source
    assert tuple(inspect.signature(audit.audit).parameters) == ("root", "official_metadata")
    with pytest.raises(SystemExit):
        audit.main(["unused-build", "--skip-loader-smoke"])
