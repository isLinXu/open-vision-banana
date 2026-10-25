"""Data ablations must keep the model, optimizer and evaluation controls fixed."""

import json
import runpy
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_res_ablation_configs_only_change_data_and_output_paths():
    from open_vision_banana.training.config import load_config

    recipes = []
    for suffix in ("base", "rea2seg", "rea2seg_llmseg"):
        path = ROOT / f"configs/experiments/qwen_image_21_res_{suffix}.yaml"
        config = load_config(config_path=path)
        assert config.tasks == ["referring"] and config.vision_ratio == 1.0
        raw = yaml.safe_load(path.read_text())
        assert raw["training"]["steps"] == 10000
        assert raw["data"]["pair_manifest"].endswith("/training.jsonl")
        del raw["data"]["dir"], raw["data"]["pair_manifest"], raw["checkpoint"]["output_dir"]
        recipes.append(raw)
    assert recipes[0] == recipes[1] == recipes[2]


def test_download_supplement_uses_existing_restorer(tmp_path, monkeypatch):
    import huggingface_hub

    from open_vision_banana import res_bundle

    folder = "supplements/reasoning-v1/llmseg-coco5k"
    snapshot = tmp_path / "cache"
    source = snapshot / folder
    source.mkdir(parents=True)
    (source / "bundle.json").write_text(
        json.dumps({"format": res_bundle.FORMAT, "files": [{"path": "assets/part-00000.tar"}]})
    )
    calls = []

    def hub_file(repo, filename, **kwargs):
        assert filename == folder + "/bundle.json"
        return str(source / "bundle.json")

    def snapshot_download(repo, **kwargs):
        calls.append(kwargs)
        return str(snapshot)

    def restore(bundle, output):
        assert bundle == source and output == tmp_path / "out"
        return {"restored": True}

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub_file)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(res_bundle, "restore_bundle", restore)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_res_dataset.py",
            "--repo-id",
            "example/res",
            "--revision",
            "a" * 40,
            "--subfolder",
            folder,
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    runpy.run_path(str(ROOT / "scripts/download_res_dataset.py"), run_name="__main__")
    assert calls[0]["allow_patterns"] == [
        folder + "/bundle.json",
        folder + "/assets/part-00000.tar",
    ]


def test_download_subfolder_rejects_traversal_before_network(tmp_path, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *a, **kw: pytest.fail("network called")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_res_dataset.py",
            "--repo-id",
            "example/res",
            "--revision",
            "a" * 40,
            "--subfolder",
            "../other",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(ValueError, match="Unsafe"):
        runpy.run_path(str(ROOT / "scripts/download_res_dataset.py"), run_name="__main__")
