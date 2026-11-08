"""Fail-closed dataset preparation, paired configs, and consumed-sample accounting."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from open_vision_banana.res_bundle import sha256_file
from open_vision_banana.training.config import load_config
from open_vision_banana.training.exposure import ExposureAudit
from open_vision_banana.training.trainer import ManifestVisionDataset, _collate_fn
from scripts import prepare_lingbot_experiments as prep
from scripts.run_lingbot_experiment import build_command

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def data(tmp_path, monkeypatch):
    base = tmp_path / "base"
    ling = tmp_path / "ling"
    base.mkdir()
    ling.mkdir()
    rows = []
    for i, (task, split) in enumerate((t, s) for t in sorted(prep.TASKS) for s in ("train", "val")):
        image = base / f"base-{i}.png"
        Image.new("RGB", (8, 8), (i * 9, 20, 30)).save(image)
        rows.append(
            dict(input=image.name, target=image.name, task=task, split=split, prompt=prep.PROMPT)
        )
        if task == "depth":
            native = base / f"base-{i}-depth.npy"
            np.save(native, np.ones((8, 8), dtype=np.float32))
            rows[-1]["metric_depth"] = native.name
    jsonl(base / "pairs.jsonl", rows)
    Image.new("RGB", (8, 8), (240, 50, 60)).save(ling / "input.png")
    Image.new("RGB", (8, 8), (0, 150, 0)).save(ling / "target.png")
    np.save(ling / "depth.npy", np.ones((8, 8), np.float32))
    row = dict(
        input="input.png",
        target="target.png",
        metric_depth="depth.npy",
        task="depth",
        split="train",
        sample_id="lingbot-test",
        source_dataset="lingbot-depth",
        source_subset="RobbySim",
        source_view="rrt_view",
        source_scene="room-a",
        depth_units="metres",
        depth_convention="planar_z_depth",
        prompt=prep.PROMPT,
    )
    jsonl(ling / "pairs.jsonl", [row])
    assets = [
        dict(path=n, bytes=(ling / n).stat().st_size, sha256=sha256_file(ling / n))
        for n in ("input.png", "target.png", "depth.npy")
    ]
    jsonl(ling / "assets.jsonl", assets)
    for file, status in (("build_report.json", "complete"), ("validation_report.json", "passed")):
        dump(ling / file, dict(status=status, dataset_fingerprint=prep.FINGERPRINT))
    monkeypatch.setattr(prep, "ROWS", 1)
    monkeypatch.setattr(prep, "PAIRS_SHA", sha256_file(ling / "pairs.jsonl"))
    monkeypatch.setattr(prep, "ASSETS_SHA", sha256_file(ling / "assets.jsonl"))
    return base / "pairs.jsonl", ling, tmp_path / "paired"


def prepare(data):
    base, ling, output = data
    return prep.prepare(
        base, ling, output, sha256_file(base), depth_rows=1, replay_rows=1, hash_assets=True
    )


def test_pair_preserves_rows_splits_paths_and_can_load(data):
    report = prepare(data)
    base, ling, out = data
    old = prep.read_jsonl(out / "full_baseline/pairs.jsonl")
    new = prep.read_jsonl(out / "full_plus_lingbot/pairs.jsonl")
    # Sibling output roots have identical relative bases: the old records are unchanged.
    assert new[:-1] == old and len(new) == 13
    assert report["manifests"]["full_plus_lingbot"]["task_split_counts"]["depth/train"] == 2
    assert report["external_benchmark_audit"] == "not performed here"
    ds = ManifestVisionDataset(
        out / "full_plus_lingbot/pairs.jsonl",
        32,
        "train",
        {"depth"},
        depth_colormaps=["cube_edge"],
        seed=42,
    )
    batch = _collate_fn([ds[0], ds[1]])
    assert [x["source"] for x in batch["experiment_audit"]] == ["baseline", "lingbot-depth-v1"]
    assert batch["experiment_audit"][1]["palette"] == "cube_edge"
    assert (out / "experiment-data.json").is_file()
    with pytest.raises(FileExistsError):
        prepare(data)


@pytest.mark.parametrize("failure", ["missing", "tamper", "partial", "wrong_revision", "overlap"])
def test_reject_before_output(data, failure):
    base, ling, out = data
    if failure == "missing":
        (base.parent / "base-0.png").unlink()
    elif failure == "tamper":
        (ling / "depth.npy").write_bytes(b"corrupt")
    elif failure == "partial":
        (ling / "BUILD_INCOMPLETE").touch()
    elif failure == "wrong_revision":
        (ling / "pairs.jsonl").write_text("{}\n")
    else:
        rows = prep.read_jsonl(base)
        rows[1]["input"] = rows[0]["input"]
        jsonl(base, rows)
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare(data)
    assert not out.exists()


def test_depth_only_loader_blocks_incomplete_restore(tmp_path):
    (tmp_path / "BUILD_INCOMPLETE").touch()
    with pytest.raises(ValueError, match="incomplete"):
        ManifestVisionDataset(tmp_path / "pairs.jsonl", 32, "train", {"depth"})


def test_metadata_preserves_pixels_prompts_and_dataset_rng(data):
    prepare(data)
    path = data[2] / "full_plus_lingbot/pairs.jsonl"
    rows = prep.read_jsonl(path)
    for row in rows:
        for key in ("experiment_source", "experiment_sample_id", "source_scene"):
            row.pop(key, None)
    plain_path = path.with_name("plain.jsonl")
    jsonl(plain_path, rows)
    args = dict(
        resolution=32,
        split="train",
        tasks={"depth"},
        depth_colormaps=["cube_edge", "viridis"],
        seed=42,
    )
    tagged = ManifestVisionDataset(path, **args)
    plain = ManifestVisionDataset(plain_path, **args)
    for i in [0, 1, 1, 0, 1]:
        a, b = tagged[i], plain[i]
        a.pop("experiment_audit")
        assert a.keys() == b.keys()
        for key in a:
            if isinstance(a[key], torch.Tensor):
                assert torch.equal(a[key], b[key])
            else:
                assert a[key] == b[key]
        assert tagged.rng.getstate() == plain.rng.getstate()


def test_audit_no_rng_effect_and_resume_is_separate(tmp_path):
    state = torch.random.get_rng_state().clone()
    a = ExposureAudit(tmp_path, 0, 0)
    b = {
        "experiment_audit": [
            dict(source="lingbot", task="depth", id="a", scene="room", palette="cube_edge")
        ]
    }
    a.observe(b)
    a.observe(b)
    a.dropped_updates += 1
    a.flush(1, complete=True)
    report = json.loads(a.path.read_text())
    assert report["consumed_counts"]["lingbot/depth"] == 2
    assert report["unique_counts"]["lingbot/depth"] == 1
    assert report["dropped_updates"] == 1
    assert torch.equal(state, torch.random.get_rng_state())
    assert ExposureAudit(tmp_path, 0, 1).path != a.path


def test_configs_are_paired():
    for prefix in "DPM":
        a, b = [
            load_config(config_path=ROOT / f"configs/experiments/lingbot_{prefix.lower()}{i}.yaml")
            for i in (0, 1)
        ]
        assert a.steps == b.steps == 10000
        assert a.batch_size * a.grad_accum * 8 == 32
        assert a.save_every == a.eval_every == 2000 and a.max_samples_per_task is None
        da, db = a.to_dict(), b.to_dict()
        for key in ("data_dir", "pair_manifest", "output_dir"):
            da.pop(key)
            db.pop(key)
        assert da == db


def test_launcher_integrity_and_accumulation(data, tmp_path):
    prepare(data)
    parent = tmp_path / "parent"
    (parent / "lora_adapter").mkdir(parents=True)
    config = load_config(config_path=ROOT / "configs/experiments/lingbot_d0.yaml")
    dump(parent / "metadata.json", {"adapter_weight_source": "online"})
    dump(
        parent / "lora_adapter/adapter_config.json",
        dict(r=32, lora_alpha=64, use_rslora=True, target_modules=config.lora_targets),
    )
    (parent / "lora.pt").write_bytes(b"test-state")
    model = tmp_path / "model"
    model.mkdir()
    dump(model / "model_index.json", {})
    kwargs = dict(base_model=model, base_revision="a" * 40, world_size=4)
    command, audit = build_command(
        "D1", data[2], parent, sha256_file(parent / "lora.pt"), tmp_path / "run", **kwargs
    )
    assert command[command.index("--grad-accum") + 1] == "8"
    assert audit["global_batch"] == 32 and not (tmp_path / "run").exists()
    (data[2] / "full_baseline/pairs.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="manifest changed"):
        build_command(
            "D1", data[2], parent, sha256_file(parent / "lora.pt"), tmp_path / "run", **kwargs
        )
