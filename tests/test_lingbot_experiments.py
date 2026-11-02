"""Fail-closed dataset preparation, paired configs, and consumed-sample accounting."""

import hashlib
import json
import os
import sys
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
from scripts import run_lingbot_experiment as run
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


@pytest.mark.parametrize(
    "names, steps, every, warmup",
    [
        (("d0", "d1"), 10000, 2000, 500),
        (("p0", "p1"), 10000, 2000, 500),
        (("fresh_baseline", "fresh_plus_lingbot"), 20000, 2500, 1000),
    ],
)
def test_configs_are_paired(names, steps, every, warmup):
    a, b = [load_config(config_path=ROOT / f"configs/experiments/lingbot_{n}.yaml") for n in names]
    assert a.steps == b.steps == steps
    assert a.warmup_steps == b.warmup_steps == warmup and a.ema_warmup == 500
    assert a.batch_size * a.grad_accum * 8 == 32
    assert a.save_every == a.eval_every == every and a.max_samples_per_task is None
    da, db = a.to_dict(), b.to_dict()
    for key in ("data_dir", "pair_manifest", "output_dir"):
        da.pop(key)
        db.pop(key)
    assert da == db


def legacy_parent(tmp_path):
    """A pinned-parent checkpoint and a base snapshot accepted by the D/P launcher."""
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
    return parent, model


def test_launcher_integrity_and_accumulation(data, tmp_path):
    prepare(data)
    parent, model = legacy_parent(tmp_path)
    kwargs = dict(base_model=model, base_revision="a" * 40, world_size=4)
    command, audit = build_command(
        "D1", data[2], parent, sha256_file(parent / "lora.pt"), tmp_path / "run", **kwargs
    )
    resume = command.index("--resume")
    assert command[resume : resume + 4] == [
        "--resume",
        str(parent),
        "--resume-weights-only",
        "--grad-accum",
    ]
    assert command[command.index("--grad-accum") + 1] == "8"
    assert audit["global_batch"] == 32 and not (tmp_path / "run").exists()
    (data[2] / "full_baseline/pairs.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="manifest changed"):
        build_command(
            "D1", data[2], parent, sha256_file(parent / "lora.pt"), tmp_path / "run", **kwargs
        )


def test_prepare_keeps_test_split_and_parallel_hashes_match(data):
    base, ling, output = data
    rows = [json.loads(line) for line in base.read_text().splitlines()]
    rows.append(dict(rows[-1], split="test"))
    image = base.parent / "held-out-test.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(image)
    rows[-1]["input"] = rows[-1]["target"] = image.name
    jsonl(base, rows)
    report = prep.prepare(
        base, ling, output, sha256_file(base), depth_rows=1, replay_rows=1, workers=2
    )
    counts = report["manifests"]["full_baseline"]["task_split_counts"]
    assert counts[f"{rows[-1]['task']}/test"] == 1
    paths = [base.parent / r["input"] for r in rows]
    assert prep.pixel_hashes(paths, 2) == prep.pixel_hashes(paths, 1)


def test_prepare_rejects_unknown_split(data):
    base, ling, output = data
    rows = [json.loads(line) for line in base.read_text().splitlines()]
    rows[0]["split"] = "dev"
    jsonl(base, rows)
    with pytest.raises(ValueError, match="task/split"):
        prep.prepare(base, ling, output, sha256_file(base), depth_rows=1, replay_rows=1)


WEIGHTS = b"weights"


def snapshot(root, *, merged=None, shard=True):
    (root / "transformer").mkdir(parents=True)
    dump(root / "model_index.json", {})
    if shard:
        (root / "transformer/shard-1.safetensors").write_bytes(WEIGHTS)
    if merged is not None:
        dump(root / "merge_metadata.json", merged)
    return root


def merged_metadata(**overrides):
    meta = dict(
        adapter_sha256=run.PARENT_ADAPTER_SHA256,
        base_revision="a" * 40 + " (transformer shard hashes checked)",
        transformer_shards={"shard-1.safetensors": hashlib.sha256(WEIGHTS).hexdigest()},
    )
    return {**meta, **overrides}


@pytest.fixture
def snapshots(tmp_path):
    return snapshot(tmp_path / "E", merged=merged_metadata()), snapshot(tmp_path / "Q")


@pytest.mark.parametrize(
    "arm, kwargs, expect",
    [
        (
            "M0",
            {},
            dict(pool="full_baseline", mode="lora", init="E", steps=20000, rank=32, res=512),
        ),
        (
            "M1",
            {},
            dict(pool="full_plus_lingbot", mode="lora", init="E", steps=20000, rank=32, res=512),
        ),
        (
            "Q0",
            {},
            dict(pool="full_baseline", mode="lora", init="Q", steps=20000, rank=32, res=512),
        ),
        (
            "Q1",
            {},
            dict(pool="full_plus_lingbot", mode="lora", init="Q", steps=20000, rank=32, res=512),
        ),
        (
            "F-M1",
            {"lr": 5e-6},
            dict(pool="full_plus_lingbot", mode="full", init="E", steps=15000, rank=None, res=512),
        ),
        (
            "K-Q0",
            {"lora_rank": 64},
            dict(pool="full_baseline", mode="lora", init="Q", steps=20000, rank=64, res=512),
        ),
        (
            "K-Q1",
            {"lora_rank": 128},
            dict(pool="full_plus_lingbot", mode="lora", init="Q", steps=20000, rank=128, res=512),
        ),
        (
            "R-M0",
            {},
            dict(pool="full_baseline", mode="lora", init="E", steps=20000, rank=32, res=768),
        ),
        (
            "R-M1",
            {},
            dict(pool="full_plus_lingbot", mode="lora", init="E", steps=20000, rank=32, res=768),
        ),
    ],
)
def test_fresh_arms(data, tmp_path, snapshots, arm, kwargs, expect):
    prepare(data)
    init_dir = snapshots[0] if expect["init"] == "E" else snapshots[1]
    command, audit = run.build_fresh_command(
        arm,
        data[2],
        tmp_path / "run",
        init_dir=init_dir,
        base_revision="a" * 40,
        verify_weights=True,
        **kwargs,
    )
    flag = lambda name: command[command.index(name) + 1] if name in command else None  # noqa: E731
    assert "--resume" not in command and "--resume-weights-only" not in command
    assert flag("--pair-manifest").endswith(f"{expect['pool']}/pairs.jsonl")
    assert flag("--steps") == ("15000" if expect["mode"] == "full" else None)
    assert audit["steps"] == expect["steps"] and flag("--grad-accum") == "4"
    assert flag("--config").endswith(f"lingbot_fresh_{expect['pool'].removeprefix('full_')}.yaml")
    assert audit["family"] == "fresh"
    assert flag("--train-mode") == ("full" if expect["mode"] == "full" else None)
    assert flag("--lr") == (repr(kwargs["lr"]) if "lr" in kwargs else None)
    assert flag("--lora-rank") == (str(kwargs["lora_rank"]) if "lora_rank" in kwargs else None)
    assert flag("--resolution") == ("768" if expect["res"] == 768 else None)
    assert audit["mode"] == expect["mode"] and audit["pool"] == expect["pool"]
    assert audit["lora_rank"] == expect["rank"] and audit["resolution"] == expect["res"]
    assert audit["initialization"]["init_source"] == expect["init"]
    assert audit["initialization"]["merged"] is (expect["init"] == "E")
    check = audit["initialization"]["weights_check"]
    assert check == (
        "sha256 against pins"
        if expect["init"] == "E"
        else "sha256 recorded, no reference to compare (pass --weight-pins)"
    )
    assert audit["global_batch"] == 32 and not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "arm, kwargs, match",
    [
        ("M1", {"lr": 1e-5}, "--lr is required for F-M1"),
        ("F-M1", {}, "--lr is required for F-M1"),
        ("F-M1", {"lr": 0.0}, "must be positive"),
        ("Q0", {"lora_rank": 64}, "--lora-rank is required"),
        ("K-Q1", {}, "--lora-rank is required"),
        ("K-Q1", {"lora_rank": 32}, "rank grid"),
        ("X1", {}, "Unknown experiment arm"),
        ("M1", {"world_size": 3}, "global batch 32"),
    ],
)
def test_fresh_arm_arguments(data, tmp_path, snapshots, arm, kwargs, match):
    prepare(data)
    with pytest.raises(ValueError, match=match):
        run.build_fresh_command(
            arm, data[2], tmp_path / "run", init_dir=snapshots[0], base_revision="a" * 40, **kwargs
        )


def test_initializations_cannot_be_swapped(tmp_path, snapshots):
    e_dir, q_dir = snapshots
    with pytest.raises(ValueError, match="needs a merged snapshot"):
        run.check_initialization("E", q_dir, "a" * 40)
    with pytest.raises(ValueError, match="unmerged official base"):
        run.check_initialization("Q", e_dir, "a" * 40)
    with pytest.raises(ValueError, match="E or Q"):
        run.check_initialization("Z", q_dir, "a" * 40)
    assert run.check_initialization("Q", q_dir, "a" * 40)["weights_check"].startswith("sizes")


@pytest.mark.parametrize(
    "meta, match",
    [
        (dict(adapter_sha256="0" * 64), "pinned E2B parent"),
        (dict(base_revision="b" * 40), "different base revision"),
        (dict(transformer_shards={"other.safetensors": "0"}), "differ from merge_metadata"),
        (dict(transformer_shards={"shard-1.safetensors": "0" * 64}), "hashes differ"),
    ],
)
def test_merged_snapshot_identity(tmp_path, meta, match):
    model = snapshot(tmp_path / "E", merged=merged_metadata(**meta))
    with pytest.raises(ValueError, match=match):
        run.check_initialization("E", model, "a" * 40, verify_weights=True)


def test_weight_pins_and_shards(tmp_path, snapshots):
    _, q_dir = snapshots
    good = {"shard-1.safetensors": sha256_file(q_dir / "transformer/shard-1.safetensors")}
    record = run.check_initialization("Q", q_dir, "a" * 40, verify_weights=True, pins=good)
    assert record["transformer_shard_sha256"] == good
    with pytest.raises(ValueError, match="hashes differ"):
        run.check_initialization(
            "Q", q_dir, "a" * 40, verify_weights=True, pins={"shard-1.safetensors": "0"}
        )
    with pytest.raises(ValueError, match="Pinned transformer shards differ"):
        run.check_initialization("Q", q_dir, "a" * 40, pins={"x.safetensors": "0"})
    empty = snapshot(tmp_path / "empty", shard=False)
    with pytest.raises(ValueError, match="No transformer shards"):
        run.check_initialization("Q", empty, "a" * 40)


def test_legacy_launcher_rejects_fresh_arms(data, tmp_path):
    with pytest.raises(ValueError, match="parent-continuation"):
        build_command(
            "M1",
            data[2],
            tmp_path,
            "0",
            tmp_path / "run",
            base_model=tmp_path,
            base_revision="a" * 40,
        )


def test_cli_fresh_arm_prints_and_executes(data, tmp_path, snapshots, monkeypatch, capsys):
    prepare(data)
    calls = []
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    pins = tmp_path / "pins.json"
    dump(
        pins, {"shard-1.safetensors": sha256_file(snapshots[1] / "transformer/shard-1.safetensors")}
    )
    out = tmp_path / "run"
    argv = [
        "run",
        "--arm",
        "K-Q1",
        "--lora-rank",
        "64",
        "--data-root",
        str(data[2]),
        "--output-dir",
        str(out),
        "--base-model-revision",
        "a" * 40,
        "--init-dir",
        str(snapshots[1]),
        "--weight-pins",
        str(pins),
        "--verify-weights",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    run.main()
    assert "--lora-rank 64" in capsys.readouterr().out and not calls and not out.exists()
    monkeypatch.setattr(sys, "argv", [*argv, "--execute"])
    run.main()
    ((command, kwargs),) = calls
    launch = json.loads((out / "lingbot-launch.json").read_text())
    assert launch["arm"] == "K-Q1" and launch["initialization"]["init_source"] == "Q"
    assert kwargs["env"]["OPEN_VISION_BANANA_MODEL_PATH"] == str(snapshots[1].resolve())
    assert kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(ROOT / "src")


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--arm", "M1"], "take --init-dir"),
        (["--arm", "M1", "--init-dir", "x", "--parent", "p"], "never a parent adapter"),
        (["--arm", "D1"], "need --parent"),
    ],
)
def test_cli_rejects_mixed_families(tmp_path, monkeypatch, capsys, extra, message):
    argv = [
        "run",
        "--data-root",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "o"),
        "--base-model-revision",
        "a" * 40,
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        run.main()
    assert message in capsys.readouterr().err


def test_cli_legacy_arm_prints(data, tmp_path, monkeypatch, capsys):
    prepare(data)
    parent, model = legacy_parent(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--arm",
            "D0",
            "--data-root",
            str(data[2]),
            "--output-dir",
            str(tmp_path / "run"),
            "--base-model-revision",
            "a" * 40,
            "--parent",
            str(parent),
            "--parent-sha256",
            sha256_file(parent / "lora.pt"),
            "--base-model-dir",
            str(model),
        ],
    )
    run.main()
    assert "--resume-weights-only" in capsys.readouterr().out


def test_e_pins_cannot_override_merge_metadata(snapshots):
    e_dir, _ = snapshots
    with pytest.raises(ValueError, match="Pins disagree"):
        run.check_initialization("E", e_dir, "a" * 40, pins={"shard-1.safetensors": "0" * 64})
    same = {"shard-1.safetensors": hashlib.sha256(WEIGHTS).hexdigest()}
    record = run.check_initialization("E", e_dir, "a" * 40, verify_weights=True, pins=same)
    assert record["weights_check"] == "sha256 against pins"
