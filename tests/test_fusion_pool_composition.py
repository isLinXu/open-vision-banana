"""Synthetic composition contracts; no images, models, network or GPU required."""
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts/compose_fusion_training_pool.py"
spec = importlib.util.spec_from_file_location("fusion_composer", MODULE)
composer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(composer)


@pytest.fixture
def pool(tmp_path):
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"synthetic existence-only asset")

    def row(task, name, split="train"):
        return dict(
            task=task,
            sample_id=name,
            source_dataset=name,
            split=split,
            input="asset.bin",
            target="asset.bin",
            source_image="provenance-only.jpg",
        )

    old = row("depth", "old")
    baseline = [
        old,
        row("referring", "old_ref"),
        row("normal", "normal"),
        row("depth", "original_hypersim"),
        row("generation", "replay"),
        row("normal", "heldout_normal", "val"),
        row("referring", "test", "test"),
    ]
    records = {"baseline": baseline, "old_hypersim": [old]}
    for role in composer.ROLES[2:]:
        task = sorted(composer.ROLE_TASKS[role])[0]
        records[role] = [row(task, role)]
    records["full_res"].append(row("referring", "res_val", "val"))
    entries = {}
    for role, rows in records.items():
        path = tmp_path / f"{role}.jsonl"
        composer.write_rows(path, rows)
        entries[role] = dict(
            manifest=path.name,
            root=".",
            sha256=composer.digest(path),
            expected_train=sum(r["split"] == "train" for r in rows),
        )
    path = tmp_path / "spec.json"
    config = dict(components=entries, expected_removed_referring=1)
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, config, tmp_path


def rewrite(pool, role, transform):
    path, config, root = pool
    manifest = root / config["components"][role]["manifest"]
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    transform(rows)
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    config["components"][role]["sha256"] = composer.digest(manifest)
    path.write_text(json.dumps(config))


def test_replacements_paths_and_heldout(pool):
    path, _, root = pool
    report = composer.compose(path, root / "out")
    rows = [json.loads(x) for x in (root / "out/pairs.jsonl").read_text().splitlines()]
    training = [json.loads(x) for x in (root / "out/training.jsonl").read_text().splitlines()]
    assert len(training) == 10
    assert all(r["split"] in {"train", "val"} for r in training)
    assert [r["sample_id"] for r in training if r["split"] == "val"] == ["heldout_normal"]
    assert report["train_rows"] == 9
    assert report["baseline_heldout_rows"] == 2
    assert report["component_heldout_rows"] == 1
    assert report["training_ready"] is False
    assert not (root / "out/COMPLETE.json").exists()
    assert not (root / "out/BUILD_INCOMPLETE").exists()
    assert all(r["split"] == "train" and r["task"] != "normal" for r in rows)
    assert "original_hypersim" in {r["sample_id"] for r in rows}
    assert "old" not in {r["sample_id"] for r in rows}
    assert all(Path(r["input"]).is_absolute() for r in rows)
    assert all(r["source_image"] == "provenance-only.jpg" for r in rows)
    with pytest.raises(FileExistsError):
        composer.compose(path, root / "out")


@pytest.mark.parametrize(
    "role,key,value",
    [
        ("llmseg", "input", "missing"),
        ("indoor", "valid_mask", "missing"),
        ("full_res", "split", "testA"),
        ("llmseg", "source_split", "val"),
        ("coconut", "task", "unknown"),
        ("old_hypersim", "sample_id", "absent"),
    ],
)
def test_invalid_source_fails_without_publication(pool, role, key, value):
    rewrite(pool, role, lambda rows: rows[0].update({key: value}))
    path, _, root = pool
    with pytest.raises((ValueError, FileNotFoundError)):
        composer.compose(path, root / "out")
    assert not (root / "out").exists()


def test_sha_and_count_fail(pool):
    path, config, root = pool
    config["components"]["baseline"]["sha256"] = "0" * 64
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="SHA"):
        composer.compose(path, root / "out")


def test_no_baseline_reconstruction(pool):
    path, config, root = pool
    config["components"]["baseline"]["manifest"] = "absent.jsonl"
    path.write_text(json.dumps(config))
    with pytest.raises(FileNotFoundError):
        composer.compose(path, root / "out")


def test_required_counts_and_roles(pool):
    path, config, root = pool
    config["components"]["llmseg"]["expected_train"] = 100
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="expected_train"):
        composer.compose(path, root / "out")
    del config["components"]["llmseg"]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="role"):
        composer.compose(path, root / "out")


def test_official_training_case_is_preserved(pool, tmp_path):
    spec_path, spec_data, _ = pool
    path = Path(spec_data["components"]["indoor"]["manifest"])
    if not path.is_absolute():
        path = spec_path.parent / path
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["source_split"] = "Training"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    spec_data["components"]["indoor"]["sha256"] = composer.digest(path)
    spec_path.write_text(json.dumps(spec_data))
    output = tmp_path / "official-case"
    composer.compose(spec_path, output)
    output_rows = [json.loads(line) for line in (output / "pairs.jsonl").read_text().splitlines()]
    assert any(row.get("source_split") == "Training" for row in output_rows)


def test_target_only_generation_is_not_rewritten(pool):
    path, _, root = pool

    def remove_generation_input(rows):
        for row in rows:
            if row["task"] == "generation":
                row.pop("input")

    rewrite(pool, "baseline", remove_generation_input)
    composer.compose(path, root / "target-only")
    rows = [json.loads(x) for x in (root / "target-only/pairs.jsonl").read_text().splitlines()]
    generation = next(row for row in rows if row["task"] == "generation")
    assert "input" not in generation
    assert Path(generation["target"]).is_file()


def test_removed_baseline_assets_need_not_exist(pool):
    def stale(rows):
        for row in rows:
            if row["sample_id"] in {"old", "old_ref", "normal"}:
                row["input"] = "unavailable-old-input"
                row["target"] = "unavailable-old-target"
                row["valid_mask"] = "unavailable-old-mask"

    rewrite(pool, "baseline", stale)
    path, _, root = pool
    result = composer.compose(path, root / "out")
    assert result["train_rows"] == 9


@pytest.mark.parametrize("sample_id", ["original_hypersim", "heldout_normal", "test"])
def test_retained_baseline_assets_still_required(pool, sample_id):
    def stale(rows):
        for row in rows:
            if row["sample_id"] == sample_id:
                row["target"] = "missing-retained-target"

    rewrite(pool, "baseline", stale)
    path, _, root = pool
    with pytest.raises(FileNotFoundError):
        composer.compose(path, root / "out")
    assert not (root / "out").exists()
