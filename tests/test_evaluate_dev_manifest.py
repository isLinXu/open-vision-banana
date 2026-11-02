"""Dev-set index freezing, metric sums, geometry and shard merging (no model needed)."""

import json
import math

import pytest
import torch

from scripts import evaluate_dev_manifest as dev


def write_manifest(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def rows():
    out = []
    for i in range(30):
        out.append(
            dict(
                task="depth",
                split="val",
                input=f"d{i}.png",
                target=f"t{i}.png",
                prompt="p",
                target_kind="tartan" if i < 20 else "hyper",
            )
        )
    for i in range(10):
        out.append(
            dict(
                task="semantic",
                split="val",
                input=f"s{i}.png",
                target=f"st{i}.png",
                prompt=f"s{i}",
                source="ade20k",
            )
        )
    out.append(dict(task="depth", split="train", input="x.png", target="y.png", prompt="p"))
    out.append(dict(task="referring", split="test", input="r.png", target="rt.png", prompt="r"))
    return out


def test_parse_counts():
    assert dev.parse_counts("depth=3, semantic=2") == {"depth": 3, "semantic": 2}
    for bad in ("depth", "depth=0", "=3", "depth=x"):
        with pytest.raises(ValueError):
            dev.parse_counts(bad)


def test_allocate_is_proportional_and_capped():
    assert dev.allocate({"a": 20, "b": 10}, 6) == {"a": 4, "b": 2}
    assert (
        dev.allocate({"a": 1, "b": 10}, 5) == {"a": 1, "b": 4}
        or dev.allocate({"a": 1, "b": 10}, 5)["a"] <= 1
    )
    assert sum(dev.allocate({"a": 2, "b": 1}, 10).values()) == 3


def test_build_index_is_deterministic_val_only(tmp_path):
    m = write_manifest(tmp_path / "pairs.jsonl", rows())
    a = dev.build_index(m, {"depth": 6, "semantic": 3}, "salt")
    b = dev.build_index(m, {"depth": 6, "semantic": 3}, "salt")
    assert a == b
    assert a["tasks"]["depth"]["by_source"] == {"tartan": 4, "hyper": 2}
    assert [e["position"] for e in a["entries"]] == list(range(9))
    assert all(e["input"] not in ("x.png", "r.png") for e in a["entries"])
    assert dev.build_index(m, {"depth": 6}, "other")["entries"] != a["entries"][:6]
    with pytest.raises(ValueError, match="No val rows"):
        dev.build_index(m, {"instance": 1}, "salt")


def test_index_guards_manifest_and_cli(tmp_path, capsys):
    m = write_manifest(tmp_path / "pairs.jsonl", rows())
    out = tmp_path / "index.json"
    assert (
        dev.main(["index", "--manifest", str(m), "--counts", "depth=4", "--output", str(out)]) == 0
    )
    assert "sha256" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        dev.main(["index", "--manifest", str(m), "--counts", "depth=4", "--output", str(out)])
    assert len(dev.load_index(out, m)["entries"]) == 4
    m.write_text(m.read_text() + "\n")
    with pytest.raises(ValueError, match="Manifest differs"):
        dev.load_index(out, m)
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    with pytest.raises(ValueError, match="Not a dev index"):
        dev.load_index(bad, m)


def test_shard_entries():
    entries = [{"position": i} for i in range(10)]
    shards = [dev.shard_entries(entries, 3, k) for k in range(3)]
    assert sorted(e["position"] for s in shards for e in s) == list(range(10))
    with pytest.raises(ValueError):
        dev.shard_entries(entries, 3, 3)


def test_depth_geometry_matches_center_crop():
    depth = torch.arange(8 * 4, dtype=torch.float32).reshape(1, 1, 4, 8)
    out = dev.depth_geometry(depth, (8, 4), 4, True)
    assert out.shape[-2:] == (4, 4)
    assert torch.equal(out, depth[..., :, 2:6])
    assert dev.depth_geometry(depth, (8, 4), 2, False).shape[-2:] == (2, 2)
    small = torch.ones(1, 1, 2, 4)
    assert dev.depth_geometry(small, (8, 4), 4, True).shape[-2:] == (4, 4)


def test_depth_sums_cap_clamp_and_bins():
    gt = torch.tensor([[0.5, 2.0], [4.0, 100.0]])
    pred = torch.tensor([[0.5, 4.0], [float("inf"), 1.0]])
    m = dev.depth_sums(pred, gt)
    assert m["pixels"] == 3 and m["pred_out_of_range"] == 1
    assert math.isclose(m["abs_rel"], 0 + 1.0 + (80.0 - 4.0) / 4.0)
    assert m["delta1"] == 1
    assert set(m["bins"]) == {"0-1", "2-3", "3-5"}
    assert dev.depth_sums(pred, gt, cap=1.0)["pixels"] == 1
    assert dev.depth_sums(pred, torch.zeros(2, 2)) == {"pixels": 0, "pred_out_of_range": 0}


def test_referring_and_semantic_sums():
    p = torch.tensor([[True, False], [True, True]])
    g = torch.tensor([[True, True], [False, True]])
    assert dev.referring_sums(p, g) == {"intersection": 2, "union": 4, "iou": 0.5}
    assert dev.referring_sums(p & False, g & False)["iou"] == 1.0
    s = dev.semantic_sums(
        torch.tensor([[0, 1], [1, 1]]), torch.tensor([[0, 0], [1, 1]]), ["a", "b", "c"]
    )
    assert s["per_class"] == {"a": [1, 2], "b": [2, 3]} and s["correct"] == 3 and s["pixels"] == 4


def record(position, task, metrics, status="ok", **extra):
    return dict(position=position, task=task, metrics=metrics, status=status, **extra)


def test_aggregate_all_tasks():
    gt = torch.tensor([[1.0, 2.0]])
    depth = dev.depth_sums(torch.tensor([[1.0, 2.0]]), gt)
    recs = [
        record(
            0,
            "depth",
            depth,
            metrics_cap10=dev.depth_sums(torch.tensor([[1.0, 2.0]]), gt, 10.0),
            gt_floor=depth,
        ),
        record(1, "referring", {"intersection": 1, "union": 2, "iou": 0.5}),
        record(2, "referring", {"intersection": 3, "union": 3, "iou": 1.0}),
        record(3, "semantic", {"per_class": {"a": [1, 2]}, "correct": 3, "pixels": 4}),
        record(4, "instance", {"AP50": 0.5}),
        record(5, "generation", {"clip_text_score": 0.3}),
        record(6, "generation", {}, status="failed"),
    ]
    out = dev.aggregate(recs)
    assert out["depth"]["raw"]["abs_rel"] == 0 and out["depth"]["raw"]["delta1"] == 1
    assert out["depth"]["raw_cap10m"]["rmse"] == 0 and out["depth"]["gt_floor"]["delta1"] == 1
    assert out["referring"]["cIoU"] == 0.8 and out["referring"]["gIoU"] == 0.75
    assert out["semantic"]["mIoU"] == 0.5 and out["semantic"]["pixel_accuracy"] == 0.75
    assert out["instance"]["AP50"] == 0.5
    assert out["generation"]["failed"] == 1 and out["generation"]["proxy"]["clip_text_score"] == 0.3


def test_merge_coverage_duplicates_and_failures(tmp_path, capsys):
    m = write_manifest(tmp_path / "pairs.jsonl", rows())
    index = tmp_path / "index.json"
    dev.main(["index", "--manifest", str(m), "--counts", "semantic=2", "--output", str(index)])
    ok = {"per_class": {"a": [1, 1]}, "correct": 1, "pixels": 1}
    s0 = tmp_path / "s0.jsonl"
    s0.write_text(
        json.dumps(record(0, "semantic", ok)) + "\n" + json.dumps(record(0, "semantic", ok)) + "\n"
    )
    out = tmp_path / "summary.json"
    meta = tmp_path / "meta.json"
    meta.write_text('{"job": "x"}')
    assert (
        dev.main(["merge", "--index", str(index), "--shards", str(s0), "--output-json", str(out)])
        == 3
    )
    result = json.loads(out.read_text())
    assert result["coverage"]["missing"] == 1 and result["coverage"]["duplicates_ignored"] == 1
    s1 = tmp_path / "s1.jsonl"
    s1.write_text(
        json.dumps(record(1, "semantic", {}, status="failed"))
        + "\n"
        + json.dumps(record(1, "semantic", ok))
        + "\n"
    )
    assert (
        dev.main(
            [
                "merge",
                "--index",
                str(index),
                "--shards",
                str(s0),
                str(s1),
                "--meta",
                str(meta),
                "--output-json",
                str(out),
            ]
        )
        == 0
    )
    result = json.loads(out.read_text())
    assert result["complete"] and result["meta"] == {"job": "x"}
    capsys.readouterr()
