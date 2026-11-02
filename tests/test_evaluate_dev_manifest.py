"""Dev-set index freezing, metric sums, geometry and shard merging (no model needed)."""

import json
import math
from collections import defaultdict

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
    assert dev.allocate({"a": 1, "b": 10}, 5) == {"a": 0, "b": 5}
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


def test_depth_geometry_matches_dataset_crop(tmp_path):
    """The metric map lands on the same pixels as ManifestVisionDataset's val canvas."""
    from PIL import Image

    from open_vision_banana.training.trainer import ManifestVisionDataset

    w, h = 37, 23
    xs = torch.arange(w, dtype=torch.float32).repeat(h, 1)
    ys = torch.arange(h, dtype=torch.float32).unsqueeze(1).repeat(1, w)
    code = torch.stack([xs, ys, torch.zeros_like(xs)]) / 255.0
    Image.fromarray((code.permute(1, 2, 0).numpy() * 255).round().astype("uint8")).save(
        tmp_path / "in.png"
    )
    write_manifest(
        tmp_path / "pairs.jsonl",
        [
            dict(
                task="semantic",
                split="val",
                input="in.png",
                target="in.png",
                prompt="p",
                mapping={"a": [0, 0, 0]},
            )
        ],
    )
    for resolution in (8, 16):
        ds = ManifestVisionDataset(
            tmp_path / "pairs.jsonl",
            resolution,
            "val",
            {"semantic"},
            aspect_preserving_crop=True,
            data_root=tmp_path,
        )
        canvas = (ds[0]["target_rgb"] * 255).round()
        mapped = dev.depth_geometry(code.unsqueeze(0) * 255, (w, h), resolution, True)[0]
        assert torch.equal(canvas[:2], mapped[:2].round())


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


def referring_state(pred, gt):
    from open_vision_banana.evaluation.referring import ReferringAccumulator
    from open_vision_banana.evaluation.shards import accumulator_state

    accumulator = ReferringAccumulator()
    accumulator.update(torch.tensor(pred), torch.tensor(gt))
    return accumulator_state(accumulator)


def test_semantic_sums():
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
        record(1, "referring", referring_state([[True, False]], [[True, True]])),
        record(2, "referring", referring_state([[True, True, True]], [[True, True, True]])),
        record(3, "semantic", {"per_class": {"a": [1, 2]}, "correct": 3, "pixels": 4}),
        record(4, "instance", {"AP50": 0.5}),
        record(5, "generation", {"clip_text_score": 0.3}),
        record(6, "generation", {}, status="failed"),
    ]
    out = dev.aggregate(recs)
    assert out["depth"]["raw"]["abs_rel"] == 0 and out["depth"]["raw"]["delta1"] == 1
    assert out["depth"]["raw_cap10m"]["rmse"] == 0 and out["depth"]["gt_floor"]["delta1"] == 1
    assert out["referring"]["cIoU"] == 0.8 and out["referring"]["gIoU"] == 0.75
    assert out["referring"]["samples"] == 2
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


class _FakeGenerator:
    """Returns the input image; raises / returns NaN for marked prompts."""

    def generate(self, prompt, image, **kwargs):
        if "boom" in prompt:
            raise RuntimeError("boom")
        if "nan" in prompt:
            return torch.full_like(image, float("nan"))
        return image.clone()


class _FakeClip:
    def __init__(self):
        self.vision_model = lambda pixel_values: type(
            "O", (), {"pooler_output": pixel_values.flatten(1)[:, :4]}
        )()
        self.text_model = lambda input_ids, attention_mask: type(
            "O", (), {"pooler_output": input_ids[:, :4].float()}
        )()
        self.visual_projection = self.text_projection = lambda x: x

    def to(self, device):
        return self

    def eval(self):
        return self


class _FakeProcessor:
    def __call__(self, text, images, **kwargs):
        import numpy as np

        pixels = torch.stack(
            [torch.from_numpy(np.asarray(i, np.float32)).permute(2, 0, 1) for i in images]
        )
        return _Batch(
            pixel_values=pixels + 1.0,
            input_ids=torch.ones(1, 8, dtype=torch.long),
            attention_mask=torch.ones(1, 8, dtype=torch.long),
        )


class _Batch(dict):
    def to(self, device):
        return self


def eval_fixture(tmp_path):
    import numpy as np
    from PIL import Image

    from open_vision_banana.core.depth_mapper import DepthRGBMapper
    from open_vision_banana.data.res_contract import build_binary_referring_prompt

    def png(name, array):
        Image.fromarray(np.asarray(array, np.uint8)).save(tmp_path / name)
        return name

    rgb = png("img.png", np.random.default_rng(0).integers(0, 255, (24, 32, 3)))
    depth = DepthRGBMapper(colormap="cube_edge").depth_to_rgb(torch.full((1, 1, 24, 32), 2.5))
    depth_png = png("depth.png", (depth[0].permute(1, 2, 0).numpy() * 255).round())
    mask = np.zeros((24, 32, 3))
    mask[4:12, 4:20] = 255
    mask_png = png("mask.png", mask)
    seg = np.zeros((24, 32, 3))
    seg[:, 16:] = (255, 0, 0)
    seg_png = png("seg.png", seg)
    rows = [
        dict(
            task="depth",
            split="val",
            input=rgb,
            target=depth_png,
            prompt="Generate a metric depth visualization of this image.",
            target_kind="tartan",
        ),
        dict(
            task="depth",
            split="val",
            input=rgb,
            target=depth_png,
            prompt="boom depth",
            target_kind="tartan",
        ),
        dict(
            task="depth",
            split="val",
            input=rgb,
            target=depth_png,
            prompt="nan depth",
            target_kind="tartan",
        ),
        dict(
            task="referring",
            split="val",
            input=rgb,
            target=mask_png,
            expression="the box",
            encoding="binary_white_black_v1",
            prompt=build_binary_referring_prompt("the box"),
        ),
        dict(
            task="semantic",
            split="val",
            input=rgb,
            target=seg_png,
            prompt="seg",
            mapping={"wall": [0, 0, 0], "bed": [255, 0, 0]},
        ),
        dict(
            task="instance",
            split="val",
            input=rgb,
            target=mask_png,
            prompt="inst",
            background_color=[0, 0, 0],
            category="box",
        ),
        dict(task="generation", split="val", input=rgb, target=rgb, prompt="Add a tree"),
    ]
    manifest = write_manifest(tmp_path / "pairs.jsonl", rows)
    index = tmp_path / "index.json"
    dev.main(
        [
            "index",
            "--manifest",
            str(manifest),
            "--counts",
            "depth=3,referring=1,semantic=1,instance=1,generation=1",
            "--output",
            str(index),
        ]
    )
    return manifest, index


def test_eval_end_to_end_on_cpu(tmp_path, monkeypatch):
    import transformers

    from open_vision_banana.inference import pipeline

    manifest, index = eval_fixture(tmp_path)
    loads = []
    monkeypatch.setattr(
        pipeline.VisionBanana,
        "from_pretrained",
        classmethod(
            lambda cls, **kw: loads.append(kw) or type("M", (), {"generator": _FakeGenerator()})()
        ),
    )
    monkeypatch.setattr(
        transformers.CLIPModel, "from_pretrained", classmethod(lambda cls, *a, **k: _FakeClip())
    )
    monkeypatch.setattr(
        transformers.CLIPProcessor,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: _FakeProcessor()),
    )
    shards = []
    for k in range(2):
        out = tmp_path / f"shard{k}.jsonl"
        args = [
            "eval",
            "--manifest",
            str(manifest),
            "--index",
            str(index),
            "--device",
            "cpu",
            "--resolution",
            "16",
            "--num-shards",
            "2",
            "--shard-index",
            str(k),
            "--output-jsonl",
            str(out),
            "--save-dir",
            str(tmp_path / "png"),
        ]
        assert dev.main(args) == 0
        shards.append(out)
    records = [r for s in shards for r in dev.read_jsonl(s)]
    assert len(records) == 7 and len(loads) == 2 and loads[0]["device"] == "cpu"
    by_task = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r)
    failed = sorted(r["error"] for r in records if r["status"] == "failed")
    assert len(failed) == 2 and {r["task"] for r in records if r["status"] == "failed"} == {"depth"}
    assert failed[0].startswith("RuntimeError: boom") and "non-finite" in failed[1]
    assert "AP@0.5" in by_task["instance"][0]["metrics"]
    depth = next(r for r in by_task["depth"] if r["status"] == "ok")
    assert depth["gt_floor"]["delta1"] == depth["gt_floor"]["pixels"] > 0
    assert by_task["referring"][0]["metrics"]["kind"] == "referring"
    assert set(by_task["semantic"][0]["metrics"]["per_class"]) <= {"wall", "bed"}
    assert by_task["generation"][0]["metrics"]["clip_image_to_reference"] > 0.99
    assert len(list((tmp_path / "png").glob("*_pred.png"))) == 5
    summary = tmp_path / "summary.json"
    assert (
        dev.main(
            [
                "merge",
                "--index",
                str(index),
                "--shards",
                *map(str, shards),
                "--output-json",
                str(summary),
            ]
        )
        == 3
    )
    metrics = json.loads(summary.read_text())["metrics"]
    assert metrics["depth"]["failed"] == 2 and metrics["referring"]["samples"] == 1
    # A rerun retries only the failed samples; a fully scored shard never loads the model.
    loads.clear()
    assert dev.main(args) == 0 and len(loads) == 1
    ok_only = tmp_path / "ok.jsonl"
    ok_only.write_text("".join(json.dumps(r) + "\n" for r in records if r["status"] == "ok"))
    monkeypatch.setattr(
        pipeline.VisionBanana,
        "from_pretrained",
        classmethod(lambda cls, **kw: pytest.fail("loaded")),
    )
    done = [
        "eval",
        "--manifest",
        str(manifest),
        "--index",
        str(index),
        "--device",
        "cpu",
        "--resolution",
        "16",
        "--num-shards",
        "1",
        "--shard-index",
        "0",
        "--output-jsonl",
        str(ok_only),
    ]
    ok_only.write_text("".join(json.dumps({**r, "status": "ok"}) + "\n" for r in records))
    assert dev.main(done) == 0


def test_dataset_prompt_matches_dataset():
    from open_vision_banana.data.res_contract import build_binary_referring_prompt

    row = dict(expression="the box", encoding="binary_white_black_v1")
    assert dev.dataset_prompt(row, "referring") == build_binary_referring_prompt("the box")
    assert dev.dataset_prompt({"prompt": "p"}, "semantic") == "p"
    assert dev.dataset_prompt({}, "depth")
