"""Paper 2D / 3D / generation adapters and CLI grouping."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from open_vision_banana.data import build_dataset
from open_vision_banana.data.datasets import SACoInstanceDataset
from open_vision_banana.data.paper_eval_datasets import (
    GenAIBenchDataset,
    ImgEditDataset,
    ReasonSegDataset,
)
from open_vision_banana.evaluation.benchmark import (
    DEPTH_EVAL_CAPS,
    apply_depth_benchmark_protocol,
    canonical_depth_dataset,
)
from open_vision_banana.evaluation.paper_capabilities import (
    PAPER_CAPABILITIES,
    capability_for_benchmark,
)
from scripts.evaluate_paper_benchmarks import (
    NATIVE_GT_DATASETS,
    build_jobs,
    preflight_jobs,
    selected_benchmarks,
)


def test_capability_groups_match_paper_axes() -> None:
    assert selected_benchmarks("2d") == {"semantic", "referring", "instance"}
    assert selected_benchmarks("3d") == {"depth", "normal"}
    assert selected_benchmarks("generation") == {"generation"}
    assert "nuscenes" in PAPER_CAPABILITIES["3d"]
    assert PAPER_CAPABILITIES["2d"]["refcocog"]["paper"] == 0.738
    assert capability_for_benchmark("depth", "nuscenes") == "3d"
    assert capability_for_benchmark("referring", "reasonseg") == "2d"


def test_build_jobs_wires_paper_roots(tmp_path: Path) -> None:
    args = type(
        "Args",
        (),
        {
            "benchmark": "all",
            "depth_root": None,
            "normal_root": None,
            "nyu_root": tmp_path / "nyu",
            "ibims_root": None,
            "eth3d_root": None,
            "diode_root": None,
            "kitti_root": None,
            "nuscenes_root": tmp_path / "nuscenes",
            "scannet_root": None,
            "vkitti_root": None,
            "cityscapes_root": tmp_path / "cityscapes",
            "refcocog_root": tmp_path / "refcocog",
            "reasonseg_root": tmp_path / "reasonseg",
            "saco_root": tmp_path / "saco",
            "genai_bench_root": tmp_path / "genai",
            "imgedit_root": tmp_path / "imgedit",
        },
    )()
    jobs = build_jobs(args)
    keys = {(bench, name) for bench, name, _root, _task in jobs}
    assert ("depth", "nyu") in keys
    assert ("depth", "nuscenes") in keys
    assert ("semantic", "cityscapes") in keys
    assert ("referring", "refcocog") in keys
    assert ("referring", "reasonseg") in keys
    assert ("instance", "saco") in keys
    assert ("generation", "genai_bench") in keys
    assert ("generation", "imgedit") in keys


def test_every_native_resolution_job_has_a_native_adapter_key(tmp_path: Path) -> None:
    # --native-resolution covers depth, normal, referring and semantic jobs; each job's adapter
    # key must be in NATIVE_GT_DATASETS or the run exits before scoring (Cityscapes did).
    roots = ("nyu", "ibims", "eth3d", "diode", "kitti", "nuscenes", "scannet", "vkitti")
    roots += ("cityscapes", "refcocog", "reasonseg", "saco", "genai_bench", "imgedit")
    fields = {f"{name}_root": tmp_path / name for name in roots}
    fields |= {"benchmark": "all", "depth_root": None, "normal_root": None}
    jobs = build_jobs(type("Args", (), fields)())
    native = [job for job in jobs if job[0] in {"depth", "normal", "referring", "semantic"}]
    assert {(bench, name) for bench, name, _root, _key in native} >= {("semantic", "cityscapes")}
    assert [job for job in native if job[3] not in NATIVE_GT_DATASETS] == []


def test_reasonseg_loads_lisa_json_and_polygon(tmp_path: Path) -> None:
    split = tmp_path / "val"
    split.mkdir()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(split / "frame.jpg")
    (split / "frame.json").write_text(
        json.dumps(
            {
                "text": "the object a person would sit on",
                "is_sentence": True,
                "shapes": [{"points": [[1, 1], [6, 1], [6, 6], [1, 6]]}],
            }
        ),
        encoding="utf-8",
    )
    dataset = ReasonSegDataset(tmp_path, split="val", resolution=8, native_gt=True)
    assert len(dataset) == 1
    item = dataset[0]
    assert "person would sit on" in item["expression"]
    assert item["gt_mask"].shape == (8, 8)
    assert bool(item["gt_mask_native"].any())


def test_reasonseg_factory_key(tmp_path: Path) -> None:
    dataset = build_dataset("reasonseg", str(tmp_path), split="val", resolution=8)
    assert isinstance(dataset, ReasonSegDataset)
    assert len(dataset) == 0


def test_saco_gold_query_jsonl(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (8, 8), (4, 5, 6)).save(images / "scene.jpg")
    mask = np.zeros((8, 8, 3), dtype=np.uint8)
    mask[2:6, 2:6] = (255, 0, 0)
    Image.fromarray(mask).save(images / "scene_mask.png")
    (tmp_path / "queries.jsonl").write_text(
        json.dumps(
            {
                "image": "images/scene.jpg",
                "query": "a red box",
                "present": True,
                "mask": "images/scene_mask.png",
            }
        )
        + "\n"
        + json.dumps({"image": "images/scene.jpg", "query": "a zebra", "present": False})
        + "\n",
        encoding="utf-8",
    )
    dataset = SACoInstanceDataset(tmp_path, split="val", resolution=8)
    assert len(dataset) == 2
    positive = dataset[0]
    negative = dataset[1]
    assert positive["object_name"] == "a red box"
    assert positive["gt_present"] is True
    assert float(positive["target_rgb"].sum()) > 0
    assert negative["gt_present"] is False
    assert float(negative["target_rgb"].sum()) == 0


def test_nuscenes_depth_benchmark_builds_and_returns_metric_gt(tmp_path: Path) -> None:
    rgb_dir = tmp_path / "val" / "samples" / "CAM_FRONT"
    depth_dir = tmp_path / "val" / "depth"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    Image.new("RGB", (6, 4), (8, 9, 10)).save(rgb_dir / "cam.jpg")
    np.save(depth_dir / "cam.npy", np.full((4, 6), 12.5, dtype=np.float32))
    args = type(
        "Args",
        (),
        {"benchmark": "depth", "depth_root": None, "nuscenes_root": tmp_path}
        | {f"{name}_root": None for name in ("nyu", "ibims", "eth3d", "diode", "kitti")},
    )()
    jobs = build_jobs(args)
    # build_dataset's first argument is ``task``; a {"task": "depth"} adapter kwarg clashed.
    preflight = preflight_jobs(jobs, split="val", resolution=8)
    assert preflight["ok"], preflight
    assert [job["samples"] for job in preflight["jobs"]] == [1]
    dataset = build_dataset(jobs[0][3], str(tmp_path), split="val", resolution=8, native_gt=True)
    assert dataset.TASK_TYPE == "depth"
    item = dataset[0]
    assert item["gt_depth"].shape[0] == 1
    assert float(item["gt_depth"].mean()) == pytest.approx(12.5)
    assert "gt_depth_native" in item


def test_nuscenes_uses_80m_cap_without_garg_crop() -> None:
    assert canonical_depth_dataset("nuscenes") == "nuscenes"
    assert DEPTH_EVAL_CAPS["nuscenes"] == 80.0
    pred = torch.tensor([[[[200.0, 20.0]]]])
    gt = torch.tensor([[[[20.0, 20.0]]]])
    capped, _, mask = apply_depth_benchmark_protocol("nuscenes", pred, gt)
    assert float(capped.max()) == 80.0
    assert bool(mask.all())


def test_genai_bench_prompt_only_and_imgedit_pairs(tmp_path: Path) -> None:
    genai = tmp_path / "genai"
    genai.mkdir()
    (genai / "prompts.json").write_text(
        json.dumps({"0": "a red bicycle on cobblestones"}),
        encoding="utf-8",
    )
    prompts = GenAIBenchDataset(genai, split="val", resolution=8)
    assert len(prompts) == 1
    assert prompts.has_images is False
    with pytest.raises(RuntimeError, match="human pairwise"):
        prompts[0]

    edits = tmp_path / "imgedit"
    images = edits / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (8, 8), (1, 2, 3)).save(images / "src.png")
    (edits / "metadata.jsonl").write_text(
        json.dumps({"image": "images/src.png", "instruction": "make it night"}) + "\n",
        encoding="utf-8",
    )
    dataset = ImgEditDataset(edits, split="val", resolution=8)
    assert dataset.has_images is True
    item = dataset[0]
    assert item["prompt"] == "make it night"
    assert item["image"].shape == (3, 8, 8)
