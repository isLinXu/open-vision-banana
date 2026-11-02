"""COCO / LVIS -> instance-segmentation training rows (paper Section 3.1, Appendix A)."""

from __future__ import annotations

import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from open_vision_banana.core.segmentation import SegmentationDecoder
from open_vision_banana.inference.prompt import PromptBuilder
from scripts.build_instance_manifest import (
    BACKGROUND,
    MIN_COLOR_DISTANCE,
    PALETTE,
    _dist,
    build_instance_manifest,
)

pytest.importorskip("pycocotools")


def _rect(x0: int, y0: int, x1: int, y1: int) -> list[list[float]]:
    return [[x0, y0, x1, y0, x1, y1, x0, y1]]


def _write_coco(tmp_path: Path, *, lvis: bool = False) -> tuple[Path, Path]:
    images_dir = tmp_path / ("coco" if lvis else "train2017")
    (images_dir / "train2017" if lvis else images_dir).mkdir(parents=True)
    for image_id in (1, 2, 3):
        name = f"{image_id:012d}.jpg"
        path = (images_dir / "train2017" / name) if lvis else (images_dir / name)
        Image.new("RGB", (64, 48), (90, 90, 90)).save(path)

    def image_entry(image_id: int, **extra) -> dict:
        entry = {"id": image_id, "height": 48, "width": 64}
        if lvis:
            entry["coco_url"] = f"http://images.cocodataset.org/train2017/{image_id:012d}.jpg"
            entry.setdefault("not_exhaustive_category_ids", [])
            entry.setdefault("neg_category_ids", [])
        else:
            entry["file_name"] = f"{image_id:012d}.jpg"
        entry.update(extra)
        return entry

    images = [
        image_entry(1),
        image_entry(2, **({"not_exhaustive_category_ids": [1]} if lvis else {})),
        image_entry(3),
    ]
    annotations = [
        # image 1: three dogs (one occluding a bigger one) + one cat
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "segmentation": _rect(2, 2, 30, 40),
            "area": 1064,
            "iscrowd": 0,
        },
        {
            "id": 2,
            "image_id": 1,
            "category_id": 1,
            "segmentation": _rect(10, 10, 20, 20),
            "area": 100,
            "iscrowd": 0,
        },
        {
            "id": 3,
            "image_id": 1,
            "category_id": 1,
            "segmentation": _rect(40, 4, 60, 44),
            "area": 800,
            "iscrowd": 0,
        },
        {
            "id": 4,
            "image_id": 1,
            "category_id": 2,
            "segmentation": _rect(34, 30, 38, 46),
            "area": 64,
            "iscrowd": 0,
        },
        # image 2: one dog (LVIS marks dog not exhaustively annotated here)
        {
            "id": 5,
            "image_id": 2,
            "category_id": 1,
            "segmentation": _rect(0, 0, 20, 20),
            "area": 400,
            "iscrowd": 0,
        },
        # image 3: a crowd of dogs -> ambiguous, must be skipped for category 1
        {
            "id": 6,
            "image_id": 3,
            "category_id": 1,
            "segmentation": _rect(0, 0, 10, 10),
            "area": 100,
            "iscrowd": 0,
        },
        {
            "id": 7,
            "image_id": 3,
            "category_id": 1,
            "segmentation": {"counts": [0, 3072], "size": [48, 64]},
            "area": 3072,
            "iscrowd": 1,
        },
        {
            "id": 8,
            "image_id": 3,
            "category_id": 2,
            "segmentation": _rect(20, 20, 40, 40),
            "area": 400,
            "iscrowd": 0,
        },
    ]
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "dog"}, {"id": 2, "name": "cat"}],
    }
    ann_path = tmp_path / ("lvis_v1_train.json" if lvis else "instances_train2017.json")
    ann_path.write_text(json.dumps(payload), encoding="utf-8")
    return ann_path, images_dir


def test_palette_is_decodable_by_the_paper_tolerance():
    assert len(PALETTE) >= 32
    assert min(_dist(a, b) for a, b in itertools.combinations(PALETTE, 2)) >= MIN_COLOR_DISTANCE
    assert min(_dist(c, BACKGROUND) for c in PALETTE) >= MIN_COLOR_DISTANCE
    assert MIN_COLOR_DISTANCE > 2 * 14  # two tau-drifted instances still separate


def test_coco_rows_render_targets_the_paper_decoder_recovers(tmp_path: Path):
    ann_path, images_dir = _write_coco(tmp_path)
    out = tmp_path / "instance.jsonl"
    audit = build_instance_manifest(
        ann_path, images_dir, tmp_path / "targets", out, data_root=tmp_path, val_fraction=0.0
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    by_key = {(r["image_id"], r["category"]): r for r in rows}

    # image 3 / dog is a crowd pair and is dropped; everything else is kept
    assert audit["skipped_crowd_pairs"] == 1
    assert set(by_key) == {(1, "dog"), (1, "cat"), (2, "dog"), (3, "cat")}
    assert all(r["task"] == "instance" and r["split"] == "train" for r in rows)
    assert audit["instances_per_row_histogram"] == {"1": 3, "3": 1}

    dog = by_key[(1, "dog")]
    assert dog["prompt"] == PromptBuilder.build_instance_prompt("dog")
    assert dog["prompt"] == (
        "Generate an instance segmentation visualization of this image. Each dog is colored differently."
    )
    assert not Path(dog["input"]).is_absolute() and not Path(dog["target"]).is_absolute()

    target = np.asarray(Image.open(tmp_path / dog["target"]).convert("RGB"))
    assert target.shape == (48, 64, 3)
    assert tuple(target[0, 0]) == BACKGROUND
    colors = {tuple(c) for c in target.reshape(-1, 3)} - {BACKGROUND}
    assert len(colors) == 3 and colors <= set(PALETTE)
    # the small dog (10..20) is painted on top of the large one (2..30)
    assert tuple(target[15, 15]) != tuple(target[5, 5])

    rgb = torch.from_numpy(target).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    instances = SegmentationDecoder.paper_reproduction().decode_instance(rgb)[0]
    assert len(instances) == 3


def test_images_outside_the_data_root_get_manifest_relative_paths(tmp_path: Path, monkeypatch):
    # E0 layout: COCO images under data/raw, targets and manifest under data/prep/inst (cwd-relative)
    from open_vision_banana.data.manifest import resolve_manifest_path

    monkeypatch.chdir(tmp_path)
    ann_path, images_dir = _write_coco(tmp_path / "raw")
    prep = Path("prep") / "inst"
    build_instance_manifest(
        ann_path.relative_to(tmp_path),
        images_dir.relative_to(tmp_path),
        prep / "targets",
        prep / "instance.jsonl",
        data_root=prep,
        val_fraction=0.0,
    )
    rows = [json.loads(line) for line in (prep / "instance.jsonl").read_text().splitlines()]
    assert rows and all(row["input"].startswith("../../raw/") for row in rows)
    assert all(row["target"].startswith("targets/") for row in rows)
    for row in rows:
        assert resolve_manifest_path(row["input"], prep / "instance.jsonl").is_file()
        assert resolve_manifest_path(row["target"], prep / "instance.jsonl").is_file()


def test_lvis_format_skips_not_exhaustive_and_honours_image_id_exclusion(tmp_path: Path):
    ann_path, images_dir = _write_coco(tmp_path, lvis=True)
    refs = tmp_path / "refs(umd).p"
    with refs.open("wb") as handle:
        pickle.dump([{"image_id": 3, "split": "val"}, {"image_id": 1, "split": "train"}], handle)

    out = tmp_path / "instance_lvis.jsonl"
    audit = build_instance_manifest(
        ann_path,
        images_dir,
        tmp_path / "targets",
        out,
        fmt="lvis",
        data_root=tmp_path,
        exclude_image_ids=[refs],
        min_instances=1,
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    keys = {(r["image_id"], r["category"]) for r in rows}
    assert audit["skipped_not_exhaustive_pairs"] == 1  # image 2 / dog
    assert audit["rejected_excluded_image_id"] == 2  # image 3: dog pair and cat pair
    assert audit["excluded_image_id_count"] == 1  # the train ref does not exclude
    assert keys == {(1, "dog"), (1, "cat")}
    assert all(r["input"].startswith("coco/train2017/") for r in rows)
    assert all(r["source_dataset"] == "lvis" for r in rows)


def test_instance_filters_and_val_split(tmp_path: Path):
    ann_path, images_dir = _write_coco(tmp_path)
    out = tmp_path / "multi.jsonl"
    audit = build_instance_manifest(
        ann_path,
        images_dir,
        tmp_path / "targets",
        out,
        data_root=tmp_path,
        min_instances=2,
        val_fraction=0.5,
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert audit["rows_written"] == 1 and rows[0]["num_instances"] == 3
    assert audit["skipped_too_few_instances"] == 3

    with pytest.raises(ValueError, match="fmt"):
        build_instance_manifest(ann_path, images_dir, tmp_path / "t", out, fmt="voc")


def test_fig3_background_styles_render_prompt_and_target_in_sync(tmp_path: Path):
    from scripts.build_instance_manifest import BACKGROUND_STYLES, build_instance_palette

    ann_path, images_dir = _write_coco(tmp_path)
    out = tmp_path / "styled.jsonl"
    audit = build_instance_manifest(
        ann_path,
        images_dir,
        tmp_path / "targets",
        out,
        data_root=tmp_path,
        background_styles=("named", "rgb", "hex"),
        seed=3,
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows and all("background_style_black" not in k for k in audit)
    assert sum(audit.get(f"background_style_{s}", 0) for s in BACKGROUND_STYLES) == len(rows)

    decoder = SegmentationDecoder.paper_reproduction()
    for row in rows:
        bg = tuple(row["background_color"])
        target = np.asarray(Image.open(tmp_path / row["target"]).convert("RGB"))
        # The corner is background and the prompt names that very colour
        assert tuple(target[47, 63]) == bg
        assert row["prompt"] in (
            PromptBuilder.build_instance_prompt(row["category"], bg, 1, background_style=s)
            for s in ("named", "rgb", "hex")
        ) or row["prompt"] in (
            PromptBuilder.build_instance_prompt(row["category"], bg, 2, background_style=s)
            for s in ("named", "rgb", "hex")
        )
        assert "circular" not in row["prompt"]
        # Instance colours stay >= 60 from this background and decode with it
        colors = {tuple(c) for c in target.reshape(-1, 3)} - {bg}
        assert len(colors) == row["num_instances"]
        assert all(_dist(c, bg) >= MIN_COLOR_DISTANCE for c in colors)
        assert colors <= set(build_instance_palette(background=bg))
        rgb = torch.from_numpy(target).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        assert len(decoder.decode_instance(rgb, background_color=bg)[0]) == row["num_instances"]

    with pytest.raises(ValueError, match="background styles"):
        build_instance_manifest(
            ann_path, images_dir, tmp_path / "t", out, background_styles=("neon",)
        )


def test_instance_templates_match_fig3_wording():
    assert PromptBuilder.build_instance_prompt("garlic") == (
        "Generate an instance segmentation visualization of this image. Each garlic is colored differently."
    )
    assert PromptBuilder.build_instance_prompt(
        "croissant", (88, 50, 82), 1, background_style="rgb"
    ) == (
        "This image is a segmentation task derived from the input. The croissant instances are each "
        "represented by a unique, solid color. Background is RGB(88, 50, 82)."
    )
    assert PromptBuilder.build_instance_prompt(
        "basketball", (16, 170, 5), 2, background_style="hex"
    ) == (
        "This image shows segmentation masks for the basketball from the input image. The background "
        "is set to #10aa05. Each basketball is represented by a different color."
    )
    assert PromptBuilder.build_instance_prompt("ball", (255, 255, 255), 2).endswith(
        "The background is set to white color. Each ball is represented by a different color."
    )
    with pytest.raises(ValueError, match="black background"):
        PromptBuilder.build_instance_prompt("dog", (255, 255, 255), 0)
