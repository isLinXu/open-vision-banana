"""Open-vocabulary semantic target generation."""

import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from open_vision_banana.data.openvocab_semantic import (
    BACKGROUND_NAME,
    NAMED_COLORS,
    STYLES,
    class_phrase,
    color_distance,
    decode_nearest,
    sample_distinct_colors,
    sample_openvocab_semantic,
)

CLASSES = ["road", "traffic_light", "sky", "person", "car", "tree", "wall", "bus"]


def _labels() -> np.ndarray:
    labels = np.full((32, 32), -1, dtype=np.int64)  # -1 = ignore
    labels[:16, :] = 2  # sky
    labels[16:, :16] = 0  # road
    labels[16:, 16:24] = 3  # person
    labels[16:20, 24:28] = 4  # car (small)
    return labels


def test_distinct_colors_respect_min_distance():
    rng = random.Random(0)
    colors = sample_distinct_colors(10, rng, min_distance=64.0)
    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            assert color_distance(a, b) >= 64.0


def test_distinct_colors_raise_when_infeasible():
    with pytest.raises(ValueError):
        sample_distinct_colors(500, random.Random(0), min_distance=64.0)


def _assert_pairwise_apart(colors, min_distance):
    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            assert color_distance(a, b) >= min_distance


def test_distinct_colors_allow_fewer_returns_the_colors_that_fit():
    colors = sample_distinct_colors(500, random.Random(0), min_distance=64.0, allow_fewer=True)
    assert 0 < len(colors) < 500
    _assert_pairwise_apart(colors, 64.0)


def test_crowded_json_named_legend_falls_back_to_tuples_and_keeps_every_class():
    # 20 visible classes need 21 colors; at most 16 named colors are 90 apart.
    names = [f"class_{i}" for i in range(20)]
    labels = np.repeat(np.arange(20), 5)[None, :].repeat(10, axis=0)
    for seed in range(20):
        sample = sample_openvocab_semantic(
            labels,
            names,
            random.Random(seed),
            ignore_index=-1,
            max_present=20,
            max_absent=0,
            style_weights={"json_named": 1.0},
        )
        assert sample.style == "json_tuple"
        assert set(sample.mapping) == {class_phrase(n) for n in names} | {BACKGROUND_NAME}
        _assert_pairwise_apart(list(sample.mapping.values()), 90.0)


def test_json_named_legend_that_fits_stays_named():
    for seed in range(20):
        sample = sample_openvocab_semantic(
            _labels(),
            CLASSES,
            random.Random(seed),
            ignore_index=-1,
            max_absent=0,
            style_weights={"json_named": 1.0},
        )
        assert sample.style == "json_named"
        assert set(sample.mapping.values()) <= set(NAMED_COLORS.values())


def test_named_remap_of_an_18_class_legend_falls_back_to_tuples():
    from open_vision_banana.training.trainer import ManifestVisionDataset

    # all 18 named colors: brown and olive are 63 apart, so they cannot all be redrawn at 64
    mapping = {f"class_{i}": rgb for i, rgb in enumerate(NAMED_COLORS.values())}
    ds = object.__new__(ManifestVisionDataset)
    ds.rng = random.Random(0)
    ds.semantic_prompt_styles = ["json_named"]
    ds.semantic_color_remap_prob = 1.0
    _, prompt = ds._augment_semantic(torch.zeros(3, 4, 4), "", mapping)
    assert "<" in prompt  # json_tuple legend
    for name in mapping:
        assert name in prompt


@pytest.mark.parametrize("style", STYLES)
def test_prompt_mentions_every_class_and_color(style):
    rng = random.Random(1)
    sample = sample_openvocab_semantic(
        _labels(), CLASSES, rng, ignore_index=-1, style_weights={style: 1.0}
    )
    assert sample.style == style
    assert BACKGROUND_NAME in sample.mapping
    for phrase, (r, g, b) in sample.mapping.items():
        assert phrase in sample.prompt
        if style == "hex":
            assert f"#{r:02X}{g:02X}{b:02X}" in sample.prompt
        elif style == "json_named":
            names = [n for n, rgb in NAMED_COLORS.items() if rgb == (r, g, b)]
            assert names and any(f'"{n}"' in sample.prompt for n in names)
        else:
            assert f"{r}, {g}, {b}" in sample.prompt


@pytest.mark.parametrize("style", STYLES)
def test_target_roundtrips_under_nearest_color_decoding(style):
    labels = _labels()
    sample = sample_openvocab_semantic(
        labels, CLASSES, random.Random(3), ignore_index=-1, style_weights={style: 1.0}
    )
    names = list(sample.mapping)
    decoded = decode_nearest(sample.target_rgb, sample.mapping)
    expected_phrase = {i: n.replace("_", " ") for i, n in enumerate(CLASSES)}
    for idx in (0, 2, 3, 4):
        phrase = expected_phrase[idx]
        if phrase not in sample.mapping:
            continue  # json_named may truncate when the named pool is small
        region = labels == idx
        assert (decoded[region] == names.index(phrase)).all()
    # Ignored pixels are painted as background.
    assert (
        (decoded[labels == -1] == names.index(BACKGROUND_NAME)).all()
        if (labels == -1).any()
        else True
    )


def test_mapping_is_resampled_across_variants():
    labels = _labels()
    maps = {
        tuple(
            sorted(
                sample_openvocab_semantic(
                    labels, CLASSES, random.Random(s), ignore_index=-1
                ).mapping.items()
            )
        )
        for s in range(8)
    }
    assert len(maps) > 4  # colors/classes/styles actually vary


def test_deterministic_given_rng():
    a = sample_openvocab_semantic(_labels(), CLASSES, random.Random(7), ignore_index=-1)
    b = sample_openvocab_semantic(_labels(), CLASSES, random.Random(7), ignore_index=-1)
    assert a.prompt == b.prompt and (a.target_rgb == b.target_rgb).all()


def test_absent_distractors_in_prompt_but_not_in_target():
    labels = _labels()
    found = False
    for seed in range(30):
        s = sample_openvocab_semantic(
            labels,
            CLASSES,
            random.Random(seed),
            ignore_index=-1,
            max_absent=6,
            style_weights={"json_tuple": 1.0},
        )
        for phrase in s.absent_classes:
            found = True
            color = np.asarray(s.mapping[phrase])
            assert not (s.target_rgb == color).all(axis=-1).any()
            assert phrase in s.prompt
    assert found


def test_tiny_classes_fall_into_background():
    labels = np.zeros((100, 100), dtype=np.int64)
    labels[0, 0] = 3  # 1 pixel of "person" = 0.01% < 0.1%
    s = sample_openvocab_semantic(labels, CLASSES, random.Random(0), ignore_index=-1, max_absent=0)
    assert "person" not in s.present_classes
    assert (s.target_rgb[0, 0] == np.asarray(s.mapping[BACKGROUND_NAME])).all()


def _load_builder():
    path = (
        Path(__file__).resolve().parent.parent / "scripts" / "build_openvocab_semantic_manifest.py"
    )
    spec = importlib.util.spec_from_file_location("build_openvocab_semantic_manifest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_refuses_benchmark_source():
    builder = _load_builder()
    with pytest.raises(SystemExit):
        builder.assert_not_benchmark("Cityscapes", allow=False)
    builder.assert_not_benchmark("Cityscapes", allow=True)
    builder.assert_not_benchmark("ade20k", allow=False)


def test_builder_split_is_stable_per_image():
    builder = _load_builder()
    assert builder.split_for("img_001", 0.5) == builder.split_for("img_001", 0.5)
    splits = {builder.split_for(f"img_{i}", 0.2) for i in range(200)}
    assert splits == {"train", "val"}


def test_builder_end_to_end_and_manifest_loadable(tmp_path, monkeypatch):
    builder = _load_builder()
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for i in range(3):
        Image.fromarray(np.full((32, 32, 3), 100 + i, dtype=np.uint8)).save(images / f"im{i}.jpg")
        # label PNG with offset 1 (0 = "other"/ignored), classes 1..8
        raw = (_labels() + 1).clip(min=0).astype(np.uint8)
        Image.fromarray(raw).save(labels / f"im{i}.png")
    classes = tmp_path / "classes.json"
    classes.write_text(json.dumps(CLASSES))
    out = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build",
            "--images",
            str(images),
            "--labels",
            str(labels),
            "--classes",
            str(classes),
            "--source-name",
            "toy",
            "--out-dir",
            str(out),
            "--label-offset",
            "1",
            "--ignore-index",
            "-1",
            "--variants",
            "2",
            "--val-fraction",
            "0.0",
        ],
    )
    assert builder.main() == 0
    rows = [json.loads(line) for line in (out / "pairs.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    assert len({r["prompt"] for r in rows}) > 1  # prompts differ per sample
    assert all(r["task"] == "semantic" and r["split"] == "train" for r in rows)
    assert all((out / r["target"]).is_file() for r in rows)

    try:
        from open_vision_banana.training.trainer import ManifestVisionDataset
    except Exception as exc:  # optional heavy deps missing in minimal envs
        pytest.skip(f"trainer import unavailable: {exc}")
    ds = ManifestVisionDataset(
        out / "pairs.jsonl", resolution=32, split="train", tasks={"semantic"}
    )
    assert len(ds) == 6
    assert {s["prompt"] for s in ds.samples} == {
        r["prompt"] for r in rows
    }  # per-row prompt preserved
    item = ds[0]
    assert item["task"] == "semantic" and item["target_rgb"].shape == (3, 32, 32)


def test_normalize_task_prompt_default_forces_cityscapes_legend():
    from open_vision_banana.task_prompts import SEMANTIC_SEGMENTATION_PROMPT, normalize_task_prompt

    assert normalize_task_prompt("semantic", "custom") == SEMANTIC_SEGMENTATION_PROMPT
    assert normalize_task_prompt("depth", "custom") == "custom"


def test_normalize_task_prompt_keep_custom_only_for_nonempty_semantic():
    from open_vision_banana.task_prompts import SEMANTIC_SEGMENTATION_PROMPT, normalize_task_prompt

    assert normalize_task_prompt("semantic", "my mapping", keep_custom=True) == "my mapping"
    assert (
        normalize_task_prompt("semantic", "   ", keep_custom=True) == SEMANTIC_SEGMENTATION_PROMPT
    )
    assert (
        normalize_task_prompt("segmentation", "", keep_custom=True) == SEMANTIC_SEGMENTATION_PROMPT
    )


@pytest.mark.parametrize("keep", [False, True])
def test_legacy_datasets_keep_prompts_only_when_opted_in(tmp_path, keep):
    from open_vision_banana.data.datasets import V3FlatDataset
    from open_vision_banana.task_prompts import SEMANTIC_SEGMENTATION_PROMPT
    from open_vision_banana.training.tuner import VisionTaskDataset

    root = tmp_path / "semantic"
    for d in ("images", "targets", "prompts"):
        (root / d).mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(root / "images" / "x.png")
    Image.new("RGB", (8, 8)).save(root / "targets" / "x.png")
    (root / "prompts" / "x.txt").write_text('Generate ... {"sky": <1, 2, 3>}.', encoding="utf-8")

    expected = '{"sky": <1, 2, 3>}' if keep else SEMANTIC_SEGMENTATION_PROMPT
    flat = V3FlatDataset(root, task_label="semantic", resolution=8, keep_semantic_prompts=keep)
    assert (
        (expected in flat.samples[0]["prompt"]) if keep else flat.samples[0]["prompt"] == expected
    )
    tuned = VisionTaskDataset(str(root), resolution=8, task="semantic", keep_semantic_prompts=keep)
    assert (
        (expected in tuned.samples[0]["prompt"]) if keep else tuned.samples[0]["prompt"] == expected
    )
