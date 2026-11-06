import numpy as np
import pytest

from open_vision_banana.coconut_preparation import (
    eligible_groups,
    image_split,
    parse_legend,
    render_targets,
)


def source(labels):
    return {
        "num_instances": len(labels),
        "conversations": [
            {
                "from": "gpt",
                "value": ", ".join(
                    f"<p>{name}<color>({r},{g},{b})</color></p>" for name, (r, g, b) in labels
                )
                + "<SEG>",
            }
        ],
    }


def test_ambiguous_food_not_a_negative_and_same_class_union():
    entries = parse_legend(
        source([("bowl-0", (255, 0, 0)), ("bowl-1", (0, 255, 0)), ("food", (0, 0, 255))])
    )
    rgb = np.zeros((30, 30, 3), np.uint8)
    rgb[:10, :10] = (255, 0, 0)
    rgb[10:20, :10] = (0, 255, 0)
    rgb[:10, 10:20] = (0, 0, 255)
    groups, reasons, _ = eligible_groups(rgb, entries)
    assert set(groups) == {"bowl"}
    assert reasons == [{"category": "food", "reason": "unreviewed_category"}]
    outputs = list(render_targets(rgb, groups, 42))
    semantic, target = outputs[0]
    assert semantic["absent_classes"] == []
    assert np.all(target[:20, :10] == semantic["mapping"]["bowl"])
    assert np.all(target[:10, 10:20] == semantic["mapping"]["background"])
    instance, target = outputs[1]
    assert instance["num_instances"] == 2
    assert not np.array_equal(target[0, 0], target[10, 0])


def test_missing_or_tiny_instance_rejects_whole_category():
    entries = parse_legend(
        source([("person-0", (255, 0, 0)), ("person-1", (0, 255, 0)), ("dog-0", (0, 0, 255))])
    )
    rgb = np.zeros((30, 30, 3), np.uint8)
    rgb[:10, :10] = (255, 0, 0)
    rgb[:10, 10:20] = (0, 0, 255)
    groups, reasons, _ = eligible_groups(rgb, entries)
    assert set(groups) == {"dog"}
    assert reasons[0]["reason"] == "empty_or_tiny_region"
    assert list(render_targets(rgb, groups, 1))[1][0]["num_instances"] == 1


def test_high_black_fraction_is_not_rejection():
    rgb = np.zeros((100, 100, 3), np.uint8)
    rgb[:10, :10] = (255, 0, 0)
    entries = parse_legend(source([("dog-0", (255, 0, 0))]))
    groups, reasons, black = eligible_groups(rgb, entries)
    assert groups and not reasons and black == 0.99


def test_unknown_colors_fail_closed():
    rgb = np.ones((10, 10, 3), np.uint8)
    with pytest.raises(ValueError, match="absent from legend"):
        eligible_groups(rgb, parse_legend(source([("dog-0", (255, 0, 0))])))


def test_duplicate_or_background_legend_rejected():
    for labels in [[("dog-0", (0, 0, 0))], [("dog-0", (255, 0, 0)), ("cat-0", (255, 0, 0))]]:
        with pytest.raises(ValueError):
            parse_legend(source(labels))


def test_existing_train_identity_stays_train_and_pixel_split_stable():
    assert image_split("abc", 9, {9: "train"}, 0.99) == "train"
    assert image_split("abc", 9, {}) == image_split("abc", 10, {})


def test_aliases_merge_class_but_not_instances():
    entries = parse_legend(source([("person-0", (255, 0, 0)), ("people-1", (0, 255, 0))]))
    rgb = np.zeros((20, 20, 3), np.uint8)
    rgb[:10] = (255, 0, 0)
    rgb[10:] = (0, 255, 0)
    groups, _, _ = eligible_groups(rgb, entries)
    assert list(groups) == ["person"] and len(groups["person"]) == 2


def test_cli_build_validate_and_bundle_roundtrip(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    from PIL import Image

    from open_vision_banana.res_bundle import package_dataset, restore_bundle

    upstream = tmp_path / "upstream"
    rows, pairs = [], []
    for idx in (9, 25, 30):
        inp = f"coco2017/train2017/{idx:012d}.jpg"
        seg = f"coconut/coconut_b/genseg/{idx:012d}.png"
        for name in (inp, seg):
            (upstream / "media" / name).parent.mkdir(parents=True, exist_ok=True)
        # All images deliberately identical: one held out, one retained, one deduplicated.
        Image.new("RGB", (32, 32), (120, 70, 30)).save(upstream / "media" / inp)
        target = np.zeros((32, 32, 3), np.uint8)
        target[4:20, 4:20] = (255, 0, 0)
        Image.fromarray(target).save(upstream / "media" / seg)
        rows.append(dict(source([("dog-0", (255, 0, 0))]), image_id=idx, image=inp, seg=seg))
        pairs.append({"id": idx, "image": [inp, seg]})
    for name, records in (
        (
            "training/cleaned_manifests/segmentation/coconut_b_total_unified_genseg_edit_regenerated.jsonl",
            rows,
        ),
        ("image_text_image/manifests/coconut_b.jsonl", pairs),
    ):
        path = upstream / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
    excluded = tmp_path / "excluded.json"
    excluded.write_text("[30]")
    root = Path(__file__).resolve().parents[1]
    prepared = tmp_path / "prepared"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/prepare_coconut.py"),
            "--source-root",
            str(upstream),
            "--output-dir",
            str(prepared),
            "--exclude-image-ids",
            str(excluded),
            "--workers",
            "1",
            "--max-images",
            "3",
        ],
        check=True,
        capture_output=True,
    )
    emitted = [json.loads(line) for line in (prepared / "pairs.jsonl").read_text().splitlines()]
    assert {r["task"] for r in emitted} == {"semantic", "instance"}
    assert {r["source_image_id"] for r in emitted} == {9}
    report = json.loads((prepared / "build_report.json").read_text())
    assert report["rejections"] == {"heldout_image_id": 1, "duplicate_decoded_rgb": 1}
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/validate_coconut.py"),
            str(prepared),
            "--source-media",
            str(upstream / "media"),
            "--workers",
            "1",
        ],
        check=True,
        capture_output=True,
    )
    bundle, restored = tmp_path / "bundle", tmp_path / "restored"
    package_dataset(prepared, bundle, 1024**2)
    restore_bundle(bundle, restored)
    assert (restored / "pairs.jsonl").read_bytes() == (prepared / "pairs.jsonl").read_bytes()
    for row in emitted:
        assert (restored / row["target"]).read_bytes() == (prepared / row["target"]).read_bytes()
