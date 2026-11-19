"""scripts/convert_modelscope_to_ovb.py: PSG grounded captions -> referring, pHash audit, compose."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from open_vision_banana.referring_contract import (
    BINARY_REFERRING_ENCODING,
    is_binary_referring_prompt,
)

spec = importlib.util.spec_from_file_location(
    "convert_ms", Path(__file__).parents[1] / "scripts/convert_modelscope_to_ovb.py"
)
convert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert)

UMBRELLA, PERSON1, PERSON2, GRASS = (149, 161, 172), (255, 255, 255), (226, 0, 252), (189, 254, 10)


def _caption(*phrases):
    body = " and ".join(f"<p>{p}<color>({r}, {g}, {b})</color></p>" for p, (r, g, b) in phrases)
    return {
        "conversations": [
            {"from": "human", "value": "<image>Describe."},
            {"from": "gpt", "value": body},
        ]
    }


def _source(tmp_path, colors, legend):
    """A 64x64 COCO-named image and its PSG mask: one horizontal band per color."""
    media = tmp_path / "media"
    (media / "coco2017/train2017").mkdir(parents=True)
    (media / "PSG/gcg/train2017").mkdir(parents=True)
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(
        media / "coco2017/train2017/000000417720.jpg"
    )
    mask = np.zeros((64, 64, 3), np.uint8)
    for i, color in enumerate(colors):
        mask[i * 16 : (i + 1) * 16] = color
    Image.fromarray(mask).save(media / "PSG/gcg/train2017/000000417720.png")
    row = dict(
        _caption(*legend),
        image="coco2017/train2017/000000417720.jpg",
        seg="PSG/gcg/train2017/000000417720.png",
        caption="two people under an umbrella on the grass",
    )
    out = tmp_path / "out"
    convert.prepare_output(out)
    return row, media, out, mask


def test_numbered_phrases_are_ambiguous():
    phrases = [
        ("person-1", PERSON1),
        ("person-2", PERSON2),
        ("umbrella", UMBRELLA),
        ("grass", GRASS),
    ]
    assert convert.unique_phrases(phrases) == [("umbrella", UMBRELLA), ("grass", GRASS)]
    # a lone person-1 still means the source numbered several people
    assert convert.unique_phrases([("person-1", PERSON1), ("tree", GRASS)]) == [("tree", GRASS)]


def test_parse_rejects_malformed_or_black_legends():
    with pytest.raises(ValueError, match="malformed"):
        convert.parse_gcg_phrases(
            {"conversations": [{"from": "gpt", "value": "<p>cat<color>(1, 2)</color></p>"}]}
        )
    with pytest.raises(ValueError, match="black"):
        convert.parse_gcg_phrases(_caption(("shadow", (0, 0, 0))))


def test_gcg_work_writes_exact_binary_targets(tmp_path):
    legend = [
        ("person-1", PERSON1),
        ("person-2", PERSON2),
        ("umbrella", UMBRELLA),
        ("grass", GRASS),
    ]
    row, media, out, mask = _source(tmp_path, [PERSON1, PERSON2, UMBRELLA, GRASS], legend)
    result = convert.gcg_work((row, str(media), str(out), 0.0, 3, 64, 2e-4))
    assert "error" not in result and result["unique_phrases"] == 2
    rows = result["rows"]
    assert sorted(r["expression"] for r in rows) == ["grass", "umbrella"]
    for r in rows:
        assert r["encoding"] == BINARY_REFERRING_ENCODING and r["task"] == "referring"
        assert is_binary_referring_prompt(r["prompt"], r["expression"])
        target = np.array(Image.open(out / r["target"]))
        color = UMBRELLA if r["expression"] == "umbrella" else GRASS
        expected = np.where(np.all(mask == color, axis=2)[..., None], 255, 0)
        assert np.array_equal(target, np.broadcast_to(expected, target.shape))
        assert set(np.unique(target)) <= {0, 255}
    copied = (out / rows[0]["input"]).read_bytes()
    assert copied == (media / row["image"]).read_bytes()  # source bytes, no re-encode


def test_gcg_work_rejects_mask_colors_outside_the_legend(tmp_path):
    row, media, out, _ = _source(tmp_path, [UMBRELLA, (9, 9, 9)], [("umbrella", UMBRELLA)])
    result = convert.gcg_work((row, str(media), str(out), 0.0, 3, 64, 2e-4))
    assert "absent from the caption legend" in result["error"]


def test_tiny_regions_are_not_targets(tmp_path):
    row, media, out, _ = _source(
        tmp_path, [UMBRELLA, GRASS], [("umbrella", UMBRELLA), ("grass", GRASS)]
    )
    result = convert.gcg_work((row, str(media), str(out), 0.0, 3, 16 * 64 + 1, 0.0))
    assert result["rows"] == []  # every band has 16 x 64 pixels, below the limit


def test_phash_matches_near_duplicates_only(tmp_path):
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    Image.fromarray(base).save(tmp_path / "a.png")
    Image.fromarray(base).resize((96, 96)).save(tmp_path / "a_small.jpg", quality=90)
    Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)).save(tmp_path / "b.png")
    a, small, b = (convert.phash(tmp_path / n) for n in ("a.png", "a_small.jpg", "b.png"))
    dist = convert.hamming(np.array([a], np.uint64), np.array([a, small, b], np.uint64))[0]
    assert dist[0] == 0 and dist[1] <= 6 and dist[2] > 6


def test_audit_and_compose(tmp_path):
    root = tmp_path / "conv"
    (root / "images").mkdir(parents=True)
    rng = np.random.default_rng(2)
    leaked = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(leaked).save(root / "images/1.png")
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(root / "images/2.png")
    rows = [
        dict(
            task="referring",
            input="images/1.png",
            target="t1.png",
            split="train",
            source_dataset="psg_gcg",
        ),
        dict(
            task="referring",
            input="images/2.png",
            target="t2.png",
            split="train",
            source_dataset="psg_gcg",
        ),
        dict(
            task="referring",
            input="images/2.png",
            target="t3.png",
            split="val",
            source_dataset="psg_gcg",
        ),
    ]
    (root / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    evals = tmp_path / "eval"
    evals.mkdir()
    Image.fromarray(leaked).save(evals / "x.jpg", quality=95)
    report = tmp_path / "audit.json"
    convert.main(["audit", "--manifest", str(root / "pairs.jsonl"), "--eval-root", f"refcocog={evals}",
                  "--workers", "1", "--output", str(report)])  # fmt: skip
    hits = json.loads(report.read_text())["hits"]["refcocog"]
    assert [Path(h["input"]).name for h in hits] == ["1.png"]
    base = tmp_path / "base.jsonl"
    base.write_text(
        json.dumps(dict(task="depth", input="/d/i.png", target="/d/t.png", split="val")) + "\n"
    )
    out = tmp_path / "train.jsonl"
    convert.main(["compose", "--base", str(base), "--add", str(root / "pairs.jsonl"),
                  "--drop", str(report.with_suffix(".drop.json")), "--output", str(out)])  # fmt: skip
    composed = [json.loads(line) for line in out.read_text().splitlines()]
    assert composed[0]["task"] == "depth"  # base rows untouched, val included
    assert [(r["input"], r["target"]) for r in composed[1:]] == [
        (str(root / "images/2.png"), str(root / "t2.png"))
    ]
    summary = json.loads(out.with_suffix(".compose.json").read_text())
    assert summary["counts"]["psg_gcg"] == {"referring": 1}


def test_coconut_rows_are_named_after_their_subset():
    rows = convert.relabel_coconut_rows(
        [dict(sample_id="coconut_b:4441588:semantic:0", source_dataset="coconut_b")], "coconut_l"
    )
    assert rows[0]["sample_id"] == "coconut_l:4441588:semantic:0"
    assert rows[0]["source_dataset"] == "coconut_l"


def test_compose_refuses_a_drop_list_that_matches_nothing(tmp_path):
    root = tmp_path / "conv"
    root.mkdir()
    row = dict(
        task="referring",
        input="images/2.png",
        target="t2.png",
        split="train",
        source_dataset="psg_gcg",
    )
    (root / "pairs.jsonl").write_text(json.dumps(row) + "\n")
    base = tmp_path / "base.jsonl"
    base.write_text(
        json.dumps(dict(task="depth", input="i.png", target="t.png", split="train")) + "\n"
    )
    drop = tmp_path / "drop.json"
    drop.write_text(json.dumps(["/elsewhere/images/2.png"]))
    out = tmp_path / "train.jsonl"
    with pytest.raises(ValueError, match="match no row"):
        convert.main(["compose", "--base", str(base), "--add", str(root / "pairs.jsonl"), "--drop", str(drop),
                      "--output", str(out)])  # fmt: skip
    drop.write_text("[]")
    convert.main(["compose", "--base", str(base), "--add", str(root / "pairs.jsonl"), "--drop", str(drop),
                  "--output", str(out)])  # fmt: skip
    first = json.loads(out.read_text().splitlines()[0])
    assert first["input"] == str(tmp_path / "i.png")  # base rows resolved against the base manifest


def test_gcg_skips_a_second_caption_row_of_the_same_image(tmp_path, monkeypatch):
    rows = [
        dict(image="coco2017/train2017/000000000001.jpg"),
        dict(image="coco2017/train2017/000000000001.jpg"),
    ]
    monkeypatch.setattr(convert, "read_jsonl", lambda path: iter(rows))
    monkeypatch.setattr(convert, "pool_map", lambda fn, items, workers, chunksize: iter(()))
    out = tmp_path / "out"
    convert.main(["gcg", "--source-root", str(tmp_path), "--media-root", str(tmp_path), "--output-dir", str(out),
                  "--workers", "1"])  # fmt: skip
    report = json.loads((out / "build_report.json").read_text())
    assert report["converted_images"] == 1 and report["rejections"] == {"duplicate_image_id": 1}


def test_phash_reads_nyu_h5_samples(tmp_path):
    h5py = pytest.importorskip("h5py")
    rgb = np.random.default_rng(3).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    with h5py.File(tmp_path / "00001.h5", "w") as f:
        f["rgb"] = rgb.transpose(2, 0, 1)
    Image.fromarray(rgb).save(tmp_path / "same.png")
    assert convert.phash(tmp_path / "00001.h5") == convert.phash(tmp_path / "same.png")
