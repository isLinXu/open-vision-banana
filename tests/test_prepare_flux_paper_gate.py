"""Tests for the FLUX paper-method gate dataset layout."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts.archive.pre_opensource.prepare_flux_paper_gate_data import prepare_flux_paper_gate


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(path)


def test_prepare_flux_paper_gate_symlinks_expected_layout(tmp_path: Path) -> None:
    hypersim = tmp_path / "hypersim"
    cityscapes = tmp_path / "cityscapes"
    generation = tmp_path / "generation"
    output = tmp_path / "gate"

    hypersim_rows = []
    for index in range(3):
        name = f"{index:06d}_frame.000{index}..png"
        image = hypersim / "images" / name
        target = hypersim / "targets" / name
        _write_png(image)
        _write_png(target)
        hypersim_rows.append({"input": str(image), "target": str(target)})
    (hypersim / "pairs.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in hypersim_rows)
    )

    city_rows = []
    for index in range(2):
        name = f"cityscapes_{index:06d}.png"
        image = cityscapes / "images" / name
        target = cityscapes / "targets" / name
        _write_png(image)
        _write_png(target)
        city_rows.append({"input": str(image), "target": str(target)})
    (cityscapes / "pairs.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in city_rows)
    )

    (generation / "images").mkdir(parents=True, exist_ok=True)
    captions = []
    for index in range(2):
        filename = f"aug_{index:06d}.jpg"
        Image.new("RGB", (8, 8), color=(1, 2, 3)).save(generation / "images" / filename)
        captions.append({"filename": filename, "caption": f"caption {index}"})
    (generation / "captions.json").write_text(json.dumps(captions))

    manifest = prepare_flux_paper_gate(
        output,
        hypersim_root=hypersim,
        cityscapes_root=cityscapes,
        generation_root=generation,
        depth_limit=2,
    )

    assert manifest["depth_pairs"] == 2
    assert manifest["segmentation_pairs"] == 2
    assert manifest["generation_pairs"] == 2
    assert (output / "depth" / "images" / "000000_frame.0000..png").is_symlink()
    assert (output / "segmentation" / "targets" / "cityscapes_000001.png").is_file()
    prompts = [
        json.loads(line)
        for line in (output / "generation" / "prompts.jsonl").read_text().splitlines()
    ]
    assert prompts[0] == {"image": "aug_000000.jpg", "prompt": "caption 0"}
    assert not (output / "depth" / "images" / "000002_frame.0002..png").exists()
