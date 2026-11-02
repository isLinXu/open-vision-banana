"""Review item 6b/6c: non-black Fig. 3 backgrounds must reach the Appendix A decoder.

The instance builder records ``background_color`` per row; previously the
dataset / batch dropped it and the evaluator decoded with the default black, so a
white-background single-instance target decoded as *two* instances. ``num_instances``
also recorded raw annotation counts rather than visible instances.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from open_vision_banana.core.segmentation import SegmentationDecoder
from open_vision_banana.evaluation.evaluator import (
    VisionEvaluator,
    _decode_instances_with_background,
)
from open_vision_banana.training.trainer import ManifestVisionDataset, _collate_fn

BACKGROUNDS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "rgb": (88, 50, 82),
    "hex": (16, 232, 168),
}
GREEN = (0, 255, 0)


def _target(background: tuple[int, int, int], size: int = 64) -> torch.Tensor:
    arr = np.empty((size, size, 3), np.uint8)
    arr[...] = background
    arr[16:48, 16:48] = GREEN
    return torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0) / 255.0


class _EchoModel:
    """predict() returns the target itself, so the metric isolates the decoder path."""

    def __init__(self) -> None:
        self.current = None

    def eval(self):
        return self

    def predict(self, **kwargs):
        return {"rgb_output": self.current}


@pytest.mark.parametrize("name", sorted(BACKGROUNDS))
def test_evaluator_decodes_one_instance_on_every_background(name):
    background = BACKGROUNDS[name]
    target = _target(background)
    decoder = SegmentationDecoder.paper_reproduction()
    # the bug: default-black decoding sees the bright background as an instance
    default_count = len(decoder.decode_instance(target)[0])
    explicit_count = len(decoder.decode_instance(target, background_color=background)[0])
    assert explicit_count == 1
    if name != "black":
        assert default_count == 2

    # the evaluator's decode path: with the collated background both pred and GT give 1 instance;
    # without it (legacy batches) the bright background becomes a second "instance" on both sides,
    # which an echo model hides (AP stays 1.0) but a real prediction would be scored against a wrong GT.
    with_bg_pred, with_bg_gt = _decode_instances_with_background(
        decoder, target, target, [background]
    )
    assert len(with_bg_pred[0]) == 1 and len(with_bg_gt[0]) == 1
    legacy_pred, legacy_gt = _decode_instances_with_background(decoder, target, target, None)
    assert len(legacy_gt[0]) == (1 if name == "black" else 2)

    model = _EchoModel()
    model.current = target
    evaluator = VisionEvaluator(model, device="cpu")
    batch = {
        "image": torch.zeros(1, 3, 64, 64),
        "target_rgb": target,
        "task": ["instance"],
        "background_color": [background],
    }
    metrics = evaluator.evaluate([batch], task="instance")
    assert metrics["AP@0.5"] == pytest.approx(1.0) and metrics["pmF1"] == pytest.approx(1.0)


def test_manifest_background_rides_from_row_to_batch(tmp_path: Path):
    (tmp_path / "img").mkdir()
    rows = []
    for name, background in BACKGROUNDS.items():
        Image.new("RGB", (64, 64), (100, 100, 100)).save(tmp_path / "img" / f"{name}.jpg")
        Image.fromarray(
            (_target(background)[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        ).save(tmp_path / "img" / f"{name}_t.png")
        rows.append(
            {
                "task": "instance",
                "input": f"img/{name}.jpg",
                "target": f"img/{name}_t.png",
                "split": "train",
                "prompt": "Generate an instance segmentation visualization of this image. Each box is colored differently.",
                "background_color": list(background),
                "category": "box",
                "num_instances": 1,
            }
        )
    rows.append(
        {
            "task": "instance",
            "input": "img/black.jpg",
            "target": "img/black_t.png",
            "split": "train",
            "prompt": "Generate an instance segmentation visualization of this image. Each box is colored differently.",
        }
    )  # legacy row
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ds = ManifestVisionDataset(manifest, 64, "train", {"instance"}, data_root=tmp_path)
    items = [ds[i] for i in range(len(ds))]
    assert [it.get("background_color") for it in items[:4]] == [BACKGROUNDS[n] for n in BACKGROUNDS]
    assert "background_color" not in items[4]
    batch = _collate_fn(items)
    assert batch["background_color"] == [
        *BACKGROUNDS.values(),
        (0, 0, 0),
    ]  # legacy row defaults to black
    assert "background_color" not in _collate_fn(
        [items[4]]
    )  # all-legacy batches keep the historical keys
    # and every target still decodes to exactly one instance through the evaluator path
    decoder = SegmentationDecoder.paper_reproduction()
    for rgb, bg in zip(batch["target_rgb"], batch["background_color"], strict=True):
        assert len(decoder.decode_instance(rgb.unsqueeze(0), background_color=bg)[0]) == 1


def test_num_instances_counts_visible_instances(tmp_path: Path):
    pytest.importorskip("pycocotools")
    from scripts.build_instance_manifest import build_instance_manifest

    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (64, 48), (90, 90, 90)).save(images / "000000000001.jpg")

    def rect(x0, y0, x1, y1):
        return [[x0, y0, x1, y0, x1, y1, x0, y1]]

    # a large instance fully covered by a smaller one painted on top -> only 2 of 3 remain visible
    anns = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "segmentation": rect(2, 2, 30, 46),
            "area": 1200,
            "iscrowd": 0,
            "bbox": [2, 2, 28, 44],
        },
        {
            "id": 2,
            "image_id": 1,
            "category_id": 1,
            "segmentation": rect(40, 10, 60, 40),
            "area": 600,
            "iscrowd": 0,
            "bbox": [40, 10, 20, 30],
        },
        {
            "id": 3,
            "image_id": 1,
            "category_id": 1,
            "segmentation": rect(42, 12, 58, 38),
            "area": 400,
            "iscrowd": 0,
            "bbox": [42, 12, 16, 26],
        },
    ]
    # make #3 cover #2 entirely by giving it the same polygon but painting later (smaller area)
    anns[2]["segmentation"] = rect(40, 10, 60, 40)
    anns[2]["area"] = 599
    payload = {
        "images": [{"id": 1, "file_name": "000000000001.jpg", "width": 64, "height": 48}],
        "annotations": anns,
        "categories": [{"id": 1, "name": "box"}],
    }
    ann_file = tmp_path / "instances.json"
    ann_file.write_text(json.dumps(payload))
    out = tmp_path / "out" / "instance.jsonl"
    build_instance_manifest(
        ann_file, images, tmp_path / "out" / "targets", out, data_root=tmp_path / "out"
    )
    row = json.loads(out.read_text().splitlines()[0])
    assert row["num_annotations"] == 3 and row["num_instances"] == 2
