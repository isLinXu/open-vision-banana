"""Regression tests for metrics and encodings used in the paper tables."""

from __future__ import annotations

import json

import pytest
import torch
from PIL import Image

from open_vision_banana.core.depth_mapper import (
    CUBE_EDGE_KEYPOINTS,
    DepthRGBMapper,
    get_available_colormaps,
)
from open_vision_banana.evaluation.instance import (
    compute_image_level_mcc,
    evaluate_saco,
)
from open_vision_banana.evaluation.referring import ReferringAccumulator, evaluate_referring
from open_vision_banana.inference.prompt import PromptBuilder


def _instance(mask: torch.Tensor) -> dict:
    return {"mask": mask.bool()}


def test_saco_perfect_positive_and_negative_queries() -> None:
    mask = torch.tensor([[1, 1], [0, 0]], dtype=torch.bool)
    metrics = evaluate_saco(
        pred_instances=[[_instance(mask)], []],
        gt_instances=[[_instance(mask)], []],
        pred_present=[True, False],
        gt_present=[True, False],
    )

    assert metrics["pmF1"] == pytest.approx(1.0)
    assert metrics["IL_MCC"] == pytest.approx(1.0)
    assert metrics["cgF1"] == pytest.approx(100.0)
    assert metrics["queries"] == 2
    assert metrics["positive_queries"] == 1


def test_saco_cgf1_uses_paper_product_not_geometric_mean() -> None:
    mask = torch.ones(2, 2, dtype=torch.bool)
    metrics = evaluate_saco(
        pred_instances=[[_instance(mask)], [], [_instance(mask)], []],
        gt_instances=[[_instance(mask)], [], [_instance(mask)], []],
        pred_present=[True, True, True, False],
        gt_present=[True, False, True, False],
    )

    # Confusion: TP=2, TN=1, FP=1, FN=0 -> MCC = 2/sqrt(12).
    expected_mcc = 2.0 / (12.0 ** 0.5)
    assert metrics["pmF1"] == pytest.approx(1.0)
    assert metrics["IL_MCC"] == pytest.approx(expected_mcc)
    assert metrics["cgF1"] == pytest.approx(100.0 * expected_mcc)


def test_image_level_mcc_validates_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_image_level_mcc([True], [True, False])


def test_referring_ciou_and_giou_use_dataset_level_aggregation() -> None:
    gt = torch.tensor(
        [
            [[1, 0], [0, 0]],
            [[1, 1], [0, 0]],
        ],
        dtype=torch.bool,
    )
    pred = torch.tensor(
        [
            [[1, 0], [0, 0]],
            [[1, 0], [0, 1]],
        ],
        dtype=torch.bool,
    )
    accumulator = ReferringAccumulator(iou_threshold=0.5)
    accumulator.update(pred[:1], gt[:1])
    accumulator.update(pred[1:], gt[1:])
    metrics = accumulator.compute()

    assert metrics["cIoU"] == pytest.approx(0.5)  # (1 + 1) / (1 + 3)
    assert metrics["gIoU"] == pytest.approx((1.0 + 1.0 / 3.0) / 2.0)
    assert metrics["IoU"] == metrics["gIoU"]
    assert metrics["Precision@0.5"] == pytest.approx(0.5)
    assert metrics["samples"] == 2


def test_referring_batch_precision_is_per_query() -> None:
    gt = torch.tensor([[[1]], [[1]]], dtype=torch.bool)
    pred = torch.tensor([[[1]], [[0]]], dtype=torch.bool)
    metrics = evaluate_referring(pred, gt, iou_threshold=0.5)
    assert metrics["Precision@0.5"] == pytest.approx(0.5)


def test_cube_edge_matches_figure_5_hilbert_vertices() -> None:
    expected_colors = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    assert torch.allclose(CUBE_EDGE_KEYPOINTS[:, 0], torch.linspace(0.0, 1.0, 8))
    assert torch.equal(CUBE_EDGE_KEYPOINTS[:, 1:], expected_colors)
    assert "legacy_cube_edge" in get_available_colormaps()


def test_corrected_cube_edge_round_trip() -> None:
    mapper = DepthRGBMapper(colormap="cube_edge")
    depth = torch.logspace(-2, 2, 200).reshape(1, 1, 20, 10)
    recovered = mapper.rgb_to_depth(mapper.depth_to_rgb(depth))
    assert torch.allclose(recovered, depth, rtol=2e-4, atol=2e-4)


def test_cityscapes_evaluation_prompt_contains_all_nineteen_classes() -> None:
    from open_vision_banana.data.datasets import CityscapesSegDataset

    color_map = CityscapesSegDataset.CITYSCAPES_COLORS
    prompt = PromptBuilder.build_semantic_prompt(color_map)
    assert len(color_map) == 19
    for class_name, (red, green, blue) in color_map.items():
        assert f'"{class_name}": <{red}, {green}, {blue}>' in prompt


def test_cityscapes_subset_prompt_expands_to_canonical_legend() -> None:
    prompt = PromptBuilder.build_semantic_prompt(
        {"road": (128, 64, 128), "sky": (70, 130, 180)}
    )
    assert '"bicycle": <119, 11, 32>' in prompt
    assert prompt.count('<') == 19


def test_manifest_cityscapes_prompt_matches_inference_contract(tmp_path) -> None:
    """A prompt omitted from a manifest must use the full Cityscapes legend."""
    from scripts.train_sana_512 import ManifestVisionDataset
    from open_vision_banana.data.datasets import CityscapesSegDataset

    image = tmp_path / "input.png"
    target = tmp_path / "target.png"
    Image.new("RGB", (4, 4), (0, 0, 0)).save(image)
    Image.new("RGB", (4, 4), (128, 64, 128)).save(target)
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "task": "cityscapes",
                "split": "train",
                "input": image.name,
                "target": target.name,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = ManifestVisionDataset(
        manifest_path=manifest,
        resolution=4,
        split="train",
        tasks={"semantic"},
    )
    expected = PromptBuilder.build_semantic_prompt(CityscapesSegDataset.CITYSCAPES_COLORS)
    assert len(dataset) == 1
    assert dataset[0]["prompt"] == expected
    assert '"road": <128, 64, 128>' in dataset[0]["prompt"]
