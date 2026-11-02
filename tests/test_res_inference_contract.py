"""RES protocol regression tests without loading diffusion model weights."""

import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from open_vision_banana.core.referring import decode_referring_mask
from open_vision_banana.data.res_dataset import RESManifestDataset
from open_vision_banana.evaluation import evaluator as evaluator_module
from open_vision_banana.evaluation.evaluator import VisionEvaluator
from open_vision_banana.inference.decoder import VisualizationDecoder
from open_vision_banana.inference.generator import _tensor_to_pil
from open_vision_banana.inference.pipeline import VisionBanana
from open_vision_banana.inference.prompt import PromptBuilder
from open_vision_banana.referring_contract import (
    BINARY_REFERRING_ENCODING,
    LEGACY_REFERRING_ENCODING,
    build_binary_referring_prompt,
)


def _two_regions():
    rgb = torch.zeros(1, 3, 9, 11)
    rgb[:, :, 1:4, 1:4] = 1
    rgb[:, :, 7, 9] = 1
    return rgb


def _pipeline(rgb):
    model = object.__new__(VisionBanana)
    model.training_prompts = {"referring": "Segment a previous training example.", "depth": "DEPTH"}
    model.prompt_builder = PromptBuilder()
    model.decoder = VisualizationDecoder()
    model.generator = SimpleNamespace(
        generate=lambda **kwargs: rgb.clone(),
        encode_prompts=lambda prompts: {"prompt_embeds": None, "pooled_prompt_embeds": None},
    )
    return model


def test_binary_keeps_disconnected_and_single_pixel_regions():
    rgb = _two_regions()
    mask = VisualizationDecoder().decode_referring(rgb)["mask"]
    assert mask.dtype == torch.bool
    assert mask.shape == (1, 1, 9, 11)
    assert mask.sum() == 10
    assert mask[0, 0, 7, 9]


def test_binary_threshold_matches_historical_evaluator_boundary():
    rgb = torch.tensor([0.49, 0.5, 0.51]).view(1, 1, 1, 3).expand(1, 3, 1, 3)
    assert decode_referring_mask(rgb).flatten().tolist() == [False, False, True]


def test_legacy_opt_in_preserves_largest_component():
    pytest.importorskip("scipy")
    mask = decode_referring_mask(_two_regions(), encoding=LEGACY_REFERRING_ENCODING)
    assert mask.sum() == 9
    assert not mask[0, 0, 7, 9]


def test_nonwhite_color_call_still_uses_color_distance():
    rgb = _two_regions()
    rgb[:, 1:] = 0
    decoder = VisualizationDecoder()
    implicit = decoder.decode_referring(rgb, target_color=(255, 0, 0), threshold=0.1)["mask"]
    explicit = decoder.decode_referring(
        rgb,
        target_color=(255, 0, 0),
        threshold=0.1,
        encoding=LEGACY_REFERRING_ENCODING,
    )["mask"]
    assert torch.equal(implicit, explicit)
    assert implicit[0, 0, 1, 1]
    assert not implicit[0, 0, 0, 0]


@pytest.mark.parametrize("encoding", ["unknown", "binary_white_black_v2"])
def test_unknown_protocol_fails(encoding):
    with pytest.raises(ValueError, match="Unknown referring encoding"):
        decode_referring_mask(_two_regions(), encoding=encoding)


def test_binary_rejects_conflicting_nonwhite_target():
    with pytest.raises(ValueError, match="white target_color"):
        decode_referring_mask(
            _two_regions(), target_color=(255, 0, 0), encoding=BINARY_REFERRING_ENCODING
        )


def test_default_prompt_uses_current_expression_and_shared_binary_template():
    model = _pipeline(_two_regions())
    first = model._build_prompt("referring", expression="the left dog")
    second = model._build_prompt("referring", expression="both right dogs")
    assert first == build_binary_referring_prompt("the left dog")
    assert "both right dogs" in second
    assert "The left dog is represented by a solid white" in first
    assert "previous training example" not in first
    assert model._build_prompt("depth") == "DEPTH"


@pytest.mark.parametrize("expression", [None, "", "   "])
def test_checkpoint_prompt_cannot_bypass_missing_expression(expression):
    with pytest.raises(ValueError, match="expression"):
        _pipeline(_two_regions())._build_prompt("referring", expression=expression)


def test_custom_color_and_explicit_legacy_are_reflected_in_prompts():
    model = _pipeline(_two_regions())
    custom = model._build_prompt("referring", expression="the dog", target_color=(255, 0, 0))
    assert "<255, 0, 0>" in custom
    legacy = model._build_prompt(
        "referring", expression="the dog", referring_encoding=LEGACY_REFERRING_ENCODING
    )
    assert "other objects are rendered in green" in legacy


@pytest.mark.parametrize("encoding", [None, BINARY_REFERRING_ENCODING, LEGACY_REFERRING_ENCODING])
def test_predict_and_evaluator_use_identical_masks(monkeypatch, encoding):
    rgb = _two_regions()
    rgb[:, :, 5, 5] = 0.6  # Distinguishes binary intensity and legacy color distance.
    model = _pipeline(rgb)
    kwargs = {} if encoding is None else {"referring_encoding": encoding}
    result = model.predict(torch.zeros_like(rgb), "referring", expression="the pieces", **kwargs)
    observed = []
    original = evaluator_module.evaluate_referring

    def capture(pred, gt):
        observed.append(pred.clone())
        return original(pred, gt)

    monkeypatch.setattr(evaluator_module, "evaluate_referring", capture)
    VisionEvaluator(model, device="cpu").evaluate(
        [{"image": torch.zeros_like(rgb), "target_rgb": rgb, "expression": ["the pieces"]}],
        task="referring",
        **kwargs,
    )
    assert torch.equal(observed[0], result["mask"].squeeze(1))
    assert result["metadata"]["referring_encoding"] == (encoding or BINARY_REFERRING_ENCODING)


def test_evaluator_keeps_gt_components_even_for_legacy_predictions(monkeypatch):
    rgb = _two_regions()
    observed = []
    original = evaluator_module.evaluate_referring

    def capture(pred, gt):
        observed.append(gt.clone())
        return original(pred, gt)

    monkeypatch.setattr(evaluator_module, "evaluate_referring", capture)
    VisionEvaluator(_pipeline(rgb), device="cpu").evaluate_batch(
        torch.zeros_like(rgb),
        rgb,
        "referring",
        expression="the pieces",
        referring_encoding=LEGACY_REFERRING_ENCODING,
    )
    assert observed[0].sum() == 10


def test_batch_explicit_prompts_support_the_same_protocol_options():
    rgb = _two_regions().expand(2, -1, -1, -1)
    model = _pipeline(rgb)
    results = model.predict_batch(
        torch.zeros_like(rgb),
        "referring",
        prompts=["first", "second"],
        referring_encoding=BINARY_REFERRING_ENCODING,
        referring_threshold=0.9,
    )
    assert len(results) == 2
    assert all(result["mask"].sum() == 10 for result in results)
    assert all(
        result["metadata"]["referring_encoding"] == BINARY_REFERRING_ENCODING for result in results
    )


def test_streaming_forwards_color_encoding_and_threshold():
    rgb = torch.zeros(1, 3, 4, 4)
    rgb[:, 0] = 0.8
    model = _pipeline(rgb)
    result = model.predict_streaming(
        torch.zeros_like(rgb),
        "referring",
        expression="red target",
        target_color=(255, 0, 0),
        referring_encoding=LEGACY_REFERRING_ENCODING,
        referring_threshold=0.3,
    )
    assert result["mask"].all()
    assert "<255, 0, 0>" in result["prompt"]


@pytest.mark.parametrize("threshold, expected", [(0.5, True), (0.7, False)])
def test_tiled_referring_forwards_threshold_and_preserves_protocol(threshold, expected):
    rgb = torch.full((1, 3, 8, 12), 0.6)
    model = _pipeline(rgb)
    model.generator.generate = lambda **kwargs: kwargs["image"]
    result = model.predict_tiled(
        rgb,
        "referring",
        tile_size=6,
        overlap=2,
        expression="the region",
        referring_encoding=BINARY_REFERRING_ENCODING,
        referring_threshold=threshold,
    )
    assert bool(result["mask"].all()) == expected
    assert result["metadata"]["referring_encoding"] == BINARY_REFERRING_ENCODING
    assert torch.equal(decode_referring_mask(result["rgb_output"]), result["mask"])


def test_tiled_custom_color_rgb_is_encoded_with_the_requested_color():
    rgb = torch.zeros(1, 3, 8, 12)
    rgb[:, 0] = 1
    model = _pipeline(rgb)
    model.generator.generate = lambda **kwargs: kwargs["image"]
    result = model.predict_tiled(
        rgb,
        "referring",
        tile_size=6,
        overlap=2,
        expression="the region",
        target_color=(255, 0, 0),
        referring_threshold=0.1,
    )
    assert result["mask"].all()
    assert result["rgb_output"][:, 0].all()
    assert not result["rgb_output"][:, 1:].any()
    assert torch.equal(result["visualization"], result["rgb_output"])
    assert result["metadata"]["referring_encoding"] == LEGACY_REFERRING_ENCODING


def test_manifest_evaluation_preserves_bright_gray_192_through_generator(tmp_path):
    Image.new("RGB", (3, 2), (192, 192, 192)).save(tmp_path / "image.png")
    Image.new("L", (3, 2), 255).save(tmp_path / "mask.png")
    row = {
        "sample_id": "gray192",
        "task": "referring",
        "split": "val",
        "source_dataset": "refcoco",
        "source_protocol": "unc",
        "source_split": "val",
        "input": "image.png",
        "target": "mask.png",
        "width": 3,
        "height": 2,
        "expression": "the region",
        "prompt": build_binary_referring_prompt("the region"),
        "encoding": BINARY_REFERRING_ENCODING,
    }
    (tmp_path / "pairs.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    dataset = RESManifestDataset(tmp_path, split="val", resolution=2)
    # All normalized pixels are positive, which defeated generator's heuristic.
    assert dataset[0]["image"].min() > 0
    seen = []
    model = _pipeline(torch.ones(1, 3, 2, 2))

    def generate(**kwargs):
        seen.append(_tensor_to_pil(kwargs["image"][0]).getpixel((0, 0)))
        return torch.ones_like(kwargs["image"])

    model.generator.generate = generate
    metrics = VisionEvaluator(model, device="cpu").evaluate(
        DataLoader(dataset, batch_size=1),
        task="referring",
        native_resolution=True,
    )
    assert seen == [(192, 192, 192)]
    assert metrics["cIoU"] == pytest.approx(1.0)


@pytest.mark.parametrize("task, image_range", [("depth", "minus_one_one"), ("referring", None)])
def test_other_benchmarks_and_unmarked_res_keep_input_values(task, image_range):
    rgb = torch.full((1, 3, 2, 2), 0.25)
    seen = []
    model = _pipeline(torch.ones_like(rgb))

    def generate(**kwargs):
        seen.append(kwargs["image"].clone())
        return torch.ones_like(kwargs["image"])

    model.generator.generate = generate
    batch = {"image": rgb, "target_rgb": torch.ones_like(rgb), "expression": "the region"}
    if image_range is not None:
        batch["image_range"] = [image_range]
    VisionEvaluator(model, device="cpu").evaluate([batch], task=task)
    assert torch.equal(seen[0], rgb)


def test_referring_prompt_is_the_paper_template_everywhere():
    """Rea2Seg / RES builders, inference and evaluation share one Fig. 4 (c)/(d) string."""
    from scripts.build_rea2seg_manifest import referring_prompt

    from open_vision_banana.data.res_contract import resolve_referring_prompt
    from open_vision_banana.referring_contract import (
        BINARY_REFERRING_TEMPLATE,
        is_binary_referring_prompt,
    )

    expected = (
        "This image shows segmentation masks from the given image. The background is "
        "black color. The red car is represented by a solid white."
    )
    assert BINARY_REFERRING_TEMPLATE.format(expression="red car") == expected
    assert build_binary_referring_prompt("the red car") == expected
    assert build_binary_referring_prompt("  red car ") == expected
    assert PromptBuilder.build_referring_prompt("the red car") == expected
    assert referring_prompt("the red car") == expected
    assert resolve_referring_prompt({"expression": "the red car"}) == expected
    assert "The the" not in expected

    # Pre-template manifests stay loadable but are not re-emitted.
    old = "Segment the red car in solid white. Render all other pixels black."
    assert is_binary_referring_prompt(old, "the red car")
    assert resolve_referring_prompt(
        {"expression": "the red car", "prompt": old, "encoding": BINARY_REFERRING_ENCODING}
    ) == old
    with pytest.raises(ValueError, match="canonical"):
        resolve_referring_prompt(
            {"expression": "the red car", "prompt": "Paint it white.", "encoding": BINARY_REFERRING_ENCODING}
        )

    # Fig. 4 (a) and the legacy green-distractor wording remain reachable explicitly.
    legacy = PromptBuilder.build_referring_prompt("the red car", encoding=LEGACY_REFERRING_ENCODING)
    assert legacy.startswith("A segmentation map image.") and "green" in legacy
    for template in PromptBuilder.REFERRING_TEMPLATES:
        assert "Other objects are" not in template and "all other pixels" not in template
