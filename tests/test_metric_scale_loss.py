"""Tests for the opt-in metric scale ablation."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.evaluation.benchmark import DepthAccumulator
from open_vision_banana.evaluation.shards import accumulator_state
from open_vision_banana.training.config import build_arg_parser, load_config
from open_vision_banana.training.metric_scale_loss import metric_scale_loss
from open_vision_banana.training.trainer import ManifestVisionDataset, Trainer, _collate_fn
from scripts.merge_eval_shards import merge_shards


def test_metric_scale_penalizes_global_bias_and_ignores_invalid_pixels():
    gt = torch.full((1, 1, 8, 8), 2.0)
    mask = torch.ones_like(gt, dtype=torch.bool)
    mask[..., 0, 0] = False
    gt[..., 0, 0] = float("nan")
    factor = torch.tensor(2.0, requires_grad=True)
    depth = torch.full_like(gt, 2.0) * factor
    rgb = DepthRGBMapper().depth_to_rgb(depth) * 2 - 1
    loss = metric_scale_loss(rgb, gt, mask)
    loss.backward()
    assert loss.item() > 0.1
    assert factor.grad.item() > 0
    rgb_correct = DepthRGBMapper().depth_to_rgb(torch.full_like(gt, 2.0)) * 2 - 1
    assert metric_scale_loss(rgb_correct, gt, mask).item() < 1e-4
    white = torch.ones_like(rgb_correct, requires_grad=True)
    white_loss = metric_scale_loss(white, gt, mask)
    white_loss.backward()
    assert torch.isfinite(white_loss)
    assert torch.isfinite(white.grad).all()


def test_metric_dataset_uses_same_crop_and_requires_both_native_assets(tmp_path):
    depth = np.tile(np.arange(1000, 1024, dtype=np.uint16), (12, 1))
    mask = np.full_like(depth, 255, dtype=np.uint8)
    mask[6, 12] = 0
    red = (depth - 1000).astype(np.uint8)
    rgb = np.stack([red, red, red], axis=-1)
    Image.fromarray(rgb).save(tmp_path / "rgb.png")
    Image.fromarray(depth).save(tmp_path / "metric.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")
    base = {"task": "depth", "split": "train", "input": "rgb.png", "target": "rgb.png"}
    native = dict(base, metric_depth="metric.png", valid_mask="mask.png", metric_depth_scale=1000)
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row) for row in (native, dict(base, metric_depth="metric.png"))),
        encoding="utf-8",
    )
    options = {"aspect_preserving_crop": True, "seed": 42}
    enabled = ManifestVisionDataset(
        manifest, 12, "train", {"depth"}, **options, load_metric_supervision=True
    )
    control = ManifestVisionDataset(manifest, 12, "train", {"depth"}, **options)
    sample = enabled[0]
    reference = control[0]
    torch.testing.assert_close(sample["image"], reference["image"], rtol=0, atol=0)
    torch.testing.assert_close(sample["target_rgb"], reference["target_rgb"], rtol=0, atol=0)
    metres, valid = sample["metric_depth_supervision"]
    assert not valid.all()
    torch.testing.assert_close(
        metres[valid], 1 + (sample["target_rgb"][:1][valid] * 255).round() / 1000
    )
    assert "metric_depth_supervision" not in reference
    assert _collate_fn([sample, enabled[1]])["metric_depth_supervision"][1] is None


@pytest.mark.parametrize("checkpointing", [False, True])
def test_sft_scale_hook_decodes_only_eligible_endpoints_and_backpropagates(checkpointing):
    gt = torch.full((1, 4, 4), 2.0)
    noisy = DepthRGBMapper().depth_to_rgb(gt.unsqueeze(0)).expand(2, -1, -1, -1) * 2 - 1
    velocity = torch.full_like(noisy, 0.1, requires_grad=True)
    timesteps = torch.tensor([0.4, 0.9])

    def decode(endpoint):
        assert endpoint.shape[0] == 1
        torch.testing.assert_close(endpoint, noisy[:1] + (1 - timesteps[0]) * velocity[:1])
        return endpoint

    trainer = SimpleNamespace(
        backend=SimpleNamespace(family="qwen-image-2.1", decode_latents=decode),
        config=SimpleNamespace(gradient_checkpointing=checkpointing),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    batch = dict(task=["depth", "referring"], metric_depth_supervision=[(gt, gt > 0), None])
    loss, count = Trainer._metric_scale_loss(trainer, batch, velocity, noisy, timesteps)
    loss.backward()
    assert count == 1 and velocity.grad[0].abs().sum().item() > 0
    assert velocity.grad[1].abs().sum().item() == 0


def test_scale_loss_config_is_off_by_default_and_rejects_other_palettes():
    parser = build_arg_parser()
    assert load_config(parser.parse_args([])).depth_scale_loss_weight == 0
    with pytest.raises(ValueError, match="requires metric_power and cube_edge"):
        load_config(
            parser.parse_args(["--depth-scale-loss-weight", ".01", "--depth-colormaps", "plasma"])
        )


def test_scale_ratio_sidecar_preserves_shard_order_and_existing_metrics():
    shards = []
    for index, factor in enumerate([2.0, 0.5]):
        accumulator = DepthAccumulator()
        ratios = []
        truth = torch.ones(1, 1, 2, 2)
        accumulator.update(truth * factor, truth, scale_ratios=ratios)
        shards.append(
            {
                "metadata": {},
                "benchmarks": {
                    "depth/nyu": {
                        "samples": 1,
                        "depth_scale_ratios": ratios,
                        "shard": dict(
                            index=index, count=2, start=index, stop=index + 1, of_samples=2
                        ),
                        "accumulator_states": dict(
                            main=accumulator_state(accumulator),
                            relative=None,
                            skipped_samples=0,
                            depth_curve="metric_power",
                            depth_units="metres",
                        ),
                    }
                },
            }
        )
    row = merge_shards(shards[::-1])["benchmarks"]["depth/nyu"]
    assert row["depth_scale_ratios"] == [2.0, 0.5]
    assert row["decoded_metrics"]["delta_1.25"] == 0
    assert row["decoded_metrics"]["delta_1.25_aligned"] == 1
    del shards[1]["benchmarks"]["depth/nyu"]["depth_scale_ratios"]
    with pytest.raises(ValueError, match="incomplete depth scale ratios"):
        merge_shards(shards)
