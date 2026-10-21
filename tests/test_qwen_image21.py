"""Qwen-Image-2.1 and Qwen-Image-Edit baseline training, dynamic prompts, and manifest rebalancing."""

import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.model_registry import (
    get_model_config,
    resolve_model_config,
    vae_decode,
    vae_encode,
)
from open_vision_banana.task_prompts import SEMANTIC_SEGMENTATION_PROMPT
from open_vision_banana.training import qwen_image21
from open_vision_banana.training.config import load_config
from open_vision_banana.training.edit_forward import edit_latent_layout, qwen21_edit_forward
from open_vision_banana.training.flow_matching import FlowScheduler
from open_vision_banana.training.model_backend import ModelBackend
from open_vision_banana.training.trainer import (
    ManifestVisionDataset,
    parse_semantic_legend_from_prompt,
)
from scripts.build_qwen_baseline_manifest import build_qwen_baseline_manifest

C = 64
CONFIG = get_model_config("qwen-image-2.1")


class _Vae:
    """AutoencoderKLQwenImage21 stand-in: 16x down, 64 latent channels, mean 0.5, std 2."""

    def __init__(self):
        self.config = types.SimpleNamespace(z_dim=C, latents_mean=[0.5] * C, latents_std=[2.0] * C)
        self.encoded = None
        self.decoded = None

    def encode(self, x):
        self.encoded = x
        mode = torch.full((x.shape[0], C, 1, x.shape[-2] // 16, x.shape[-1] // 16), 4.5)
        return types.SimpleNamespace(latent_dist=types.SimpleNamespace(mode=lambda: mode))

    def decode(self, z, return_dict=False):
        self.decoded = z
        return (torch.zeros(z.shape[0], 4, 1, 16 * z.shape[-2], 16 * z.shape[-1]),)


class _Pipe:
    def __init__(self):
        self.calls = []

    def _get_qwen_prompt_embeds(self, prompt, image, device, dtype=None):
        self.calls.append((prompt, image))
        pad = torch.tensor([[False, True, True, False, False]])
        return torch.randn(1, 5, 8), torch.ones(1, 5, dtype=torch.long), pad


def _backend(module=None, pipe=None, vae=None, model_name="qwen-image-2.1"):
    cfg = get_model_config(model_name)
    backend = object.__new__(ModelBackend)
    backend.device = torch.device("cpu")
    backend.dtype = torch.float32
    backend.model_name = model_name
    backend.model_config = cfg
    backend.family = cfg.family
    backend.peft_model = module
    backend.flow_scheduler = FlowScheduler()
    backend.pipe = pipe
    backend.vae = vae
    backend._qwen21_logged = False
    return backend


def test_qwen_family_registry_has_512_tokens_and_full_lora_targets():
    for name in ("qwen-image", "qwen-image-edit", "qwen-image-edit-2509", "qwen-image-2.1"):
        cfg = get_model_config(name)
        assert cfg.text_encoder_max_length == 512
        for proj in ("to_q", "to_k", "to_v", "to_out.0"):
            assert proj in cfg.lora_target_modules

    assert CONFIG.pipeline_class == "QwenImage21Pipeline"
    assert CONFIG.family == "qwen-image-2.1"
    assert CONFIG.instruction_editing and CONFIG.vae_latent_channels == C
    assert edit_latent_layout(CONFIG) == "qwen21"
    assert resolve_model_config("/models/Qwen-Image-2.1").name == "qwen-image-2.1"


def test_vae_encode_adds_opaque_alpha_and_normalizes_per_channel():
    vae = _Vae()
    images = torch.rand(2, 3, 32, 48) * 2 - 1
    latents = vae_encode(vae, images, CONFIG)
    assert vae.encoded.shape == (2, 4, 1, 32, 48)
    torch.testing.assert_close(vae.encoded[:, :3, 0], images)
    assert torch.all(vae.encoded[:, 3] == 1)
    torch.testing.assert_close(latents, torch.full((2, C, 2, 3), 2.0))


def test_vae_decode_undoes_normalization_and_keeps_rgb():
    vae = _Vae()
    out = vae_decode(vae, torch.zeros(1, C, 2, 3), CONFIG)
    assert out.shape == (1, 3, 32, 48)
    torch.testing.assert_close(vae.decoded, torch.full((1, C, 1, 2, 3), 0.5))


def test_edit_forward_puts_source_first_feeds_sigma_and_negates():
    calls = []

    def backbone(**kwargs):
        calls.append(kwargs)
        return (kwargs["hidden_states"] * 3,)

    h, w = 4, 6
    noisy, context = torch.randn(1, C, h, w), torch.randn(1, C, h, w)
    text = {
        "encoder_hidden_states": torch.randn(1, 4, 8),
        "image_pad_mask": torch.tensor([[False, True, True, False]]),
    }
    out = qwen21_edit_forward(backbone, noisy, context, torch.tensor([0.25]), text)

    kwargs = calls[0]
    hidden = kwargs["hidden_states"]
    torch.testing.assert_close(hidden[:, : h * w], context.flatten(2).transpose(1, 2))
    torch.testing.assert_close(hidden[:, h * w :], noisy.flatten(2).transpose(1, 2))
    torch.testing.assert_close(kwargs["timestep"], torch.tensor([0.75]))
    assert kwargs["img_shapes"] == [[(1, h, w), (1, h, w)]]
    assert kwargs["img_mask"].shape == (1, 4 + h * w // 4) and kwargs["img_mask"][:, 4:].all()
    assert kwargs["encoder_hidden_states_mask"] is None
    torch.testing.assert_close(out, -3 * noisy)


def test_backend_encodes_with_source_for_both_qwen21_and_qwen_edit():
    pipe21 = _Pipe()
    backend21 = _backend(module=lambda **kw: (kw["hidden_states"],), pipe=pipe21, vae=_Vae())
    images = torch.zeros(1, 3, 32, 32)
    cond21 = backend21.encode_edit_text(["depth"], images)
    assert pipe21.calls[0][0] == ["depth"] and pipe21.calls[0][1][0].size == (32, 32)
    assert "image_pad_mask" in cond21

    pipe_edit = _Pipe()
    backend_edit = _backend(pipe=pipe_edit, model_name="qwen-image-edit-2509")
    cond_edit = backend_edit.encode_edit_text(["semantic"], images)
    assert pipe_edit.calls[0][0] == ["semantic"] and pipe_edit.calls[0][1][0].size == (32, 32)
    assert "encoder_hidden_states" in cond_edit


def test_manifest_dataset_dynamic_depth_colormaps_and_semantic_remapping(tmp_path: Path):
    img_dir = tmp_path / "images"
    tgt_dir = tmp_path / "targets"
    img_dir.mkdir()
    tgt_dir.mkdir()

    Image.new("RGB", (32, 32), (100, 100, 100)).save(img_dir / "a.png")
    # Save a valid metric cube_edge depth target at 3.0m
    depth_rgb = DepthRGBMapper(colormap="cube_edge").depth_to_rgb(torch.full((1, 1, 32, 32), 3.0))[0]
    Image.fromarray((depth_rgb.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(
        tgt_dir / "d.png"
    )
    # Save a 2-class semantic target with road=(128, 64, 128) and sky=(70, 130, 180)
    sem_arr = np.zeros((32, 32, 3), dtype=np.uint8)
    sem_arr[:16, :] = (128, 64, 128)
    sem_arr[16:, :] = (70, 130, 180)
    Image.fromarray(sem_arr).save(tgt_dir / "s.png")

    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "task": "depth",
                "input": "images/a.png",
                "target": "targets/d.png",
                "split": "train",
                "prompt": "Generate a metric depth visualization of this image.",
            }
        )
        + "\n"
        + json.dumps(
            {
                "task": "semantic",
                "input": "images/a.png",
                "target": "targets/s.png",
                "split": "train",
                "prompt": SEMANTIC_SEGMENTATION_PROMPT,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    depth_ds = ManifestVisionDataset(
        manifest,
        32,
        "train",
        {"depth"},
        data_root=tmp_path,
        depth_colormaps=["viridis"],
        seed=42,
    )
    depth_sample = depth_ds[0]
    assert "viridis" in depth_sample["prompt"]
    decoded_v = DepthRGBMapper(colormap="viridis").rgb_to_depth(
        depth_sample["target_rgb"].unsqueeze(0)
    )
    assert torch.allclose(decoded_v, torch.full_like(decoded_v, 3.0), atol=0.15)

    sem_ds = ManifestVisionDataset(
        manifest,
        32,
        "train",
        {"semantic"},
        data_root=tmp_path,
        semantic_prompt_styles=["json_tuple"],
        semantic_color_remap_prob=0.999,
        seed=7,
    )
    sem_sample = sem_ds[0]
    new_legend = parse_semantic_legend_from_prompt(sem_sample["prompt"])
    assert len(new_legend) == 19
    assert new_legend["road"] != (128, 64, 128)
    top_pixel = tuple(
        (sem_sample["target_rgb"][:, 0, 0] * 255.0).round().to(torch.int32).tolist()
    )
    assert top_pixel == new_legend["road"]


def test_build_qwen_baseline_manifest_filters_gray_depth_and_upsamples_cityscapes(tmp_path: Path):
    img_dir = tmp_path / "images"
    tgt_dir = tmp_path / "targets"
    img_dir.mkdir()
    tgt_dir.mkdir()

    Image.new("RGB", (16, 16), (50, 60, 70)).save(img_dir / "in.png")
    # 1. Invalid grayscale pseudo-depth target
    Image.new("RGB", (16, 16), (180, 180, 180)).save(tgt_dir / "bad_depth.png")
    # 2. Valid cube-edge metric depth target
    good_rgb = DepthRGBMapper(colormap="cube_edge").depth_to_rgb(torch.full((1, 1, 16, 16), 2.5))[0]
    Image.fromarray((good_rgb.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(
        tgt_dir / "good_depth.png"
    )
    Image.new("RGB", (16, 16), (128, 64, 128)).save(tgt_dir / "sem.png")

    base = tmp_path / "base.jsonl"
    rows = [
        {"task": "depth", "input": "images/in.png", "target": "targets/bad_depth.png", "split": "train", "prompt": "Generate a relative depth map."},
        {"task": "depth", "input": "images/in.png", "target": "targets/good_depth.png", "split": "train", "prompt": "Generate a relative depth map."},
        {"task": "semantic", "input": "images/in.png", "target": "targets/sem.png", "split": "train", "prompt": SEMANTIC_SEGMENTATION_PROMPT},
    ]
    for i in range(9):
        rows.append(
            {
                "task": "semantic",
                "input": "images/in.png",
                "target": "targets/sem.png",
                "split": "train",
                "prompt": f'Generate a semantic segmentation visualization image, using this color mapping: {{"obj_{i}": <255, 0, 0>}}.',
            }
        )
    base.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    out = tmp_path / "out.jsonl"
    audit = build_qwen_baseline_manifest(base, out, data_root=tmp_path, min_cityscapes_share=0.50)
    assert audit["rejected_non_metric_depth_rows"] == 1
    assert audit["task_split_counts"]["depth/train"] == 1
    assert audit["cityscapes_train_initial"] == 1
    assert audit["cityscapes_train_final"] == 9
    assert abs(audit["cityscapes_train_share"] - 0.50) < 1e-6


def test_qwen_paper_aligned_configs_parse_cleanly():
    root = Path(__file__).resolve().parents[1]
    for name in (
        "qwen_image_21_shared_baseline.yaml",
        "qwen_image_21_shared_smoke.yaml",
        "qwen_image_21_paper_aligned.yaml",
        "qwen_image_edit_paper_aligned.yaml",
    ):
        cfg = load_config(config_path=root / "configs" / "experiments" / name)
        assert cfg.batch_size == 1
        assert cfg.grad_accum == 32
        if "paper_aligned" in name:
            assert cfg.depth_colormaps == ["cube_edge", "plasma", "inferno", "viridis", "grayscale"]
            assert cfg.semantic_prompt_styles == ["json_tuple", "natural", "hex"]
            assert cfg.semantic_color_remap_prob == 0.5
            assert cfg.aspect_preserving_crop is True


def test_referring_prompt_and_decoder_match_white_on_black_contract():
    from open_vision_banana.inference.decoder import VisualizationDecoder
    from open_vision_banana.inference.prompt import PromptBuilder

    prompt = PromptBuilder.build_referring_prompt("the red bicycle")
    assert "the red bicycle" in prompt
    assert "white" in prompt.lower()
    assert "black" in prompt.lower()
    assert "green" not in prompt.lower()

    # Two disjoint white regions on black background must both be preserved
    rgb = torch.zeros(1, 3, 32, 32)
    rgb[:, :, 2:6, 2:6] = 1.0
    rgb[:, :, 20:28, 20:28] = 1.0
    decoded = VisualizationDecoder().decode_referring(rgb)
    mask = decoded["mask"]
    assert bool(mask[0, 0, 3, 3]) is True
    assert bool(mask[0, 0, 22, 22]) is True
    assert bool(mask[0, 0, 12, 12]) is False


def test_aspect_preserving_crop_and_generation_ratio_config(tmp_path: Path):
    from open_vision_banana.training.config import TrainingConfig

    # 1. vision_ratio < 1.0 is accepted when 'generation' is in tasks
    cfg = TrainingConfig(
        tasks=["depth", "semantic", "generation"],
        vision_ratio=0.1,
    )
    cfg.validate()
    assert cfg.vision_ratio == 0.1

    # 2. aspect_preserving_crop on a 2:1 image (64x32 -> crop 32x32 without squashing)
    img_dir = tmp_path / "images"
    tgt_dir = tmp_path / "targets"
    img_dir.mkdir()
    tgt_dir.mkdir()
    # Left half (x < 32) is red, right half (x >= 32) is blue
    wide = np.zeros((32, 64, 3), dtype=np.uint8)
    wide[:, :32] = (255, 0, 0)
    wide[:, 32:] = (0, 0, 255)
    Image.fromarray(wide).save(img_dir / "w.png")
    Image.fromarray(wide).save(tgt_dir / "w.png")

    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "task": "semantic",
                "input": "images/w.png",
                "target": "targets/w.png",
                "split": "val",
                "prompt": SEMANTIC_SEGMENTATION_PROMPT,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ds = ManifestVisionDataset(
        manifest,
        resolution=32,
        split="val",
        tasks={"semantic"},
        data_root=tmp_path,
        aspect_preserving_crop=True,
    )
    sample = ds[0]
    assert sample["target_rgb"].shape == (3, 32, 32)
    # Center crop of 64x32 at resolution 32 is x in [16, 48): x=0..15 is red, x=16..31 is blue
    assert torch.allclose(sample["target_rgb"][:, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(sample["target_rgb"][:, 0, 31], torch.tensor([0.0, 0.0, 1.0]))

