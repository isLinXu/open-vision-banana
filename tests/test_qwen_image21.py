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
    # Cityscapes upsampling is an in-domain feature; strict refuses it (tested below)
    audit = build_qwen_baseline_manifest(base, out, data_root=tmp_path, min_cityscapes_share=0.50, protocol="in_domain")
    assert audit["protocol"] == "in_domain" and audit["zero_shot_claim_allowed"] is False
    assert audit["rejected_non_metric_depth_rows"] == 1
    assert audit["task_split_counts"]["depth/train"] == 1
    assert audit["cityscapes_train_initial"] == 1
    assert audit["cityscapes_train_final"] == 9
    assert abs(audit["cityscapes_train_share"] - 0.50) < 1e-6


def test_build_qwen_baseline_manifest_enforces_zero_shot_protocol(tmp_path: Path):
    from scripts.build_qwen_baseline_manifest import benchmark_leak

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.new("RGB", (16, 16), (50, 60, 70)).save(img_dir / "in.png")
    good_rgb = DepthRGBMapper(colormap="cube_edge").depth_to_rgb(torch.full((1, 1, 16, 16), 2.5))[0]
    depth_png = (good_rgb.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    for sub in ("vkitti2/Scene01/depth", "hypersim/ai_001/depth", "nyu_depth_v2/train"):
        d = tmp_path / sub
        d.mkdir(parents=True)
        Image.fromarray(depth_png).save(d / "d.png")

    rows = [
        {"task": "depth", "input": "images/in.png", "target": "vkitti2/Scene01/depth/d.png", "split": "train"},
        {"task": "depth", "input": "images/in.png", "target": "hypersim/ai_001/depth/d.png", "split": "train"},
        {"task": "depth", "input": "images/in.png", "target": "nyu_depth_v2/train/d.png", "split": "train"},
        {"task": "referring", "input": "refcocog/train2014/x.jpg", "target": "images/in.png", "split": "train"},
    ]
    base = tmp_path / "base.jsonl"
    base.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    strict = build_qwen_baseline_manifest(base, tmp_path / "strict.jsonl", data_root=tmp_path)
    assert strict["rejected_benchmark_leak_rows"] == {"vkitti": 1, "nyu": 1, "refcocog": 1}
    assert strict["task_split_counts"] == {"depth/train": 1}

    with pytest.raises(ValueError, match="does not permit allow_benchmarks"):
        build_qwen_baseline_manifest(base, tmp_path / "x.jsonl", data_root=tmp_path, allow_benchmarks=["vkitti"])
    relaxed = build_qwen_baseline_manifest(
        base, tmp_path / "relaxed.jsonl", data_root=tmp_path, allow_benchmarks=["vkitti"], protocol="in_domain"
    )
    assert relaxed["rejected_benchmark_leak_rows"] == {"nyu": 1, "refcocog": 1}
    assert relaxed["task_split_counts"]["depth/train"] == 2
    # "virtual_kitti/" also contains "kitti/": an allowed corpus must not be re-flagged as KITTI
    assert benchmark_leak({"target": "virtual_kitti/Scene01/depth/d.png"}, {"vkitti"}) is None
    with pytest.raises(ValueError, match="allow_benchmarks"):
        build_qwen_baseline_manifest(base, tmp_path / "x.jsonl", allow_benchmarks=["bogus"], protocol="in_domain")


def test_strict_protocol_rejects_cityscapes_train_rows_and_manifest(tmp_path: Path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.new("RGB", (16, 16), (50, 60, 70)).save(img_dir / "in.png")
    rows = [
        {"task": "semantic", "input": "cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png",
         "target": "images/in.png", "split": "train", "prompt": SEMANTIC_SEGMENTATION_PROMPT},
        {"task": "semantic", "input": "images/in.png", "target": "images/in.png", "split": "train",
         "prompt": 'Generate a semantic segmentation visualization image, using this color mapping: {"wall": <255, 0, 0>}.'},
    ]
    base = tmp_path / "base.jsonl"
    base.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    strict = build_qwen_baseline_manifest(base, tmp_path / "s.jsonl", data_root=tmp_path)
    assert strict["rejected_benchmark_leak_rows"] == {"cityscapes": 1}
    assert strict["task_split_counts"] == {"semantic/train": 1}
    with pytest.raises(ValueError, match="min_cityscapes_share == 0"):
        build_qwen_baseline_manifest(base, tmp_path / "x.jsonl", data_root=tmp_path, min_cityscapes_share=0.5)
    with pytest.raises(ValueError, match="Cityscapes train manifest"):
        build_qwen_baseline_manifest(base, tmp_path / "x.jsonl", data_root=tmp_path, cityscapes_manifest=base)
    in_domain = build_qwen_baseline_manifest(base, tmp_path / "d.jsonl", data_root=tmp_path, protocol="in_domain")
    assert in_domain["rejected_benchmark_leak_rows"] == {} and in_domain["task_split_counts"] == {"semantic/train": 2}
    assert in_domain["zero_shot_claim_allowed"] is False
    with pytest.raises(ValueError, match="protocol must be"):
        build_qwen_baseline_manifest(base, tmp_path / "x.jsonl", protocol="loose")


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
            assert cfg.semantic_prompt_styles == ["json_named", "json_tuple", "natural", "hex"]
            assert cfg.semantic_color_remap_prob == 0.5
            assert cfg.aspect_preserving_crop is True


def test_referring_prompt_and_decoder_match_white_on_black_contract():
    from open_vision_banana.inference.decoder import VisualizationDecoder
    from open_vision_banana.inference.prompt import PromptBuilder

    prompt = PromptBuilder.build_referring_prompt("the red bicycle")
    assert "red bicycle" in prompt.casefold()
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



def test_build_pipeline_requires_the_processor():
    pytest.importorskip("diffusers")
    with pytest.raises(RuntimeError, match="Qwen3VLProcessor"):
        qwen_image21.build_pipeline(types.SimpleNamespace(processor=None))


def test_encode_condition_feeds_the_source_to_the_text_encoder():
    pipe = _Pipe()
    source = Image.new("RGB", (32, 32))
    cond = qwen_image21.encode_condition(pipe, ["depth"], [source], "cpu", torch.float32)
    assert pipe.calls == [(["depth"], [source])]
    assert set(cond) == {"encoder_hidden_states", "image_pad_mask"}
    assert cond["image_pad_mask"].sum() == 2


def test_encode_condition_rejects_batches():
    with pytest.raises(ValueError, match="batch_size 1"):
        qwen_image21.encode_condition(_Pipe(), ["a", "b"], [None, None], "cpu", torch.float32)


def test_backend_forward_routes_the_qwen21_edit_forward():
    def backbone(**kwargs):
        return (kwargs["hidden_states"],)

    backend = _backend(module=backbone, pipe=_Pipe(), vae=_Vae())
    latents = backend.encode_image(torch.zeros(1, 3, 32, 32))
    cond = {
        "encoder_hidden_states": torch.randn(1, 4, 8),
        "image_pad_mask": torch.tensor([[False, True, True, False]]),
    }
    noisy = torch.randn(1, C, 2, 2)
    out = backend.forward(noisy, torch.tensor([0.5]), cond, 1, context_latents=latents)
    torch.testing.assert_close(out, -noisy)


def test_cfg_dropout_blanks_the_prompt_and_keeps_the_source():
    pipe = _Pipe()
    backend = _backend(pipe=pipe)
    images = torch.zeros(1, 3, 32, 32)
    backend.encode_edit_text(["depth"], images, cfg_dropout_rate=0.0)
    backend.encode_edit_text(["depth"], images, cfg_dropout_rate=1.0)
    assert [call[0] for call in pipe.calls] == [["depth"], [""]]
    assert all(len(call[1]) == 1 for call in pipe.calls)


@pytest.mark.parametrize(
    ("batch_size", "resolution", "message"),
    [(4, 512, "batch_size 1"), (1, 528, "multiple of 32"), (1, 224, "multiple of 32")],
)
def test_training_config_is_checked_before_loading(batch_size, resolution, message):
    config = types.SimpleNamespace(batch_size=batch_size, resolution=resolution)
    with pytest.raises(ValueError, match=message):
        qwen_image21.check_training_config(config)
    qwen_image21.check_training_config(types.SimpleNamespace(batch_size=1, resolution=512))


def test_quantized_loading_is_refused(monkeypatch):
    def load_pipeline(*args, **kwargs):
        raise AssertionError("the model must not load")

    monkeypatch.setattr("open_vision_banana.training.model_backend.load_pipeline", load_pipeline)
    with pytest.raises(ValueError, match="Quantized loading"):
        ModelBackend("qwen-image-2.1", device=torch.device("cpu"), quantize="4bit")


class _FakeQwen21Pipeline:
    """Stands in for diffusers.QwenImage21Pipeline in loader tests."""

    def __init__(self, **modules):
        self.modules = modules

    @classmethod
    def from_pretrained(cls, source, **kwargs):
        return types.SimpleNamespace(
            transformer=torch.nn.Linear(2, 2),
            vae=torch.nn.Linear(2, 2),
            text_encoder=torch.nn.Linear(2, 2),
            processor="qwen3-vl-processor",
            scheduler="scheduler",
        )


def test_load_pipeline_keeps_the_processor(monkeypatch):
    diffusers = pytest.importorskip("diffusers")

    from open_vision_banana.model_registry import load_pipeline

    monkeypatch.setattr(diffusers, "QwenImage21Pipeline", _FakeQwen21Pipeline, raising=False)
    components = load_pipeline(CONFIG, device="cpu", dtype=torch.float32)
    assert components.processor == "qwen3-vl-processor"
    assert components.scheduler == "scheduler"


def test_build_pipeline_wraps_the_loaded_components(monkeypatch):
    diffusers = pytest.importorskip("diffusers")

    monkeypatch.setattr(diffusers, "QwenImage21Pipeline", _FakeQwen21Pipeline, raising=False)
    parts = types.SimpleNamespace(scheduler="s", vae="v", text_encoder="t", processor="p", unet="u")
    pipe = qwen_image21.build_pipeline(parts)
    assert pipe.modules == {
        "scheduler": "s",
        "vae": "v",
        "text_encoder": "t",
        "processor": "p",
        "transformer": "u",
    }


def test_backend_builds_the_prompt_pipeline_once(monkeypatch):
    from open_vision_banana.training import model_backend

    components = types.SimpleNamespace(unet=torch.nn.Linear(2, 2), vae=_Vae(), processor="p")
    built = []
    monkeypatch.setattr(model_backend, "load_pipeline", lambda config, **kwargs: components)
    monkeypatch.setattr(model_backend, "apply_lora", lambda backbone, **kwargs: backbone)
    monkeypatch.setattr(
        model_backend.qwen_image21, "build_pipeline", lambda parts: built.append(parts) or "pipe"
    )
    backend = ModelBackend("qwen-image-2.1", device=torch.device("cpu"), dtype=torch.float32)
    assert backend.pipe == "pipe" and built == [components]
    assert backend._qwen21_logged is False


def test_trainer_rejects_batches_before_loading_the_model(monkeypatch, tmp_path):
    import open_vision_banana.training.trainer as module
    from open_vision_banana.training.config import TrainingConfig

    loaded = []
    monkeypatch.setattr(module.Trainer, "_build_datasets", lambda self: (None, None))
    monkeypatch.setattr(module, "ModelBackend", lambda **kwargs: loaded.append(kwargs))
    config = TrainingConfig(
        model_name="qwen-image-2.1", batch_size=4, device="cpu", output_dir=str(tmp_path)
    )
    with pytest.raises(ValueError, match="batch_size 1"):
        module.Trainer(config)
    assert not loaded


def test_encode_edit_text_keeps_the_other_layouts(monkeypatch):
    calls = []
    hunyuan = object.__new__(ModelBackend)
    hunyuan.model_config = get_model_config("hunyuan-image3-instruct")
    monkeypatch.setattr(
        ModelBackend, "encode_hunyuan_edit", lambda self, p, i: calls.append(("hunyuan", p)) or {}
    )
    hunyuan.encode_edit_text(["depth"], torch.zeros(1, 3, 8, 8), cfg_dropout_rate=0.5)

    text_only = object.__new__(ModelBackend)
    text_only.model_config = get_model_config("sana-1600m")
    text_only.device, text_only.dtype = torch.device("cpu"), torch.float32
    monkeypatch.setattr(
        ModelBackend, "encode_text", lambda self, p, d, t: calls.append(("text", p)) or {}
    )
    monkeypatch.setattr(
        ModelBackend,
        "apply_cfg_dropout",
        lambda self, cond, rate, n: calls.append(("dropout", rate, n)) or cond,
    )
    text_only.encode_edit_text(["depth"], torch.zeros(1, 3, 8, 8), cfg_dropout_rate=0.1)
    assert calls == [("hunyuan", ["depth"]), ("text", ["depth"]), ("dropout", 0.1, 1)]


def test_build_qwen_baseline_manifest_excludes_refcocog_eval_image_ids(tmp_path: Path):
    """RefCOCOg val images are COCO train2014 files: the path audit passes, the id audit must not."""
    import pickle

    from scripts.build_qwen_baseline_manifest import (
        coco_image_id_from_path,
        load_excluded_image_ids,
    )

    assert coco_image_id_from_path("coco/train2014/COCO_train2014_000000581921.jpg") == 581921
    assert coco_image_id_from_path("imgs/000000000042.png") == 42
    assert coco_image_id_from_path("hypersim/ai_001/frame.0001.png") is None  # not COCO naming
    assert coco_image_id_from_path("cityscapes/aachen_000000_000019_leftImg8bit.png") is None

    img_dir = tmp_path / "coco" / "train2014"
    img_dir.mkdir(parents=True)
    for image_id in (7, 8, 9):
        Image.new("RGB", (16, 16), (50, 60, 70)).save(img_dir / f"COCO_train2014_{image_id:012d}.jpg")
    Image.new("RGB", (16, 16), (128, 64, 128)).save(tmp_path / "sem.png")

    refs = tmp_path / "refs(umd).p"
    with refs.open("wb") as handle:
        pickle.dump(
            [
                {"image_id": 7, "split": "val"},
                {"image_id": 8, "split": "train"},  # train refs are not eval images
                {"image_id": 9, "split": "test"},
            ],
            handle,
        )
    txt = tmp_path / "more.txt"
    txt.write_text("# comment\nCOCO_train2014_000000000008.jpg\n", encoding="utf-8")
    assert load_excluded_image_ids([refs]) == {7, 9}
    assert load_excluded_image_ids([refs, txt]) == {7, 8, 9}

    rows = [
        {"task": "semantic", "input": f"coco/train2014/COCO_train2014_{i:012d}.jpg", "target": "sem.png", "split": "train", "prompt": SEMANTIC_SEGMENTATION_PROMPT}
        for i in (7, 8, 9)
    ]
    base = tmp_path / "base.jsonl"
    base.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    # strict: COCO rows without an exclusion source are refused outright (review item 1b)
    with pytest.raises(ValueError, match="carry COCO image ids but no exclude_image_ids"):
        build_qwen_baseline_manifest(base, tmp_path / "a.jsonl", data_root=tmp_path)
    no_ids = build_qwen_baseline_manifest(base, tmp_path / "a.jsonl", data_root=tmp_path, protocol="in_domain")
    assert no_ids["task_split_counts"] == {"semantic/train": 3}
    assert no_ids["rejected_benchmark_leak_rows"] == {}  # plain COCO paths are not a benchmark pattern

    with_ids = build_qwen_baseline_manifest(
        base, tmp_path / "b.jsonl", data_root=tmp_path, min_cityscapes_share=0.0, exclude_image_ids=[refs]
    )
    assert with_ids["rejected_excluded_image_id_rows"] == 2
    assert with_ids["excluded_image_id_count"] == 2
    assert with_ids["task_split_counts"] == {"semantic/train": 1}
    assert with_ids["protocol"] == "strict" and with_ids["zero_shot_claim_allowed"] is True
    assert with_ids["evaluation_image_id_overlap_by_source"] == {str(refs): 0}
    assert with_ids["train_val_coco_image_id_overlap"] == 0
    assert len(with_ids["manifest_sha256"]) == 64
    kept = [json.loads(line) for line in (tmp_path / "b.jsonl").read_text().splitlines()]
    assert kept[0]["input"].endswith("000000000008.jpg")


def test_paper_aligned_configs_mix_generation_replay():
    root = Path(__file__).resolve().parents[1]
    for name in ("qwen_image_21_paper_aligned.yaml", "qwen_image_edit_paper_aligned.yaml"):
        cfg = load_config(config_path=root / "configs" / "experiments" / name)
        assert "generation" in cfg.tasks
        assert 0.0 < cfg.vision_ratio < 1.0
        cfg.validate()


def test_generation_replay_builder_t2i_and_edit(tmp_path: Path):
    from scripts.build_generation_replay_manifest import build_generation_replay_manifest

    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for name in ("a.jpg", "b.jpg", "b_edit.jpg", "c.jpg"):
        Image.new("RGB", (8, 8), (10, 20, 30)).save(imgs / name)

    t2i = tmp_path / "captions.jsonl"
    t2i.write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                {"image": "imgs/a.jpg", "caption": "a photo of a cat"},
                {"image": "imgs/missing.jpg", "caption": "gone"},
                {"image": "imgs/c.jpg", "caption": "   "},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "gen_t2i.jsonl"
    audit = build_generation_replay_manifest(t2i, out, kind="t2i", data_root=tmp_path)
    assert audit["rows_written"] == 1 and audit["skipped_missing_file"] == 1
    assert audit["skipped_missing_text"] == 1
    row = json.loads(out.read_text().splitlines()[0])
    assert row == {"task": "generation", "target": "imgs/a.jpg", "prompt": "a photo of a cat", "split": "train"}

    edit = tmp_path / "edits.csv"
    edit.write_text(
        "source,target,instruction\n"
        "imgs/b.jpg,imgs/b_edit.jpg,make the sky purple\n"
        "imgs/a.jpg,imgs/a.jpg,identity should be dropped\n",
        encoding="utf-8",
    )
    out2 = tmp_path / "gen_edit.jsonl"
    audit2 = build_generation_replay_manifest(
        edit, out2, kind="edit", data_root=tmp_path, val_fraction=0.5
    )
    assert audit2["rows_written"] == 1 and audit2["skipped_identity_pairs"] == 1
    row2 = json.loads(out2.read_text().splitlines()[0])
    assert row2["input"] == "imgs/b.jpg" and row2["target"] == "imgs/b_edit.jpg"
    assert row2["prompt"] == "make the sky purple"

    # --append keeps the earlier rows
    build_generation_replay_manifest(t2i, out2, kind="t2i", data_root=tmp_path, append=True)
    assert len(out2.read_text().splitlines()) == 2


def test_editor_rejects_identity_generation_replay_before_loading(monkeypatch, tmp_path):
    import open_vision_banana.training.trainer as module
    from open_vision_banana.training.config import TrainingConfig

    img = tmp_path / "x.png"
    Image.new("RGB", (32, 32)).save(img)
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(
        json.dumps({"task": "generation", "target": "x.png", "prompt": "a cat", "split": "train"})
        + "\n"
        + json.dumps({"task": "depth", "input": "x.png", "target": "x.png", "split": "train"})
        + "\n",
        encoding="utf-8",
    )
    loaded = []
    monkeypatch.setattr(module, "ModelBackend", lambda **kwargs: loaded.append(kwargs))
    config = TrainingConfig(
        model_name="qwen-image-2.1",
        batch_size=1,
        device="cpu",
        output_dir=str(tmp_path / "out"),
        data_dir=str(tmp_path),
        pair_manifest=str(manifest),
        tasks=["depth", "generation"],
        vision_ratio=0.5,
        num_workers=0,
    )
    with pytest.raises(ValueError, match="instruction editor"):
        module.Trainer(config)
    assert not loaded

    # A text-to-image backbone may replay T2I rows (input == target is the convention).
    assert module._reject_identity_generation_replay(
        types.SimpleNamespace(datasets=[]), "sana-1600m"
    ) is None
