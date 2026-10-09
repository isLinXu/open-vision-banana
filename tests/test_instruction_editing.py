"""Instruction-editing models beyond FLUX.1 Kontext.

Inference goes through each model's official pipeline. Training stays limited
to editors that set in_context_conditioning.
"""

import json

import pytest
import torch
from PIL import Image

from open_vision_banana.model_registry import _auto_detect_family, get_model_config


EDIT_MODELS = (
    "qwen-image-edit",
    "qwen-image-edit-2509",
    "longcat-image-edit",
    "flux2-klein",
    "joyimage-edit",
    "joyimage-edit-plus",
    "instruct-pix2pix",
    "sdxl-instruct-pix2pix",
    "hunyuan-image3-instruct",
    "hunyuan-image3-instruct-distil",
)


def test_edit_models_are_inference_only():
    kontext = get_model_config("flux-kontext")
    assert kontext.instruction_editing and kontext.in_context_conditioning
    assert not get_model_config("flux-schnell").instruction_editing
    assert not get_model_config("qwen-image").instruction_editing
    assert not get_model_config("flux2-dev").instruction_editing
    assert not get_model_config("hunyuan-image").instruction_editing

    expected = {
        "qwen-image-edit": "QwenImageEditPipeline",
        "qwen-image-edit-2509": "QwenImageEditPlusPipeline",
        "longcat-image-edit": "LongCatImageEditPipeline",
        "flux2-klein": "Flux2KleinPipeline",
        "joyimage-edit": "JoyImageEditPipeline",
        "joyimage-edit-plus": "JoyImageEditPlusPipeline",
        "instruct-pix2pix": "StableDiffusionInstructPix2PixPipeline",
        "sdxl-instruct-pix2pix": "StableDiffusionXLInstructPix2PixPipeline",
        "hunyuan-image3-instruct": "HunyuanImage3Instruct",
        "hunyuan-image3-instruct-distil": "HunyuanImage3Instruct",
    }
    for name, pipeline_class in expected.items():
        cfg = get_model_config(name)
        assert cfg.instruction_editing
        assert not cfg.in_context_conditioning
        assert cfg.pipeline_class == pipeline_class
        assert cfg.lora_target_modules


def test_auto_detect_edit_pipeline(tmp_path):
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "LongCatImageEditPipeline"})
    )
    cfg = _auto_detect_family(str(tmp_path))
    assert cfg is not None
    assert cfg.pipeline_class == "LongCatImageEditPipeline"
    assert cfg.instruction_editing
    assert not cfg.in_context_conditioning


def test_registered_editors_have_a_training_forward():
    from open_vision_banana.training.edit_forward import edit_latent_layout
    from open_vision_banana.training.model_backend import reject_unimplemented_instruction_edit

    reject_unimplemented_instruction_edit(get_model_config("flux-kontext"))
    for name in EDIT_MODELS:
        cfg = get_model_config(name)
        assert edit_latent_layout(cfg)
        reject_unimplemented_instruction_edit(cfg)


class _Out:
    def __init__(self, size):
        self.images = [Image.new("RGB", size, (10, 20, 30))]


def _generator(model_name, pipe):
    from open_vision_banana.inference.generator import BaseGenerator

    gen = object.__new__(BaseGenerator)
    gen.model_config = get_model_config(model_name)
    gen._t2i_pipe = pipe
    gen.device = "cpu"
    gen.source_conditioning_config = None
    gen._prompt_cache = {}
    return gen


def test_qwen_edit_uses_true_cfg_and_not_kontext_resize():
    calls = []

    class _Pipe:
        def __call__(
            self,
            image,
            prompt,
            num_inference_steps,
            height,
            width,
            true_cfg_scale,
            negative_prompt,
            generator=None,
        ):
            calls.append(
                {
                    "image": image,
                    "prompt": prompt,
                    "num_inference_steps": num_inference_steps,
                    "height": height,
                    "width": width,
                    "true_cfg_scale": true_cfg_scale,
                    "negative_prompt": negative_prompt,
                    "generator": generator,
                }
            )
            return _Out((width, height))

    gen = _generator("qwen-image-edit", _Pipe())
    src = Image.new("RGB", (70, 50))
    out = gen.generate(prompt="make it snow", image=src, num_inference_steps=1, guidance_scale=0.0, seed=3)
    kw = calls[0]
    assert kw["image"] is src
    assert kw["prompt"] == "make it snow"
    assert (kw["height"], kw["width"]) == (48, 64)
    assert kw["num_inference_steps"] == 50
    assert kw["true_cfg_scale"] == 4.0
    assert kw["negative_prompt"] == " "
    assert "_auto_resize" not in kw
    assert out.shape == (1, 3, 50, 70)


def test_longcat_edit_omits_height_and_width():
    calls = []

    class _Pipe:
        def __call__(self, image, prompt, num_inference_steps, guidance_scale, generator=None):
            calls.append(
                {
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "keys": {"image", "prompt", "num_inference_steps", "guidance_scale"},
                }
            )
            return _Out(image.size)

    gen = _generator("longcat-image-edit", _Pipe())
    out = gen._generate_instruction_edit(
        prompt="depth",
        image=Image.new("RGB", (64, 48)),
        num_inference_steps=1,
        guidance_scale=0.0,
        seed=None,
    )
    assert calls[0]["num_inference_steps"] == 50
    assert calls[0]["guidance_scale"] == 4.5
    assert "height" not in calls[0]
    assert out.shape == (1, 3, 48, 64)


def test_flux2_klein_passes_guidance_scale():
    calls = []

    class _Pipe:
        def __call__(self, image, prompt, height, width, num_inference_steps, guidance_scale, generator=None):
            calls.append(
                {
                    "height": height,
                    "width": width,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                }
            )
            return _Out((width, height))

    gen = _generator("flux2-klein", _Pipe())
    gen._generate_instruction_edit(
        prompt="edit",
        image=Image.new("RGB", (32, 32)),
        num_inference_steps=1,
        guidance_scale=0.0,
        seed=1,
    )
    assert calls[0] == {
        "height": 32,
        "width": 32,
        "num_inference_steps": 50,
        "guidance_scale": 4.0,
    }


def test_kontext_route_still_disables_auto_resize():
    calls = []

    class _Pipe:
        def __call__(self, **kw):
            calls.append(kw)
            return _Out((kw["width"], kw["height"]))

    gen = _generator("flux-kontext", _Pipe())
    gen.generate(
        prompt="depth",
        image=Image.new("RGB", (32, 32)),
        num_inference_steps=1,
        guidance_scale=0.0,
        seed=None,
    )
    assert calls[0]["_auto_resize"] is False
    assert calls[0]["num_inference_steps"] == 30
    assert calls[0]["guidance_scale"] == 2.5


def test_joyimage_edit_plus_passes_a_one_image_list():
    calls = []

    class _Pipe:
        def __call__(self, images, prompt, height, width, num_inference_steps, guidance_scale, generator=None):
            calls.append(
                {
                    "n_images": len(images),
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "height": height,
                    "width": width,
                }
            )
            return _Out((width, height))

    gen = _generator("joyimage-edit-plus", _Pipe())
    out = gen._generate_instruction_edit(
        prompt="add a hat",
        image=Image.new("RGB", (40, 40)),
        num_inference_steps=1,
        guidance_scale=0.0,
        seed=None,
    )
    assert calls[0] == {
        "n_images": 1,
        "num_inference_steps": 30,
        "guidance_scale": 4.0,
        "height": 32,
        "width": 32,
    }
    assert out.shape == (1, 3, 40, 40)


def test_instruct_pix2pix_keeps_image_guidance_default():
    calls = []

    class _Pipe:
        def __call__(
            self,
            prompt,
            image,
            num_inference_steps,
            guidance_scale,
            image_guidance_scale=1.5,
            generator=None,
        ):
            calls.append(
                {
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "image_guidance_scale": image_guidance_scale,
                    "image": image,
                }
            )
            return _Out(image.size)

    gen = _generator("instruct-pix2pix", _Pipe())
    src = Image.new("RGB", (32, 24))
    out = gen._generate_instruction_edit(
        prompt="make it snow",
        image=src,
        num_inference_steps=1,
        guidance_scale=0.0,
        seed=None,
    )
    assert calls[0]["image"] is src
    assert calls[0]["num_inference_steps"] == 100
    assert calls[0]["guidance_scale"] == 7.5
    assert calls[0]["image_guidance_scale"] == 1.5
    assert out.shape == (1, 3, 24, 32)


def test_hunyuan_image3_instruct_calls_generate_image(tmp_path):
    from open_vision_banana.inference.generator import resolve_hunyuan_image3_source

    calls = []

    class _Model:
        def generate_image(self, **kwargs):
            calls.append(kwargs)
            return "cot", [Image.new("RGB", (32, 32), (1, 2, 3))]

    dotted = tmp_path / "HunyuanImage-3.0-Instruct"
    dotted.mkdir()
    with pytest.raises(ValueError, match="contains"):
        resolve_hunyuan_image3_source(str(dotted))
    plain = tmp_path / "HunyuanImage-3-Instruct"
    plain.mkdir()
    assert resolve_hunyuan_image3_source(str(plain)) == str(plain)

    gen = _generator("hunyuan-image3-instruct", _Model())
    src = Image.new("RGB", (30, 20), (9, 9, 9))
    out = gen._generate_instruction_edit(
        prompt="change the background",
        image=src,
        num_inference_steps=1,
        guidance_scale=0.0,
        seed=4,
    )
    kw = calls[0]
    assert kw["prompt"] == "change the background"
    assert kw["seed"] == 4
    assert kw["diff_infer_steps"] == 50
    assert kw["infer_align_image_size"] is True
    assert kw["bot_task"] == "think_recaption"
    assert kw["use_system_prompt"] == "en_unified"
    assert len(kw["image"]) == 1
    assert kw["image"][0].endswith("source.png")
    assert out.shape == (1, 3, 20, 30)

    distil = _generator("hunyuan-image3-instruct-distil", _Model())
    distil._generate_instruction_edit(
        prompt="edit",
        image=src,
        num_inference_steps=1,
        guidance_scale=0.0,
        seed=None,
    )
    assert calls[1]["diff_infer_steps"] == 8
    assert calls[1]["seed"] is None
