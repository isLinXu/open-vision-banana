"""Edit training forwards on the real diffusers classes, built tiny so no weights are downloaded."""

import pytest
import torch

diffusers = pytest.importorskip("diffusers")

from open_vision_banana.model_registry import get_model_config
from open_vision_banana.training.flow_matching import FlowScheduler
from open_vision_banana.training.model_backend import ModelBackend


def _backend(name, module):
    backend = object.__new__(ModelBackend)
    backend.device = torch.device("cpu")
    backend.dtype = torch.float32
    backend.model_name = name
    backend.model_config = get_model_config(name)
    backend.family = backend.model_config.family
    backend.peft_model = module
    backend.flow_scheduler = FlowScheduler()
    return backend


def _finite(name, module, text, latent_channels=4):
    module.eval()
    backend = _backend(name, module)
    noisy = torch.randn(1, latent_channels, 8, 8)
    context = torch.randn(1, latent_channels, 8, 8)
    with torch.no_grad():
        out = backend.forward(noisy, torch.tensor([0.3]), text, 1, context_latents=context)
    assert out.shape == (1, latent_channels, 8, 8), name
    assert torch.isfinite(out).all(), name


def test_edit_forwards_execute_on_real_diffusers_modules():
    text16 = {"encoder_hidden_states": torch.randn(1, 4, 16)}
    qwen = diffusers.QwenImageTransformer2DModel(
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        axes_dims_rope=(2, 2, 4),
        joint_attention_dim=16,
        in_channels=16,
        out_channels=4,
        guidance_embeds=False,
    )
    _finite("qwen-image-edit", qwen, text16)
    _finite("qwen-image-edit-2509", qwen, text16)

    longcat = diffusers.LongCatImageTransformer2DModel(
        num_layers=1,
        num_single_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        axes_dims_rope=[2, 2, 4],
        joint_attention_dim=16,
        in_channels=16,
        pooled_projection_dim=16,
    )
    _finite("longcat-image-edit", longcat, text16)

    flux2 = diffusers.Flux2Transformer2DModel(
        num_layers=1,
        num_single_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        axes_dims_rope=(2, 2, 2, 2),
        joint_attention_dim=16,
        in_channels=4,
        guidance_embeds=False,
    )
    _finite("flux2-klein", flux2, text16)

    joy = diffusers.JoyImageEditTransformer3DModel(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=2,
        rope_dim_list=[4, 6, 6],
        text_dim=8,
        in_channels=4,
        patch_size=[1, 2, 2],
    )
    _finite("joyimage-edit", joy, {"encoder_hidden_states": torch.randn(1, 4, 8)})
    _finite("joyimage-edit-plus", joy, {"encoder_hidden_states": torch.randn(1, 4, 8)})

    pix2pix = diffusers.UNet2DConditionModel(
        sample_size=8,
        in_channels=8,
        out_channels=4,
        block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=16,
        layers_per_block=1,
        attention_head_dim=8,
    )
    _finite("instruct-pix2pix", pix2pix, text16)

    pooled_dim = 8
    time_dim = 32
    sdxl = diffusers.UNet2DConditionModel(
        sample_size=8,
        in_channels=8,
        out_channels=4,
        block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=16,
        attention_head_dim=8,
        addition_embed_type="text_time",
        addition_time_embed_dim=time_dim,
        projection_class_embeddings_input_dim=6 * time_dim + pooled_dim,
    )
    _finite(
        "sdxl-instruct-pix2pix",
        sdxl,
        {**text16, "pooled_embeds": torch.randn(1, pooled_dim)},
    )
