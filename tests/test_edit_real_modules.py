"""Edit training forwards on the real diffusers classes, built tiny so no weights are downloaded."""

import pytest
import torch

diffusers = pytest.importorskip("diffusers")

from open_vision_banana.model_registry import get_model_config  # noqa: E402
from open_vision_banana.training.flow_matching import FlowScheduler  # noqa: E402
from open_vision_banana.training.model_backend import ModelBackend  # noqa: E402


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

    flux3 = diffusers.Flux3Transformer2DModel(
        num_layers=1,
        num_single_layers=1,
        joint_attention_dim=16,
        attention_head_dim=8,
        num_attention_heads=2,
        axes_dims_rope=(2, 2, 2, 2),
    )
    _finite("flux3-klein", flux3, text16, latent_channels=128)

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


class _MossStandIn(torch.nn.Module):
    """Records MOSSImageTransformer call args and echoes a [B, C, H, W] map."""

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        return_dict=True,
        skip_layers=None,
    ):
        self.calls = {
            "shape": tuple(hidden_states.shape),
            "timestep": timestep.clone(),
            "mask": encoder_attention_mask,
            "skip_layers": skip_layers,
        }
        return (hidden_states,) if not return_dict else type(
            "Out", (), {"sample": hidden_states}
        )()


def test_moss_panel_edit_forward_plumbing():
    """MOSS is T2I-only: the panel forward must paste [source | target] on one
    canvas, feed scheduler timesteps (sigma * 1000), and keep the right half."""
    backend = _backend("moss-image", _MossStandIn())
    noisy = torch.randn(1, 128, 8, 8)
    context = torch.randn(1, 128, 8, 8)
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    text = {
        "encoder_hidden_states": torch.randn(1, 4, 16),
        "attention_mask": mask,
    }
    with torch.no_grad():
        out = backend.forward(noisy, torch.tensor([0.25]), text, 1, context_latents=context)
    calls = backend.peft_model.calls
    assert calls["shape"] == (1, 128, 8, 16)  # [source | target] along width
    assert torch.equal(calls["timestep"], torch.tensor([750.0]))  # sigma 0.75 -> t*1000
    assert calls["mask"].dtype == torch.bool and calls["mask"].eq(mask).all()
    assert out.shape == noisy.shape
    # The edit wrapper negates native v = noise - x0 into the repo's v = x0 - noise.
    assert torch.equal(out, -noisy)  # echo: right half is the noisy target


def test_flux2_vae_patchified_encode_decode_roundtrip():
    """flux2/moss/flux3 share AutoencoderKLFlux2: latents are 2x2-patchified to 128
    channels and normalised by the VAE's running BN stats."""
    from open_vision_banana.model_registry import vae_decode, vae_encode

    vae = diffusers.AutoencoderKLFlux2(
        block_out_channels=(32, 32),
        layers_per_block=1,
        latent_channels=32,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
    ).eval()
    image = torch.rand(1, 3, 64, 64) * 2 - 1
    # bf16 images against the float32 VAE (as MOSS loads it) must encode too
    assert (
        vae_encode(vae, image.to(torch.bfloat16), get_model_config("moss-image")).dtype
        == torch.float32
    )
    for name in ("flux2-klein", "moss-image", "flux3-klein"):
        cfg = get_model_config(name)
        with torch.no_grad():
            z = vae_encode(vae, image, cfg)
            out = vae_decode(vae, z, cfg)
        assert z.shape == (1, 128, 16, 16), name
        raw = vae.encode(image).latent_dist.mode()
        b, c, h, w = raw.shape
        packed = (
            raw.reshape(b, c, h // 2, 2, w // 2, 2)
            .permute(0, 1, 3, 5, 2, 4)
            .reshape(b, c * 4, h // 2, w // 2)
        )
        bn = vae.bn
        ref = (packed - bn.running_mean.view(1, -1, 1, 1)) / (
            bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
        ).sqrt()
        assert torch.allclose(z, ref, atol=1e-5), name
        assert out.shape == image.shape and torch.isfinite(out).all(), name
