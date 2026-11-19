"""MOSS-Image passes the manifest preflight checks that only look at its registry entry."""

from open_vision_banana.data.validation import _is_attention_lora_target
from open_vision_banana.model_registry import get_model_config


def test_moss_projection_names_count_as_attention_targets():
    assert _is_attention_lora_target("proj.q")
    assert _is_attention_lora_target("qkv")
    assert _is_attention_lora_target("to_q")  # the existing markers still match
    assert not _is_attention_lora_target("proj.ffn_down")
    assert not _is_attention_lora_target("proj_out")
    assert any(
        _is_attention_lora_target(t) for t in get_model_config("moss-image").lora_target_modules
    )


def test_moss_registers_the_length_its_encoder_truncates_at():
    assert get_model_config("moss-image").text_encoder_max_length == 4096


def test_patchified_encode_accepts_images_in_another_dtype():
    import pytest

    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    from open_vision_banana.model_registry import vae_encode

    vae = diffusers.AutoencoderKLFlux2(
        block_out_channels=(32, 32),
        layers_per_block=1,
        latent_channels=32,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
    ).eval()  # float32, as the MOSS pipeline loads it
    image = (torch.rand(1, 3, 64, 64) * 2 - 1).to(torch.bfloat16)
    with torch.no_grad():
        z = vae_encode(vae, image, get_model_config("moss-image"))
    assert z.shape == (1, 128, 16, 16) and torch.isfinite(z).all()
