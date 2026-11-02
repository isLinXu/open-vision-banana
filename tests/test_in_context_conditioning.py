"""FLUX.1-Kontext-style in-context conditioning in ModelBackend (tiny random model, CPU)."""

import pytest
import torch

from open_vision_banana.model_registry import get_model_config
from open_vision_banana.training.flow_matching import FlowScheduler, pack_latents, unpack_latents

diffusers = pytest.importorskip("diffusers")
ModelBackend = pytest.importorskip("open_vision_banana.training.model_backend").ModelBackend

C, H, W = 16, 8, 8  # VAE latent channels / size; packed token dim = 4 * C
TOKENS = (H // 2) * (W // 2)


def _tiny_transformer(seed: int = 0):
    torch.manual_seed(seed)
    return diffusers.FluxTransformer2DModel(
        patch_size=1,
        in_channels=4 * C,
        out_channels=4 * C,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        pooled_projection_dim=24,
        guidance_embeds=True,
        axes_dims_rope=(4, 6, 6),
    ).eval()


def _backend(model_name: str, transformer):
    backend = object.__new__(ModelBackend)
    backend.device = torch.device("cpu")
    backend.dtype = torch.float32
    backend.model_name = model_name
    backend.model_config = get_model_config(model_name)
    backend.family = backend.model_config.family
    backend.peft_model = transformer
    backend.flow_scheduler = FlowScheduler()
    return backend


def _text(batch: int = 2):
    return {
        "encoder_hidden_states": torch.randn(batch, 5, 32),
        "pooled_projections": torch.randn(batch, 24),
        "text_ids": torch.zeros(5, 3),
    }


def test_kontext_is_registered_as_in_context_model():
    cfg = get_model_config("flux-kontext")
    assert cfg.in_context_conditioning and cfg.family == "flux"
    assert cfg.pipeline_class == "FluxKontextPipeline"
    assert not get_model_config("flux-schnell").in_context_conditioning


def test_forward_matches_manual_kontext_call_and_flips_frame():
    transformer = _tiny_transformer()
    backend = _backend("flux-kontext", transformer)
    noisy, context = torch.randn(2, C, H, W), torch.randn(2, C, H, W)
    t = torch.tensor([0.3, 0.8])
    text = _text()

    out = backend.forward(noisy, t, text, 2, context_latents=context)
    assert out.shape == (2, C, H, W)

    # Independent reference: concatenate tokens, ids with axis-0 = 1 for context,
    # native timestep sigma = 1 - t, keep target tokens, negate.
    ids = torch.zeros(TOKENS, 3)
    ids[:, 1] = torch.arange(H // 2).repeat_interleave(W // 2).float()
    ids[:, 2] = torch.arange(W // 2).repeat(H // 2).float()
    ctx_ids = ids.clone()
    ctx_ids[:, 0] = 1
    with torch.no_grad():
        ref = transformer(
            hidden_states=torch.cat([pack_latents(noisy, 2), pack_latents(context, 2)], dim=1),
            timestep=1.0 - t,
            guidance=torch.full((2,), float(backend.model_config.default_guidance_scale)),
            pooled_projections=text["pooled_projections"],
            encoder_hidden_states=text["encoder_hidden_states"],
            txt_ids=text["text_ids"],
            img_ids=torch.cat([ids, ctx_ids], dim=0),
        ).sample
    expected = unpack_latents(-ref[:, :TOKENS], H, W)
    torch.testing.assert_close(out, expected)


def test_context_changes_prediction_and_receives_gradient():
    backend = _backend("flux-kontext", _tiny_transformer(1))
    noisy = torch.randn(2, C, H, W)
    t = torch.tensor([0.5, 0.5])
    text = _text()
    ctx_a = torch.randn(2, C, H, W, requires_grad=True)
    out_a = backend.forward(noisy, t, text, 2, context_latents=ctx_a)
    out_b = backend.forward(noisy, t, text, 2, context_latents=torch.randn(2, C, H, W))
    assert not torch.allclose(out_a, out_b)
    out_a.square().mean().backward()
    assert ctx_a.grad is not None and ctx_a.grad.abs().sum() > 0


def test_context_rejected_for_text_to_image_model():
    backend = _backend("flux-schnell", _tiny_transformer())
    with pytest.raises(ValueError, match="in-context"):
        backend.forward(
            torch.randn(1, C, H, W),
            torch.tensor([0.5]),
            _text(1),
            1,
            context_latents=torch.randn(1, C, H, W),
        )


def test_noise_source_flow_is_consistent_with_native_target():
    """With img2img off, repo velocity (target - noise) == -(native noise - target)."""
    backend = _backend("flux-kontext", _tiny_transformer())
    target, source = torch.randn(3, C, H, W), torch.randn(3, C, H, W)
    fb = backend.prepare_flow_batch(target, source, 3, img2img_flow=False)
    t = fb["timesteps"].view(-1, 1, 1, 1)
    sigma = 1.0 - t
    noise = fb["noise"]
    torch.testing.assert_close(fb["noisy_latents"], (1 - sigma) * target + sigma * noise)
    torch.testing.assert_close(fb["velocity_target"], -(noise - target))
