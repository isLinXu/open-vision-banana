"""Training-side conditioning for Qwen-Image-2.1 (``QwenImage21Pipeline``).

The training forward reproduces the pipeline's image-editing mode with one source image:

* the source goes through the Qwen3-VL text encoder as visual tokens, via the
  pipeline's own ``_get_qwen_prompt_embeds`` (raw template, un-normalized last hidden
  state, system tokens dropped), which also returns the ``<|image_pad|>`` mask;
* the source is VAE-encoded as an RGBA reference (alpha 1) with the pipeline's
  ``_encode_vae_image`` (argmax latent, ``(z - mean) / std``) and fed to the
  transformer as tokens placed before the target tokens;
* the transformer takes sigma in [0, 1] (1 = noise) and predicts ``noise - x0``.

This repo's flow time ``t`` runs from 0 (noise) to 1 (target) with velocity
``target - noise``, so the forward feeds ``1 - t`` and negates the prediction, as the
FLUX Kontext path does.
"""

from __future__ import annotations

from typing import Any

import torch

LAYOUT = "qwen21"


def build_pipeline(components) -> Any:
    """A QwenImage21Pipeline around already-loaded components, for its encode helpers."""
    from diffusers import QwenImage21Pipeline

    if components.processor is None:
        raise RuntimeError("Qwen-Image-2.1 needs its Qwen3VLProcessor; load_pipeline did not keep it")
    return QwenImage21Pipeline(
        scheduler=components.scheduler,
        vae=components.vae,
        text_encoder=components.text_encoder,
        processor=components.processor,
        transformer=components.unet,
    )


def _rgba(images: torch.Tensor) -> torch.Tensor:
    """[B, 3, H, W] in [-1, 1] -> [B, 4, 1, H, W] with an opaque alpha channel (+1)."""
    alpha = torch.ones_like(images[:, :1])
    return torch.cat([images, alpha], dim=1).unsqueeze(2)


@torch.no_grad()
def vae_encode(pipe, images: torch.Tensor) -> torch.Tensor:
    """[B, 3, H, W] in [-1, 1] -> normalized latents [B, 64, H/16, W/16]."""
    vae = pipe.vae
    latents = pipe._encode_vae_image(_rgba(images).to(device=vae.device, dtype=vae.dtype), generator=None)
    return latents.squeeze(2)


def vae_decode(pipe, latents: torch.Tensor) -> torch.Tensor:
    """Normalized latents [B, 64, h, w] -> RGB images in [-1, 1] (alpha dropped)."""
    vae = pipe.vae
    channels = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean).view(1, channels, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, channels, 1, 1, 1)
    z = latents.unsqueeze(2).to(vae.dtype)
    z = z * std.to(z.device, z.dtype) + mean.to(z.device, z.dtype)
    return vae.decode(z, return_dict=False)[0][:, :3, 0]


@torch.no_grad()
def encode_condition(pipe, prompts: list[str], images: torch.Tensor, device, dtype) -> dict[str, Any]:
    """Prompt + source image -> text-encoder states with the source as visual tokens.

    ``images`` is [1, 3, H, W] in [-1, 1]. One sample per call: the transformer reads
    the token layout from the first sample, so a batch would have to share prompt
    length as well as image size.
    """
    if len(prompts) != 1 or images.shape[0] != 1:
        raise ValueError("Qwen-Image-2.1 training uses batch_size 1; raise gradient_accumulation instead")
    from .model_backend import _tensors_to_pils

    embeds, mask, image_pad_mask = pipe._get_qwen_prompt_embeds(
        prompt=list(prompts), image=_tensors_to_pils(images), device=device
    )
    return {
        "encoder_hidden_states": embeds.to(device=device, dtype=dtype),
        # All ones for a single prompt; the pipeline passes None in that case.
        "attention_mask": None if mask.all() else mask,
        "image_pad_mask": image_pad_mask,
    }


def edit_forward(
    backbone: torch.nn.Module,
    noisy_latents: torch.Tensor,
    context_latents: torch.Tensor,
    timesteps: torch.Tensor,
    text_cond: dict[str, Any],
) -> torch.Tensor:
    """Target prediction in this repo's frame from [source tokens, noisy target tokens]."""
    batch, channels, height, width = noisy_latents.shape
    target_tokens = height * width
    ctx_h, ctx_w = context_latents.shape[-2:]

    def tokens(latents):
        return latents.reshape(batch, channels, -1).transpose(1, 2)

    hidden = torch.cat([tokens(context_latents), tokens(noisy_latents)], dim=1)
    image_pad_mask = text_cond["image_pad_mask"]
    img_mask = torch.cat(
        [image_pad_mask, image_pad_mask.new_ones(image_pad_mask.shape[0], target_tokens // 4)], dim=1
    )
    sigma = (1.0 - timesteps).to(noisy_latents.dtype)
    predicted = backbone(
        hidden_states=hidden,
        encoder_hidden_states=text_cond["encoder_hidden_states"],
        encoder_hidden_states_mask=text_cond.get("attention_mask"),
        timestep=sigma,
        img_shapes=[[(1, ctx_h, ctx_w), (1, height, width)]] * batch,
        img_mask=img_mask,
        return_dict=False,
    )[0]
    target = predicted[:, -target_tokens:].transpose(1, 2).reshape(batch, channels, height, width)
    return -target
