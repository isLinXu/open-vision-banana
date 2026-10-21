"""Training forwards for instruction-editing models.

Each function builds the tensor layout the official denoiser uses, then returns
a spatial prediction of the target only. The flow forwards work in their
pipeline's native frame: ``timesteps`` is sigma (1 = pure noise) and the output is
the transformer's ``noise - x0``; ``ModelBackend._forward_edit`` converts from and
to the repo's flow time. FLUX.1 Kontext does the same conversion inside
``ModelBackend.forward``.
"""

from __future__ import annotations

from typing import Any

import torch

from ..model_registry import build_added_cond_kwargs
from .flow_matching import pack_latents, unpack_latents


def edit_latent_layout(model_config) -> str | None:
    """How this checkpoint consumes the source image during a training forward."""
    return {
        "FluxKontextPipeline": "kontext",
        "QwenImageEditPipeline": "qwen_packed",
        "QwenImageEditPlusPipeline": "qwen_packed",
        "QwenImage21Pipeline": "qwen21",
        "LongCatImageEditPipeline": "longcat_packed",
        "Flux2KleinPipeline": "flux2_packed",
        "JoyImageEditPipeline": "joy_slots",
        "JoyImageEditPlusPipeline": "joy_slots",
        "StableDiffusionInstructPix2PixPipeline": "channel_concat",
        "StableDiffusionXLInstructPix2PixPipeline": "channel_concat",
        "HunyuanImage3Instruct": "hunyuan",
    }.get(getattr(model_config, "pipeline_class", None))


def _guidance_tensor(guidance, batch: int, reference: torch.Tensor) -> torch.Tensor | None:
    if guidance is None:
        return None
    if isinstance(guidance, torch.Tensor):
        return guidance
    return torch.full((batch,), float(guidance), device=reference.device, dtype=reference.dtype)


def qwen_edit_forward(
    backbone: torch.nn.Module,
    noisy_latents: torch.Tensor,
    context_latents: torch.Tensor,
    timesteps: torch.Tensor,
    text_cond: dict[str, Any],
    guidance: float | None,
) -> torch.Tensor:
    """Qwen-Image-Edit: pack 2x2, append the source tokens, pass ``img_shapes``.

    The pipeline feeds ``scheduler_timestep / 1000``. ``timesteps`` is already
    that unit interval. The prediction is sliced back to the target tokens.
    """
    batch, _, height, width = noisy_latents.shape
    packed_h, packed_w = height // 2, width // 2
    target_tokens = packed_h * packed_w
    hidden = torch.cat(
        [pack_latents(noisy_latents, batch), pack_latents(context_latents, batch)],
        dim=1,
    )
    img_shapes = [[(1, packed_h, packed_w), (1, packed_h, packed_w)]] * batch
    text = text_cond["encoder_hidden_states"]
    mask = text_cond.get("attention_mask")
    if mask is None:
        mask = torch.ones(text.shape[:2], device=text.device, dtype=torch.long)
    predicted = backbone(
        hidden_states=hidden,
        timestep=timesteps,
        guidance=_guidance_tensor(guidance, batch, noisy_latents),
        encoder_hidden_states_mask=mask,
        encoder_hidden_states=text,
        img_shapes=img_shapes,
        return_dict=False,
    )[0]
    return unpack_latents(predicted[:, :target_tokens], height, width)


def longcat_edit_forward(
    backbone: torch.nn.Module,
    noisy_latents: torch.Tensor,
    context_latents: torch.Tensor,
    timesteps: torch.Tensor,
    text_cond: dict[str, Any],
) -> torch.Tensor:
    """LongCat-Image-Edit: packed tokens, modality 1 for the target and 2 for the source."""
    batch, _, height, width = noisy_latents.shape
    packed_h, packed_w = height // 2, width // 2
    target_tokens = packed_h * packed_w
    text = text_cond["encoder_hidden_states"]
    text_len = int(text.shape[1])
    hidden = torch.cat(
        [pack_latents(noisy_latents, batch), pack_latents(context_latents, batch)],
        dim=1,
    )
    target_ids = _longcat_image_ids(1, packed_h, packed_w, text_len, noisy_latents.device)
    source_ids = _longcat_image_ids(2, packed_h, packed_w, text_len, noisy_latents.device)
    predicted = backbone(
        hidden_states=hidden,
        timestep=timesteps,
        guidance=None,
        encoder_hidden_states=text,
        txt_ids=_longcat_text_ids(text_cond, text_len, noisy_latents.device),
        img_ids=torch.cat([target_ids, source_ids], dim=0),
        return_dict=False,
    )[0]
    return unpack_latents(predicted[:, :target_tokens], height, width)


def _longcat_text_ids(text_cond: dict[str, Any], length: int, device) -> torch.Tensor:
    provided = text_cond.get("text_ids")
    if provided is not None:
        return provided
    ids = torch.zeros(length, 3, device=device)
    ids[:, 1] = torch.arange(length, device=device)
    ids[:, 2] = torch.arange(length, device=device)
    return ids


def _longcat_image_ids(modality_id: int, height: int, width: int, text_len: int, device) -> torch.Tensor:
    ids = torch.zeros(height, width, 3, device=device)
    ids[..., 0] = modality_id
    ids[..., 1] = torch.arange(height, device=device)[:, None] + text_len
    ids[..., 2] = torch.arange(width, device=device)[None, :] + text_len
    return ids.reshape(height * width, 3)


def flux2_edit_forward(
    backbone: torch.nn.Module,
    noisy_latents: torch.Tensor,
    context_latents: torch.Tensor,
    timesteps: torch.Tensor,
    text_cond: dict[str, Any],
) -> torch.Tensor:
    """FLUX.2 Klein: flatten to tokens and append the reference with time index 10."""
    batch, channels, height, width = noisy_latents.shape
    target = _flatten_latents(noisy_latents)
    source = _flatten_latents(context_latents)
    hidden = torch.cat([target, source], dim=1)
    target_ids = _flux2_ids(height, width, time_index=0, batch=batch, device=noisy_latents.device)
    source_ids = _flux2_ids(height, width, time_index=10, batch=batch, device=noisy_latents.device)
    text = text_cond["encoder_hidden_states"]
    predicted = backbone(
        hidden_states=hidden,
        timestep=timesteps,
        guidance=None,
        encoder_hidden_states=text,
        txt_ids=_flux2_text_ids(text_cond, text),
        img_ids=torch.cat([target_ids, source_ids], dim=1),
        return_dict=False,
    )[0]
    return _unflatten_latents(predicted[:, : height * width], channels, height, width)


def _flatten_latents(latents: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = latents.shape
    return latents.flatten(2).transpose(1, 2).reshape(batch, height * width, channels)


def _unflatten_latents(tokens: torch.Tensor, channels: int, height: int, width: int) -> torch.Tensor:
    return tokens.transpose(1, 2).reshape(tokens.shape[0], channels, height, width)


def qwen21_edit_forward(
    backbone: torch.nn.Module,
    noisy_latents: torch.Tensor,
    context_latents: torch.Tensor,
    timesteps: torch.Tensor,
    text_cond: dict[str, Any],
) -> torch.Tensor:
    """Qwen-Image-2.1: [source tokens, target tokens], unpatched, in the repo frame.

    QwenImage21Pipeline feeds sigma in [0, 1] (1 = noise) and its transformer predicts
    ``noise - x0``. The repo's t runs 0 (noise) -> 1 (target) with velocity
    ``target - noise``, so this feeds ``1 - t`` and negates, as the Kontext path does.
    ``img_mask`` appends one slot per 2x2 group of target latents to the image-pad mask.
    """
    batch, channels, height, width = noisy_latents.shape
    target_tokens = height * width
    image_pad_mask = text_cond["image_pad_mask"]
    predicted = backbone(
        hidden_states=torch.cat(
            [_flatten_latents(context_latents), _flatten_latents(noisy_latents)], dim=1
        ),
        encoder_hidden_states=text_cond["encoder_hidden_states"],
        encoder_hidden_states_mask=None,
        timestep=(1.0 - timesteps).to(noisy_latents.dtype),
        img_shapes=[[(1, *context_latents.shape[-2:]), (1, height, width)]] * batch,
        img_mask=torch.cat(
            [image_pad_mask, image_pad_mask.new_ones(batch, target_tokens // 4)], dim=1
        ),
        return_dict=False,
    )[0]
    return -_unflatten_latents(predicted[:, -target_tokens:], channels, height, width)


def _flux2_text_ids(text_cond: dict[str, Any], text: torch.Tensor) -> torch.Tensor:
    provided = text_cond.get("text_ids")
    if provided is not None:
        return provided
    batch, length, _ = text.shape
    ids = torch.zeros(batch, length, 4, device=text.device, dtype=text.dtype)
    ids[..., 3] = torch.arange(length, device=text.device, dtype=text.dtype)
    return ids


def _flux2_ids(height: int, width: int, time_index: int, batch: int, device) -> torch.Tensor:
    # (T, H, W, L) with L fixed at 0. The reference image uses T = 10.
    coords = torch.cartesian_prod(
        torch.tensor([time_index], device=device),
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        torch.tensor([0], device=device),
    )
    return coords.unsqueeze(0).expand(batch, -1, -1)


def joy_edit_forward(
    backbone: torch.nn.Module,
    noisy_latents: torch.Tensor,
    context_latents: torch.Tensor,
    timesteps: torch.Tensor,
    text_cond: dict[str, Any],
) -> torch.Tensor:
    """JoyImage edit: reference slot first, noised target slot second, keep the last slot.

    The pipeline feeds the scheduler timestep, sigma * 1000.
    """
    source_slot = context_latents.unsqueeze(1).unsqueeze(3)
    noise_slot = noisy_latents.unsqueeze(1).unsqueeze(3)
    hidden = torch.cat([source_slot, noise_slot], dim=1)
    predicted = backbone(
        hidden_states=hidden,
        timestep=timesteps.float() * 1000,  # float32: bf16 steps by 4 above 512
        encoder_hidden_states=text_cond["encoder_hidden_states"],
        return_dict=False,
    )[0]
    target_slot = predicted[:, -1]
    if target_slot.dim() == 5:
        target_slot = target_slot.squeeze(2)
    return target_slot


def channel_concat_edit_forward(
    backbone: torch.nn.Module,
    noisy_latents: torch.Tensor,
    context_latents: torch.Tensor,
    timesteps: torch.Tensor,
    text_cond: dict[str, Any],
    model_config,
) -> torch.Tensor:
    """InstructPix2Pix: concatenate the clean source onto the noised target channels."""
    batch, _, height, width = noisy_latents.shape
    hidden = torch.cat([noisy_latents, context_latents], dim=1)
    pooled = text_cond.get("pooled_embeds")
    scale = int(getattr(model_config, "vae_downsample_factor", 8) or 8)
    added = build_added_cond_kwargs(
        model_config,
        pooled,
        (height * scale, width * scale),
        noisy_latents.device,
        noisy_latents.dtype,
        batch,
    )
    kwargs = {"added_cond_kwargs": added} if added is not None else {}
    if text_cond.get("attention_mask") is not None and getattr(model_config, "needs_attention_mask", False):
        kwargs["encoder_attention_mask"] = text_cond["attention_mask"]
    output = backbone(
        hidden,
        timesteps * 1000,
        text_cond["encoder_hidden_states"],
        **kwargs,
    )
    return output.sample if hasattr(output, "sample") else output


def hunyuan_edit_forward(
    model: torch.nn.Module,
    noisy_latents: torch.Tensor,
    context_latents: torch.Tensor,
    timesteps: torch.Tensor,
    text_cond: dict[str, Any],
) -> torch.Tensor:
    """One HunyuanImage-3.0-Instruct diffusion step.

    ``text_cond['hunyuan_inputs']`` is the token batch from the model's
    preprocess. The noised target is ``images``. The source VAE latent is
    ``cond_vae_images`` at timestep 0, which is how a clean condition is marked.
    The model's scheduler feeds ``sigma * 1000``.
    """
    prepared = text_cond.get("hunyuan_inputs")
    if not prepared:
        raise ValueError(
            "HunyuanImage-3.0-Instruct training forward needs text_cond['hunyuan_inputs'] "
            "(input_ids, image_mask, timesteps_index, cond_vae_image_mask, cond_timesteps_index)."
        )
    cond_timesteps = torch.zeros(
        context_latents.shape[0], device=context_latents.device, dtype=context_latents.dtype
    )
    output = model(
        input_ids=prepared["input_ids"],
        attention_mask=prepared.get("attention_mask"),
        rope_image_info=prepared.get("rope_image_info"),
        images=noisy_latents,
        image_mask=prepared["image_mask"],
        timesteps=timesteps.float() * 1000,  # float32: bf16 steps by 4 above 512
        timesteps_index=prepared["timesteps_index"],
        cond_vae_images=context_latents,
        cond_vae_image_mask=prepared["cond_vae_image_mask"],
        cond_timesteps=cond_timesteps,
        cond_timesteps_index=prepared["cond_timesteps_index"],
        mode="gen_image",
        first_step=True,
        use_cache=False,
    )
    return output.diffusion_prediction
