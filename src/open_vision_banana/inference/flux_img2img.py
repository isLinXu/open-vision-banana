"""FLUX source-to-target flow-matching img2img.

FLUX LoRAs in this repo are trained with the custom Euler flow in
``InstructionTuner._training_step_flow``:

    z_t = (1 - t) * z_input + t * z_target
    v*  = z_target - z_input

``FluxImg2ImgPipeline`` instead noises the input on schnell's distilled
schedule and denoises toward a generated image. That path does not keep
spatial correspondence with the source, so vision-task panels look like
unrelated samples. Inference must integrate the same ODE as training:

    z ← z_input
    z ← z + dt * v_θ(z, t, c)
"""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from ..model_registry import ModelConfig, vae_decode, vae_encode


def images_to_flux_source(
    image: torch.Tensor | Image.Image | list[Image.Image],
    device: torch.device | str,
) -> torch.Tensor:
    """Normalize generate() image inputs to ``[B, 3, H, W]`` in ``[-1, 1]``."""
    if isinstance(image, Image.Image):
        image = [image]
    if isinstance(image, list):
        import numpy as np

        tensors = []
        for pil in image:
            arr = np.array(pil.convert("RGB"), dtype="float32") / 255.0
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
        tensor = torch.stack(tensors, dim=0)
        return (tensor * 2.0 - 1.0).to(device=device)

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Unsupported FLUX image type: {type(image)}")

    tensor = image.float()
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 4 or tensor.shape[1] != 3:
        raise ValueError(f"Expected [B, 3, H, W] or [3, H, W], got {tuple(image.shape)}")
    if float(tensor.min()) >= 0.0:
        tensor = tensor * 2.0 - 1.0
    return tensor.to(device=device)


def pack_flux_latents(
    latents: torch.Tensor,
    batch_size: int,
    num_channels: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Pack VAE latents for the FLUX transformer.

    ``[B, C, H, W]`` → ``[B, (H/2)*(W/2), C*4]``
    """
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)


def unpack_flux_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Unpack FLUX transformer output back to VAE latent layout.

    ``[B, (H/2)*(W/2), C*4]`` → ``[B, C, H, W]``
    """
    batch_size, _num_patches, channels = latents.shape
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, channels // 4, height, width)


def prepare_flux_latent_image_ids(
    height: int,
    width: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build FLUX packed-grid position IDs ``[H*W, 3]``."""
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
    return latent_image_ids.reshape(height * width, 3).to(device=device, dtype=dtype)


def flux_guidance_tensor(
    transformer: Any,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    guidance: float | None = None,
) -> torch.Tensor | None:
    """Guidance embedding for distilled FLUX.1-dev / Kontext transformers.

    Schnell has ``guidance_embeds=False`` and must not receive this tensor.
    Training (`train_flux.py`, ``ModelBackend``) feeds 50.0, which the
    transformer then multiplies by 1000.
    """
    if not bool(getattr(getattr(transformer, "config", None), "guidance_embeds", False)):
        return None
    scale = 50.0 if guidance is None or guidance <= 0 else float(guidance)
    return torch.full((batch_size,), scale, device=device, dtype=dtype)


def encode_text_flux(pipe: Any, prompts: list[str], device: torch.device) -> dict[str, torch.Tensor | None]:
    """Encode prompts with FluxPipeline.encode_prompt, matching training."""
    encoders = [getattr(pipe, "text_encoder", None), getattr(pipe, "text_encoder_2", None)]
    original_dtypes = []
    for encoder in encoders:
        quantized = bool(
            getattr(encoder, "is_loaded_in_8bit", False)
            or getattr(encoder, "is_loaded_in_4bit", False)
        )
        if encoder is not None and not quantized:
            original_dtypes.append(next(encoder.parameters()).dtype)
            encoder.to(torch.float32)
        else:
            original_dtypes.append(None)
    try:
        text_outputs = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=prompts,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
    finally:
        for encoder, dtype in zip(encoders, original_dtypes, strict=False):
            if encoder is not None and dtype is not None:
                encoder.to(dtype)

    prompt_embeds = text_outputs[0]
    pooled_prompt_embeds = text_outputs[1] if len(text_outputs) > 1 else None
    text_ids = text_outputs[2] if len(text_outputs) > 2 else None
    return {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "text_ids": text_ids,
    }


@torch.no_grad()
def flow_matching_inference(
    pipe: Any,
    config: ModelConfig,
    input_images: torch.Tensor,
    prompts: list[str],
    device: torch.device,
    model_dtype: torch.dtype,
    num_steps: int = 4,
    guidance: float | None = None,
) -> torch.Tensor:
    """Euler-integrate from input latents toward the prompted vision target.

    Returns ``[B, 3, H, W]`` in ``[0, 1]``.
    """
    vae = pipe.vae
    transformer = pipe.transformer
    vae_dtype = model_dtype
    for parameter in vae.parameters():
        if parameter.device.type != "meta":
            vae_dtype = parameter.dtype
            break
    batch_size = input_images.shape[0]

    input_latents = vae_encode(vae, input_images.to(dtype=vae_dtype), config)
    z = input_latents.to(dtype=model_dtype)
    text = encode_text_flux(pipe, prompts, device)
    prompt_embeds = text["prompt_embeds"].to(dtype=model_dtype)
    pooled = text["pooled_prompt_embeds"]
    if pooled is None:
        pooled = torch.zeros(batch_size, 768, device=device, dtype=model_dtype)
    else:
        pooled = pooled.to(dtype=model_dtype)
    text_ids = text["text_ids"]
    if text_ids is not None:
        text_ids = text_ids.to(dtype=model_dtype)

    dt = 1.0 / max(int(num_steps), 1)
    for step in range(max(int(num_steps), 1)):
        t_tensor = torch.full((batch_size,), step * dt, device=device, dtype=model_dtype)
        latent_h, latent_w = z.shape[-2], z.shape[-1]
        latent_channels = z.shape[1]
        if latent_channels == 16:
            hidden = pack_flux_latents(z, batch_size, latent_channels, latent_h, latent_w)
            img_ids = prepare_flux_latent_image_ids(
                latent_h // 2,
                latent_w // 2,
                device,
                z.dtype,
            )
        else:
            hidden = z
            img_ids = None

        call = {
            "hidden_states": hidden.to(dtype=model_dtype),
            "timestep": t_tensor,
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled,
            "txt_ids": text_ids,
            "img_ids": img_ids,
            "return_dict": False,
        }
        guidance_tensor = flux_guidance_tensor(
            transformer, batch_size, device, model_dtype, guidance
        )
        if guidance_tensor is not None:
            call["guidance"] = guidance_tensor
        output = transformer(**call)
        velocity = output[0] if isinstance(output, (tuple, list)) else output.sample
        if latent_channels == 16:
            velocity = unpack_flux_latents(velocity, latent_h, latent_w)
        z = z + dt * velocity.to(dtype=z.dtype)

    # Sequential CPU offload parks VAE weights on ``meta``. Decode on the
    # inference device so accelerate can move the module onto it.
    decoded = vae_decode(vae, z.to(device=device, dtype=vae_dtype), config)
    return ((decoded.float() + 1.0) / 2.0).clamp(0.0, 1.0)
