"""SANA source-to-target flow-matching img2img.

SANA LoRAs in this repo are trained with the custom Euler flow used by
``scripts/train_sana_512.py``. The pinned Diffusers build has no
``SanaImg2ImgPipeline``, and ``SanaPipeline`` rejects ``image=``. Vision
task inference must therefore integrate the same ODE as training:

    z_{t+dt} = scheduler.step(v_θ(z_t, t, c), t, z_t)

starting from VAE latents of the input image.
"""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from ..model_registry import ModelConfig, PipelineComponents
from ..training.sana_flow import (
    denormalize_vae_latents,
    native_sana_timestep,
    normalize_vae_latents,
)


def images_to_sana_source(
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
        raise TypeError(f"Unsupported SANA image type: {type(image)}")

    tensor = image.float()
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 4 or tensor.shape[1] != 3:
        raise ValueError(f"Expected [B, 3, H, W] or [3, H, W], got {tuple(image.shape)}")
    if float(tensor.min()) >= 0.0:
        tensor = tensor * 2.0 - 1.0
    return tensor.to(device=device)


def resolve_sana_img2img_settings(
    *,
    num_inference_steps: int,
    guidance_scale: float,
    default_steps: int = 20,
) -> tuple[int, float]:
    """Map FLUX-schnell defaults onto the SANA training contract.

    ``BaseGenerator.generate`` historically defaults to 1 step and CFG 0.0
    (schnell). For SANA, CFG 0.0 would decode the unconditional branch, and
    1 step is not the trained schedule.
    """
    steps = default_steps if num_inference_steps <= 1 else num_inference_steps
    cfg = 1.0 if guidance_scale <= 0.0 else guidance_scale
    return steps, cfg


def pipeline_components_from_sana(
    pipe: Any,
    config: ModelConfig,
) -> PipelineComponents:
    """Build ``PipelineComponents`` from a loaded ``SanaPipeline``."""
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        raise AttributeError("SANA pipeline is missing transformer")
    return PipelineComponents(
        unet=transformer,
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        text_encoder_2=getattr(pipe, "text_encoder_2", None),
        tokenizer=pipe.tokenizer,
        config=config,
        scheduler=getattr(pipe, "scheduler", None),
    )


def encode_text_sana(
    components: PipelineComponents,
    prompts: list[str],
    device: torch.device,
    dtype: torch.dtype,
    max_length: int = 300,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode text prompts using Gemma-2-2B-IT, matching training."""
    tokenizer = components.tokenizer
    text_encoder = components.text_encoder
    if tokenizer is None or text_encoder is None:
        raise RuntimeError("SANA pipeline did not provide tokenizer/text_encoder")

    tokenized = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)

    with torch.no_grad():
        outputs = text_encoder(input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "last_hidden_state"):
            encoder_hidden_states = outputs.last_hidden_state
        else:
            encoder_hidden_states = outputs[0]

    return encoder_hidden_states.to(dtype), attention_mask.to(dtype)


def vae_encode_sana(
    vae: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode images to latents using DC-AE.

    Input: ``[B, 3, H, W]`` in ``[-1, 1]``.
    """
    del device
    with torch.no_grad():
        enc_out = vae.encode(images.to(dtype=vae.dtype))
        if hasattr(enc_out, "latent_dist"):
            latents = enc_out.latent_dist.mode()
        elif hasattr(enc_out, "latent"):
            latents = enc_out.latent
        elif hasattr(enc_out, "sample"):
            latents = enc_out.sample
        elif isinstance(enc_out, torch.Tensor):
            latents = enc_out
        else:
            for attr in ("latents", "encoded", "z"):
                if hasattr(enc_out, attr):
                    latents = getattr(enc_out, attr)
                    break
            else:
                raise TypeError(
                    f"VAE encode returned {type(enc_out).__name__} with no known "
                    f"latent attribute. Available: {dir(enc_out)}"
                )

    return normalize_vae_latents(latents.to(dtype), vae)


def vae_decode_sana(
    vae: torch.nn.Module,
    latents: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Decode latents to images using DC-AE.

    Output: ``[B, 3, H, W]`` in ``[-1, 1]``.
    """
    del device
    with torch.no_grad():
        dec_out = vae.decode(denormalize_vae_latents(latents, vae).to(dtype=vae.dtype))
        if hasattr(dec_out, "sample"):
            images = dec_out.sample
        elif isinstance(dec_out, torch.Tensor):
            images = dec_out
        else:
            for attr in ("image", "images", "decoded", "x"):
                if hasattr(dec_out, attr):
                    images = getattr(dec_out, attr)
                    break
            else:
                raise TypeError(
                    f"VAE decode returned {type(dec_out).__name__} with no known "
                    f"sample attribute. Available: {dir(dec_out)}"
                )

    return images


@torch.no_grad()
def flow_matching_inference(
    peft_model,
    vae,
    components: PipelineComponents,
    input_images: torch.Tensor,
    prompts: list[str],
    device: torch.device,
    model_dtype: torch.dtype,
    num_steps: int = 20,
    use_img2img: bool = True,
    max_length: int = 300,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """Run flow-matching inference to generate target images from inputs.

    For img2img flow: starts from input latents, integrates ODE toward target.
    For noise->target flow: starts from pure noise, integrates toward target.

    Returns: ``[B, 3, H, W]`` in ``[0, 1]``.
    """
    prompt_embeds, attention_mask = encode_text_sana(
        components, prompts, device, model_dtype, max_length
    )
    uncond_embeds = None
    uncond_attention_mask = None
    if guidance_scale != 1.0:
        uncond_embeds, uncond_attention_mask = encode_text_sana(
            components, [""] * len(prompts), device, model_dtype, max_length
        )

    input_latents = vae_encode_sana(vae, input_images, device, model_dtype)
    current_latents = input_latents.clone() if use_img2img else torch.randn_like(input_latents)
    batch_size = current_latents.shape[0]

    # The public evaluator accepts one task prompt for a whole image batch.
    # Gemma returns a single prompt embedding in that case, while the Sana
    # attention processor requires both hidden states and masks to have the
    # latent batch dimension.  Broadcast only the unambiguous single-prompt
    # case and fail loudly for any other mismatch.
    if prompt_embeds.shape[0] == 1 and batch_size > 1:
        prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)
        attention_mask = attention_mask.expand(batch_size, -1)
    elif prompt_embeds.shape[0] != batch_size:
        raise ValueError(
            "SANA prompt batch does not match image batch: "
            f"prompts={prompt_embeds.shape[0]}, images={batch_size}"
        )
    if uncond_embeds is not None:
        if uncond_embeds.shape[0] == 1 and batch_size > 1:
            uncond_embeds = uncond_embeds.expand(batch_size, -1, -1)
            uncond_attention_mask = uncond_attention_mask.expand(batch_size, -1)
        elif uncond_embeds.shape[0] != batch_size:
            raise ValueError(
                "SANA unconditional prompt batch does not match image batch: "
                f"prompts={uncond_embeds.shape[0]}, images={batch_size}"
            )

    amp_dtype = torch.bfloat16 if device.type == "mps" else model_dtype
    use_autocast = device.type == "mps" or (device.type == "cuda" and model_dtype != torch.float32)

    scheduler = components.scheduler
    if scheduler is None:
        raise RuntimeError("SANA pipeline did not provide its native flow scheduler")
    scheduler.set_timesteps(num_steps, device=device)

    for timestep in scheduler.timesteps:
        t = timestep.expand(batch_size)

        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_autocast,
        ):
            predicted_velocity = peft_model(
                current_latents,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=attention_mask,
                timestep=native_sana_timestep(t, batch_size, peft_model),
                return_dict=False,
            )[0]
            if uncond_embeds is not None:
                unconditional_velocity = peft_model(
                    current_latents,
                    encoder_hidden_states=uncond_embeds,
                    encoder_attention_mask=uncond_attention_mask,
                    timestep=native_sana_timestep(t, batch_size, peft_model),
                    return_dict=False,
                )[0]
                predicted_velocity = unconditional_velocity + guidance_scale * (
                    predicted_velocity - unconditional_velocity
                )

        current_latents = scheduler.step(
            predicted_velocity.float(), timestep, current_latents, return_dict=False
        )[0]

    predicted_images = vae_decode_sana(vae, current_latents, device, model_dtype)
    predicted_images = (predicted_images.float() + 1.0) / 2.0
    return predicted_images.clamp(0.0, 1.0)
