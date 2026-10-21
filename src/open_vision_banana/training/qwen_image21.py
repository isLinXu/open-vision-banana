"""Source-image conditioning for Qwen-Image-2.1 and Qwen-Image-Edit training.

The pipelines' image-editing modes give the source image to the Qwen3-VL / Qwen2.5-VL
text encoder as visual tokens; the training text condition is built with the pipeline's
own ``_get_qwen_prompt_embeds`` (raw template, un-normalized last hidden state, system
tokens dropped). For Qwen-Image-2.1 this also returns the ``<|image_pad|>`` mask the
transformer uses to place the reference latents. The VAE side lives in
``model_registry.vae_encode`` and the transformer forwards in ``edit_forward``.
"""

from __future__ import annotations

from typing import Any

import torch


def check_training_config(config) -> None:
    """Reject settings the Qwen-Image-2.1 forward cannot honour, before loading the model.

    The transformer reads one token layout per batch (batch size 1), and the pipeline
    snaps images to multiples of 32 so that the Qwen3-VL visual-token grid equals the
    latent grid / 2; 256 px is the processor's minimum area per side.
    """
    if config.batch_size != 1:
        raise ValueError(
            "Qwen-Image-2.1 training uses batch_size 1; raise gradient_accumulation instead"
        )
    if config.resolution % 32 or config.resolution < 256:
        raise ValueError("Qwen-Image-2.1 needs a resolution that is a multiple of 32, at least 256")


def build_pipeline(components) -> Any:
    """A QwenImage21Pipeline around already-loaded components, for its prompt encoder."""
    from diffusers import QwenImage21Pipeline

    if components.processor is None:
        raise RuntimeError(
            "Qwen-Image-2.1 needs its Qwen3VLProcessor; load_pipeline did not keep it"
        )
    return QwenImage21Pipeline(
        scheduler=components.scheduler,
        vae=components.vae,
        text_encoder=components.text_encoder,
        processor=components.processor,
        transformer=components.unet,
    )


def build_qwen_edit_pipeline(components, pipeline_class: str) -> Any | None:
    """Build QwenImageEditPipeline / QwenImageEditPlusPipeline when processor is present."""
    if getattr(components, "processor", None) is None:
        return None
    import diffusers

    pipe_cls = getattr(diffusers, pipeline_class, None)
    if pipe_cls is None:
        return None
    try:
        return pipe_cls(
            scheduler=components.scheduler,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            processor=components.processor,
            transformer=components.unet,
        )
    except TypeError:
        return None


@torch.no_grad()
def encode_condition(pipe, prompts: list[str], images: list, device, dtype) -> dict[str, Any]:
    """Prompt + source image (PIL) -> text-encoder states with the source as visual tokens.

    One sample per call: the transformer reads the token layout from the first sample,
    so a batch would have to share prompt length as well as image size. With one
    prompt nothing is padded, so no attention mask is needed (the pipeline passes None).
    """
    if len(prompts) != 1 or len(images) != 1:
        raise ValueError(
            "Qwen-Image-2.1 training uses batch_size 1; raise gradient_accumulation instead"
        )
    embeds, _mask, image_pad_mask = pipe._get_qwen_prompt_embeds(
        prompt=prompts, image=images, device=device
    )
    return {"encoder_hidden_states": embeds.to(dtype=dtype), "image_pad_mask": image_pad_mask}


@torch.no_grad()
def encode_qwen_edit_condition(
    pipe, prompts: list[str], images: list, device, dtype
) -> dict[str, Any]:
    """Prompt + source image (PIL) -> Qwen2.5-VL hidden states + attention mask for Qwen-Image-Edit."""
    outputs = pipe._get_qwen_prompt_embeds(
        prompt=prompts, image=images, device=device, dtype=dtype
    )
    embeds = outputs[0]
    mask = outputs[1] if len(outputs) > 1 else None
    result: dict[str, Any] = {"encoder_hidden_states": embeds.to(device=device, dtype=dtype)}
    if mask is not None:
        result["attention_mask"] = mask.to(device=device)
    return result
