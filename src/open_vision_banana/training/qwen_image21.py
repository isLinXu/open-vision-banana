"""Source-image conditioning for Qwen-Image-2.1 (``QwenImage21Pipeline``) training.

The pipeline's image-editing mode gives the source image to the Qwen3-VL text encoder
as visual tokens; the training text condition is built with the pipeline's own
``_get_qwen_prompt_embeds`` (raw template, un-normalized last hidden state, system
tokens dropped), which also returns the ``<|image_pad|>`` mask the transformer uses to
place the reference latents. The VAE side lives in ``model_registry.vae_encode`` and the
transformer forward in ``edit_forward.qwen21_edit_forward``.
"""

from __future__ import annotations

from typing import Any

import torch


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
