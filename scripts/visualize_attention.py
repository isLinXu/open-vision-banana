#!/usr/bin/env python3
"""Attention visualization for Vision Banana / FLUX.1 transformer.

Extracts and visualizes cross-attention maps from the transformer during
inference, showing which image regions the model attends to for each
text token.  Useful for debugging prompt adherence and spatial reasoning.

Usage:
    # Visualize attention for a single image + prompt
    python scripts/visualize_attention.py \
        --image input.jpg \
        --prompt "depth estimation" \
        --output output/attn_vis/

    # Visualize specific layers (default: all)
    python scripts/visualize_attention.py \
        --image input.jpg \
        --prompt "segmentation of a dog" \
        --layers 18,19,20,21 \
        --output output/attn_vis/

    # Generate a grid of attention maps for a list of tokens
    python scripts/visualize_attention.py \
        --image input.jpg \
        --prompt "the man in pink shirt" \
        --tokens "man,shirt,pink" \
        --output output/attn_vis/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AttentionExtractor:
    """Hook-based attention extractor for diffusers transformer."""

    def __init__(self, model: torch.nn.Module, layer_names: list[str] | None = None) -> None:
        self.attention_maps: dict[str, list[torch.Tensor]] = {}
        self.hooks: list[Any] = []
        self._register_hooks(model, layer_names)

    def _register_hooks(self, model: torch.nn.Module, layer_names: list[str] | None) -> None:
        """Register forward hooks on transformer attention layers."""
        for name, module in model.named_modules():
            if layer_names is not None and name not in layer_names:
                continue
            # Hook on modules that perform attention (attn1 = self-attn, attn2 = cross-attn)
            if hasattr(module, "to_q") and hasattr(module, "to_k"):
                hook = module.register_forward_hook(self._attention_hook(name))
                self.hooks.append(hook)

    def _attention_hook(self, name: str):
        def hook(module, input, output):
            # Extract attention scores from the module's forward pass
            # This is a generic hook that works with diffusers attention processors
            if hasattr(module, "attention_scores"):
                attn = module.attention_scores
                if attn is not None:
                    self.attention_maps.setdefault(name, []).append(attn.detach().cpu())
        return hook

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_maps(self, layer: str | None = None) -> dict[str, list[torch.Tensor]]:
        if layer is not None:
            return {layer: self.attention_maps.get(layer, [])}
        return self.attention_maps

    def aggregate_maps(self, layer: str | None = None) -> torch.Tensor | None:
        """Average attention maps across timesteps for a given layer or all layers."""
        maps_dict = self.get_maps(layer)
        if not maps_dict:
            return None
        all_maps = []
        for layer_maps in maps_dict.values():
            if layer_maps:
                all_maps.append(torch.stack(layer_maps).mean(dim=0))
        if not all_maps:
            return None
        return torch.stack(all_maps).mean(dim=0)


def visualize_attention_grid(
    image: np.ndarray,
    attention_maps: torch.Tensor,
    token_indices: list[int],
    token_labels: list[str],
    output_path: Path,
    alpha: float = 0.6,
) -> None:
    """Create a grid visualization of attention maps overlaid on the image."""
    num_tokens = len(token_indices)
    cols = min(4, num_tokens)
    rows = (num_tokens + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    H, W = image.shape[:2]
    attn_avg = attention_maps.mean(dim=0)  # [num_heads, H, W] or [H, W]

    for idx, token_idx in enumerate(token_indices):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]

        if attn_avg.dim() == 3:
            attn_map = attn_avg[:, token_idx, :].mean(dim=0)
        else:
            attn_map = attn_avg[token_idx]

        # Reshape to spatial grid (assuming square patches)
        spatial_size = int(attn_map.shape[-1] ** 0.5)
        if spatial_size * spatial_size == attn_map.shape[-1]:
            attn_map = attn_map.reshape(spatial_size, spatial_size)
        else:
            # Fallback: interpolate to a reasonable size
            attn_map = F.interpolate(
                attn_map.view(1, 1, -1, 1),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

        attn_map = attn_map.numpy()
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        ax.imshow(image)
        ax.imshow(
            plt.cm.hot(attn_map),
            alpha=alpha * attn_map,
            extent=(0, W, H, 0),
        )
        label = token_labels[idx] if idx < len(token_labels) else f"token_{token_idx}"
        ax.set_title(f'"{label}"')
        ax.axis("off")

    # Hide unused subplots
    for idx in range(num_tokens, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved attention visualization to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize FLUX.1 transformer attention maps")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--output", type=str, required=True, help="Output directory for visualizations")
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated layer indices to visualize")
    parser.add_argument("--tokens", type=str, default=None, help="Comma-separated token words to highlight")
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-schnell", help="Model ID")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.6, help="Attention overlay opacity")
    parser.add_argument("--num-inference-steps", type=int, default=4, help="FLUX schnell steps")
    parser.add_argument("--dry-run", action="store_true", help="Run without a real model (synthetic attention)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image)

    if args.dry_run:
        logger.info("Dry-run mode: generating synthetic attention maps")
        # Simulate attention maps for demonstration without loading a model
        H, W = image_np.shape[:2]
        spatial = 16 * 16  # 16x16 patch grid
        num_tokens = 8
        synthetic_attn = torch.randn(1, spatial, num_tokens).softmax(dim=1)
        token_labels = (args.tokens or "depth,estimation,object,scene,region,pixel,token,attn").split(",")
        visualize_attention_grid(
            image_np,
            synthetic_attn,
            list(range(num_tokens)),
            token_labels,
            output_dir / "attention_grid.png",
            alpha=args.alpha,
        )
        print(f"✅ Dry-run attention visualization saved to {output_dir / 'attention_grid.png'}")
        return 0

    # Real model loading (requires diffusers + FLUX weights)
    try:
        from diffusers import FluxPipeline
    except ImportError:
        logger.error("diffusers not installed. Run: pip install -e '.[inference]'")
        return 1

    logger.info("Loading %s...", args.model)
    pipe = FluxPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    pipe = pipe.to(device)

    # Extract attention
    extractor = AttentionExtractor(pipe.transformer)

    logger.info("Running inference with prompt: %r", args.prompt)
    result = pipe(
        prompt=args.prompt,
        image=image,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=0.0,  # FLUX schnell doesn't use CFG
    ).images[0]

    # Aggregate and visualize
    aggregated = extractor.aggregate_maps()
    if aggregated is None:
        logger.warning("No attention maps extracted. Ensure diffusers version supports attention hooks.")
        extractor.remove_hooks()
        return 1

    extractor.remove_hooks()

    token_labels = []
    if args.tokens:
        token_labels = [t.strip() for t in args.tokens.split(",")]

    visualize_attention_grid(
        image_np,
        aggregated,
        list(range(min(aggregated.shape[-1], 8))),
        token_labels,
        output_dir / "attention_grid.png",
        alpha=args.alpha,
    )

    # Save the generated image alongside
    result.save(output_dir / "generated.png")
    logger.info("Done. Output saved to %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
