#!/usr/bin/env python3
"""Convert Sana-1600M .pth checkpoint to diffusers format.

Uses Sana-600M_512px_diffusers as template for VAE, text_encoder, tokenizer, scheduler.
Only converts the transformer weights from .pth to diffusers format.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

# Paths
PTH_PATH = Path("/media/linxu/thinkHub/ckpts/Efficient-Large-Model/Sana_1600M_512px/checkpoints/Sana_1600M_512px.pth")
TEMPLATE_PATH = Path("/media/linxu/thinkHub/ckpts/Efficient-Large-Model/Sana_600M_512px_diffusers")
OUTPUT_PATH = Path("/media/linxu/thinkHub/ckpts/Efficient-Large-Model/Sana_1600M_512px_diffusers")


def convert_transformer_config():
    """Create transformer config for Sana-1600M."""
    return {
        "_class_name": "SanaTransformer2DModel",
        "_diffusers_version": "0.32.0",
        "attention_head_dim": 70,
        "caption_channels": 2304,
        "cross_attention_dim": 2240,
        "dropout": 0.0,
        "in_channels": 32,
        "mlp_ratio": 2.5,
        "num_attention_heads": 32,
        "num_cross_attention_heads": 32,
        "num_layers": 20,
        "out_channels": 32,
        "patch_size": 1,
        "sample_size": 32,
        "upcast_attention": False,
    }


def convert_model_index():
    """Create model_index.json for diffusers pipeline."""
    return {
        "_class_name": "SanaPipeline",
        "_diffusers_version": "0.32.0",
        "_name_or_path": "Efficient-Large-Model/Sana_1600M_512px",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "Gemma2Model"],
        "tokenizer": ["transformers", "GemmaTokenizer"],
        "transformer": ["diffusers", "SanaTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderDC"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", type=Path, default=PTH_PATH)
    parser.add_argument("--template", type=Path, default=TEMPLATE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    print(f"Loading {args.pth}...")
    ckpt = torch.load(args.pth, map_location="cpu")
    state_dict = ckpt["state_dict"]
    print(f"  Loaded {len(state_dict)} keys")

    # Create output directories
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "transformer").mkdir(exist_ok=True)
    (args.output / "vae").mkdir(exist_ok=True)
    (args.output / "text_encoder").mkdir(exist_ok=True)
    (args.output / "tokenizer").mkdir(exist_ok=True)
    (args.output / "scheduler").mkdir(exist_ok=True)

    # Convert transformer weights
    print("Converting transformer weights...")
    # Map .pth keys to diffusers format
    # .pth: blocks.0.attn.qkv.weight -> transformer: blocks.0.attn.qkv.weight (same)
    # .pth: pos_embed -> transformer: pos_embed (same)
    transformer_sd = {}
    for k, v in state_dict.items():
        # Remove 'module.' prefix if present
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        transformer_sd[new_key] = v

    # Save transformer as safetensors
    save_file(transformer_sd, args.output / "transformer" / "diffusion_pytorch_model.safetensors")
    print(f"  Saved {len(transformer_sd)} keys to transformer/")

    # Copy VAE, text_encoder, tokenizer, scheduler from template
    print("Copying VAE, text_encoder, tokenizer, scheduler from template...")
    for subdir in ["vae", "text_encoder", "tokenizer", "scheduler"]:
        src = args.template / subdir
        dst = args.output / subdir
        if src.exists():
            # Copy all files except .bin (use .safetensors only)
            for f in src.glob("*"):
                if f.is_file() and not f.name.endswith(".bin"):
                    shutil.copy2(f, dst / f.name)
            print(f"  Copied {subdir}/")

    # Write configs
    print("Writing configs...")
    with open(args.output / "transformer" / "config.json", "w") as f:
        json.dump(convert_transformer_config(), f, indent=2)

    with open(args.output / "model_index.json", "w") as f:
        json.dump(convert_model_index(), f, indent=2)

    # Copy other files
    for f in ["LICENSE", "README.md", "configuration.json"]:
        src = args.template / f
        if src.exists():
            shutil.copy2(src, args.output / f)

    print(f"\nDone! Converted model saved to {args.output}")
    print("\nDirectory structure:")
    for item in sorted(args.output.rglob("*")):
        if item.is_file():
            rel = item.relative_to(args.output)
            size = item.stat().st_size / (1024**2)
            print(f"  {rel}: {size:.1f} MB")


if __name__ == "__main__":
    main()
