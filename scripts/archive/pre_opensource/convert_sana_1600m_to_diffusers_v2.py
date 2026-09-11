#!/usr/bin/env python3
"""Convert Sana-1600M .pth checkpoint to diffusers format (corrected key mapping).

Sana .pth keys -> diffusers keys:
  blocks.N.attn.qkv.weight -> transformer_blocks.N.attn1.{to_q,to_k,to_v}.weight (split)
  blocks.N.attn.proj -> transformer_blocks.N.attn1.to_out.0
  blocks.N.cross_attn.q_linear -> transformer_blocks.N.attn2.to_q
  blocks.N.cross_attn.kv_linear -> transformer_blocks.N.attn2.{to_k,to_v} (split)
  blocks.N.cross_attn.proj -> transformer_blocks.N.attn2.to_out.0
  blocks.N.mlp.inverted_conv -> transformer_blocks.N.ff.conv_inverted
  blocks.N.mlp.depth_conv -> transformer_blocks.N.ff.conv_depth
  blocks.N.mlp.point_conv -> transformer_blocks.N.ff.conv_point
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

PTH_PATH = Path("/media/linxu/thinkHub/ckpts/Efficient-Large-Model/Sana_1600M_512px/checkpoints/Sana_1600M_512px.pth")
TEMPLATE_PATH = Path("/media/linxu/thinkHub/ckpts/Efficient-Large-Model/Sana_600M_512px_diffusers")
OUTPUT_PATH = Path("/media/linxu/thinkHub/ckpts/Efficient-Large-Model/Sana_1600M_512px_diffusers_v2")


def convert_key_mapping():
    """Return mapping from .pth keys to diffusers keys."""
    mapping = {}
    num_blocks = 20

    for i in range(num_blocks):
        pfx = f"blocks.{i}"
        dfx = f"transformer_blocks.{i}"
        # Self-attention: qkv -> to_q, to_k, to_v (need split)
        mapping[f"{pfx}.attn.qkv.weight"] = f"{dfx}.attn1.qkv.weight"  # special handling
        mapping[f"{pfx}.attn.qkv.bias"] = f"{dfx}.attn1.qkv.bias"  # special handling
        mapping[f"{pfx}.attn.proj.weight"] = f"{dfx}.attn1.to_out.0.weight"
        mapping[f"{pfx}.attn.proj.bias"] = f"{dfx}.attn1.to_out.0.bias"
        # Cross-attention: q_linear -> to_q, kv_linear -> to_k, to_v (need split)
        mapping[f"{pfx}.cross_attn.q_linear.weight"] = f"{dfx}.attn2.to_q.weight"
        mapping[f"{pfx}.cross_attn.q_linear.bias"] = f"{dfx}.attn2.to_q.bias"
        mapping[f"{pfx}.cross_attn.kv_linear.weight"] = f"{dfx}.attn2.kv.weight"  # special handling
        mapping[f"{pfx}.cross_attn.kv_linear.bias"] = f"{dfx}.attn2.kv.bias"  # special handling
        mapping[f"{pfx}.cross_attn.proj.weight"] = f"{dfx}.attn2.to_out.0.weight"
        mapping[f"{pfx}.cross_attn.proj.bias"] = f"{dfx}.attn2.to_out.0.bias"
        # MLP
        mapping[f"{pfx}.mlp.inverted_conv.conv.weight"] = f"{dfx}.ff.conv_inverted.weight"
        mapping[f"{pfx}.mlp.inverted_conv.conv.bias"] = f"{dfx}.ff.conv_inverted.bias"
        mapping[f"{pfx}.mlp.depth_conv.conv.weight"] = f"{dfx}.ff.conv_depth.weight"
        mapping[f"{pfx}.mlp.depth_conv.conv.bias"] = f"{dfx}.ff.conv_depth.bias"
        mapping[f"{pfx}.mlp.point_conv.conv.weight"] = f"{dfx}.ff.conv_point.weight"
        # Scale-shift table
        mapping[f"{pfx}.scale_shift_table"] = f"{dfx}.scale_shift_table"

    # Other mappings
    mapping["pos_embed"] = "pos_embed"  # may not be used in diffusers
    mapping["x_embedder.proj.weight"] = "patch_embed.proj.weight"
    mapping["x_embedder.proj.bias"] = "patch_embed.proj.bias"
    mapping["t_embedder.mlp.0.weight"] = "time_embed.emb.timestep_embedder.linear_1.weight"
    mapping["t_embedder.mlp.0.bias"] = "time_embed.emb.timestep_embedder.linear_1.bias"
    mapping["t_embedder.mlp.2.weight"] = "time_embed.emb.timestep_embedder.linear_2.weight"
    mapping["t_embedder.mlp.2.bias"] = "time_embed.emb.timestep_embedder.linear_2.bias"
    mapping["t_block.1.weight"] = "time_embed.linear.weight"
    mapping["t_block.1.bias"] = "time_embed.linear.bias"
    mapping["y_embedder.y_embedding"] = "y_embedder.y_embedding"  # may not be used
    mapping["y_embedder.y_proj.fc1.weight"] = "caption_projection.linear_1.weight"
    mapping["y_embedder.y_proj.fc1.bias"] = "caption_projection.linear_1.bias"
    mapping["y_embedder.y_proj.fc2.weight"] = "caption_projection.linear_2.weight"
    mapping["y_embedder.y_proj.fc2.bias"] = "caption_projection.linear_2.bias"
    mapping["attention_y_norm.weight"] = "caption_norm.weight"
    mapping["final_layer.scale_shift_table"] = "scale_shift_table"
    mapping["final_layer.linear.weight"] = "proj_out.weight"
    mapping["final_layer.linear.bias"] = "proj_out.bias"

    return mapping


def split_qkv(weight, dim):
    """Split qkv weight [3*dim, dim] into q, k, v each [dim, dim]."""
    q, k, v = weight.chunk(3, dim=0)
    return q, k, v


def split_kv(weight, dim):
    """Split kv weight [2*dim, dim] into k, v each [dim, dim]."""
    k, v = weight.chunk(2, dim=0)
    return k, v


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
    mapping = convert_key_mapping()
    dim = 2240  # Sana-1600M hidden dim

    transformer_sd = {}
    for pth_key, diff_key in mapping.items():
        if pth_key not in state_dict:
            print(f"  WARNING: {pth_key} not found, skipping")
            continue

        w = state_dict[pth_key]

        # Special handling for qkv/kv split
        if ".attn.qkv.weight" in pth_key:
            q, k, v = split_qkv(w, dim)
            base = diff_key.replace(".qkv.weight", "")
            transformer_sd[f"{base}.to_q.weight"] = q
            transformer_sd[f"{base}.to_k.weight"] = k
            transformer_sd[f"{base}.to_v.weight"] = v
        elif ".attn.qkv.bias" in pth_key:
            q, k, v = w.chunk(3, dim=0)
            base = diff_key.replace(".qkv.bias", "")
            transformer_sd[f"{base}.to_q.bias"] = q
            transformer_sd[f"{base}.to_k.bias"] = k
            transformer_sd[f"{base}.to_v.bias"] = v
        elif ".cross_attn.kv_linear.weight" in pth_key:
            k, v = split_kv(w, dim)
            base = diff_key.replace(".kv.weight", "")
            transformer_sd[f"{base}.to_k.weight"] = k
            transformer_sd[f"{base}.to_v.weight"] = v
        elif ".cross_attn.kv_linear.bias" in pth_key:
            k, v = w.chunk(2, dim=0)
            base = diff_key.replace(".kv.bias", "")
            transformer_sd[f"{base}.to_k.bias"] = k
            transformer_sd[f"{base}.to_v.bias"] = v
        else:
            transformer_sd[diff_key] = w

    print(f"  Converted {len(transformer_sd)} keys")

    # Save transformer as safetensors
    save_file(transformer_sd, args.output / "transformer" / "diffusion_pytorch_model.safetensors")

    # Copy VAE, text_encoder, tokenizer, scheduler from template
    print("Copying VAE, text_encoder, tokenizer, scheduler from template...")
    for subdir in ["vae", "text_encoder", "tokenizer", "scheduler"]:
        src = args.template / subdir
        dst = args.output / subdir
        if src.exists():
            for f in src.glob("*"):
                if f.is_file() and not f.name.endswith(".bin"):
                    shutil.copy2(f, dst / f.name)
            print(f"  Copied {subdir}/")

    # Write configs
    print("Writing configs...")
    config = {
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
    }
    with open(args.output / "transformer" / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    model_index = {
        "_class_name": "SanaPipeline",
        "_diffusers_version": "0.32.0",
        "_name_or_path": "Efficient-Large-Model/Sana_1600M_512px",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "Gemma2Model"],
        "tokenizer": ["transformers", "GemmaTokenizer"],
        "transformer": ["diffusers", "SanaTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderDC"],
    }
    with open(args.output / "model_index.json", "w") as f:
        json.dump(model_index, f, indent=2)

    # Copy other files
    for f in ["LICENSE", "README.md", "configuration.json"]:
        src = args.template / f
        if src.exists():
            shutil.copy2(src, args.output / f)

    print(f"\nDone! Converted model saved to {args.output}")


if __name__ == "__main__":
    main()
