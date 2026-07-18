#!/usr/bin/env python3
"""Convert ComfyUI-format SD3-Medium safetensors to diffusers pipeline directory.

Uses diffusers' built-in conversion functions for proper key remapping:
  - convert_sd3_transformer_checkpoint_to_diffusers (MMDiT key remapping)
  - convert_sd3_t5_checkpoint_to_diffusers (T5 key remapping)

Manually builds CLIP text encoders (already in HF format) and VAE.

Usage:
  python scripts/convert_sd3_to_diffusers.py
  python scripts/convert_sd3_to_diffusers.py --output /path/to/sd3_diffusers
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# ─── Paths ────────────────────────────────────────────────────────────
MS_SNAPSHOT = Path.home() / "MyWork/models/ms_cache/models/AI-ModelScope--stable-diffusion-3-medium/snapshots/master"
T5_CACHE = Path.home() / "MyWork/models/ms_cache_t5/models/AI-ModelScope--stable-diffusion-3-medium/snapshots/master"
DEFAULT_OUTPUT = Path.home() / "MyWork/models/sd3-medium-diffusers"


def find_file(candidates: list[Path], label: str) -> Path:
    """Find the first existing file from candidates."""
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{label} not found. Tried: {[str(c) for c in candidates]}"
    )


def write_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))


def build_transformer(main_state: dict, output_dir: Path, dtype: torch.dtype) -> None:
    """Convert MMDiT weights using diffusers' built-in conversion function."""
    from diffusers.loaders.single_file_utils import (
        convert_sd3_transformer_checkpoint_to_diffusers,
    )

    print("  Building transformer (MMDiT)...")
    trans_dir = output_dir / "transformer"
    trans_dir.mkdir(parents=True, exist_ok=True)

    # Detect config params BEFORE calling conversion (it mutates the dict via pop)
    mmdit_keys = [k for k in main_state.keys() if k.startswith("model.diffusion_model.")]
    # caption_projection_dim = context_embedder.weight.shape[0]
    emb_key = "model.diffusion_model.context_embedder.weight"
    caption_projection_dim = main_state[emb_key].shape[0]
    # num_layers from joint_blocks
    layer_nums = set()
    for k in mmdit_keys:
        if "joint_blocks" in k:
            parts = k.split(".")
            layer_nums.add(int(parts[3]))
    num_layers = max(layer_nums) + 1
    # dual_attention_layers
    dual = set()
    for k in mmdit_keys:
        if "attn2" in k and "joint_blocks" in k:
            parts = k.split(".")
            dual.add(int(parts[3]))
    dual_attention_layers = sorted(dual)
    # qk_norm
    has_qk_norm = any("ln_q" in k for k in mmdit_keys)

    # Now call conversion (it will pop keys from main_state)
    converted = convert_sd3_transformer_checkpoint_to_diffusers(main_state)

    print(f"    caption_projection_dim={caption_projection_dim}, num_layers={num_layers}")
    print(f"    dual_attention_layers={dual_attention_layers}, qk_norm={has_qk_norm}")
    print(f"    {len(converted)} converted tensors")

    # Cast to target dtype
    converted = {k: v.to(dtype) for k, v in converted.items()}
    save_file(converted, str(trans_dir / "diffusion_pytorch_model.safetensors"))

    config = {
        "_class_name": "SD3Transformer2DModel",
        "_diffusers_version": "0.30.0",
        "sample_size": 128,
        "patch_size": 2,
        "in_channels": 16,
        "num_layers": num_layers,
        "attention_head_dim": 64,
        "num_attention_heads": 24,  # 24 * 64 = 1536 inner dim
        "joint_attention_dim": 4096,
        "caption_projection_dim": caption_projection_dim,
        "pooled_projection_dim": 2048,
        "out_channels": 16,
        "pos_embed_max_size": 192,
        "dual_attention_layers": dual_attention_layers,
        "qk_norm": "rms_norm" if has_qk_norm else None,
    }
    write_config(trans_dir / "config.json", config)


def build_vae(main_state: dict, output_dir: Path, dtype: torch.dtype) -> None:
    """Extract and save AutoencoderKL weights."""
    print("  Building VAE...")
    vae_dir = output_dir / "vae"
    vae_dir.mkdir(parents=True, exist_ok=True)

    # ComfyUI keys: first_stage_model.* -> diffusers keys: no prefix
    vae_state = {}
    prefix = "first_stage_model."
    for key, tensor in main_state.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            vae_state[new_key] = tensor.to(dtype)

    print(f"    {len(vae_state)} tensors")
    save_file(vae_state, str(vae_dir / "diffusion_pytorch_model.safetensors"))

    config = {
        "_class_name": "AutoencoderKL",
        "_diffusers_version": "0.30.0",
        "act_fn": "silu",
        "block_out_channels": [128, 256, 512, 512],
        "down_block_types": [
            "DownEncoderBlock2D", "DownEncoderBlock2D",
            "DownEncoderBlock2D", "DownEncoderBlock2D",
        ],
        "force_upcast": False,
        "in_channels": 3,
        "latent_channels": 16,
        "latents_mean": None,
        "latents_std": None,
        "layers_per_block": 2,
        "mid_block_add_attention": True,
        "norm_num_groups": 32,
        "out_channels": 3,
        "scaling_factor": 0.3611,
        "shift_factor": 0.1159,
        "up_block_types": [
            "UpDecoderBlock2D", "UpDecoderBlock2D",
            "UpDecoderBlock2D", "UpDecoderBlock2D",
        ],
        "use_post_quantum": False,
        "use_quantum": False,
    }
    write_config(vae_dir / "config.json", config)


def build_clip_text_encoder(
    ckpt_path: Path,
    output_dir: Path,
    encoder_name: str,
    dtype: torch.dtype,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    intermediate_size: int,
    projection_dim: int,
) -> None:
    """Build CLIPTextModelWithProjection from ComfyUI-format safetensors."""
    print(f"  Building {encoder_name}...")
    enc_dir = output_dir / encoder_name
    enc_dir.mkdir(parents=True, exist_ok=True)

    state = load_file(str(ckpt_path))
    # Keys are already in HF format: text_model.* and text_projection.*
    state = {k: v.to(dtype) for k, v in state.items()}
    print(f"    {len(state)} tensors, projection_dim={projection_dim}")
    save_file(state, str(enc_dir / "model.safetensors"))

    # projection_dim must be inside text_config for transformers to use it correctly
    config = {
        "_class_name": "CLIPTextModelWithProjection",
        "_diffusers_version": "0.30.0",
        "projection_dim": projection_dim,
        "text_config": {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_attention_heads": num_heads,
            "num_hidden_layers": num_layers,
            "vocab_size": 49408,
            "hidden_act": "quick_gelu" if hidden_size == 768 else "gelu",
            "max_position_embeddings": 77,
            "layer_norm_eps": 1e-05,
            "attention_dropout": 0.0,
            "hidden_dropout": 0.0,
            "initializer_factor": 1.0,
            "initializer_range": 0.02,
            "pad_token_id": 49407,
            "bos_token_id": 49406,
            "eos_token_id": 49407,
            "projection_dim": projection_dim,
        },
    }
    write_config(enc_dir / "config.json", config)


def build_t5_encoder(ckpt_path: Path, output_dir: Path, dtype: torch.dtype) -> None:
    """Build T5EncoderModel from ComfyUI-format safetensors.

    The standalone t5xxl_fp16.safetensors already has keys in 'encoder.*' format,
    which is the standard diffusers T5EncoderModel format. No conversion needed.
    """
    print("  Building text_encoder_3 (T5-XXL)...")
    enc_dir = output_dir / "text_encoder_3"
    enc_dir.mkdir(parents=True, exist_ok=True)

    state = load_file(str(ckpt_path))
    print(f"    Loaded {len(state)} tensors (already in encoder.* format)")

    # Keys are already in diffusers format (encoder.block.*, encoder.embed_tokens, etc.)
    # Just cast dtype
    state = {k: v.to(dtype) for k, v in state.items()}
    save_file(state, str(enc_dir / "model.safetensors"))

    config = {
        "d_model": 4096,
        "d_ff": 10240,
        "num_layers": 24,
        "num_decoder_layers": 0,
        "num_heads": 64,
        "relative_attention_num_buckets": 32,
        "relative_attention_max_distance": 128,
        "vocab_size": 32128,
        "layer_norm_epsilon": 1e-06,
        "dropout_rate": 0.0,
        "feed_forward_proj": "gated-gelu",
        "is_gated_act": True,
        "use_cache": False,
        "tie_word_embeddings": False,
        "model_type": "t5",
        "architectures": ["T5EncoderModel"],
        "is_encoder_decoder": True,
        "d_kv": 64,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "decoder_start_token_id": 0,
    }
    write_config(enc_dir / "config.json", config)


def build_tokenizers(output_dir: Path) -> None:
    """Download tokenizer files from known open-access HF repos."""
    print("  Building tokenizers...")
    import urllib.request

    # CLIP tokenizer (from openai/clip-vit-base-patch32 - open access)
    clip_tokenizer_url = "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main"

    for tdir_name in ["tokenizer", "tokenizer_2"]:
        tdir = output_dir / tdir_name
        tdir.mkdir(parents=True, exist_ok=True)
        for fname in ["vocab.json", "merges.txt", "special_tokens_map.json",
                       "tokenizer_config.json"]:
            fpath = tdir / fname
            if not fpath.exists():
                try:
                    urllib.request.urlretrieve(f"{clip_tokenizer_url}/{fname}", str(fpath))
                    print(f"    {tdir_name}/{fname} ✓")
                except Exception as e:
                    print(f"    {tdir_name}/{fname} FAILED: {e}")
                    if fname == "tokenizer_config.json":
                        write_config(fpath, {"model_max_length": 77, "tokenizer_class": "CLIPTokenizer"})
                    elif fname == "special_tokens_map.json":
                        write_config(fpath, {
                            "bos_token": "<|startoftext|>",
                            "eos_token": "<|endoftext|>",
                            "pad_token": "!",
                            "unk_token": "<|endoftext|>",
                        })

    # T5 tokenizer
    t5_tokenizer_url = "https://huggingface.co/google-t5/t5-base/resolve/main"
    tdir = output_dir / "tokenizer_3"
    tdir.mkdir(parents=True, exist_ok=True)
    for fname in ["spiece.model", "special_tokens_map.json", "tokenizer_config.json"]:
        fpath = tdir / fname
        if not fpath.exists():
            try:
                urllib.request.urlretrieve(f"{t5_tokenizer_url}/{fname}", str(fpath))
                print(f"    tokenizer_3/{fname} ✓")
            except Exception as e:
                print(f"    tokenizer_3/{fname} FAILED: {e}")
                if fname == "tokenizer_config.json":
                    write_config(fpath, {"model_max_length": 256, "tokenizer_class": "T5TokenizerFast"})
                elif fname == "special_tokens_map.json":
                    write_config(fpath, {
                        "eos_token": "</s>",
                        "unk_token": "<unk>",
                        "pad_token": "<pad>",
                    })


def build_scheduler(output_dir: Path) -> None:
    """Write scheduler config."""
    print("  Building scheduler...")
    sdir = output_dir / "scheduler"
    sdir.mkdir(parents=True, exist_ok=True)
    config = {
        "_class_name": "FlowMatchEulerDiscreteScheduler",
        "_diffusers_version": "0.30.0",
        "num_train_timesteps": 1000,
        "shift": 3.0,
        "use_dynamic_shifting": False,
        "base_shift": 0.0,
        "base_image_seq_len": 256,
        "max_image_seq_len": 4096,
        "invert_sigmas": False,
    }
    write_config(sdir / "scheduler_config.json", config)


def main():
    parser = argparse.ArgumentParser(description="Convert SD3 safetensors to diffusers format")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output directory for diffusers pipeline")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    output_dir = Path(args.output)
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    save_dtype = dtype_map[args.dtype]

    # Locate input files
    main_ckpt = find_file([MS_SNAPSHOT / "sd3_medium.safetensors"], "Main checkpoint")
    clip_l = find_file([MS_SNAPSHOT / "text_encoders/clip_l.safetensors"], "CLIP-L")
    clip_g = find_file([MS_SNAPSHOT / "text_encoders/clip_g.safetensors"], "CLIP-G")
    t5xxl = find_file(
        [
            T5_CACHE / "text_encoders/t5xxl_fp16.safetensors",
            MS_SNAPSHOT / "text_encoders/t5xxl_fp16.safetensors",
        ],
        "T5-XXL fp16",
    )

    print(f"Main checkpoint:  {main_ckpt}")
    print(f"CLIP-L:           {clip_l}")
    print(f"CLIP-G:           {clip_g}")
    print(f"T5-XXL fp16:      {t5xxl}")
    print(f"Output:           {output_dir}")
    print(f"Dtype:            {args.dtype}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load main checkpoint
    print("Loading main checkpoint...")
    t0 = time.time()
    main_state = load_file(str(main_ckpt))
    print(f"  Loaded {len(main_state)} keys in {time.time()-t0:.1f}s\n")

    # Build model_index.json
    print("Building diffusers pipeline directory...")
    model_index = {
        "_class_name": "StableDiffusion3Pipeline",
        "_diffusers_version": "0.30.0",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "CLIPTextModelWithProjection"],
        "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
        "text_encoder_3": ["transformers", "T5EncoderModel"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "tokenizer_2": ["transformers", "CLIPTokenizer"],
        "tokenizer_3": ["transformers", "T5TokenizerFast"],
        "transformer": ["diffusers", "SD3Transformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    write_config(output_dir / "model_index.json", model_index)

    # Build each component
    build_transformer(main_state, output_dir, save_dtype)
    build_vae(main_state, output_dir, save_dtype)
    build_clip_text_encoder(
        clip_l, output_dir, "text_encoder", save_dtype,
        hidden_size=768, num_layers=12, num_heads=12,
        intermediate_size=3072, projection_dim=768,
    )
    build_clip_text_encoder(
        clip_g, output_dir, "text_encoder_2", save_dtype,
        hidden_size=1280, num_layers=32, num_heads=20,
        intermediate_size=5120, projection_dim=1280,
    )
    build_t5_encoder(t5xxl, output_dir, save_dtype)
    build_scheduler(output_dir)
    build_tokenizers(output_dir)

    # Free memory
    del main_state

    # Verify
    print("\n✅ Conversion complete! Directory structure:")
    total_size = 0
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"  {f.relative_to(output_dir)} ({size_mb:.1f} MB)")
    print(f"\nTotal: {total_size:.1f} MB ({total_size/1024:.1f} GB)")


if __name__ == "__main__":
    main()
