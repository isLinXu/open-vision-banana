#!/usr/bin/env python3
"""Merge SANA-1600M LoRA adapter into the base model and save.

Strategy: load transformer on CPU, merge LoRA, save merged transformer,
then copy remaining pipeline components (text_encoder, vae, scheduler, etc.)
from the base model to avoid loading the full pipeline into memory.

Usage:
    python scripts/merge_sana_lora.py
"""

import argparse
import gc
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from diffusers.models.transformers.sana_transformer import SanaTransformer2DModel


def merge_lora(base_model_path: str, lora_adapter_path: str,
               output_dir: str, dtype: str = "bfloat16") -> None:
    """Load base transformer, merge LoRA, save merged pipeline."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype]
    base_path = Path(base_model_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Load transformer on CPU
    print(f"[1/5] Loading base transformer (CPU) from {base_model_path} ...")
    transformer = SanaTransformer2DModel.from_pretrained(
        base_path,
        subfolder="transformer",
        torch_dtype=torch_dtype,
    )

    # Step 2: Load LoRA adapter via PEFT
    print(f"[2/5] Loading LoRA adapter from {lora_adapter_path} ...")
    transformer = PeftModel.from_pretrained(
        transformer, lora_adapter_path, adapter_name="vision_banana"
    )

    adapter_config = json.loads(
        (Path(lora_adapter_path) / "adapter_config.json").read_text()
    )
    print(f"  LoRA: r={adapter_config['r']}, alpha={adapter_config['lora_alpha']}, "
          f"rsLoRA={adapter_config.get('use_rslora', False)}, "
          f"targets={adapter_config['target_modules']}")

    # Step 3: Merge LoRA weights
    print("[3/5] Merging LoRA weights into base transformer ...")
    transformer = transformer.merge_and_unload()

    # Step 4: Save merged transformer
    transformer_dir = output_path / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    print(f"[4/5] Saving merged transformer to {transformer_dir} ...")
    transformer.save_pretrained(str(transformer_dir))

    del transformer
    gc.collect()

    # Step 5: Copy remaining pipeline components from base model
    print(f"[5/5] Copying pipeline components from base model ...")
    skip_dirs = {"transformer"}
    for item in base_path.iterdir():
        if item.name in skip_dirs:
            continue
        dst = output_path / item.name
        if dst.exists():
            continue
        if item.is_dir():
            shutil.copytree(str(item), str(dst))
        else:
            shutil.copy2(str(item), str(dst))
        print(f"  Copied: {item.name}")

    # Write merge metadata
    merge_meta = {
        "base_model": base_model_path,
        "lora_adapter": lora_adapter_path,
        "lora_rank": adapter_config["r"],
        "lora_alpha": adapter_config["lora_alpha"],
        "use_rslora": adapter_config.get("use_rslora", False),
        "target_modules": adapter_config["target_modules"],
        "dtype": dtype,
        "merge_method": "peft.merge_and_unload",
        "tasks": ["depth", "normal", "segmentation"],
        "training_data": "SNVC-50M v5_full clean",
        "training_steps": 10500,
        "val_loss": 0.03247,
        "description": (
            "SANA-1600M with Vision Banana multi-task LoRA merged "
            "(depth + normal + segmentation). "
            "Trained on SNVC-50M v5_full clean data, 10500 steps. "
            "Best val_loss=0.0325."
        ),
    }
    (output_path / "merge_metadata.json").write_text(
        json.dumps(merge_meta, indent=2)
    )

    # Report output size
    total_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    print(f"\nDone! Merged model saved to: {output_path}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Merge SANA LoRA into base model")
    parser.add_argument(
        "--base-model",
        default="Efficient-Large-Model/Sana_1600M_512px_diffusers",
        help="Path to base SANA diffusers model",
    )
    parser.add_argument(
        "--lora-adapter",
        default="output/sana_1600m_3task_full_clean/checkpoint-10500_best/lora_adapter",
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--output-dir",
        default="output/sana_1600m_multitask_v1_merged",
        help="Output directory for merged model",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Dtype for the merged model",
    )
    args = parser.parse_args()

    merge_lora(args.base_model, args.lora_adapter, args.output_dir, args.dtype)


if __name__ == "__main__":
    main()
