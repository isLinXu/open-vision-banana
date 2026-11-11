#!/usr/bin/env python3
"""Janus v4 inference script: depth prediction with the trained model.

Usage:
    # single-image inference
    PYTHONPATH=src:/tmp/Janus python scripts/infer_v4.py --input path/to/image.jpg --output output/pred.png

    # batch inference (a directory)
    PYTHONPATH=src:/tmp/Janus python scripts/infer_v4.py --input path/to/images/ --output output/predictions/

    # choose a checkpoint
    PYTHONPATH=src:/tmp/Janus python scripts/infer_v4.py --input image.jpg --ckpt output/janus_continuous_rgb_v4/ckpt_best
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, "/tmp/Janus")

# ── Default config ───────────────────────────────────────────────────────
DEFAULT_CKPT = str(PROJECT_ROOT / "output" / "janus_continuous_rgb_v4" / "ckpt_best")
MODEL_ID = "deepseek-ai/Janus-1.3B"
RES = 384
IMAGE_TOKEN_NUM = 576
DTYPE = torch.float32
TASK_PROMPT = "Generate a metric depth visualization of this image."


class RGBHead(nn.Module):
    """RGB head matching the training one (manual normalization)."""

    def __init__(self, input_dim: int, image_token_num: int, image_size: int, dropout: float = 0.0):
        super().__init__()
        self.image_token_num = image_token_num
        self.image_size = image_size
        self.patch_size = image_size // int(math.sqrt(image_token_num))
        out_dim = self.patch_size * self.patch_size * 3
        mid_dim = min(input_dim, 512)

        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_var", torch.ones(input_dim))

        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, out_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        b = hidden_states.shape[0]
        # normalize with running stats at inference
        mean = self.running_mean
        var = self.running_var
        x = (hidden_states - mean) / (var.sqrt() + 1e-6)
        x = self.proj(x)
        x = torch.sigmoid(x)
        x = x.view(b, self.image_token_num, self.patch_size, self.patch_size, 3)
        x = x.permute(0, 4, 1, 2, 3)
        h = w = int(math.sqrt(self.image_token_num))
        x = x.reshape(b, 3, h * self.patch_size, w * self.patch_size)
        return x


def _resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(ckpt_dir: str, device: str):
    """Load Janus + LoRA + RGB head."""
    import json

    import janus.models.modeling_vlm as janus_vlm
    for cfg_cls in [janus_vlm.VisionConfig, janus_vlm.AlignerConfig,
                    janus_vlm.GenVisionConfig, janus_vlm.GenAlignerConfig,
                    janus_vlm.GenHeadConfig]:
        if hasattr(cfg_cls, '__dataclass_fields__'):
            if 'params' in cfg_cls.__dataclass_fields__:
                cfg_cls.__dataclass_fields__['params'].default = None

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from janus.models.modeling_vlm import MultiModalityConfig, MultiModalityCausalLM
    AutoConfig.register("multi_modality", MultiModalityConfig)
    AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)

    # read the training config
    config_path = Path(ckpt_dir) / "train_config.json"
    if config_path.exists():
        with open(config_path) as f:
            train_cfg = json.load(f)
        model_id = train_cfg.get("model_id", MODEL_ID)
        dropout = train_cfg.get("dropout", 0.0)
        feature_src = train_cfg.get("feature_src", "vision")
        print(f"[cfg] from train_config: model={model_id}, feature_src={feature_src}")
    else:
        model_id = MODEL_ID
        dropout = 0.0
        feature_src = "vision"

    # load the base model
    print(f"[load] loading base model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=DTYPE, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = model.to(device)
    model.eval()

    # load the LoRA
    adapter_dir = Path(ckpt_dir) / "lora"
    if adapter_dir.exists():
        from peft import PeftModel
        model.language_model = PeftModel.from_pretrained(model.language_model, str(adapter_dir))
        model.language_model.eval()
        print(f"[load] LoRA loaded from {adapter_dir}")

    # load the RGB head
    vision_dim = 1024
    rgb_head = RGBHead(vision_dim, IMAGE_TOKEN_NUM, RES, dropout=0.0).to(device, DTYPE)
    rgb_head_path = Path(ckpt_dir) / "rgb_head.pt"
    if rgb_head_path.exists():
        rgb_head.load_state_dict(torch.load(rgb_head_path, map_location=device, weights_only=True))
        rgb_head.eval()
        print(f"[load] RGB Head loaded from {rgb_head_path}")
    else:
        print(f"[warn] RGB Head not found at {rgb_head_path}")

    return model, tokenizer, rgb_head, feature_src


@torch.no_grad()
def predict(model, tokenizer, rgb_head, image_path: str, device: str,
            feature_src: str = "vision") -> tuple[torch.Tensor, Image.Image]:
    """Depth → RGB prediction for one image.

    Returns:
        (pred_rgb_tensor, input_pil_image)
    """
    from torchvision import transforms

    # load and preprocess the image
    pil_img = Image.open(image_path).convert("RGB")
    original_size = pil_img.size

    transform = transforms.Compose([
        transforms.Resize((RES, RES)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    inp_tensor = transform(pil_img)

    # prepare the inputs
    prompt = TASK_PROMPT
    p_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    p_len = p_ids.shape[1]

    pixel_values = inp_tensor.unsqueeze(0).unsqueeze(0).to(device, DTYPE)
    seq_len = p_len + IMAGE_TOKEN_NUM
    full_input_ids = torch.cat([
        p_ids,
        torch.full((1, IMAGE_TOKEN_NUM), -1, device=device, dtype=torch.long),
    ], dim=1)
    images_seq_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
    images_seq_mask[0, p_len:] = True
    images_emb_mask = torch.ones(1, 1, IMAGE_TOKEN_NUM, dtype=torch.bool, device=device)

    # Vision features
    vis_features = model.vision_model(pixel_values.squeeze(1))  # [1, 576, 1024]

    if feature_src == "vision":
        features = vis_features
    elif feature_src == "llm":
        input_embeds = model.prepare_inputs_embeds(
            input_ids=full_input_ids,
            pixel_values=pixel_values,
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask,
        )
        outputs = model.language_model.model(
            inputs_embeds=input_embeds,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]
        features = hidden_states[:, p_len:, :]
    elif feature_src == "both":
        input_embeds = model.prepare_inputs_embeds(
            input_ids=full_input_ids,
            pixel_values=pixel_values,
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask,
        )
        outputs = model.language_model.model(
            inputs_embeds=input_embeds,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]
        llm_hidden = hidden_states[:, p_len:, :]
        features = torch.cat([vis_features, llm_hidden], dim=-1)
    else:
        features = vis_features

    pred_rgb = rgb_head(features).squeeze(0).clamp(0, 1).cpu()
    return pred_rgb, pil_img


def predict_to_panel(model, tokenizer, rgb_head, image_path: str, device: str,
                     feature_src: str = "vision", target_path: str | None = None) -> Image.Image:
    """Build the comparison panel: [input | prediction | (optional target)]."""
    pred_rgb, pil_img = predict(model, tokenizer, rgb_head, image_path, device, feature_src)

    if target_path:
        target_img = Image.open(target_path).convert("RGB")
        panel = Image.new("RGB", (RES * 3, RES))
    else:
        panel = Image.new("RGB", (RES * 2, RES))

    panel.paste(pil_img.resize((RES, RES)), (0, 0))
    panel.paste(Image.fromarray(
        (pred_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
    ).resize((RES, RES)), (RES, 0))

    if target_path:
        target_img = Image.open(target_path).convert("RGB")
        panel.paste(target_img.resize((RES, RES)), (RES * 2, 0))

    return panel


def main():
    parser = argparse.ArgumentParser(description="Janus v4 depth→RGB inference")
    parser.add_argument("--input", "-i", required=True,
                        help="Input image path or directory")
    parser.add_argument("--output", "-o", default="output/v4_predictions",
                        help="Output path")
    parser.add_argument("--ckpt", "-c", default=DEFAULT_CKPT,
                        help="Checkpoint directory")
    parser.add_argument("--panel", action="store_true",
                        help="Build an input|pred|target comparison panel")
    parser.add_argument("--target-dir",
                        help="Target image directory (for comparison)")
    args = parser.parse_args()

    device = _resolve_device()
    print(f"[cfg] device={device}  ckpt={args.ckpt}")

    model, tokenizer, rgb_head, feature_src = load_model(args.ckpt, device)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # collect the input images
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = sorted([f for f in input_path.iterdir() if f.suffix.lower() in exts])
        print(f"[data] found {len(image_files)} images in {input_path}")
    else:
        print(f"[error] input not found: {input_path}")
        return

    t0 = time.time()
    for img_path in image_files:
        stem = img_path.stem

        if args.panel:
            target = None
            if args.target_dir:
                target_candidates = list(Path(args.target_dir).glob(f"{stem}*"))
                if target_candidates:
                    target = str(target_candidates[0])
            panel = predict_to_panel(model, tokenizer, rgb_head, str(img_path), device,
                                     feature_src, target_path=target)
            out_file = output_path / f"{stem}_panel.png"
            panel.save(str(out_file))
        else:
            pred_rgb, _ = predict(model, tokenizer, rgb_head, str(img_path), device, feature_src)
            pred_pil = Image.fromarray(
                (pred_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
            )
            out_file = output_path / f"{stem}_depth.png"
            pred_pil.save(str(out_file))

        print(f"[pred] {img_path.name} → {out_file.name}")

    elapsed = time.time() - t0
    print(f"\n[done] {len(image_files)} images in {elapsed:.1f}s ({elapsed/max(1,len(image_files)):.1f}s/img)")
    print(f"[done] output → {output_path}")


if __name__ == "__main__":
    main()
