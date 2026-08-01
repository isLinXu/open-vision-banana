#!/usr/bin/env python3
"""Janus-1.3B open-vision-banana 完整训练 —— 所有 576 tokens.

改进：
- 训练所有 576 VQ tokens（teacher forcing）
- 输入 = (image, prompt) → LLM → gen_head → VQ token logits
- 目标 = depth RGB → VQ encode → 576 tokens
- 损失 = CrossEntropy(所有 576 个位置)

用法:
    PYTHONPATH=src:/tmp/Janus python scripts/janus_vision_banana_full.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, "/tmp/Janus")  # Janus 自定义代码

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.evaluation.relative_depth import evaluate_relative_depth_rgb
from open_vision_banana.training.tuner import VisionTaskDataset, _collate_fn

# ── Config ───────────────────────────────────────────────────────────────
MODEL_ID = os.environ.get("VB_MODEL", "deepseek-ai/Janus-1.3B")
DATA_DIR = os.environ.get(
    "VB_DATA",
    "/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v3/depth",
)
STEPS = int(os.environ.get("VB_STEPS", "2000"))
NSAMPLES = int(os.environ.get("VB_NSAMPLES", "8"))
RES = int(os.environ.get("VB_RES", "384"))
BATCH = int(os.environ.get("VB_BATCH", "1"))
LR = float(os.environ.get("VB_LR", "1e-4"))
RANK = int(os.environ.get("VB_RANK", "64"))
ALPHA = int(os.environ.get("VB_ALPHA", "128"))
OUT = Path(os.environ.get("VB_OUT", str(PROJECT_ROOT / "output" / "janus_vision_banana_full")))
OUT.mkdir(parents=True, exist_ok=True)

DTYPE = torch.float32  # MPS safe
VISION_PROMPT = "Generate a metric depth visualization of this image."
IMAGE_TOKEN_NUM = 576  # 24x24 for 384x384 with patch16


def _resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    torch.manual_seed(42)
    device = _resolve_device()

    print(f"[cfg] model={MODEL_ID}  device={device}  dtype={DTYPE}")
    print(f"[cfg] steps={STEPS}  nsamples={NSAMPLES}  res={RES}  batch={BATCH}")
    print(f"[cfg] lr={LR}  rank={RANK}  alpha={ALPHA}  out={OUT}")

    # 1) 加载 Janus-1.3B ───────────────────────────────────────────────
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from janus.models.modeling_vlm import MultiModalityConfig, MultiModalityCausalLM
    AutoConfig.register("multi_modality", MultiModalityConfig)
    AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = model.to(device)
    print(f"[model] loaded in {time.time()-t0:.1f}s")
    print(f"[model] type={type(model).__name__}")
    print(f"[model] params={sum(p.numel() for p in model.parameters()):,}")

    # 2) 数据集 ───────────────────────────────────────────────────────
    ds = VisionTaskDataset(data_path=DATA_DIR, resolution=RES, task="depth")
    n_total = len(ds)
    print(f"[data] loaded {n_total} depth samples @ {RES}px")
    if n_total < NSAMPLES:
        raise RuntimeError(f"Need {NSAMPLES} samples, have {n_total}")

    subset = torch.utils.data.Subset(ds, list(range(NSAMPLES)))
    loader = torch.utils.data.DataLoader(
        subset, batch_size=BATCH, shuffle=True, num_workers=0,
        pin_memory=False, collate_fn=_collate_fn, drop_last=True,
    )

    # 3) LoRA on language_model ─────────────────────────────────────────
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    peft_model = get_peft_model(model.language_model, lora_config)
    n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    n_total_p = sum(p.numel() for p in peft_model.parameters())
    print(f"[lora] trainable = {n_train:,}  ({100*n_train/n_total_p:.2f}%)")

    # 冻结 vision 和 generation 部分
    model.vision_model.requires_grad_(False)
    model.gen_vision_model.requires_grad_(False)
    model.aligner.requires_grad_(False)
    model.gen_aligner.requires_grad_(False)
    model.gen_head.requires_grad_(False)
    model.gen_embed.requires_grad_(False)

    # 4) Optimizer ─────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )

    def lr_lambda(step: int) -> float:
        warmup = max(1, STEPS // 30)
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, STEPS - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # 5) 训练循环 ─────────────────────────────────────────────────────
    print(f"\n[train] Janus open-vision-banana FULL: {STEPS} steps")
    print(f"[train] prompt: {VISION_PROMPT[:80]}...")

    # 预编码 prompt
    prompt_ids = tokenizer(VISION_PROMPT, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    losses: list[float] = []
    it = iter(loader)
    t_train = time.time()

    for step in range(1, STEPS + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        opt.zero_grad(set_to_none=True)

        # 取 batch（batch_size=1 简化）
        image = batch["image"][0:1].to(device, DTYPE)  # [-1, 1]
        target_rgb = batch["target_rgb"][0:1].to(device, DTYPE)  # [0, 1]

        # 1. 图像理解路径: pixel_values → vision_model → aligner
        pixel_values = image.unsqueeze(1)  # [1, 1, 3, H, W]

        # 2. 目标图像 → VQ tokens (576 个)
        with torch.no_grad():
            target_for_vq = target_rgb * 2 - 1
            quant, emb_loss, info = model.gen_vision_model.encode(target_for_vq)
            target_tokens = info[2]  # [576]

        # 3. 构造输入 embeds（图像 + prompt）
        seq_len = prompt_len + IMAGE_TOKEN_NUM
        full_input_ids = torch.cat([
            prompt_ids,
            torch.full((1, IMAGE_TOKEN_NUM), -1, device=device, dtype=torch.long),
        ], dim=1)
        images_seq_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
        images_seq_mask[0, prompt_len:] = True
        images_emb_mask = torch.ones(1, 1, IMAGE_TOKEN_NUM, dtype=torch.bool, device=device)

        input_embeds = model.prepare_inputs_embeds(
            input_ids=full_input_ids,
            pixel_values=pixel_values,
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask,
        )

        # 4. 前向传播
        outputs = model.language_model.model(
            inputs_embeds=input_embeds,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state  # [1, T, D]

        # 5. 生成图像 token logits（所有 576 个位置）
        # hidden_states[:, prompt_len:, :] 对应 576 个图像 token 位置
        gen_hidden = hidden_states[:, prompt_len:, :]  # [1, 576, D]
        gen_logits = model.gen_head(gen_hidden)  # [1, 576, image_token_size]

        # 6. 损失：CrossEntropy(所有 576 个位置)
        # target_tokens [576] → [1, 576]
        target_tokens_batch = target_tokens.unsqueeze(0)  # [1, 576]
        loss = F.cross_entropy(
            gen_logits.reshape(-1, gen_logits.shape[-1]),  # [576, image_token_size]
            target_tokens_batch.reshape(-1),  # [576]
        )

        if not torch.isfinite(loss).all():
            print(f"[train] step {step:>4}: NON-FINITE loss — skip", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())

        if step % 30 == 0 or step == 1:
            recent = sum(losses[-30:]) / len(losses[-30:])
            print(f"[train] step {step:>4}/{STEPS}  loss={loss.item():.4f}  "
                  f"avg30={recent:.4f}  lr={sched.get_last_lr()[0]:.2e}  "
                  f"({(time.time()-t_train)/step:.2f}s/step)", flush=True)
        # 每 500 步存中间 checkpoint
        if step % 500 == 0:
            mid_ckpt = OUT / f"lora_step{step}"
            mid_ckpt.mkdir(parents=True, exist_ok=True)
            peft_model.save_pretrained(str(mid_ckpt))
            print(f"[ckpt] mid-training save → {mid_ckpt}", flush=True)

    print(f"\n[train] completed {len(losses)} steps")
    print(f"[train] avg loss = {sum(losses)/len(losses):.4f}")

    # 保存 checkpoint
    ckpt_dir = OUT / "lora_final"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(ckpt_dir))
    print(f"[ckpt] saved → {ckpt_dir}")

    # 6) 推理评估 ─────────────────────────────────────────────────────
    print("\n[eval] inference on training samples...")
    mapper = DepthRGBMapper(colormap="viridis")
    results = []

    @torch.no_grad()
    def infer(sample_idx: int) -> dict:
        s = ds[sample_idx]
        inp_tensor = s["image"]  # [-1, 1]
        tgt_rgb = s["target_rgb"]  # [0, 1]
        pil_in = Image.fromarray(
            (inp_tensor.permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5)
            .mul(255).byte().numpy()
        )

        # 简化推理：用模型生成所有 576 个 token
        pixel_values = inp_tensor.unsqueeze(0).unsqueeze(0).to(device, DTYPE)
        seq_len = prompt_len + IMAGE_TOKEN_NUM
        full_input_ids = torch.cat([
            prompt_ids,
            torch.full((1, IMAGE_TOKEN_NUM), -1, device=device, dtype=torch.long),
        ], dim=1)
        images_seq_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
        images_seq_mask[0, prompt_len:] = True
        images_emb_mask = torch.ones(1, 1, IMAGE_TOKEN_NUM, dtype=torch.bool, device=device)
        input_embeds = model.prepare_inputs_embeds(
            input_ids=full_input_ids,
            pixel_values=pixel_values,
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask,
        )
        outputs = model.language_model.model(
            inputs_embeds=input_embeds,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state
        gen_hidden = hidden_states[:, prompt_len:, :]  # [1, 576, D]
        gen_logits = model.gen_head(gen_hidden)  # [1, 576, image_token_size]
        pred_tokens = torch.argmax(gen_logits, dim=-1)  # [1, 576]

        # 解码 VQ tokens → RGB
        pred_tokens_flat = pred_tokens.squeeze(0)  # [576]
        dec = model.gen_vision_model.decode_code(
            pred_tokens_flat,
            shape=[1, 8, 24, 24],  # [batch, channels, h, w]
        )
        pred_rgb = dec.squeeze(0).clamp(0, 1).cpu()  # [3, 384, 384]

        # 三联图
        panel = Image.new("RGB", (RES * 3, RES))
        panel.paste(pil_in.resize((RES, RES)), (0, 0))
        panel.paste(Image.fromarray(
            (pred_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
        ).resize((RES, RES)), (RES, 0))
        panel.paste(Image.fromarray(
            (tgt_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
        ).resize((RES, RES)), (RES * 2, 0))
        panel.save(OUT / f"panel_{sample_idx:02d}.png")

        rgb_mae = (pred_rgb.float() - tgt_rgb.float()).abs().mean().item()
        return {"idx": sample_idx, "rgb_mae": rgb_mae}

    for idx in range(min(NSAMPLES, 2)):
        r = infer(idx)
        results.append(r)
        print(f"[eval] idx={idx:>3}  rgb_mae={r['rgb_mae']:.4f}")

    avg_mae = sum(r["rgb_mae"] for r in results) / len(results)
    print(f"\n[gate] avg rgb_mae={avg_mae:.4f}")

    total_min = (time.time() - t_train) / 60
    print(f"\n[done] Janus open-vision-banana FULL complete ({total_min:.1f} min)")


if __name__ == "__main__":
    main()
