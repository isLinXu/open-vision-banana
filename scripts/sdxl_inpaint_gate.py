#!/usr/bin/env python3
"""SDXL Inpainting depth gate — 论文范式的合理开源替代.

SDXL inpainting 模型天生支持图像条件（masked_image_latent + mask），
天然具备"看图生成"能力。训练时：
  输入 = (photo, full_mask, vision_prompt) → depth_rgb
  模型在原有图像基础上，按 prompt 将遮罩区域重绘为 viridis 深度色图

与 paper-style (SSD-1B) 的关键区别：
  - inpainting 模型 UNet 有 9 通道输入（4 noisy + 4 masked + 1 mask）
  - masked_image_latent 提供图像上下文，让模型"看见"输入
  - 无需额外扩 conv_in，用模型原生条件注入

用法:
    PYTHONPATH=src python scripts/sdxl_inpaint_gate.py
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

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.evaluation.relative_depth import evaluate_relative_depth_rgb
from open_vision_banana.training.tuner import VisionTaskDataset, _collate_fn

# ── Config ───────────────────────────────────────────────────────────────
MODEL_ID = os.environ.get("VB_MODEL", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
DATA_DIR = os.environ.get(
    "VB_DATA",
    "/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v3/depth",
)
STEPS = int(os.environ.get("VB_STEPS", "200"))
NSAMPLES = int(os.environ.get("VB_NSAMPLES", "8"))
RES = int(os.environ.get("VB_RES", "256"))
BATCH = int(os.environ.get("VB_BATCH", "1"))
LR = float(os.environ.get("VB_LR", "1e-4"))
RANK = int(os.environ.get("VB_RANK", "64"))
ALPHA = int(os.environ.get("VB_ALPHA", "128"))
OUT = Path(os.environ.get("VB_OUT", str(PROJECT_ROOT / "output" / "sdxl_inpaint_gate")))
OUT.mkdir(parents=True, exist_ok=True)

VISION_PROMPT = (
    "Generate a relative depth map of this scene using the viridis RGB palette, "
    "from dark purple and blue through green to yellow."
)

DTYPE = torch.float32  # MPS safe


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

    # 1) 加载 SDXL Inpainting Pipeline ──────────────────────────────────
    from diffusers import StableDiffusionXLInpaintPipeline

    t0 = time.time()
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, variant=None
    ).to(device)
    print(f"[model] loaded in {time.time()-t0:.1f}s")

    unet = pipe.unet
    print(f"[model] UNet conv_in: {unet.conv_in.in_channels} channels "
          f"(4 noisy + 4 masked + 1 mask)")

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

    # 3) LoRA ──────────────────────────────────────────────────────────
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out",
                        "ff.net.0.proj", "ff.net.2"],
    )
    peft_model = get_peft_model(unet, lora_config)
    n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    n_total_p = sum(p.numel() for p in peft_model.parameters())
    print(f"[lora] trainable = {n_train:,}  ({100*n_train/n_total_p:.2f}%)")

    # 4) Optimizer ─────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )

    def lr_lambda(step: int) -> float:
        warmup = max(1, STEPS // 20)
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, STEPS - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    vae = pipe.vae
    vae.requires_grad_(False)
    vae.eval()

    # 预计算 VAE scaling factor（SDXL 用 0.13025）
    vae_scale = getattr(vae.config, "scaling_factor", 0.13025)

    # 5) 训练循环 ─────────────────────────────────────────────────────
    print(f"\n[train] SDXL inpainting gate: {STEPS} steps")
    print(f"[train] prompt: {VISION_PROMPT[:80]}...")

    # 预编码文本 — SDXL 需要两个 text encoder + pooled + time_ids
    text_input = pipe.tokenizer_2(
        VISION_PROMPT, padding="max_length", max_length=pipe.tokenizer_2.model_max_length,
        truncation=True, return_tensors="pt",
    )
    text_input2 = pipe.tokenizer(
        VISION_PROMPT, padding="max_length", max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        enc2_out = pipe.text_encoder_2(
            text_input.input_ids.to(device), output_hidden_states=True
        )
        prompt_embeds_2 = enc2_out.hidden_states[-2]  # [B, 77, 1280]
        pooled = enc2_out.text_embeds  # [B, 1280] — CLIPTextModelWithProjection 的投影输出
        prompt_embeds_1 = pipe.text_encoder(
            text_input2.input_ids.to(device), output_hidden_states=True
        ).hidden_states[-2]  # [B, 77, 768]
        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)  # [B, 77, 2048]

    # SDXL micro-conditioning: time_ids 格式 [6]
    # 包含: original_size(2), crops_coords_top_left(2), target_size(2)
    # inpainting pipeline 需要额外 aesthetic_score + negative_* 参数
    time_ids = torch.tensor([
        RES, RES,    # original_size
        0, 0,        # crops_coords_top_left
        RES, RES,    # target_size
    ], device=device, dtype=DTYPE).unsqueeze(0)  # [1, 6]

    added_cond_kwargs = {
        "text_embeds": pooled.to(device, DTYPE),
        "time_ids": time_ids,
    }
    print(f"[prompt] embeds: {prompt_embeds.shape}  pooled: {pooled.shape}  time_ids: {time_ids.shape}")

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
        target_rgb = batch["target_rgb"][0:1].to(device, DTYPE)  # [0, 1] 或 [-1, 1]

        with torch.no_grad():
            # 编码目标 depth RGB → latent
            z0 = vae.encode(target_rgb).latent_dist.mode() * vae_scale
            # 编码输入 photo → masked_image_latent
            masked_image_latent = vae.encode(image).latent_dist.mode() * vae_scale
            # 全遮罩 mask（整张图都要重绘）
            mask = torch.ones(
                (1, 1, z0.shape[-2] * 8, z0.shape[-1] * 8),
                device=device, dtype=DTYPE
            )
            mask = F.interpolate(mask, size=(z0.shape[-2], z0.shape[-1]))

        # 噪声
        noise = torch.randn_like(z0)
        timesteps = torch.randint(
            0, pipe.scheduler.config.num_train_timesteps, (1,), device=device
        ).long()

        # 噪声目标 latent
        noisy_z0 = pipe.scheduler.add_noise(z0, noise, timesteps)

        # 模型输入: [noisy_z0, masked_image_latent, mask] → 9ch
        model_input = torch.cat([noisy_z0, mask, masked_image_latent], dim=1)

        # 预测噪声 — SDXL 需要 added_cond_kwargs
        pred = peft_model(
            model_input,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        loss = F.mse_loss(pred, noise)
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

    if not losses:
        raise RuntimeError("No finite training steps")

    k = max(1, len(losses) // 3)
    avg_first = sum(losses[:k]) / k
    avg_last = sum(losses[-k:]) / k
    print(f"\n[train] avg(first 1/3)={avg_first:.4f}  avg(last 1/3)={avg_last:.4f}  "
          f"({'DECREASING' if avg_last < avg_first else 'NOT decreasing'})")

    ckpt_dir = OUT / "lora_final"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(ckpt_dir))
    print(f"[ckpt] saved → {ckpt_dir}")

    # 6) 推理评估 ─────────────────────────────────────────────────────
    print("\n[eval] inpainting inference on training samples...")
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

        out = pipe(
            prompt=VISION_PROMPT,
            image=pil_in,
            mask_image=Image.new("L", pil_in.size, 255),  # 全遮罩
            num_inference_steps=8,
            strength=1.0,
            guidance_scale=0.0,
            height=RES,
            width=RES,
        )
        pred_rgb = out.images[0]
        pred_rgb = torch.from_numpy(np.array(pred_rgb)).permute(2, 0, 1).float() / 255.0
        # resize 到训练分辨率
        pred_rgb = F.interpolate(
            pred_rgb.unsqueeze(0), size=(RES, RES), mode="bilinear", align_corners=False
        ).squeeze(0)

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
        depth_metrics = evaluate_relative_depth_rgb(
            pred_rgb.unsqueeze(0), tgt_rgb.unsqueeze(0), colormap="viridis"
        )
        return {
            "idx": sample_idx,
            "rgb_mae": rgb_mae,
            "depth_mae": depth_metrics.get("mae", float("nan")),
            "pearson": depth_metrics.get("pearson", float("nan")),
        }

    for idx in range(min(NSAMPLES, 2)):
        r = infer(idx)
        results.append(r)
        print(f"[eval] idx={idx:>3}  rgb_mae={r['rgb_mae']:.4f}  "
              f"depth_mae={r['depth_mae']:.4f}  pearson={r['pearson']:.4f}")

    avg_mae = sum(r["rgb_mae"] for r in results) / len(results)
    avg_pearson = sum(r["pearson"] for r in results if not math.isnan(r["pearson"])) / max(1, len(results))
    print(f"\n[gate] avg rgb_mae={avg_mae:.4f}  avg pearson={avg_pearson:.4f}")
    gate_passed = avg_mae < 0.15 and avg_pearson > 0.7
    print(f"[gate] {'PASSED' if gate_passed else 'NOT PASSED'}  "
          f"(target: rgb_mae<0.15, pearson>0.7)")

    (OUT / "gate_result.json").write_text(json.dumps({
        "model": MODEL_ID, "device": device, "dtype": str(DTYPE),
        "steps": STEPS, "nsamples": NSAMPLES, "res": RES,
        "rank": RANK, "alpha": ALPHA,
        "loss_first_third": avg_first, "loss_last_third": avg_last,
        "avg_rgb_mae": avg_mae, "avg_pearson": avg_pearson,
        "gate_passed": gate_passed, "per_sample": results,
    }, indent=2))
    print(f"[out] → {OUT / 'gate_result.json'}")

    total_min = (time.time() - t_train) / 60
    print(f"\n[done] SDXL inpainting gate complete ({total_min:.1f} min)  "
          f"{'✓ PASSED' if gate_passed else '✗ not passed'}")


if __name__ == "__main__":
    main()
