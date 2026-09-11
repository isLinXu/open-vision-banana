#!/usr/bin/env python3
"""Janus-1.3B task-specific 训练 —— 每任务单独训练，然后合并.

任务：depth/normal/instance/referring/keypoint
每任务单独训练，使用 continuous RGB 输出

用法:
    PYTHONPATH=src:/tmp/Janus python scripts/janus_task_specific.py --task depth
    PYTHONPATH=src:/tmp/Janus python scripts/janus_task_specific.py --task normal
    ...
"""
from __future__ import annotations

import argparse
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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, "/tmp/Janus")

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.training.tuner import VisionTaskDataset, _collate_fn

# ── Config ───────────────────────────────────────────────────────────────
MODEL_ID = os.environ.get("VB_MODEL", "deepseek-ai/Janus-1.3B")
DATA_ROOT = os.environ.get(
    "VB_DATA_ROOT",
    "/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v3",
)
STEPS = int(os.environ.get("VB_STEPS", "1000"))
NSAMPLES = int(os.environ.get("VB_NSAMPLES", "100"))
VAL_SAMPLES = int(os.environ.get("VB_VAL_SAMPLES", "10"))
RES = int(os.environ.get("VB_RES", "384"))
BATCH = int(os.environ.get("VB_BATCH", "1"))
LR = float(os.environ.get("VB_LR", "1e-4"))
RANK = int(os.environ.get("VB_RANK", "64"))
ALPHA = int(os.environ.get("VB_ALPHA", "128"))
DROPOUT = float(os.environ.get("VB_DROPOUT", "0.1"))
WEIGHT_DECAY = float(os.environ.get("VB_WEIGHT_DECAY", "0.01"))
PATIENCE = int(os.environ.get("VB_PATIENCE", "10"))
VAL_EVERY = int(os.environ.get("VB_VAL_EVERY", "50"))

DTYPE = torch.float32
IMAGE_TOKEN_NUM = 576

TASK_PROMPTS = {
    "depth": "Generate a metric depth visualization of this image.",
    "normal": "Generate a surface normal estimation visualization of this image.",
    "instance": "Generate an instance segmentation visualization of this image.",
    "referring": "Generate a referring expression segmentation visualization of this image.",
    "keypoint": "Generate a keypoint detection visualization of this image.",
}


class RGBHead(nn.Module):
    """RGB 预测头：LLM hidden states → RGB 值（带 dropout）."""

    def __init__(self, hidden_dim: int, image_token_num: int, image_size: int, dropout: float = 0.1):
        super().__init__()
        self.image_token_num = image_token_num
        self.image_size = image_size
        self.patch_size = image_size // int(math.sqrt(image_token_num))

        self.proj = nn.Linear(hidden_dim, self.patch_size * self.patch_size * 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        b = hidden_states.shape[0]
        x = self.dropout(hidden_states)
        x = self.proj(x)
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


def train_task(task: str, out_dir: Path) -> dict:
    """训练单个任务."""
    device = _resolve_device()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[task] {task}")
    print(f"{'='*60}")

    # 1) 数据集 ───────────────────────────────────────────────────────
    data_path = Path(DATA_ROOT) / task
    if not data_path.exists():
        raise FileNotFoundError(f"Task data not found: {data_path}")
    ds = VisionTaskDataset(data_path=str(data_path), resolution=RES, task=task)
    n_total = len(ds)
    print(f"[data] loaded {n_total} {task} samples @ {RES}px")
    if n_total < NSAMPLES + VAL_SAMPLES:
        raise RuntimeError(f"Need {NSAMPLES + VAL_SAMPLES} samples, have {n_total}")

    train_idx = list(range(NSAMPLES))
    val_idx = list(range(NSAMPLES, NSAMPLES + VAL_SAMPLES))
    train_subset = torch.utils.data.Subset(ds, train_idx)
    val_subset = torch.utils.data.Subset(ds, val_idx)
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=BATCH, shuffle=True, num_workers=0,
        pin_memory=False, collate_fn=_collate_fn, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=BATCH, shuffle=False, num_workers=0,
        pin_memory=False, collate_fn=_collate_fn, drop_last=True,
    )
    print(f"[data] train={len(train_idx)}  val={len(val_idx)}")

    # 2) 加载模型 ───────────────────────────────────────────────────────
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

    # 3) LoRA ──────────────────────────────────────────────────────────
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    peft_model = get_peft_model(model.language_model, lora_config)
    n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"[lora] trainable = {n_train:,}")

    # 冻结 vision 和 generation 部分
    model.vision_model.requires_grad_(False)
    model.gen_vision_model.requires_grad_(False)
    model.aligner.requires_grad_(False)
    model.gen_aligner.requires_grad_(False)
    model.gen_head.requires_grad_(False)
    model.gen_embed.requires_grad_(False)

    # 4) RGB Head ──────────────────────────────────────────────────────
    hidden_dim = model.config.language_config.hidden_size
    rgb_head = RGBHead(hidden_dim, IMAGE_TOKEN_NUM, RES, dropout=DROPOUT).to(device, DTYPE)

    # 5) Optimizer ─────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad] +
        [p for p in rgb_head.parameters()],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )

    def lr_lambda(step: int) -> float:
        warmup = max(1, STEPS // 30)
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, STEPS - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # 6) 验证函数 ─────────────────────────────────────────────────────
    prompt = TASK_PROMPTS.get(task, "Generate a visualization of this image.")
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    @torch.no_grad()
    def run_validation() -> float:
        peft_model.eval()
        rgb_head.eval()
        val_losses = []
        for batch in val_loader:
            image = batch["image"][0:1].to(device, DTYPE)
            target_rgb = batch["target_rgb"][0:1].to(device, DTYPE)

            pixel_values = image.unsqueeze(1)
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
            gen_hidden = hidden_states[:, prompt_len:, :]
            pred_rgb = rgb_head(gen_hidden)
            loss = F.mse_loss(pred_rgb, target_rgb)
            val_losses.append(loss.item())

        peft_model.train()
        rgb_head.train()
        return sum(val_losses) / len(val_losses)

    # 7) 训练循环 ─────────────────────────────────────────────────────
    print(f"\n[train] {task}: {STEPS} steps")

    losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float('inf')
    patience_counter = 0
    it = iter(train_loader)
    t_train = time.time()

    for step in range(1, STEPS + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        opt.zero_grad(set_to_none=True)

        image = batch["image"][0:1].to(device, DTYPE)
        target_rgb = batch["target_rgb"][0:1].to(device, DTYPE)

        pixel_values = image.unsqueeze(1)
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
        gen_hidden = hidden_states[:, prompt_len:, :]
        pred_rgb = rgb_head(gen_hidden)
        loss = F.mse_loss(pred_rgb, target_rgb)

        if not torch.isfinite(loss).all():
            print(f"[train] step {step:>4}: NON-FINITE loss — skip", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(peft_model.parameters()) + list(rgb_head.parameters()), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())

        if step % 30 == 0 or step == 1:
            recent = sum(losses[-30:]) / len(losses[-30:])
            print(f"[train] step {step:>4}/{STEPS}  loss={loss.item():.4f}  "
                  f"avg30={recent:.4f}  lr={sched.get_last_lr()[0]:.2e}  "
                  f"({(time.time()-t_train)/step:.2f}s/step)", flush=True)

        if step % VAL_EVERY == 0:
            val_loss = run_validation()
            val_losses.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_ckpt = out_dir / "ckpt_best"
                best_ckpt.mkdir(parents=True, exist_ok=True)
                peft_model.save_pretrained(str(best_ckpt / "lora"))
                torch.save(rgb_head.state_dict(), best_ckpt / "rgb_head.pt")
                marker = "  *BEST*"
            else:
                patience_counter += 1
                marker = ""
            print(f"[val] step {step:>4}  val_loss={val_loss:.4f}  best={best_val_loss:.4f}  "
                  f"patience={patience_counter}{marker}", flush=True)
            if patience_counter >= PATIENCE:
                print(f"[early] no val improvement for {PATIENCE} evals — "
                      f"early stop at step {step} (best val_loss={best_val_loss:.4f})")
                break

    if not losses:
        raise RuntimeError("No finite training steps")

    k = max(1, len(losses) // 3)
    avg_first = sum(losses[:k]) / k
    avg_last = sum(losses[-k:]) / k
    print(f"\n[train] avg(first 1/3)={avg_first:.4f}  avg(last 1/3)={avg_last:.4f}  "
          f"({'DECREASING' if avg_last < avg_first else 'NOT decreasing'})")

    # 保存最终 checkpoint
    final_ckpt = out_dir / "ckpt_final"
    final_ckpt.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(final_ckpt / "lora"))
    torch.save(rgb_head.state_dict(), final_ckpt / "rgb_head.pt")
    print(f"[ckpt] saved → {final_ckpt}")

    # 8) 推理评估 ─────────────────────────────────────────────────────
    print(f"\n[eval] inference on training samples...")
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
        gen_hidden = hidden_states[:, prompt_len:, :]
        pred_rgb = rgb_head(gen_hidden).squeeze(0).clamp(0, 1).cpu()

        panel = Image.new("RGB", (RES * 3, RES))
        panel.paste(pil_in.resize((RES, RES)), (0, 0))
        panel.paste(Image.fromarray(
            (pred_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
        ).resize((RES, RES)), (RES, 0))
        panel.paste(Image.fromarray(
            (tgt_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
        ).resize((RES, RES)), (RES * 2, 0))
        panel.save(out_dir / f"panel_{sample_idx:02d}.png")

        rgb_mae = (pred_rgb.float() - tgt_rgb.float()).abs().mean().item()
        return {"idx": sample_idx, "rgb_mae": rgb_mae}

    for idx in range(min(NSAMPLES, 2)):
        r = infer(idx)
        results.append(r)
        print(f"[eval] idx={idx:>3}  rgb_mae={r['rgb_mae']:.4f}")

    avg_mae = sum(r["rgb_mae"] for r in results) / len(results)
    print(f"\n[gate] avg rgb_mae={avg_mae:.4f}")

    total_min = (time.time() - t_train) / 60
    print(f"\n[done] {task} complete ({total_min:.1f} min)  best val_loss={best_val_loss:.4f}")

    return {
        "task": task,
        "avg_rgb_mae": avg_mae,
        "best_val_loss": best_val_loss,
        "steps": len(losses),
        "time_min": total_min,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True,
                        choices=["depth", "normal", "instance", "referring", "keypoint"],
                        help="Task to train")
    args = parser.parse_args()

    out_dir = Path(os.environ.get("VB_OUT", str(PROJECT_ROOT / "output" / "janus_task_specific"))) / args.task
    result = train_task(args.task, out_dir)

    # 保存结果
    (out_dir / "task_result.json").write_text(json.dumps(result, indent=2))
    print(f"\n[out] → {out_dir / 'task_result.json'}")


if __name__ == "__main__":
    main()
