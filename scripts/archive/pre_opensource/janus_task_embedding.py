#!/usr/bin/env python3
"""Janus-1.3B task embedding —— 学习任务特定向量注入模型.

改进：
- 为每个任务学习一个 task embedding 向量
- 注入到 LLM 的输入 embeds 中
- 帮助模型区分不同任务

用法:
    PYTHONPATH=src:/tmp/Janus python scripts/janus_task_embedding.py
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
TASKS = os.environ.get("VB_TASKS", "depth,normal,instance,referring,keypoint").split(",")
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
OUT = Path(os.environ.get("VB_OUT", str(PROJECT_ROOT / "output" / "janus_task_embedding")))
OUT.mkdir(parents=True, exist_ok=True)

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


class TaskEmbedding(nn.Module):
    """Task embedding：为每个任务学习一个向量."""

    def __init__(self, num_tasks: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_tasks, hidden_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(task_ids)


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
    print(f"[cfg] tasks={TASKS}  nsamples={NSAMPLES}  val={VAL_SAMPLES}")
    print(f"[cfg] steps={STEPS}  res={RES}  batch={BATCH}  lr={LR}  rank={RANK}  alpha={ALPHA}")
    print(f"[cfg] dropout={DROPOUT}  wd={WEIGHT_DECAY}  patience={PATIENCE}  out={OUT}")

    # 1) 加载多任务数据集 ──────────────────────────────────────────────
    task_datasets = {}
    task_loaders = {}
    task_to_id = {task: i for i, task in enumerate(TASKS)}

    for task in TASKS:
        data_path = Path(DATA_ROOT) / task
        if not data_path.exists():
            raise FileNotFoundError(f"Task data not found: {data_path}")
        ds = VisionTaskDataset(data_path=str(data_path), resolution=RES, task=task)
        task_datasets[task] = ds
        train_idx = list(range(NSAMPLES))
        val_idx = list(range(NSAMPLES, NSAMPLES + VAL_SAMPLES))
        train_subset = torch.utils.data.Subset(ds, train_idx)
        val_subset = torch.utils.data.Subset(ds, val_idx)
        task_loaders[task] = {
            "train": torch.utils.data.DataLoader(
                train_subset, batch_size=BATCH, shuffle=True, num_workers=0,
                pin_memory=False, collate_fn=_collate_fn, drop_last=True,
            ),
            "val": torch.utils.data.DataLoader(
                val_subset, batch_size=BATCH, shuffle=False, num_workers=0,
                pin_memory=False, collate_fn=_collate_fn, drop_last=True,
            ),
        }
        print(f"[data] {task}: {len(ds)} total, train={len(train_idx)}, val={len(val_idx)}")

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

    # 4) RGB Head + Task Embedding ───────────────────────────────────────
    hidden_dim = model.config.language_config.hidden_size
    rgb_head = RGBHead(hidden_dim, IMAGE_TOKEN_NUM, RES, dropout=DROPOUT).to(device, DTYPE)
    task_embedding = TaskEmbedding(len(TASKS), hidden_dim).to(device, DTYPE)
    print(f"[rgb_head] params={sum(p.numel() for p in rgb_head.parameters()):,}")
    print(f"[task_embedding] params={sum(p.numel() for p in task_embedding.parameters()):,}")

    # 5) Optimizer ─────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad] +
        [p for p in rgb_head.parameters()] +
        [p for p in task_embedding.parameters()],
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
    @torch.no_grad()
    def run_validation(task: str) -> float:
        peft_model.eval()
        rgb_head.eval()
        task_embedding.eval()
        val_losses = []
        task_id = torch.tensor([task_to_id[task]], device=device)
        task_emb = task_embedding(task_id)  # [1, D]

        for batch in task_loaders[task]["val"]:
            image = batch["image"][0:1].to(device, DTYPE)
            target_rgb = batch["target_rgb"][0:1].to(device, DTYPE)
            prompt = TASK_PROMPTS.get(task, "Generate a visualization of this image.")
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            prompt_len = prompt_ids.shape[1]

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

            # 注入 task embedding
            # input_embeds: [1, seq_len, D], task_emb: [1, D]
            # 将 task_emb 加到所有位置（广播）
            input_embeds = input_embeds + task_emb.unsqueeze(1)  # [1, seq_len, D]

            outputs = peft_model.model(
                inputs_embeds=input_embeds,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]
            gen_hidden = hidden_states[:, prompt_len:, :]
            pred_rgb = rgb_head(gen_hidden)
            loss = F.mse_loss(pred_rgb, target_rgb)
            val_losses.append(loss.item())

        peft_model.train()
        rgb_head.train()
        task_embedding.train()
        return sum(val_losses) / len(val_losses)

    # 7) 训练循环 ─────────────────────────────────────────────────────
    print(f"\n[train] Janus task embedding: {STEPS} steps")

    losses: list[float] = []
    val_losses: dict[str, list[float]] = {task: [] for task in TASKS}
    best_val_loss = float('inf')
    patience_counter = 0
    task_iters = {task: iter(task_loaders[task]["train"]) for task in TASKS}
    t_train = time.time()

    for step in range(1, STEPS + 1):
        # 随机选择任务
        task = TASKS[torch.randint(0, len(TASKS), (1,)).item()]
        try:
            batch = next(task_iters[task])
        except StopIteration:
            task_iters[task] = iter(task_loaders[task]["train"])
            batch = next(task_iters[task])

        opt.zero_grad(set_to_none=True)

        image = batch["image"][0:1].to(device, DTYPE)
        target_rgb = batch["target_rgb"][0:1].to(device, DTYPE)
        prompt = TASK_PROMPTS.get(task, "Generate a visualization of this image.")
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_len = prompt_ids.shape[1]
        task_id = torch.tensor([task_to_id[task]], device=device)
        task_emb = task_embedding(task_id)  # [1, D]

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

        # 注入 task embedding
        input_embeds = input_embeds + task_emb.unsqueeze(1)  # [1, seq_len, D]

        outputs = peft_model.model(
            inputs_embeds=input_embeds,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        gen_hidden = hidden_states[:, prompt_len:, :]
        pred_rgb = rgb_head(gen_hidden)
        loss = F.mse_loss(pred_rgb, target_rgb)

        if not torch.isfinite(loss).all():
            print(f"[train] step {step:>4}: NON-FINITE loss — skip", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(peft_model.parameters()) +
            list(rgb_head.parameters()) +
            list(task_embedding.parameters()), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())

        if step % 30 == 0 or step == 1:
            recent = sum(losses[-30:]) / len(losses[-30:])
            print(f"[train] step {step:>4}/{STEPS}  loss={loss.item():.4f}  "
                  f"avg30={recent:.4f}  lr={sched.get_last_lr()[0]:.2e}  "
                  f"task={task}  ({(time.time()-t_train)/step:.2f}s/step)", flush=True)

        if step % VAL_EVERY == 0:
            # 验证所有任务
            total_val_loss = 0.0
            for task in TASKS:
                val_loss = run_validation(task)
                val_losses[task].append(val_loss)
                total_val_loss += val_loss
            avg_val_loss = total_val_loss / len(TASKS)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_ckpt = OUT / "ckpt_best"
                best_ckpt.mkdir(parents=True, exist_ok=True)
                peft_model.save_pretrained(str(best_ckpt / "lora"))
                torch.save(rgb_head.state_dict(), best_ckpt / "rgb_head.pt")
                torch.save(task_embedding.state_dict(), best_ckpt / "task_embedding.pt")
                marker = "  *BEST*"
            else:
                patience_counter += 1
                marker = ""
            print(f"[val] step {step:>4}  avg_val_loss={avg_val_loss:.4f}  best={best_val_loss:.4f}  "
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
    final_ckpt = OUT / "ckpt_final"
    final_ckpt.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(final_ckpt / "lora"))
    torch.save(rgb_head.state_dict(), final_ckpt / "rgb_head.pt")
    torch.save(task_embedding.state_dict(), final_ckpt / "task_embedding.pt")
    print(f"[ckpt] saved → {final_ckpt}")

    # 8) 推理评估 ─────────────────────────────────────────────────────
    print("\n[eval] inference on all tasks...")
    mapper = DepthRGBMapper(colormap="viridis")
    all_results = {}

    @torch.no_grad()
    def infer_task(task: str, sample_idx: int) -> dict:
        ds = task_datasets[task]
        s = ds[sample_idx]
        inp_tensor = s["image"]  # [-1, 1]
        tgt_rgb = s["target_rgb"]  # [0, 1]
        prompt = TASK_PROMPTS.get(task, "Generate a visualization of this image.")
        pil_in = Image.fromarray(
            (inp_tensor.permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5)
            .mul(255).byte().numpy()
        )

        task_id = torch.tensor([task_to_id[task]], device=device)
        task_emb = task_embedding(task_id)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_len = prompt_ids.shape[1]

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
        input_embeds = input_embeds + task_emb.unsqueeze(1)

        outputs = peft_model.model(
            inputs_embeds=input_embeds,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
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
        panel.save(OUT / f"panel_{task}_{sample_idx:02d}.png")

        rgb_mae = (pred_rgb.float() - tgt_rgb.float()).abs().mean().item()
        return {"idx": sample_idx, "rgb_mae": rgb_mae}

    for task in TASKS:
        print(f"\n[eval] task={task}:")
        task_results = []
        for idx in range(min(NSAMPLES, 2)):
            r = infer_task(task, idx)
            task_results.append(r)
            print(f"  idx={idx:>3}  rgb_mae={r['rgb_mae']:.4f}")
        avg_mae = sum(r["rgb_mae"] for r in task_results) / len(task_results)
        all_results[task] = {"avg_rgb_mae": avg_mae, "samples": task_results}
        print(f"  → avg rgb_mae={avg_mae:.4f}")

    overall_mae = sum(r["avg_rgb_mae"] for r in all_results.values()) / len(all_results)
    print(f"\n[bench] overall avg rgb_mae={overall_mae:.4f}")

    total_min = (time.time() - t_train) / 60
    print(f"\n[done] Janus task embedding complete ({total_min:.1f} min)")
    print(f"[done] best val_loss={best_val_loss:.4f}")

    # 保存结果
    (OUT / "task_embedding_result.json").write_text(json.dumps({
        "model": MODEL_ID, "device": device,
        "tasks": TASKS, "nsamples": NSAMPLES, "val_samples": VAL_SAMPLES,
        "best_val_loss": best_val_loss,
        "per_task": all_results, "overall_rgb_mae": overall_mae,
    }, indent=2))
    print(f"[out] → {OUT / 'task_embedding_result.json'}")


if __name__ == "__main__":
    main()
