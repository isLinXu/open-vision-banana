#!/usr/bin/env python3
"""Janus-1.3B 多任务 open-vision-banana 训练.

任务：depth/normal/instance/referring/keypoint + generation
混合比例：~10% vision + 90% generation（对齐论文低比例）

用法:
    PYTHONPATH=src:/tmp/Janus python scripts/janus_multitask.py
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
VISION_PER_TASK = int(os.environ.get("VB_VISION_PER_TASK", "8"))
GEN_SAMPLES = int(os.environ.get("VB_GEN_SAMPLES", "32"))
STEPS = int(os.environ.get("VB_STEPS", "2000"))
RES = int(os.environ.get("VB_RES", "384"))
BATCH = int(os.environ.get("VB_BATCH", "1"))
LR = float(os.environ.get("VB_LR", "1e-4"))
RANK = int(os.environ.get("VB_RANK", "64"))
ALPHA = int(os.environ.get("VB_ALPHA", "128"))
VISION_RATIO = float(os.environ.get("VB_VISION_RATIO", "0.15"))
OUT = Path(os.environ.get("VB_OUT", str(PROJECT_ROOT / "output" / "janus_multitask")))
OUT.mkdir(parents=True, exist_ok=True)

DTYPE = torch.float32
IMAGE_TOKEN_NUM = 576

TASK_DEFAULT_PROMPTS = {
    "depth": "Generate a metric depth visualization of this image.",
    "normal": "Generate a surface normal estimation visualization of this image.",
    "instance": "Generate an instance segmentation visualization of this image.",
    "referring": "Generate a referring expression segmentation visualization of this image.",
    "keypoint": "Generate a keypoint detection visualization of this image.",
    "generation": "Generate an image based on the given text description.",
}


def _resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_task_dataset(task: str, n_samples: int, resolution: int) -> VisionTaskDataset:
    data_path = Path(DATA_ROOT) / task
    if not data_path.exists():
        raise FileNotFoundError(f"Task data not found: {data_path}")
    ds = VisionTaskDataset(data_path=str(data_path), resolution=resolution, task=task)
    if len(ds) < n_samples:
        raise RuntimeError(f"Task {task}: need {n_samples} samples, have {len(ds)}")
    return ds


def main() -> None:
    torch.manual_seed(42)
    device = _resolve_device()

    print(f"[cfg] model={MODEL_ID}  device={device}  dtype={DTYPE}")
    print(f"[cfg] tasks={TASKS}  vision_per_task={VISION_PER_TASK}  gen_samples={GEN_SAMPLES}")
    print(f"[cfg] steps={STEPS}  res={RES}  batch={BATCH}  lr={LR}  rank={RANK}  alpha={ALPHA}")
    print(f"[cfg] vision_ratio={VISION_RATIO}  out={OUT}")

    # 1) 加载多任务数据集 ──────────────────────────────────────────────
    task_datasets = {}
    task_loaders = {}
    total_vision = 0

    for task in TASKS:
        print(f"[data] loading {task}...")
        ds = _load_task_dataset(task, VISION_PER_TASK, RES)
        task_datasets[task] = ds
        subset = torch.utils.data.Subset(ds, list(range(VISION_PER_TASK)))
        task_loaders[task] = torch.utils.data.DataLoader(
            subset, batch_size=BATCH, shuffle=True, num_workers=0,
            pin_memory=False, collate_fn=_collate_fn, drop_last=True,
        )
        total_vision += VISION_PER_TASK
        print(f"[data] {task}: {len(ds)} total, using {VISION_PER_TASK}")

    # 加载 generation 数据
    gen_path = Path(DATA_ROOT) / "generation"
    gen_images = sorted((gen_path / "images").glob("*.jpg"))[:GEN_SAMPLES]
    gen_prompts = {}
    for img_p in gen_images:
        p = gen_path / "prompts" / f"{img_p.stem}.txt"
        if p.exists():
            gen_prompts[img_p.stem] = p.read_text().strip()

    class GenDataset(torch.utils.data.Dataset):
        def __init__(self, images, prompts, resolution):
            self.images = images
            self.prompts = prompts
            self.resolution = resolution

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_p = self.images[idx]
            img = Image.open(img_p).convert("RGB").resize(
                (self.resolution, self.resolution), Image.LANCZOS
            )
            img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            prompt = self.prompts.get(img_p.stem, TASK_DEFAULT_PROMPTS["generation"])
            return {"image": None, "target_rgb": img_t, "prompt": prompt}

    gen_ds = GenDataset(gen_images, gen_prompts, RES)
    gen_loader = torch.utils.data.DataLoader(
        gen_ds, batch_size=BATCH, shuffle=True, num_workers=0,
        pin_memory=False, collate_fn=_collate_fn, drop_last=True,
    )
    print(f"[data] generation: {len(gen_images)} images loaded")
    print(f"[data] total vision={total_vision} + gen={len(gen_images)} = {total_vision + len(gen_images)}")

    # 2) 加载 Janus-1.3B ───────────────────────────────────────────────
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
    print(f"\n[train] Janus multitask: {STEPS} steps")
    print(f"[train] vision_ratio={VISION_RATIO}")

    task_iters = {task: iter(loader) for task, loader in task_loaders.items()}
    gen_iter = iter(gen_loader)

    losses: list[float] = []
    t_train = time.time()
    step = 0

    while step < STEPS:
        step += 1

        # 按比例选择任务
        is_vision = torch.rand(1).item() < VISION_RATIO
        if is_vision:
            task = TASKS[torch.randint(0, len(TASKS), (1,)).item()]
            loader_iter = task_iters[task]
            try:
                batch = next(loader_iter)
            except StopIteration:
                task_iters[task] = iter(task_loaders[task])
                batch = next(task_iters[task])
        else:
            try:
                batch = next(gen_iter)
            except StopIteration:
                gen_iter = iter(gen_loader)
                batch = next(gen_iter)
            batch["image"] = None
            task = "generation"

        opt.zero_grad(set_to_none=True)

        # 取 batch（batch_size=1 简化）
        image = batch["image"][0:1].to(device, DTYPE) if batch["image"] is not None else None
        target_rgb = batch["target_rgb"][0:1].to(device, DTYPE)  # [0, 1]
        prompt = batch["prompt"][0] if isinstance(batch["prompt"], list) else batch["prompt"]

        # 1. 图像理解路径
        if image is not None:
            pixel_values = image.unsqueeze(1)
        else:
            # generation 模式：没有输入图像，用零填充
            pixel_values = torch.zeros(1, 1, 3, RES, RES, device=device, dtype=DTYPE)

        # 2. 目标图像 → VQ tokens
        with torch.no_grad():
            target_for_vq = target_rgb * 2 - 1
            quant, emb_loss, info = model.gen_vision_model.encode(target_for_vq)
            target_tokens = info[2]  # [576]

        # 3. 构造输入 embeds
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_len = prompt_ids.shape[1]
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
        hidden_states = outputs.last_hidden_state

        # 5. 生成图像 token logits
        gen_hidden = hidden_states[:, prompt_len:, :]
        gen_logits = model.gen_head(gen_hidden)

        # 6. 损失
        target_tokens_batch = target_tokens.unsqueeze(0)
        loss = F.cross_entropy(
            gen_logits.reshape(-1, gen_logits.shape[-1]),
            target_tokens_batch.reshape(-1),
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
                  f"task={task}  mode={'vis' if is_vision else 'gen'}  "
                  f"({(time.time()-t_train)/step:.2f}s/step)", flush=True)
        if step % 500 == 0:
            mid_ckpt = OUT / f"lora_step{step}"
            mid_ckpt.mkdir(parents=True, exist_ok=True)
            peft_model.save_pretrained(str(mid_ckpt))
            print(f"[ckpt] mid-training save → {mid_ckpt}", flush=True)

    if not losses:
        raise RuntimeError("No finite training steps")

    print(f"\n[train] completed {len(losses)} steps")
    print(f"[train] avg loss = {sum(losses)/len(losses):.4f}")

    # 保存 checkpoint
    ckpt_dir = OUT / "lora_final"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(ckpt_dir))
    print(f"[ckpt] saved → {ckpt_dir}")

    total_min = (time.time() - t_train) / 60
    print(f"\n[done] Janus multitask complete ({total_min:.1f} min)")


if __name__ == "__main__":
    main()
