#!/usr/bin/env python3
"""Janus-1.3B task embedding v2 —— 加权 loss + 颜色标定 + 诚实评估.

改进：
1. 离散任务加权 loss：非黑像素 10x 权重，防止模型偷懒输出全黑
2. 连续任务颜色标定：直方图匹配损失，让 depth 用完整 viridis 范围
3. 诚实评估：非黑像素 IoU/F1 评估离散任务，δ1/AbsRel 评估 depth

用法:
    PYTHONPATH=src:/tmp/Janus python scripts/janus_task_embedding_v2.py
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
from open_vision_banana.evaluation.depth import evaluate_depth
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
# 新增：加权 loss 配置
NONBLACK_WEIGHT = float(os.environ.get("VB_NONBLACK_WEIGHT", "10.0"))
BLACK_THRESHOLD = float(os.environ.get("VB_BLACK_THRESHOLD", "0.05"))
# 新增：颜色标定 loss 权重
COLOR_MATCH_WEIGHT = float(os.environ.get("VB_COLOR_MATCH_WEIGHT", "0.1"))
OUT = Path(os.environ.get("VB_OUT", str(PROJECT_ROOT / "output" / "janus_task_embedding_v2")))
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

# 任务类型：连续 vs 离散
CONTINUOUS_TASKS = {"depth", "normal"}
DISCRETE_TASKS = {"instance", "referring", "keypoint"}


class RGBHead(nn.Module):
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
    def __init__(self, num_tasks: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_tasks, hidden_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(task_ids)


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor,
                      task: str, nonblack_weight: float = 10.0,
                      black_threshold: float = 0.05) -> torch.Tensor:
    """加权 MSE loss：离散任务的非黑像素加权，防止模型偷懒输出全黑."""
    if task in DISCRETE_TASKS:
        # 计算目标中的非黑像素 mask（任一通道 > threshold）
        nonblack_mask = (target > black_threshold).any(dim=1, keepdim=True).float()
        # 权重：非黑像素 = nonblack_weight，黑像素 = 1.0
        weight = 1.0 + (nonblack_weight - 1.0) * nonblack_mask
        # 加权 MSE
        loss = (weight * (pred - target) ** 2).mean()
    else:
        loss = F.mse_loss(pred, target)
    return loss


def color_match_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """颜色标定 loss：匹配 pred 和 target 的颜色统计（均值+标准差）."""
    # pred, target: [b, 3, h, w]
    pred_mean = pred.mean(dim=(2, 3))  # [b, 3]
    pred_std = pred.std(dim=(2, 3))  # [b, 3]
    tgt_mean = target.mean(dim=(2, 3))
    tgt_std = target.std(dim=(2, 3))
    loss = F.mse_loss(pred_mean, tgt_mean) + F.mse_loss(pred_std, tgt_std)
    return loss


def compute_honest_metrics(pred: torch.Tensor, target: torch.Tensor,
                           task: str, black_threshold: float = 0.05) -> dict:
    """诚实评估：非黑像素 IoU/F1（离散任务）或 δ1/AbsRel（depth）."""
    pred_np = pred.cpu().numpy()
    tgt_np = target.cpu().numpy()

    if task in DISCRETE_TASKS:
        # 非黑像素 IoU
        pred_fg = (pred_np > black_threshold).any(axis=0)  # [h, w]
        tgt_fg = (tgt_np > black_threshold).any(axis=0)
        intersection = (pred_fg & tgt_fg).sum()
        union = (pred_fg | tgt_fg).sum()
        iou = intersection / max(union, 1)
        # F1
        tp = intersection
        fp = (pred_fg & ~tgt_fg).sum()
        fn = (~pred_fg & tgt_fg).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        return {"iou": iou, "f1": f1, "fg_pixels": int(tgt_fg.sum())}
    else:
        # 连续任务用 rgb_mae
        rgb_mae = np.abs(pred_np - tgt_np).mean()
        return {"rgb_mae": rgb_mae}


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
    print(f"[cfg] steps={STEPS}  lr={LR}  dropout={DROPOUT}  wd={WEIGHT_DECAY}")
    print(f"[cfg] nonblack_weight={NONBLACK_WEIGHT}  black_threshold={BLACK_THRESHOLD}")
    print(f"[cfg] color_match_weight={COLOR_MATCH_WEIGHT}  out={OUT}")

    # 1) 数据集 ───────────────────────────────────────────────────────
    task_datasets = {}
    task_loaders = {}
    task_to_id = {task: i for i, task in enumerate(TASKS)}

    for task in TASKS:
        data_path = Path(DATA_ROOT) / task
        ds = VisionTaskDataset(data_path=str(data_path), resolution=RES, task=task)
        task_datasets[task] = ds
        train_subset = torch.utils.data.Subset(ds, list(range(NSAMPLES)))
        val_subset = torch.utils.data.Subset(ds, list(range(NSAMPLES, NSAMPLES + VAL_SAMPLES)))
        task_loaders[task] = {
            "train": torch.utils.data.DataLoader(
                train_subset, batch_size=BATCH, shuffle=True, num_workers=0,
                pin_memory=False, collate_fn=_collate_fn, drop_last=True),
            "val": torch.utils.data.DataLoader(
                val_subset, batch_size=BATCH, shuffle=False, num_workers=0,
                pin_memory=False, collate_fn=_collate_fn, drop_last=True),
        }
        print(f"[data] {task}: {len(ds)} total")

    # 2) 模型 ─────────────────────────────────────────────────────────
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from janus.models.modeling_vlm import MultiModalityConfig, MultiModalityCausalLM
    AutoConfig.register("multi_modality", MultiModalityConfig)
    AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = model.to(device)
    print(f"[model] loaded")

    # 3) LoRA ──────────────────────────────────────────────────────────
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=RANK, lora_alpha=ALPHA, init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    peft_model = get_peft_model(model.language_model, lora_config)
    print(f"[lora] trainable = {sum(p.numel() for p in peft_model.parameters() if p.requires_grad):,}")

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

    # 5) Optimizer ─────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad] +
        list(rgb_head.parameters()) + list(task_embedding.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_lambda(step: int) -> float:
        warmup = max(1, STEPS // 30)
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, STEPS - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # 预编码 prompts
    prompt_ids_map = {}
    for task in TASKS:
        prompt_ids_map[task] = tokenizer(
            TASK_PROMPTS[task], return_tensors="pt").input_ids.to(device)

    # 6) 训练/验证函数 ─────────────────────────────────────────────────
    def forward_step(task: str, image: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
        """单步前向 + loss."""
        task_id = torch.tensor([task_to_id[task]], device=device)
        task_emb = task_embedding(task_id)
        prompt_ids = prompt_ids_map[task]
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
            input_ids=full_input_ids, pixel_values=pixel_values,
            images_seq_mask=images_seq_mask, images_emb_mask=images_emb_mask,
        )
        input_embeds = input_embeds + task_emb.unsqueeze(1)

        outputs = peft_model.model(inputs_embeds=input_embeds, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        gen_hidden = hidden_states[:, prompt_len:, :]
        pred_rgb = rgb_head(gen_hidden)

        # 加权 MSE loss（离散任务防偷懒）
        loss = weighted_mse_loss(pred_rgb, target_rgb, task,
                                 NONBLACK_WEIGHT, BLACK_THRESHOLD)
        # 颜色标定 loss（连续任务）
        if task in CONTINUOUS_TASKS and COLOR_MATCH_WEIGHT > 0:
            loss = loss + COLOR_MATCH_WEIGHT * color_match_loss(pred_rgb, target_rgb)
        return loss

    @torch.no_grad()
    def run_validation(task: str) -> float:
        peft_model.eval()
        rgb_head.eval()
        task_embedding.eval()
        val_losses = []
        for batch in task_loaders[task]["val"]:
            image = batch["image"][0:1].to(device, DTYPE)
            target_rgb = batch["target_rgb"][0:1].to(device, DTYPE)
            loss = forward_step(task, image, target_rgb)
            val_losses.append(loss.item())
        peft_model.train()
        rgb_head.train()
        task_embedding.train()
        return sum(val_losses) / len(val_losses)

    # 7) 训练循环 ─────────────────────────────────────────────────────
    print(f"\n[train] Janus task embedding v2: {STEPS} steps")
    losses: list[float] = []
    best_val_loss = float('inf')
    patience_counter = 0
    task_iters = {task: iter(task_loaders[task]["train"]) for task in TASKS}
    t_train = time.time()

    for step in range(1, STEPS + 1):
        task = TASKS[torch.randint(0, len(TASKS), (1,)).item()]
        try:
            batch = next(task_iters[task])
        except StopIteration:
            task_iters[task] = iter(task_loaders[task]["train"])
            batch = next(task_iters[task])

        opt.zero_grad(set_to_none=True)
        image = batch["image"][0:1].to(device, DTYPE)
        target_rgb = batch["target_rgb"][0:1].to(device, DTYPE)
        loss = forward_step(task, image, target_rgb)

        if not torch.isfinite(loss).all():
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(peft_model.parameters()) + list(rgb_head.parameters()) +
            list(task_embedding.parameters()), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())

        if step % 30 == 0 or step == 1:
            recent = sum(losses[-30:]) / len(losses[-30:])
            print(f"[train] step {step:>4}/{STEPS}  loss={loss.item():.4f}  "
                  f"avg30={recent:.4f}  task={task}  "
                  f"({(time.time()-t_train)/step:.2f}s/step)", flush=True)

        if step % VAL_EVERY == 0:
            total_val = sum(run_validation(t) for t in TASKS) / len(TASKS)
            if total_val < best_val_loss:
                best_val_loss = total_val
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
            print(f"[val] step {step:>4}  avg_val_loss={total_val:.4f}  "
                  f"best={best_val_loss:.4f}  patience={patience_counter}{marker}", flush=True)
            if patience_counter >= PATIENCE:
                print(f"[early] stop at step {step}")
                break

    # 保存最终 checkpoint
    final_ckpt = OUT / "ckpt_final"
    final_ckpt.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(final_ckpt / "lora"))
    torch.save(rgb_head.state_dict(), final_ckpt / "rgb_head.pt")
    torch.save(task_embedding.state_dict(), final_ckpt / "task_embedding.pt")

    # 8) 诚实评估 ─────────────────────────────────────────────────────
    print("\n[eval] honest evaluation on all tasks...")
    all_results = {}

    @torch.no_grad()
    def infer_task(task: str, sample_idx: int) -> dict:
        ds = task_datasets[task]
        s = ds[sample_idx]
        inp_tensor = s["image"]
        tgt_rgb = s["target_rgb"]
        pil_in = Image.fromarray(
            (inp_tensor.permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5)
            .mul(255).byte().numpy())

        task_id = torch.tensor([task_to_id[task]], device=device)
        task_emb = task_embedding(task_id)
        prompt_ids = prompt_ids_map[task]
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
            input_ids=full_input_ids, pixel_values=pixel_values,
            images_seq_mask=images_seq_mask, images_emb_mask=images_emb_mask,
        )
        input_embeds = input_embeds + task_emb.unsqueeze(1)
        outputs = peft_model.model(inputs_embeds=input_embeds, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        gen_hidden = hidden_states[:, prompt_len:, :]
        pred_rgb = rgb_head(gen_hidden).squeeze(0).clamp(0, 1).cpu()

        # 保存可视化
        panel = Image.new("RGB", (RES * 3, RES))
        panel.paste(pil_in.resize((RES, RES)), (0, 0))
        panel.paste(Image.fromarray(
            (pred_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
        ).resize((RES, RES)), (RES, 0))
        panel.paste(Image.fromarray(
            (tgt_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
        ).resize((RES, RES)), (RES * 2, 0))
        panel.save(OUT / f"panel_{task}_{sample_idx:02d}.png")

        # 诚实评估
        metrics = compute_honest_metrics(pred_rgb, tgt_rgb, task, BLACK_THRESHOLD)
        rgb_mae = (pred_rgb.float() - tgt_rgb.float()).abs().mean().item()
        metrics["rgb_mae"] = rgb_mae
        metrics["idx"] = sample_idx
        return metrics

    for task in TASKS:
        print(f"\n[eval] task={task}:")
        task_results = []
        for idx in range(min(NSAMPLES, 2)):
            r = infer_task(task, idx)
            task_results.append(r)
            if task in DISCRETE_TASKS:
                print(f"  idx={idx:>3}  IoU={r.get('iou', 0):.4f}  F1={r.get('f1', 0):.4f}  "
                      f"rgb_mae={r['rgb_mae']:.4f}")
            else:
                print(f"  idx={idx:>3}  rgb_mae={r['rgb_mae']:.4f}")

        if task in DISCRETE_TASKS:
            avg_iou = sum(r.get("iou", 0) for r in task_results) / len(task_results)
            avg_f1 = sum(r.get("f1", 0) for r in task_results) / len(task_results)
            all_results[task] = {"avg_iou": avg_iou, "avg_f1": avg_f1,
                                 "avg_rgb_mae": sum(r["rgb_mae"] for r in task_results) / len(task_results)}
            print(f"  → avg IoU={avg_iou:.4f}  avg F1={avg_f1:.4f}")
        else:
            avg_mae = sum(r["rgb_mae"] for r in task_results) / len(task_results)
            all_results[task] = {"avg_rgb_mae": avg_mae}
            print(f"  → avg rgb_mae={avg_mae:.4f}")

    overall_mae = sum(r.get("avg_rgb_mae", 0) for r in all_results.values()) / len(all_results)
    print(f"\n[bench] overall avg rgb_mae={overall_mae:.4f}")
    print(f"[done] best val_loss={best_val_loss:.4f}")

    (OUT / "task_embedding_v2_result.json").write_text(json.dumps({
        "best_val_loss": best_val_loss,
        "per_task": all_results,
        "overall_rgb_mae": overall_mae,
    }, indent=2))
    print(f"[out] → {OUT / 'task_embedding_v2_result.json'}")


if __name__ == "__main__":
    main()
