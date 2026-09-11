#!/usr/bin/env python3
"""Janus-1.3B task embedding v3 —— 全量 v3 数据 + 三路划分 + 诚实 held-out 评估.

相对 v2 的改进：
1. 全量数据：论文 5 任务（depth/normal/segmentation/instance/referring）全部样本
2. 三路划分：train / val（早停）/ test（最终诚实评估，训练与早停均不可见）
3. 任务采样按训练集大小加权，缓解小样本任务欠训
4. held-out 真实指标：depth δ1/AbsRel（colormap 解码）、normal 平均角误差、
   离散任务非黑像素 IoU/F1、全部任务 rgb_mae

用法:
    PYTHONPATH=src:/tmp/Janus python scripts/janus_task_embedding_v3.py
    VB_SMOKE=1 ...  # 20 步冒烟
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
from open_vision_banana.core.normal_mapper import NormalRGBMapper
from open_vision_banana.evaluation.depth import evaluate_depth
from open_vision_banana.training.tuner import VisionTaskDataset, _collate_fn

# ── Config ───────────────────────────────────────────────────────────────
MODEL_ID = os.environ.get("VB_MODEL", "deepseek-ai/Janus-1.3B")
DATA_ROOT = os.environ.get(
    "VB_DATA_ROOT",
    "/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v3",
)
TASKS = os.environ.get(
    "VB_TASKS", "depth,normal,segmentation,instance,referring").split(",")
STEPS = int(os.environ.get("VB_STEPS", "6000"))
VAL_SAMPLES = int(os.environ.get("VB_VAL_SAMPLES", "10"))
TEST_SAMPLES = int(os.environ.get("VB_TEST_SAMPLES", "30"))
RES = int(os.environ.get("VB_RES", "384"))
BATCH = int(os.environ.get("VB_BATCH", "1"))
LR = float(os.environ.get("VB_LR", "1e-4"))
RANK = int(os.environ.get("VB_RANK", "64"))
ALPHA = int(os.environ.get("VB_ALPHA", "128"))
DROPOUT = float(os.environ.get("VB_DROPOUT", "0.1"))
WEIGHT_DECAY = float(os.environ.get("VB_WEIGHT_DECAY", "0.01"))
PATIENCE = int(os.environ.get("VB_PATIENCE", "12"))
VAL_EVERY = int(os.environ.get("VB_VAL_EVERY", "100"))
NONBLACK_WEIGHT = float(os.environ.get("VB_NONBLACK_WEIGHT", "10.0"))
BLACK_THRESHOLD = float(os.environ.get("VB_BLACK_THRESHOLD", "0.05"))
COLOR_MATCH_WEIGHT = float(os.environ.get("VB_COLOR_MATCH_WEIGHT", "0.1"))
DEPTH_EVAL_N = int(os.environ.get("VB_DEPTH_EVAL_N", "12"))
SMOKE = os.environ.get("VB_SMOKE", "0") == "1"
OUT = Path(os.environ.get(
    "VB_OUT", str(PROJECT_ROOT / "output" / "janus_task_embedding_v3")))
OUT.mkdir(parents=True, exist_ok=True)

if SMOKE:
    STEPS, VAL_EVERY, PATIENCE = 20, 10, 999
    VAL_SAMPLES, TEST_SAMPLES, DEPTH_EVAL_N = 2, 2, 2

DTYPE = torch.float32
IMAGE_TOKEN_NUM = 576

TASK_PROMPTS = {
    "depth": "Generate a metric depth visualization of this image.",
    "normal": "Generate a surface normal estimation visualization of this image.",
    "segmentation": "Generate a semantic segmentation visualization of this image.",
    "instance": "Generate an instance segmentation visualization of this image.",
    "referring": "Generate a referring expression segmentation visualization of this image.",
    "keypoint": "Generate a keypoint detection visualization of this image.",
}

CONTINUOUS_TASKS = {"depth", "normal"}
DISCRETE_TASKS = {"segmentation", "instance", "referring", "keypoint"}


class RGBHead(nn.Module):
    def __init__(self, hidden_dim: int, image_token_num: int, image_size: int,
                 dropout: float = 0.1):
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
    if task in DISCRETE_TASKS:
        nonblack_mask = (target > black_threshold).any(dim=1, keepdim=True).float()
        weight = 1.0 + (nonblack_weight - 1.0) * nonblack_mask
        loss = (weight * (pred - target) ** 2).mean()
    else:
        loss = F.mse_loss(pred, target)
    return loss


def color_match_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_mean = pred.mean(dim=(2, 3))
    pred_std = pred.std(dim=(2, 3))
    tgt_mean = target.mean(dim=(2, 3))
    tgt_std = target.std(dim=(2, 3))
    return F.mse_loss(pred_mean, tgt_mean) + F.mse_loss(pred_std, tgt_std)


def discrete_metrics(pred: torch.Tensor, target: torch.Tensor,
                     black_threshold: float = 0.05) -> dict:
    pred_np = pred.cpu().numpy()
    tgt_np = target.cpu().numpy()
    pred_fg = (pred_np > black_threshold).any(axis=0)
    tgt_fg = (tgt_np > black_threshold).any(axis=0)
    intersection = (pred_fg & tgt_fg).sum()
    union = (pred_fg | tgt_fg).sum()
    iou = intersection / max(union, 1)
    tp = intersection
    fp = (pred_fg & ~tgt_fg).sum()
    fn = (~pred_fg & tgt_fg).sum()
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {"iou": float(iou), "f1": float(f1)}


def _resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    device = _resolve_device()

    print(f"[cfg] model={MODEL_ID}  device={device}  dtype={DTYPE}  smoke={SMOKE}")
    print(f"[cfg] tasks={TASKS}")
    print(f"[cfg] steps={STEPS}  lr={LR}  rank={RANK}  alpha={ALPHA}  "
          f"dropout={DROPOUT}  wd={WEIGHT_DECAY}")
    print(f"[cfg] val={VAL_SAMPLES}  test={TEST_SAMPLES}  res={RES}  "
          f"val_every={VAL_EVERY}  patience={PATIENCE}")
    print(f"[cfg] nonblack_weight={NONBLACK_WEIGHT}  "
          f"color_match_weight={COLOR_MATCH_WEIGHT}  out={OUT}")

    # 保存配置
    (OUT / "config.json").write_text(json.dumps({
        "model": MODEL_ID, "data_root": DATA_ROOT, "tasks": TASKS,
        "steps": STEPS, "lr": LR, "rank": RANK, "alpha": ALPHA,
        "dropout": DROPOUT, "weight_decay": WEIGHT_DECAY,
        "val_samples": VAL_SAMPLES, "test_samples": TEST_SAMPLES,
        "res": RES, "val_every": VAL_EVERY, "patience": PATIENCE,
        "nonblack_weight": NONBLACK_WEIGHT,
        "color_match_weight": COLOR_MATCH_WEIGHT, "smoke": SMOKE,
    }, indent=2))

    # 1) 数据集：三路划分 ─────────────────────────────────────────────
    # 索引布局（按数据集顺序）：[0, n_test) = test，其后 val，其余 train
    task_datasets, task_loaders, task_splits = {}, {}, {}
    task_to_id = {task: i for i, task in enumerate(TASKS)}

    for task in TASKS:
        data_path = Path(DATA_ROOT) / task
        ds = VisionTaskDataset(data_path=str(data_path), resolution=RES, task=task)
        n = len(ds)
        n_test = min(TEST_SAMPLES, max(1, n // 50))
        n_val = min(VAL_SAMPLES, max(1, n // 100))
        test_idx = list(range(0, n_test))
        val_idx = list(range(n_test, n_test + n_val))
        train_idx = list(range(n_test + n_val, n))
        task_datasets[task] = ds
        task_splits[task] = {"train": len(train_idx), "val": len(val_idx),
                             "test": len(test_idx),
                             "train_idx": train_idx, "val_idx": val_idx,
                             "test_idx": test_idx}
        task_loaders[task] = {
            "train": torch.utils.data.DataLoader(
                torch.utils.data.Subset(ds, train_idx), batch_size=BATCH,
                shuffle=True, num_workers=0, pin_memory=False,
                collate_fn=_collate_fn, drop_last=True),
            "val": torch.utils.data.DataLoader(
                torch.utils.data.Subset(ds, val_idx), batch_size=BATCH,
                shuffle=False, num_workers=0, pin_memory=False,
                collate_fn=_collate_fn, drop_last=True),
        }
        print(f"[data] {task}: total={n}  train={len(train_idx)}  "
              f"val={len(val_idx)}  test={len(test_idx)}")

    # 任务采样权重 ∝ 训练集大小
    task_weights = torch.tensor(
        [task_splits[t]["train"] for t in TASKS], dtype=torch.float32)
    task_weights = task_weights / task_weights.sum()
    print(f"[data] task sampling weights: "
          f"{ {t: round(float(w), 4) for t, w in zip(TASKS, task_weights)} }")

    # 2) 模型 ─────────────────────────────────────────────────────────
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from janus.models.modeling_vlm import MultiModalityConfig, MultiModalityCausalLM
    AutoConfig.register("multi_modality", MultiModalityConfig)
    AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = model.to(device)
    print("[model] loaded")

    # 3) LoRA ──────────────────────────────────────────────────────────
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=RANK, lora_alpha=ALPHA, init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    peft_model = get_peft_model(model.language_model, lora_config)
    n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"[lora] trainable = {n_trainable:,}")

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

    prompt_ids_map = {}
    for task in TASKS:
        prompt_ids_map[task] = tokenizer(
            TASK_PROMPTS[task], return_tensors="pt").input_ids.to(device)

    # 6) 前向 / 验证 ─────────────────────────────────────────────────
    def forward_step(task: str, image: torch.Tensor,
                     target_rgb: torch.Tensor) -> torch.Tensor:
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

        loss = weighted_mse_loss(pred_rgb, target_rgb, task,
                                 NONBLACK_WEIGHT, BLACK_THRESHOLD)
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
        return sum(val_losses) / max(len(val_losses), 1)

    # 7) 训练循环 ─────────────────────────────────────────────────────
    print(f"\n[train] Janus task embedding v3 (full data): {STEPS} steps")
    losses: list[float] = []
    best_val_loss = float("inf")
    patience_counter = 0
    task_iters = {task: iter(task_loaders[task]["train"]) for task in TASKS}
    t_train = time.time()

    for step in range(1, STEPS + 1):
        task = TASKS[torch.multinomial(task_weights, 1).item()]
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
            print(f"[warn] step {step}: non-finite loss, skipped")
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
            print(f"[train] step {step:>5}/{STEPS}  loss={loss.item():.4f}  "
                  f"avg30={recent:.4f}  task={task}  "
                  f"({(time.time() - t_train) / step:.2f}s/step)", flush=True)

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
            print(f"[val] step {step:>5}  avg_val_loss={total_val:.4f}  "
                  f"best={best_val_loss:.4f}  patience={patience_counter}{marker}",
                  flush=True)
            if patience_counter >= PATIENCE:
                print(f"[early] stop at step {step}")
                break

    train_minutes = (time.time() - t_train) / 60

    # 保存最终 checkpoint
    final_ckpt = OUT / "ckpt_final"
    final_ckpt.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(final_ckpt / "lora"))
    torch.save(rgb_head.state_dict(), final_ckpt / "rgb_head.pt")
    torch.save(task_embedding.state_dict(), final_ckpt / "task_embedding.pt")

    # 8) 加载最佳 checkpoint 做 held-out 诚实评估 ─────────────────────
    print("\n[eval] loading best ckpt for held-out test evaluation...")
    from peft import PeftModel
    peft_model = PeftModel.from_pretrained(
        model.language_model, str(OUT / "ckpt_best" / "lora"))
    rgb_head.load_state_dict(torch.load(OUT / "ckpt_best" / "rgb_head.pt",
                                        map_location=device, weights_only=True))
    task_embedding.load_state_dict(torch.load(
        OUT / "ckpt_best" / "task_embedding.pt", map_location=device,
        weights_only=True))
    peft_model.eval()
    rgb_head.eval()
    task_embedding.eval()

    depth_mapper = DepthRGBMapper().to(device)
    normal_mapper = NormalRGBMapper()

    @torch.no_grad()
    def infer_rgb(task: str, sample_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ds = task_datasets[task]
        s = ds[sample_idx]
        inp_tensor = s["image"]
        tgt_rgb = s["target_rgb"]

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
        return pred_rgb, tgt_rgb

    all_results: dict[str, dict] = {}

    for task in TASKS:
        print(f"\n[eval] task={task} (held-out test, n={len(task_splits[task]['test_idx'])}):")
        test_idx = task_splits[task]["test_idx"]
        rgb_maes, ious, f1s = [], [], []
        depth_metrics_list = []
        normal_angles = []

        for i, idx in enumerate(test_idx):
            pred_rgb, tgt_rgb = infer_rgb(task, idx)
            rgb_mae = (pred_rgb.float() - tgt_rgb.float()).abs().mean().item()
            rgb_maes.append(rgb_mae)

            if task in DISCRETE_TASKS:
                m = discrete_metrics(pred_rgb, tgt_rgb, BLACK_THRESHOLD)
                ious.append(m["iou"])
                f1s.append(m["f1"])

            # depth：colormap 解码 → 真实 δ1/AbsRel（子采样控制耗时）
            if task == "depth" and i < DEPTH_EVAL_N:
                pred_d = depth_mapper.rgb_to_depth(
                    pred_rgb.unsqueeze(0).to(device)).cpu()
                tgt_d = depth_mapper.rgb_to_depth(
                    tgt_rgb.unsqueeze(0).to(device)).cpu()
                dm = evaluate_depth(pred_d, tgt_d)
                depth_metrics_list.append(dm)
                if i == 0:
                    print(f"  [sanity] decoded target depth: "
                          f"min={tgt_d.min():.2f}  max={tgt_d.max():.2f}  "
                          f"mean={tgt_d.mean():.2f}")

            # normal：解码 → 平均角误差（度）
            if task == "normal":
                pred_n = normal_mapper.rgb_to_normal(pred_rgb.unsqueeze(0))
                tgt_n = normal_mapper.rgb_to_normal(tgt_rgb.unsqueeze(0))
                cos = F.cosine_similarity(pred_n.flatten(2), tgt_n.flatten(2), dim=1)
                ang = torch.acos(cos.clamp(-1, 1)).mean().item() * 180 / math.pi
                normal_angles.append(ang)

            # 前 3 个样本存 panel
            if i < 3:
                s = task_datasets[task][idx]
                pil_in = Image.fromarray(
                    (s["image"].permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5)
                    .mul(255).byte().numpy())
                panel = Image.new("RGB", (RES * 3, RES))
                panel.paste(pil_in.resize((RES, RES)), (0, 0))
                panel.paste(Image.fromarray(
                    (pred_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
                ).resize((RES, RES)), (RES, 0))
                panel.paste(Image.fromarray(
                    (tgt_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
                ).resize((RES, RES)), (RES * 2, 0))
                panel.save(OUT / f"panel_{task}_test{idx:04d}.png")

        res = {"n_test": len(test_idx), "avg_rgb_mae": sum(rgb_maes) / len(rgb_maes)}
        if ious:
            res["avg_iou"] = sum(ious) / len(ious)
            res["avg_f1"] = sum(f1s) / len(f1s)
            print(f"  → rgb_mae={res['avg_rgb_mae']:.4f}  "
                  f"IoU={res['avg_iou']:.4f}  F1={res['avg_f1']:.4f}")
        else:
            print(f"  → rgb_mae={res['avg_rgb_mae']:.4f}")
        if depth_metrics_list:
            for k in depth_metrics_list[0]:
                res[f"depth_{k}"] = sum(d[k] for d in depth_metrics_list) / len(depth_metrics_list)
            print(f"  → δ1={res['depth_δ<1.25']:.4f}  "
                  f"AbsRel={res['depth_AbsRel']:.4f}  "
                  f"RMSE={res['depth_RMSE']:.3f}  (n={len(depth_metrics_list)})")
        if normal_angles:
            res["normal_mean_angular_deg"] = sum(normal_angles) / len(normal_angles)
            print(f"  → mean angular error={res['normal_mean_angular_deg']:.2f}°")
        all_results[task] = res

    overall_mae = sum(r["avg_rgb_mae"] for r in all_results.values()) / len(all_results)
    print(f"\n[bench] overall avg rgb_mae={overall_mae:.4f}")
    print(f"[done] best val_loss={best_val_loss:.4f}  "
          f"train_time={train_minutes:.1f}min")

    (OUT / "task_embedding_v3_result.json").write_text(json.dumps({
        "best_val_loss": best_val_loss,
        "train_minutes": train_minutes,
        "steps_run": len(losses),
        "final_train_loss_avg30": sum(losses[-30:]) / max(len(losses[-30:]), 1),
        "per_task": all_results,
        "overall_rgb_mae": overall_mae,
        "splits": {t: {k: v for k, v in task_splits[t].items()
                       if not k.endswith("_idx")} for t in TASKS},
    }, indent=2))
    print(f"[out] → {OUT / 'task_embedding_v3_result.json'}")


if __name__ == "__main__":
    main()
