#!/usr/bin/env python3
"""Janus-1.3B continuous RGB v3: much more data + stronger regularization.

Improvements over v2:
- More training data: 500 train + 50 val (v2 had only 100+10)
- Multi-task mix: depth + normal + instance (optional)
- Stronger regularization:
  · LoRA dropout (the lora_dropout parameter)
  · RGB Head dropout
  · weight decay (AdamW)
  · early stopping (configurable patience)
  · gradient clipping
  · data augmentation (horizontal flip)
- Optional warm-start from a v2 checkpoint
- Full training history as JSON

Usage:
    # single-task depth (default)
    PYTHONPATH=src:/tmp/Janus python scripts/janus_continuous_rgb_v3.py

    # multi-task depth+normal+instance
    VB_TASKS=depth,normal,instance VB_N_PER_TASK=200 \
    PYTHONPATH=src:/tmp/Janus python scripts/janus_continuous_rgb_v3.py

    # warm start from a v2 checkpoint
    VB_RESUME=output/janus_continuous_rgb_v2/ckpt_best \
    PYTHONPATH=src:/tmp/Janus python scripts/janus_continuous_rgb_v3.py
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
from open_vision_banana.evaluation.relative_depth import evaluate_relative_depth_rgb
from open_vision_banana.training.tuner import VisionTaskDataset, _collate_fn

# ── Config ───────────────────────────────────────────────────────────────
MODEL_ID = os.environ.get("VB_MODEL", "deepseek-ai/Janus-1.3B")
DATA_ROOT = os.environ.get(
    "VB_DATA_ROOT",
    "/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v3",
)

# task config: depth / depth,normal,instance / depth,normal,instance,referring,keypoint
TASKS = os.environ.get("VB_TASKS", "depth").split(",")
N_PER_TASK = int(os.environ.get("VB_N_PER_TASK", "500"))       # training samples per task
VAL_PER_TASK = int(os.environ.get("VB_VAL_PER_TASK", "50"))    # validation samples per task

STEPS = int(os.environ.get("VB_STEPS", "2000"))
RES = int(os.environ.get("VB_RES", "384"))
BATCH = int(os.environ.get("VB_BATCH", "1"))
LR = float(os.environ.get("VB_LR", "5e-5"))         # more data → lower LR
RANK = int(os.environ.get("VB_RANK", "64"))
ALPHA = int(os.environ.get("VB_ALPHA", "128"))

# ── Regularization parameters ───────────────────────────────────────────
DROPOUT = float(os.environ.get("VB_DROPOUT", "0.15"))            # RGB Head dropout
LORA_DROPOUT = float(os.environ.get("VB_LORA_DROPOUT", "0.05"))  # LoRA dropout
WEIGHT_DECAY = float(os.environ.get("VB_WEIGHT_DECAY", "0.02"))  # AdamW weight decay
PATIENCE = int(os.environ.get("VB_PATIENCE", "15"))              # early stopping patience
VAL_EVERY = int(os.environ.get("VB_VAL_EVERY", "50"))            # validation frequency
GRAD_CLIP = float(os.environ.get("VB_GRAD_CLIP", "1.0"))        # gradient clipping
AUGMENT_FLIP = os.environ.get("VB_AUGMENT_FLIP", "1") == "1"    # horizontal flip augmentation

# optional warm-start
RESUME_CKPT = os.environ.get("VB_RESUME", "")

OUT = Path(os.environ.get("VB_OUT", str(PROJECT_ROOT / "output" / "janus_continuous_rgb_v3")))
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
    """RGB prediction head: LLM hidden states → RGB values (with dropout regularization)."""

    def __init__(self, hidden_dim: int, image_token_num: int, image_size: int, dropout: float = 0.15):
        super().__init__()
        self.image_token_num = image_token_num
        self.image_size = image_size
        self.patch_size = image_size // int(math.sqrt(image_token_num))

        # two-layer MLP + dropout for stronger regularization
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, self.patch_size * self.patch_size * 3),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        b = hidden_states.shape[0]
        x = self.proj(hidden_states)
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


def _augment_flip(image: torch.Tensor, target_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Horizontal flip augmentation (50% probability)."""
    if torch.rand(1).item() < 0.5:
        return image.flip(-1), target_rgb.flip(-1)
    return image, target_rgb


def main() -> None:
    torch.manual_seed(42)
    device = _resolve_device()

    print(f"[cfg] model={MODEL_ID}  device={device}  dtype={DTYPE}")
    print(f"[cfg] tasks={TASKS}  n_per_task={N_PER_TASK}  val_per_task={VAL_PER_TASK}")
    print(f"[cfg] steps={STEPS}  res={RES}  batch={BATCH}  lr={LR}")
    print(f"[cfg] rank={RANK}  alpha={ALPHA}")
    print(f"[cfg] dropout={DROPOUT}  lora_dropout={LORA_DROPOUT}  wd={WEIGHT_DECAY}")
    print(f"[cfg] patience={PATIENCE}  val_every={VAL_EVERY}  grad_clip={GRAD_CLIP}")
    print(f"[cfg] augment_flip={AUGMENT_FLIP}  resume={RESUME_CKPT or 'none'}")
    print(f"[cfg] out={OUT}")

    # 1) load Janus-1.3B ───────────────────────────────────────────────
    # Patch Janus config classes for Python 3.10+ dataclass compatibility
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

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = model.to(device)
    print(f"[model] loaded in {time.time()-t0:.1f}s")
    print(f"[model] type={type(model).__name__}")
    print(f"[model] params={sum(p.numel() for p in model.parameters()):,}")

    # 2) dataset ───────────────────────────────────────────────────────
    all_train_datasets = []
    all_val_datasets = []

    for task in TASKS:
        data_path = Path(DATA_ROOT) / task
        if not data_path.exists():
            print(f"[warn] task data not found: {data_path}, skipping")
            continue
        ds = VisionTaskDataset(data_path=str(data_path), resolution=RES, task=task)
        n_total = len(ds)
        need = N_PER_TASK + VAL_PER_TASK
        actual_n = min(need, n_total)
        actual_train = int(actual_n * N_PER_TASK / need)
        actual_val = actual_n - actual_train
        print(f"[data] {task}: {n_total} total @ {RES}px → train={actual_train} val={actual_val}")

        train_idx = list(range(actual_train))
        val_idx = list(range(actual_train, actual_train + actual_val))
        all_train_datasets.append(torch.utils.data.Subset(ds, train_idx))
        all_val_datasets.append(torch.utils.data.Subset(ds, val_idx))

    if not all_train_datasets:
        raise RuntimeError(f"No task data found under {DATA_ROOT} for tasks {TASKS}")

    train_combined = torch.utils.data.ConcatDataset(all_train_datasets)
    val_combined = torch.utils.data.ConcatDataset(all_val_datasets)

    train_loader = torch.utils.data.DataLoader(
        train_combined, batch_size=BATCH, shuffle=True, num_workers=0,
        pin_memory=False, collate_fn=_collate_fn, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_combined, batch_size=BATCH, shuffle=False, num_workers=0,
        pin_memory=False, collate_fn=_collate_fn, drop_last=True,
    )
    print(f"[data] combined train={len(train_combined)}  val={len(val_combined)}")

    # 3) LoRA on language_model (with dropout regularization) ──────────
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        lora_dropout=LORA_DROPOUT,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    peft_model = get_peft_model(model.language_model, lora_config)
    n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    n_total_p = sum(p.numel() for p in peft_model.parameters())
    print(f"[lora] trainable = {n_train:,}  ({100*n_train/n_total_p:.2f}%)")
    print(f"[lora] dropout = {LORA_DROPOUT}")

    # freeze the vision and generation parts
    model.vision_model.requires_grad_(False)
    model.gen_vision_model.requires_grad_(False)
    model.aligner.requires_grad_(False)
    model.gen_aligner.requires_grad_(False)
    model.gen_head.requires_grad_(False)
    model.gen_embed.requires_grad_(False)

    # 4) RGB Head (with dropout regularization) ────────────────────────
    hidden_dim = model.config.language_config.hidden_size
    rgb_head = RGBHead(hidden_dim, IMAGE_TOKEN_NUM, RES, dropout=DROPOUT).to(device, DTYPE)
    rgb_head_params = sum(p.numel() for p in rgb_head.parameters())
    print(f"[rgb_head] params={rgb_head_params:,}  dropout={DROPOUT}")

    # 5) optional warm-start ───────────────────────────────────────────
    start_step = 0
    if RESUME_CKPT and Path(RESUME_CKPT).exists():
        print(f"[resume] loading from {RESUME_CKPT}")
        adapter_dir = Path(RESUME_CKPT) / "lora"
        if adapter_dir.exists():
            from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
            adapter_state = load_peft_weights(str(adapter_dir), device=device)
            set_peft_model_state_dict(peft_model, adapter_state)
            print(f"[resume] LoRA weights loaded")
        rgb_head_path = Path(RESUME_CKPT) / "rgb_head.pt"
        if rgb_head_path.exists():
            rgb_head.load_state_dict(torch.load(rgb_head_path, map_location=device, weights_only=True))
            print(f"[resume] RGB head loaded")
        print(f"[resume] warm-start complete")

    # 6) optimizer (weight decay regularization) ───────────────────────
    trainable_params = (
        [p for p in peft_model.parameters() if p.requires_grad] +
        list(rgb_head.parameters())
    )
    opt = torch.optim.AdamW(
        trainable_params,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    def lr_lambda(step: int) -> float:
        warmup = max(1, STEPS // 20)  # longer warmup
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, STEPS - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # 7) validation function ───────────────────────────────────────────
    prompt_ids_cache: dict[str, torch.Tensor] = {}

    def _get_prompt_ids(prompt: str) -> torch.Tensor:
        if prompt not in prompt_ids_cache:
            prompt_ids_cache[prompt] = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        return prompt_ids_cache[prompt]

    @torch.no_grad()
    def run_validation() -> float:
        peft_model.eval()
        rgb_head.eval()
        val_losses = []
        for batch in val_loader:
            image = batch["image"][0:1].to(device, DTYPE)
            target_rgb = batch["target_rgb"][0:1].to(device, DTYPE)
            task = batch.get("task", ["depth"])[0] if isinstance(batch.get("task"), list) else batch.get("task", "depth")
            prompt = TASK_PROMPTS.get(task, TASK_PROMPTS["depth"])

            p_ids = _get_prompt_ids(prompt)
            p_len = p_ids.shape[1]

            pixel_values = image.unsqueeze(1)
            seq_len = p_len + IMAGE_TOKEN_NUM
            full_input_ids = torch.cat([
                p_ids,
                torch.full((1, IMAGE_TOKEN_NUM), -1, device=device, dtype=torch.long),
            ], dim=1)
            images_seq_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
            images_seq_mask[0, p_len:] = True
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
            gen_hidden = hidden_states[:, p_len:, :]
            pred_rgb = rgb_head(gen_hidden)
            loss = F.mse_loss(pred_rgb, target_rgb)
            val_losses.append(loss.item())

        peft_model.train()
        rgb_head.train()
        return sum(val_losses) / len(val_losses) if val_losses else float("inf")

    # 8) training loop ─────────────────────────────────────────────────
    print(f"\n[train] Janus continuous RGB v3: {STEPS} steps")
    print(f"[train] {len(TASKS)} task(s): {TASKS}")

    losses: list[float] = []
    val_history: list[dict] = []
    best_val_loss = float("inf")
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
        task = batch.get("task", ["depth"])[0] if isinstance(batch.get("task"), list) else batch.get("task", "depth")
        prompt = TASK_PROMPTS.get(task, TASK_PROMPTS["depth"])

        # data augmentation
        if AUGMENT_FLIP:
            image, target_rgb = _augment_flip(image, target_rgb)

        p_ids = _get_prompt_ids(prompt)
        p_len = p_ids.shape[1]

        pixel_values = image.unsqueeze(1)
        seq_len = p_len + IMAGE_TOKEN_NUM
        full_input_ids = torch.cat([
            p_ids,
            torch.full((1, IMAGE_TOKEN_NUM), -1, device=device, dtype=torch.long),
        ], dim=1)
        images_seq_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
        images_seq_mask[0, p_len:] = True
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
        gen_hidden = hidden_states[:, p_len:, :]
        pred_rgb = rgb_head(gen_hidden)
        loss = F.mse_loss(pred_rgb, target_rgb)

        if not torch.isfinite(loss).all():
            print(f"[train] step {step:>4}: NON-FINITE loss — skip", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP)
        opt.step()
        sched.step()
        losses.append(loss.item())

        if step % 30 == 0 or step == 1:
            recent = sum(losses[-30:]) / len(losses[-30:])
            print(f"[train] step {step:>4}/{STEPS}  loss={loss.item():.4f}  "
                  f"avg30={recent:.4f}  lr={sched.get_last_lr()[0]:.2e}  "
                  f"task={task}  "
                  f"({(time.time()-t_train)/step:.2f}s/step)", flush=True)

        # validation + early stopping
        if step % VAL_EVERY == 0:
            val_loss = run_validation()
            record = {
                "step": step,
                "train_loss": float(loss.item()),
                "avg30": float(sum(losses[-30:]) / len(losses[-30:])),
                "val_loss": float(val_loss),
                "lr": float(sched.get_last_lr()[0]),
                "task": task,
            }
            val_history.append(record)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # save the best checkpoint
                best_ckpt = OUT / "ckpt_best"
                best_ckpt.mkdir(parents=True, exist_ok=True)
                peft_model.save_pretrained(str(best_ckpt / "lora"))
                torch.save(rgb_head.state_dict(), best_ckpt / "rgb_head.pt")
                # save the training config for resuming
                config_snapshot = {
                    "model_id": MODEL_ID,
                    "tasks": TASKS,
                    "n_per_task": N_PER_TASK,
                    "steps": STEPS,
                    "lr": LR, "rank": RANK, "alpha": ALPHA,
                    "dropout": DROPOUT, "lora_dropout": LORA_DROPOUT,
                    "weight_decay": WEIGHT_DECAY,
                    "best_val_loss": best_val_loss,
                    "best_step": step,
                }
                with open(best_ckpt / "train_config.json", "w") as f:
                    json.dump(config_snapshot, f, indent=2)
                marker = "  *BEST*"
            else:
                patience_counter += 1
                marker = ""
            print(f"[val] step {step:>4}  val_loss={val_loss:.4f}  "
                  f"best={best_val_loss:.4f}  patience={patience_counter}/{PATIENCE}{marker}",
                  flush=True)

            if patience_counter >= PATIENCE:
                print(f"[early] no val improvement for {PATIENCE} evals — "
                      f"early stop at step {step} (best val_loss={best_val_loss:.4f})")
                break

    if not losses:
        raise RuntimeError("No finite training steps")

    # 9) training summary ──────────────────────────────────────────────
    k = max(1, len(losses) // 3)
    avg_first = sum(losses[:k]) / k
    avg_last = sum(losses[-k:]) / k
    print(f"\n[train] avg(first 1/3)={avg_first:.4f}  avg(last 1/3)={avg_last:.4f}  "
          f"({'DECREASING ✓' if avg_last < avg_first else 'NOT decreasing ✗'})")

    # save the final checkpoint
    final_ckpt = OUT / "ckpt_final"
    final_ckpt.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(final_ckpt / "lora"))
    torch.save(rgb_head.state_dict(), final_ckpt / "rgb_head.pt")
    print(f"[ckpt] saved → {final_ckpt}")

    # save the training history
    history_out = OUT / "training_history.json"
    with open(history_out, "w") as f:
        json.dump({
            "config": {
                "model_id": MODEL_ID,
                "tasks": TASKS,
                "n_per_task": N_PER_TASK,
                "val_per_task": VAL_PER_TASK,
                "steps": STEPS,
                "res": RES, "lr": LR, "rank": RANK, "alpha": ALPHA,
                "dropout": DROPOUT, "lora_dropout": LORA_DROPOUT,
                "weight_decay": WEIGHT_DECAY,
                "patience": PATIENCE, "grad_clip": GRAD_CLIP,
                "augment_flip": AUGMENT_FLIP,
            },
            "summary": {
                "total_steps": len(losses),
                "best_val_loss": best_val_loss,
                "avg_first_third": avg_first,
                "avg_last_third": avg_last,
                "stopped_early": patience_counter >= PATIENCE,
                "total_minutes": (time.time() - t_train) / 60,
            },
            "val_history": val_history,
        }, f, indent=2)
    print(f"[history] saved → {history_out}")

    # 10) inference evaluation ─────────────────────────────────────────
    print("\n[eval] inference on held-out samples...")
    mapper = DepthRGBMapper(colormap="viridis")

    # evaluate on validation set samples
    results = []
    n_eval = min(len(val_combined), 5)

    @torch.no_grad()
    def infer_from_subset(subset, sample_idx: int, task: str = "depth") -> dict:
        s = subset[sample_idx]
        inp_tensor = s["image"]
        tgt_rgb = s["target_rgb"]
        prompt = TASK_PROMPTS.get(task, TASK_PROMPTS["depth"])
        pil_in = Image.fromarray(
            (inp_tensor.permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5)
            .mul(255).byte().numpy()
        )

        p_ids = _get_prompt_ids(prompt)
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
        gen_hidden = hidden_states[:, p_len:, :]
        pred_rgb = rgb_head(gen_hidden).squeeze(0).clamp(0, 1).cpu()

        panel = Image.new("RGB", (RES * 3, RES))
        panel.paste(pil_in.resize((RES, RES)), (0, 0))
        panel.paste(Image.fromarray(
            (pred_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
        ).resize((RES, RES)), (RES, 0))
        panel.paste(Image.fromarray(
            (tgt_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
        ).resize((RES, RES)), (RES * 2, 0))
        panel.save(OUT / f"eval_{task}_{sample_idx:02d}.png")

        rgb_mae = (pred_rgb.float() - tgt_rgb.float()).abs().mean().item()
        return {"idx": sample_idx, "task": task, "rgb_mae": rgb_mae}

    # evaluate the validation set of each task
    for task_i, task in enumerate(TASKS):
        if task_i < len(all_val_datasets):
            for i in range(min(2, len(all_val_datasets[task_i]))):
                r = infer_from_subset(all_val_datasets[task_i], i, task)
                results.append(r)
                print(f"[eval] task={task}  idx={r['idx']:>3}  rgb_mae={r['rgb_mae']:.4f}")

    if results:
        avg_mae = sum(r["rgb_mae"] for r in results) / len(results)
        print(f"\n[gate] avg rgb_mae={avg_mae:.4f} across {len(results)} eval samples")

    total_min = (time.time() - t_train) / 60
    print(f"\n[done] Janus continuous RGB v3 complete ({total_min:.1f} min)")
    print(f"[done] best val_loss={best_val_loss:.4f}")
    print(f"[done] checkpoint → {OUT / 'ckpt_best'}")


if __name__ == "__main__":
    main()
