#!/usr/bin/env python3
"""MPS 多视觉任务 gate —— 对齐 Vision Banana 论文的多任务统一范式.

论文核心主张（对齐开源替代）:
  - 单一模型通过 prompt 切换处理多视觉任务（depth/normal/instance/referring/keypoint）
  - 视觉任务 RGB 化：depth→viridis, normal→RGB映射, segmentation→彩色区域
  - 混合训练：~90% generation + ~10% vision（论文未公布精确比例）
  - 轻量微调：LoRA（不破坏基座生成能力）

开源替代方案（已验证 source conditioning 可行）:
  - 基座: SSD-1B (distilled SDXL)
  - 条件: IP2P source conditioning（8ch conv_in）
  - 训练: 多任务混合 + generation 保真
  - 推理: img2img + task prompt → RGB 可视化

MPS 约束:
  - 256px 分辨率（内存）
  - batch_size=1（内存）
  - fp32（MPS 稳定）
  - 总样本数限 8 per task（gate 规模）

用法:
    PYTHONPATH=src python scripts/mps_multitask_gate.py
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
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.conditioning import (
    CONV_IN_STATE_NAME,
    write_source_conditioning_config,
)
from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.evaluation.relative_depth import evaluate_relative_depth_rgb
from open_vision_banana.inference.generator import BaseGenerator
from open_vision_banana.training.tuner import (
    InstructionTuner,
    VisionTaskDataset,
    _collate_fn,
)

# ── Config ───────────────────────────────────────────────────────────────
MODEL_ID = os.environ.get("VB_MODEL", "segmind/SSD-1B")
DATA_ROOT = os.environ.get(
    "VB_DATA_ROOT",
    "/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v3",
)
TASKS = os.environ.get("VB_TASKS", "depth,normal,instance,referring,keypoint").split(",")
VISION_PER_TASK = int(os.environ.get("VB_VISION_PER_TASK", "8"))  # 每任务训练样本数
GEN_SAMPLES = int(os.environ.get("VB_GEN_SAMPLES", "32"))  # 生成样本数
STEPS = int(os.environ.get("VB_STEPS", "2000"))
RES = int(os.environ.get("VB_RES", "256"))
BATCH = int(os.environ.get("VB_BATCH", "1"))
LR = float(os.environ.get("VB_LR", "1e-4"))
RANK = int(os.environ.get("VB_RANK", "64"))
ALPHA = int(os.environ.get("VB_ALPHA", "128"))
# 论文: vision data 比例很低（未公布精确值）
# 提高 ratio 让模型更多接触 vision 样本
VISION_RATIO = float(os.environ.get("VB_VISION_RATIO", "0.30"))
OUT = Path(os.environ.get("VB_OUT", str(PROJECT_ROOT / "output" / "mps_multitask_gate")))
OUT.mkdir(parents=True, exist_ok=True)

DTYPE = torch.float32  # MPS safe

# 任务 prompt 默认值（若数据文件缺失时使用）
TASK_DEFAULT_PROMPTS = {
    "depth": "Generate a depth map of this image. Use a power-transform color encoding where R,G,B encode metric depth values.",
    "normal": "Generate a surface normal map of this image. R=(1-x)/2*255, G=(1+y)/2*255, B=(1+z)/2*255, where x,y,z are the normalized normal vectors.",
    "instance": "Generate an instance segmentation visualization of this image. Each object is colored differently. Background is black.",
    "referring": "A segmentation map image. The area that corresponds to the referred object is rendered solid white; the other objects are rendered in black.",
    "keypoint": "Generate a keypoint detection map. Keypoints are marked as colored RGB dots on a black background.",
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
        # 取前 VISION_PER_TASK 个样本
        subset = torch.utils.data.Subset(ds, list(range(VISION_PER_TASK)))
        task_loaders[task] = torch.utils.data.DataLoader(
            subset, batch_size=BATCH, shuffle=True, num_workers=0,
            pin_memory=False, collate_fn=_collate_fn, drop_last=True,
        )
        total_vision += VISION_PER_TASK
        print(f"[data] {task}: {len(ds)} total, using {VISION_PER_TASK}")

    # 加载 generation 数据（t2i 混合 — target 就是 image 本身）
    gen_path = Path(DATA_ROOT) / "generation"
    gen_images = sorted((gen_path / "images").glob("*.jpg"))[:GEN_SAMPLES]
    gen_prompts = {}
    for img_p in gen_images:
        p = gen_path / "prompts" / f"{img_p.stem}.txt"
        if p.exists():
            gen_prompts[img_p.stem] = p.read_text().strip()

    # 手动构造 generation batch（不走 VisionTaskDataset）
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

    # 2) 加载生成模型 + source conditioning ───────────────────────────
    t0 = time.time()
    gen = BaseGenerator(MODEL_ID, device=device, dtype=DTYPE)
    print(f"[model] family={gen.model_config.family}  loaded in {time.time()-t0:.1f}s")

    # 获取目标色图统计（用于标定）—— 用 depth 任务作为参考
    depth_ds = task_datasets["depth"]
    stat_idx = list(range(0, min(8, len(depth_ds)), 1))
    tgts = torch.stack([depth_ds[i]["target_rgb"] for i in stat_idx])
    ch_mean = tuple(float(x) for x in tgts.mean(dim=(0, 2, 3)))
    ch_std = tuple(float(x) for x in tgts.std(dim=(0, 2, 3)).clamp(min=1e-4))
    print(f"[calib] target RGB mean={tuple(round(v,3) for v in ch_mean)}  "
          f"std={tuple(round(v,3) for v in ch_std)}")

    # Source conditioning tuner
    tuner = InstructionTuner(
        model=gen, task="depth",
        lora_rank=RANK, lora_alpha=ALPHA, lr=LR, use_rslora=True,
        use_ema=False, use_gradient_checkpointing=False, use_min_snr=True,
        use_img2img_flow=False, cfg_dropout_rate=0.1,
        use_source_conditioning=True,
        source_target_colormap="viridis",
        source_target_channel_mean=ch_mean,
        source_target_channel_std=ch_std,
        source_output_calibration="moments_and_project",
        source_recommended_prompt=TASK_DEFAULT_PROMPTS["depth"],
    )
    tuner.peft_model.train()
    n_train = sum(p.numel() for p in tuner.peft_model.parameters() if p.requires_grad)
    n_total_p = sum(p.numel() for p in tuner.peft_model.parameters())
    print(f"[lora] trainable = {n_train:,}  ({100*n_train/n_total_p:.2f}%)")

    # 3) Optimizer ─────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        [p for p in tuner.peft_model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )

    def lr_lambda(step: int) -> float:
        warmup = max(1, STEPS // 30)
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, STEPS - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # 4) 训练循环 — 多任务混合 ─────────────────────────────────────────
    print(f"\n[train] multi-task gate: {STEPS} steps")
    print(f"[train] vision_ratio={VISION_RATIO} (paper: very low)")

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
            # 随机选一个视觉任务
            task = TASKS[torch.randint(0, len(TASKS), (1,)).item()]
            loader_iter = task_iters[task]
            try:
                batch = next(loader_iter)
            except StopIteration:
                task_iters[task] = iter(task_loaders[task])
                batch = next(task_iters[task])
            # vision 模式: source conditioning (image + prompt → target)
        else:
            # generation 模式: t2i (text → image)
            try:
                batch = next(gen_iter)
            except StopIteration:
                gen_iter = iter(gen_loader)
                batch = next(gen_iter)
            # generation 模式: 清除 image 字段让 tuner 走 t2i
            batch["image"] = None
            task = "generation"

        opt.zero_grad(set_to_none=True)
        loss = tuner._training_step(batch)

        if not torch.isfinite(loss).all():
            print(f"[train] step {step:>4}: NON-FINITE loss — skip", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tuner.peft_model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())

        if step % 30 == 0 or step == 1:
            recent = sum(losses[-30:]) / len(losses[-30:])
            print(f"[train] step {step:>4}/{STEPS}  loss={loss.item():.4f}  "
                  f"avg30={recent:.4f}  lr={sched.get_last_lr()[0]:.2e}  "
                  f"task={task}  mode={'vis' if is_vision else 'gen'}  "
                  f"({(time.time()-t_train)/step:.2f}s/step)", flush=True)
        # 每 500 步存中间 checkpoint
        if step % 500 == 0:
            mid_ckpt = OUT / f"lora_step{step}"
            mid_ckpt.mkdir(parents=True, exist_ok=True)
            tuner.peft_model.save_pretrained(str(mid_ckpt))
            conv = tuner._source_conditioning_conv()
            torch.save(conv.state_dict(), mid_ckpt / CONV_IN_STATE_NAME)
            write_source_conditioning_config(mid_ckpt, tuner._source_conditioning_config())
            print(f"[ckpt] mid-training save → {mid_ckpt}", flush=True)

    if not losses:
        raise RuntimeError("No finite training steps")

    k = max(1, len(losses) // 3)
    avg_first = sum(losses[:k]) / k
    avg_last = sum(losses[-k:]) / k
    print(f"\n[train] avg(first 1/3)={avg_first:.4f}  avg(last 1/3)={avg_last:.4f}  "
          f"({'DECREASING' if avg_last < avg_first else 'NOT decreasing'})")

    # 保存 checkpoint
    ckpt_dir = OUT / "lora_final"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tuner.peft_model.save_pretrained(str(ckpt_dir))
    conv = tuner._source_conditioning_conv()
    torch.save(conv.state_dict(), ckpt_dir / CONV_IN_STATE_NAME)
    write_source_conditioning_config(ckpt_dir, tuner._source_conditioning_config())
    print(f"[ckpt] saved → {ckpt_dir}")

    # 5) 推理评估 — 每任务 2 样本 ──────────────────────────────────────
    print("\n[eval] inference on each task (2 samples)...")
    gen.source_conditioning_config = tuner._source_conditioning_config()
    mapper = DepthRGBMapper(colormap="viridis")

    all_results = {}

    @torch.no_grad()
    def infer_task(task: str, sample_idx: int) -> dict:
        ds = task_datasets[task]
        s = ds[sample_idx]
        inp_tensor = s["image"]  # [-1, 1]
        tgt_rgb = s["target_rgb"]  # [0, 1]
        prompt = s.get("prompt", TASK_DEFAULT_PROMPTS.get(task, ""))
        pil_in = Image.fromarray(
            (inp_tensor.permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5)
            .mul(255).byte().numpy()
        )
        out = gen.generate(
            prompt=prompt,
            image=pil_in,
            strength=0.85,
            num_inference_steps=8,
            guidance_scale=0.0,
            seed=42 + sample_idx,
        )
        pred_rgb = out[0].clamp(0, 1).cpu()

        # 三联图
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
        depth_metrics = evaluate_relative_depth_rgb(
            pred_rgb.unsqueeze(0), tgt_rgb.unsqueeze(0), colormap="viridis"
        )
        return {
            "idx": sample_idx,
            "rgb_mae": rgb_mae,
            "depth_mae": depth_metrics.get("mae", float("nan")),
            "pearson": depth_metrics.get("pearson", float("nan")),
        }

    for task in TASKS:
        print(f"\n[eval] task={task}:")
        task_results = []
        for idx in range(min(2, VISION_PER_TASK)):
            r = infer_task(task, idx)
            task_results.append(r)
            print(f"  idx={idx:>2}  rgb_mae={r['rgb_mae']:.4f}  "
                  f"depth_mae={r['depth_mae']:.4f}  pearson={r['pearson']:.4f}")
        avg_mae = sum(r["rgb_mae"] for r in task_results) / len(task_results)
        avg_pearson = sum(r["pearson"] for r in task_results if not math.isnan(r["pearson"])) / max(1, len(task_results))
        all_results[task] = {
            "avg_rgb_mae": avg_mae,
            "avg_pearson": avg_pearson,
            "samples": task_results,
        }
        print(f"  → avg rgb_mae={avg_mae:.4f}  avg pearson={avg_pearson:.4f}")

    # 汇总
    overall_mae = sum(r["avg_rgb_mae"] for r in all_results.values()) / len(all_results)
    overall_pearson = sum(
        r["avg_pearson"] for r in all_results.values() if not math.isnan(r["avg_pearson"])
    ) / max(1, len(all_results))

    print(f"\n[gate] overall avg rgb_mae={overall_mae:.4f}  avg pearson={overall_pearson:.4f}")
    gate_passed = overall_mae < 0.20 and overall_pearson > 0.6
    print(f"[gate] {'PASSED' if gate_passed else 'NOT PASSED'}  "
          f"(target: rgb_mae<0.20, pearson>0.6)")

    # 保存结果
    (OUT / "gate_result.json").write_text(json.dumps({
        "model": MODEL_ID, "device": device, "dtype": str(DTYPE),
        "tasks": TASKS, "vision_per_task": VISION_PER_TASK, "gen_samples": GEN_SAMPLES,
        "steps": STEPS, "res": RES, "rank": RANK, "alpha": ALPHA,
        "vision_ratio": VISION_RATIO,
        "loss_first_third": avg_first, "loss_last_third": avg_last,
        "per_task": all_results,
        "overall_rgb_mae": overall_mae, "overall_pearson": overall_pearson,
        "gate_passed": gate_passed,
    }, indent=2))
    print(f"[out] → {OUT / 'gate_result.json'}")

    total_min = (time.time() - t_train) / 60
    print(f"\n[done] multi-task gate complete ({total_min:.1f} min)  "
          f"{'✓ PASSED' if gate_passed else '✗ not passed'}")


if __name__ == "__main__":
    main()
