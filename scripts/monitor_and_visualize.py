#!/usr/bin/env python3
"""Checkpoint 监控 + 多任务推理可视化。

每当训练保存一个新 checkpoint，自动加载该 checkpoint 并对验证集/测试集
执行多任务推理，生成 (原图 | 预测 | GT) 三列对比可视化。

监控目录: output/ssd1b_snvc_lora/checkpoint-*
数据源:   /Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train
任务:     depth, normal, segmentation, structure(edge), generation

Usage:
  python scripts/monitor_and_visualize.py                  # 持续监控
  python scripts/monitor_and_visualize.py --once           # 只处理最新 checkpoint
  python scripts/monitor_and_visualize.py --checkpoint output/ssd1b_snvc_lora/checkpoint-200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("monitor_viz")

# ─── 配置 ─────────────────────────────────────────────────────────────────
MODEL_NAME = "segmind/SSD-1B"
CKPT_DIR = PROJECT_ROOT / "output" / "ssd1b_v5_vision_only"
DATA_DIR = Path("/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train")
OUTPUT_BASE = PROJECT_ROOT / "output" / "ssd1b_v5_viz"

# SSD-1B 是蒸馏模型，使用低步数 + 低 CFG (仅 generation 任务使用)
INFERENCE_STEPS = 4
GUIDANCE_SCALE = 1.0
STRENGTH = 0.85
RESOLUTION = 512
SEED = 42

# 多步 img2img 推理配置（视觉任务）
# 从训练数据覆盖率不足的分析可知，单步推理容易在极端噪声水平出现退化。
# 多步迭代从高噪声逐步去噪到低噪声，更接近标准 Euler 离散采样，
# 可显著提升视觉任务的预测质量。
MULTI_STEP_INFERENCE = True
MULTI_STEP_COUNT = 25  # 25 步去噪：覆盖全噪声范围，中间噪声层权重大
MULTI_STEP_SCHEDULE = "euler"  # euler 离散调度

# 单步 epsilon 预测推理参数 (视觉任务使用)
# 训练目标: model(noised_z_input, t) → epsilon
# 推理: x0 = noisy - sigma * eps_pred (标准 Euler epsilon→x0 提取)
X0_PREDICTION_TIMESTEP = 551  # 对应 sigma≈1.92, 中等噪声水平
X0_REFINE_TIMESTEP = 200  # 精修步: 对 x0 加少量噪声后再预测一次, 锐化细节
USE_REFINEMENT = False  # P0 fix: disabled — t=200 refinement was not trained
# and introduces distribution shift. Enable only after sufficient training.

# 每个任务取最后 N 张作为验证/测试（训练用了前 300 张）
VAL_SAMPLES_PER_TASK = 2
TEST_SAMPLES_PER_TASK = 2

# 任务配置: (子目录名, 任务标签, prompt)
# v5 训练覆盖全部 10 个视觉任务（各 300 样本，跳过 generation）
TASKS = {
    "depth": {
        "subdir": "depth",
        "label": "Depth",
        "prompt": "Generate a metric depth visualization of this image.",
    },
    "normal": {
        "subdir": "normal",
        "label": "Normal",
        "prompt": "Generate a surface normal visualization of this image.",
    },
    "segmentation": {
        "subdir": "segmentation",
        "label": "Segmentation",
        "prompt": "Generate a semantic segmentation visualization of the input image.",
    },
    "depth_edge": {
        "subdir": "depth_edge",
        "label": "Edge",
        "prompt": "Generate an edge detection visualization of this image.",
    },
    "keypoint": {
        "subdir": "keypoint",
        "label": "Keypoint",
        "prompt": "Generate a human keypoint detection visualization of this image.",
    },
    "object_detection": {
        "subdir": "object_detection",
        "label": "ObjDet",
        "prompt": "Generate an object detection bounding box visualization of this image.",
    },
    "referring": {
        "subdir": "referring",
        "label": "Referring",
        "prompt": "Generate a referring expression segmentation visualization of this image.",
    },
    "instance": {
        "subdir": "instance",
        "label": "Instance",
        "prompt": "Generate an instance segmentation visualization of this image.",
    },
    "structure": {
        "subdir": "structure",
        "label": "Structure",
        "prompt": "Generate a structural edge visualization of this image.",
    },
    "multiview": {
        "subdir": "multiview",
        "label": "MultiView",
        "prompt": "Generate a multi-view visualization of this image.",
    },
}

POLL_INTERVAL = 30  # 秒


# ─── 工具函数 ─────────────────────────────────────────────────────────────
def load_image(path: Path, resolution: int = 256) -> Image.Image:
    return Image.open(path).convert("RGB").resize(
        (resolution, resolution), Image.LANCZOS
    )


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.float().cpu()
    if t.dim() == 4:
        t = t[0]
    if t.min() < 0:
        t = (t + 1.0) / 2.0
    t = t.clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _render_text_image(text: str, resolution: int = 256) -> Image.Image:
    """将文本 prompt 渲染为一张图（用于 Generation 任务的 Input 列展示）。"""
    img = Image.new("RGB", (resolution, resolution), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except (IOError, OSError):
        font = ImageFont.load_default()
    # 自动换行
    import textwrap
    wrapped = textwrap.fill(text, width=38)
    draw.text((10, 10), wrapped, fill=(30, 30, 30), font=font)
    draw.text((10, resolution - 20), "[Text Prompt]", fill=(120, 120, 120), font=font)
    return img


def single_step_x0_predict(generator, input_pil: Image.Image, prompt: str,
                           timestep: int = X0_PREDICTION_TIMESTEP,
                           seed: int = SEED,
                           refine: bool = USE_REFINEMENT) -> Image.Image:
    """Epsilon-prediction推理 — 与训练目标和SSD-1B预训练完全对齐。

    训练目标: model(noised_z_input, t) → epsilon
    推理: x0 = noisy - sigma * epsilon_pred (标准 Euler epsilon→x0 提取)

    对于img2img跨图预测:
      noisy = z_cond + noise * sigma (加噪到条件latent)
      x0 = noisy - sigma * model_output (从噪声中提取目标latent)

    可选精修 (refine=True): 对提取的x0再加少量噪声(t≈200),
    再预测一次epsilon并提取x0, 可锐化细节。
    """
    return multi_step_x0_predict(
        generator, input_pil, prompt,
        num_steps=1, seed=seed, timestep_override=timestep,
    )


def multi_step_x0_predict(generator, input_pil: Image.Image, prompt: str,
                          num_steps: int = MULTI_STEP_COUNT,
                          seed: int = SEED,
                          timestep_override: int | None = None,
                          strength: float = STRENGTH) -> Image.Image:
    """多步 epsilon→x0 迭代推理 — 从高噪声逐步去噪到低噪声。

    与训练 epsilon 预测目标完全对齐的推理策略：
    1. 编码输入图到 latent (z_cond)
    2. 生成 num_steps 个从高→低的 timestep 序列（覆盖 strength 比例的噪声范围）
    3. 在每一步：
       a. 对 z_cond 加噪到 t 得到 noisy
       b. 模型预测 epsilon
       c. x0 = noisy - sigma * eps_pred (epsilon→x0 提取)
       d. 用提取的 x0 作为下一步的 condition latent（迭代精化）
    4. 最终 VAE 解码 x0

    对于 num_steps=1 且 timestep_override 指定时，退化为单步推理（向后兼容）。
    """
    pipe = generator._get_i2i_pipe()
    device = pipe.unet.device
    dtype = next(pipe.unet.parameters()).dtype

    # 1. 编码输入图到 latent space
    input_arr = np.array(input_pil, dtype=np.float32) / 255.0
    input_tensor = torch.from_numpy(input_arr).permute(2, 0, 1).unsqueeze(0)
    input_tensor = (input_tensor * 2.0 - 1.0).to(device=device, dtype=dtype)

    with torch.no_grad():
        z_cond = pipe.vae.encode(input_tensor).latent_dist.mode()
        z_cond = z_cond * pipe.vae.config.scaling_factor

    # 2. 编码 prompt (复用)
    text_out = generator.encode_prompts([prompt])
    prompt_embeds = text_out["prompt_embeds"].to(device=device, dtype=dtype)
    pooled = text_out.get("pooled_prompt_embeds")
    if pooled is not None:
        pooled = pooled.to(device=device, dtype=dtype)

    # 3. SDXL added_cond_kwargs
    add_time_ids = torch.tensor(
        [[512, 512, 0, 0, 512, 512]], device=device, dtype=dtype
    )
    added_cond_kwargs = {"text_embeds": pooled, "time_ids": add_time_ids}

    scheduler = pipe.scheduler
    # 始终初始化 scheduler sigmas（用于 index_for_timestep 查找）
    scheduler.set_timesteps(max(num_steps, 20), device=device)

    generator_rng = torch.Generator(device="cpu").manual_seed(seed)

    # 构建 timestep 序列
    cfg = getattr(scheduler, "config", scheduler)
    n_train = int(getattr(cfg, "num_train_timesteps", 1000))

    if num_steps == 1 and timestep_override is not None:
        # 单步模式：使用指定的固定 timestep（向后兼容）
        timesteps_seq = [timestep_override]
    else:
        # 多步模式：从高噪声到低噪声，覆盖 [strength * n_train, 低噪声]
        max_t = int(n_train * strength)
        # 对数间隔：在高噪声区域采样更密集
        timesteps_seq = list(np.linspace(max_t, 50, num_steps).astype(int))[::-1]
        # 从高噪声开始去噪
        timesteps_seq = sorted(timesteps_seq, reverse=True)

    # 迭代去噪
    z_current = z_cond  # 初始 condition latent

    for step_idx, t_val in enumerate(timesteps_seq):
        t = torch.tensor([t_val], device=device, dtype=torch.long)

        # 对当前 condition latent 加噪到 timestep t_val
        noise = torch.randn(z_current.shape, generator=generator_rng, device="cpu")
        noise = noise.to(device=device, dtype=dtype)
        noisy = scheduler.add_noise(z_current, noise, t)

        # scale_model_input
        try:
            step_indices = [scheduler.index_for_timestep(int(t_val), scheduler.timesteps.to(device))]
            sigma_val = scheduler.sigmas.to(device=device, dtype=dtype)[step_indices[0]]
        except (AttributeError, IndexError):
            sigma_val = torch.tensor(t_val / 100.0, device=device, dtype=dtype)
        model_input = noisy / ((sigma_val ** 2 + 1) ** 0.5)

        with torch.no_grad():
            eps_pred = pipe.unet(
                model_input, t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

        # epsilon → x0 提取
        x0_pred = noisy - sigma_val * eps_pred

        # 下一步用提取的 x0 作为新的 condition（迭代精化）
        z_current = x0_pred

    # VAE 解码最终 x0
    with torch.no_grad():
        decoded = pipe.vae.decode(z_current / pipe.vae.config.scaling_factor).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1)

    return tensor_to_pil(decoded[0])


def make_triptych(
    input_img: Image.Image,
    pred_img: Image.Image,
    target_img: Image.Image,
    task: str,
    sample_name: str,
    output_dir: Path,
) -> Path:
    """创建 Input | Prediction | GT 三列对比图。"""
    w, h = input_img.size
    label_h = 30
    canvas_w = w * 3 + 20
    canvas_h = h + label_h + 10

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except (IOError, OSError):
        font = ImageFont.load_default()

    labels = ["Input", f"Pred ({task})", "GT"]
    images = [input_img, pred_img, target_img]
    x_offsets = [0, w + 10, 2 * (w + 10)]

    for label, img, x in zip(labels, images, x_offsets):
        draw.text((x + 5, 5), label, fill=(0, 0, 0), font=font)
        canvas.paste(img, (x, label_h))

    save_path = output_dir / f"{task}_{sample_name}.png"
    canvas.save(save_path, quality=95)
    return save_path


def make_grid(all_results: list[dict], output_dir: Path, title: str) -> Path | None:
    """生成所有任务结果的汇总网格图。"""
    if not all_results:
        return None

    n = len(all_results)
    cols = 3
    cell_w, cell_h = 256, 256
    label_h = 35
    margin = 5

    grid_w = cols * (cell_w + margin) + margin
    grid_h = n * (cell_h + label_h + margin) + margin + 50

    grid = Image.new("RGB", (grid_w, grid_h), (240, 240, 240))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
        font_big = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_big = font

    draw.text((margin, 8), title, fill=(0, 0, 0), font=font_big)

    for i, item in enumerate(all_results):
        y = 50 + i * (cell_h + label_h + margin)
        task_label = f"[{item['task']}] {item['split']} — {item['name']}"
        draw.text((margin, y), task_label, fill=(50, 50, 50), font=font)

        col_labels = ["Input", "Prediction", "Ground Truth"]
        for j, (lbl, img) in enumerate(zip(col_labels, item["images"])):
            x = margin + j * (cell_w + margin)
            draw.text((x, y + 15), lbl, fill=(0, 0, 0), font=font)
            grid.paste(img.resize((cell_w, cell_h), Image.LANCZOS), (x, y + label_h))

    grid_path = output_dir / "results_grid.png"
    grid.save(grid_path, quality=95)
    return grid_path


def get_val_test_samples(task_subdir: str, task_key: str):
    """获取验证集和测试集样本（使用训练未用到的尾部数据）。

    训练使用了前 300 张 (max_per_task=300)，所以:
    - 验证集: 第 300~304 张 (index 300-304)
    - 测试集: 第 305~309 张 (index 305-309)
    对于 generation 任务 (训练用了 3000 张):
    - 验证集: 第 3000~3004 张
    - 测试集: 第 3005~3009 张
    """
    task_dir = DATA_DIR / task_subdir
    img_dir = task_dir / "images"
    tgt_dir = task_dir / "targets"

    if not img_dir.exists():
        return [], []

    all_images = sorted(img_dir.glob("*.jpg"))
    if not all_images:
        all_images = sorted(img_dir.glob("*.png"))
    if not all_images:
        return [], []

    # v5 训练: 所有视觉任务各 300 张
    train_count = 300
    remaining = all_images[train_count:]

    # 如果剩余不够，从全部数据末尾取
    if len(remaining) < VAL_SAMPLES_PER_TASK + TEST_SAMPLES_PER_TASK:
        remaining = all_images[-(VAL_SAMPLES_PER_TASK + TEST_SAMPLES_PER_TASK):]

    val_images = remaining[:VAL_SAMPLES_PER_TASK]
    test_images = remaining[VAL_SAMPLES_PER_TASK:VAL_SAMPLES_PER_TASK + TEST_SAMPLES_PER_TASK]

    def build_samples(img_list):
        samples = []
        for img_path in img_list:
            # 寻找对应 target
            tgt_path = tgt_dir / img_path.name
            if not tgt_path.exists():
                # 尝试匹配 stem
                candidates = list(tgt_dir.glob(img_path.stem + "*"))
                tgt_path = candidates[0] if candidates else None
            samples.append({"image": img_path, "target": tgt_path})
        return samples

    return build_samples(val_images), build_samples(test_images)


def get_generation_prompt(task_dir: Path, img_name: str) -> str:
    """从 prompts.jsonl 获取 generation 任务的 prompt。"""
    prompts_file = task_dir / "prompts.jsonl"
    if prompts_file.exists():
        with open(prompts_file) as f:
            for line in f:
                item = json.loads(line)
                if item.get("image") == img_name:
                    return item.get("prompt", "a photograph")
    return "a photograph"


# ─── 推理核心 ─────────────────────────────────────────────────────────────
def run_inference_on_checkpoint(ckpt_path: Path):
    """对指定 checkpoint 执行全部任务推理并生成可视化。"""
    from open_vision_banana.inference.generator import BaseGenerator

    ckpt_name = ckpt_path.name
    step_num = ckpt_name.split("-")[-1] if "-" in ckpt_name else "final"
    viz_dir = OUTPUT_BASE / ckpt_name
    viz_dir.mkdir(parents=True, exist_ok=True)
    (viz_dir / "triptychs").mkdir(exist_ok=True)

    logger.info(f"\n{'='*70}")
    logger.info(f"开始推理可视化: {ckpt_name} (step {step_num})")
    logger.info(f"{'='*70}")

    # 加载模型 + LoRA
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"加载 SSD-1B + LoRA ({ckpt_name}) on {device}...")
    t0 = time.time()
    generator = BaseGenerator(
        model_name=MODEL_NAME,
        lora_path=str(ckpt_path),
        device=device,
        dtype=torch.bfloat16,
    )
    logger.info(f"模型加载完成: {time.time()-t0:.1f}s")

    all_results = []
    metrics_summary = {}

    for task_key, task_cfg in TASKS.items():
        subdir = task_cfg["subdir"]
        label = task_cfg["label"]
        prompt = task_cfg["prompt"]

        val_samples, test_samples = get_val_test_samples(subdir, task_key)
        if not val_samples and not test_samples:
            logger.warning(f"  [{label}] 无可用验证/测试样本，跳过")
            continue

        logger.info(f"\n  [{label}] 验证集: {len(val_samples)} 张, 测试集: {len(test_samples)} 张")

        task_dir = DATA_DIR / subdir
        task_metrics = {"val": [], "test": []}

        for split_name, samples in [("val", val_samples), ("test", test_samples)]:
            split_dir = viz_dir / split_name
            split_dir.mkdir(exist_ok=True)

            for i, sample in enumerate(samples):
                img_path = sample["image"]
                tgt_path = sample["target"]
                sample_name = img_path.stem

                # 加载原图和 GT
                input_pil = load_image(img_path, RESOLUTION)

                # Generation 任务: targets 是 .txt 文本提示, 不是图片
                # GT 就是 images/ 中的参考图, Input 用文本渲染图代替
                if task_key == "generation":
                    target_pil = input_pil  # GT = 参考生成图
                elif tgt_path and tgt_path.exists() and tgt_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                    target_pil = load_image(tgt_path, RESOLUTION)
                else:
                    target_pil = input_pil

                # 确定 prompt
                if task_key == "generation":
                    task_prompt = get_generation_prompt(task_dir, img_path.name)
                    # 如果 prompts.jsonl 没有匹配, 尝试从 .txt target 读取
                    if task_prompt == "a photograph" and tgt_path and tgt_path.exists() and tgt_path.suffix == ".txt":
                        task_prompt = tgt_path.read_text().strip() or "a photograph"
                    # 为 generation 创建文本渲染的 Input 图
                    input_pil = _render_text_image(task_prompt, RESOLUTION)
                else:
                    task_prompt = prompt

                # 推理
                try:
                    t1 = time.time()
                    with torch.no_grad():
                        if task_key == "generation":
                            result = generator.generate(
                                prompt=task_prompt,
                                num_inference_steps=INFERENCE_STEPS,
                                guidance_scale=GUIDANCE_SCALE,
                                height=RESOLUTION,
                                width=RESOLUTION,
                                seed=SEED + i,
                            )
                            pred_pil = tensor_to_pil(result[0])
                        else:
                            # 多步 epsilon→x0 迭代推理（或单步，取决于 MULTI_STEP_INFERENCE）
                            if MULTI_STEP_INFERENCE:
                                pred_pil = multi_step_x0_predict(
                                    generator, input_pil, task_prompt,
                                    num_steps=MULTI_STEP_COUNT,
                                    seed=SEED + i,
                                )
                            else:
                                pred_pil = single_step_x0_predict(
                                    generator, input_pil, task_prompt,
                                    timestep=X0_PREDICTION_TIMESTEP,
                                    seed=SEED + i,
                                )
                    elapsed = time.time() - t1

                    # 计算任务感知指标
                    pred_arr = np.array(pred_pil, dtype=np.float32) / 255.0
                    tgt_arr = np.array(target_pil, dtype=np.float32) / 255.0
                    mse = float(np.mean((pred_arr - tgt_arr) ** 2))
                    mae = float(np.mean(np.abs(pred_arr - tgt_arr)))

                    # Pearson correlation (grayscale) — kept for backward compat
                    pred_gray = np.array(pred_pil.convert("L"), dtype=np.float32).flatten()
                    tgt_gray = np.array(target_pil.convert("L"), dtype=np.float32).flatten()
                    pred_norm = pred_gray - pred_gray.mean()
                    tgt_norm = tgt_gray - tgt_gray.mean()
                    denom = np.sqrt((pred_norm**2).sum() * (tgt_norm**2).sum())
                    pearson_r = float((pred_norm * tgt_norm).sum() / denom) if denom > 1e-6 else 0.0

                    # ── Task-specific metrics ──────────────────────────────
                    task_specific = {}
                    if task_key == "segmentation":
                        # mIoU via color quantization (round to 8 bins per channel)
                        pred_q = (pred_arr * 7).astype(int).clip(0, 7)
                        tgt_q = (tgt_arr * 7).astype(int).clip(0, 7)
                        # Flatten to quantized class IDs
                        pred_ids = pred_q[..., 0] * 64 + pred_q[..., 1] * 8 + pred_q[..., 2]
                        tgt_ids = tgt_q[..., 0] * 64 + tgt_q[..., 1] * 8 + tgt_q[..., 2]
                        unique = np.unique(np.concatenate([pred_ids, tgt_ids]))
                        ious = []
                        for cid in unique:
                            p = pred_ids == cid
                            t = tgt_ids == cid
                            inter = (p & t).sum()
                            union = (p | t).sum()
                            if union > 0:
                                ious.append(inter / union)
                        task_specific["miou"] = float(np.mean(ious)) if ious else 0.0
                        task_specific["pixel_acc"] = float((pred_ids == tgt_ids).mean())

                    elif task_key in ("structure", "edge"):
                        # Precision/Recall/F1 on binary edge maps
                        pred_edge = (pred_gray > 50).astype(np.uint8)
                        tgt_edge = (tgt_gray > 50).astype(np.uint8)
                        tp = ((pred_edge == 1) & (tgt_edge == 1)).sum()
                        fp = ((pred_edge == 1) & (tgt_edge == 0)).sum()
                        fn = ((pred_edge == 0) & (tgt_edge == 1)).sum()
                        prec = tp / (tp + fp + 1e-8)
                        rec = tp / (tp + fn + 1e-8)
                        task_specific["precision"] = float(prec)
                        task_specific["recall"] = float(rec)
                        task_specific["f1"] = float(2 * prec * rec / (prec + rec + 1e-8))

                    elif task_key == "normal":
                        # Angular error in degrees (RGB → XYZ normal vectors)
                        pred_n = pred_arr * 2.0 - 1.0
                        tgt_n = tgt_arr * 2.0 - 1.0
                        # Normalize
                        pred_nmag = np.linalg.norm(pred_n, axis=2, keepdims=True) + 1e-8
                        tgt_nmag = np.linalg.norm(tgt_n, axis=2, keepdims=True) + 1e-8
                        pred_normed = pred_n / pred_nmag
                        tgt_normed = tgt_n / tgt_nmag
                        cos_sim = np.sum(pred_normed * tgt_normed, axis=2).clip(-1, 1)
                        angle_deg = np.degrees(np.arccos(cos_sim))
                        task_specific["angular_error_deg"] = float(np.mean(angle_deg))
                        task_specific["median_angular_error_deg"] = float(np.median(angle_deg))

                    elif task_key == "depth":
                        # Spearman rank correlation (monotonic depth ordering)
                        from scipy.stats import spearmanr
                        sp_r, _ = spearmanr(pred_gray, tgt_gray)
                        task_specific["spearman_r"] = float(sp_r) if not np.isnan(sp_r) else 0.0
                        # Relative error (L1 in normalized depth)
                        task_specific["rel_l1"] = float(np.mean(np.abs(pred_gray - tgt_gray) / (tgt_gray + 1e-6)))

                    elif task_key == "generation":
                        # For generation, pixel-level metrics are meaningless.
                        # Report CLIP similarity if available, otherwise skip.
                        task_specific["note"] = "Generation requires CLIP/FID, not pixel metrics"

                    task_metrics[split_name].append({
                        "sample": sample_name,
                        "mse": mse,
                        "mae": mae,
                        "pearson_r": pearson_r,
                        "time": elapsed,
                        **task_specific,
                    })

                    # 保存三列对比图
                    triptych_path = make_triptych(
                        input_pil, pred_pil, target_pil,
                        label, f"{split_name}_{sample_name}",
                        split_dir,
                    )

                    all_results.append({
                        "task": label,
                        "split": split_name,
                        "name": sample_name,
                        "images": [input_pil, pred_pil, target_pil],
                    })

                    logger.info(
                        f"    [{split_name}] {sample_name}: "
                        f"r={pearson_r:.3f} MSE={mse:.4f} ({elapsed:.1f}s)"
                    )

                except Exception as e:
                    logger.error(f"    [{split_name}] {sample_name} 推理失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        # 汇总该任务的指标
        for split_name in ["val", "test"]:
            if task_metrics[split_name]:
                avg_r = np.mean([m["pearson_r"] for m in task_metrics[split_name]])
                avg_mse = np.mean([m["mse"] for m in task_metrics[split_name]])
                avg_mae = np.mean([m["mae"] for m in task_metrics[split_name]])

                summary_entry = {
                    "pearson_r": float(avg_r),
                    "mse": float(avg_mse),
                    "mae": float(avg_mae),
                    "n_samples": len(task_metrics[split_name]),
                }

                # Add task-specific metric averages
                task_metric_keys = set()
                for m in task_metrics[split_name]:
                    task_metric_keys.update(
                        k for k in m
                        if k not in ("sample", "mse", "mae", "pearson_r", "time")
                    )
                for k in sorted(task_metric_keys):
                    vals = [m[k] for m in task_metrics[split_name] if isinstance(m.get(k), (int, float))]
                    if vals:
                        summary_entry[k] = float(np.mean(vals))

                metrics_summary[f"{task_key}/{split_name}"] = summary_entry

                # Log task-specific metrics
                extra_str = ""
                for k in sorted(task_metric_keys):
                    vals = [m[k] for m in task_metrics[split_name] if isinstance(m.get(k), (int, float))]
                    if vals:
                        extra_str += f", {k}={np.mean(vals):.3f}"

                logger.info(
                    f"  [{label}/{split_name}] 平均: r={avg_r:.3f}, MSE={avg_mse:.4f}, "
                    f"MAE={avg_mae:.4f}{extra_str}"
                )

    # 生成汇总网格图
    if all_results:
        grid_path = make_grid(
            all_results, viz_dir,
            f"SSD-1B LoRA — {ckpt_name} | Steps={INFERENCE_STEPS} CFG={GUIDANCE_SCALE}"
        )
        if grid_path:
            logger.info(f"\n网格图已保存: {grid_path}")

    # 保存指标 JSON
    metrics_path = viz_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "checkpoint": ckpt_name,
            "step": step_num,
            "config": {
                "model": MODEL_NAME,
                "inference_steps": INFERENCE_STEPS,
                "guidance_scale": GUIDANCE_SCALE,
                "strength": STRENGTH,
                "resolution": RESOLUTION,
            },
            "metrics": metrics_summary,
        }, f, indent=2)
    logger.info(f"指标已保存: {metrics_path}")

    # 释放模型内存
    del generator
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    logger.info(f"\n✓ {ckpt_name} 可视化完成! 输出: {viz_dir}")
    return viz_dir


# ─── 监控逻辑 ─────────────────────────────────────────────────────────────
def find_checkpoints() -> list[Path]:
    """查找所有有效的 checkpoint 目录。"""
    if not CKPT_DIR.exists():
        return []
    ckpts = []
    for d in sorted(CKPT_DIR.glob("checkpoint-*")):
        if (d / "adapter_model.safetensors").exists():
            ckpts.append(d)
    return ckpts


def get_processed_checkpoints() -> set[str]:
    """获取已处理过的 checkpoint 名称。"""
    processed_file = OUTPUT_BASE / ".processed_checkpoints.json"
    if processed_file.exists():
        with open(processed_file) as f:
            return set(json.load(f))
    return set()


def mark_processed(ckpt_name: str):
    """标记 checkpoint 为已处理。"""
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    processed_file = OUTPUT_BASE / ".processed_checkpoints.json"
    processed = get_processed_checkpoints()
    processed.add(ckpt_name)
    with open(processed_file, "w") as f:
        json.dump(sorted(processed), f, indent=2)


def monitor_loop():
    """持续监控新 checkpoint 并自动触发推理可视化。"""
    logger.info(f"开始监控 checkpoint 目录: {CKPT_DIR}")
    logger.info(f"轮询间隔: {POLL_INTERVAL}s")
    logger.info(f"可视化输出: {OUTPUT_BASE}")
    logger.info("等待训练保存 checkpoint...")

    while True:
        ckpts = find_checkpoints()
        processed = get_processed_checkpoints()
        new_ckpts = [c for c in ckpts if c.name not in processed]

        if new_ckpts:
            for ckpt in new_ckpts:
                logger.info(f"\n检测到新 checkpoint: {ckpt.name}")
                try:
                    run_inference_on_checkpoint(ckpt)
                except Exception as e:
                    logger.error(f"处理 {ckpt.name} 失败: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # 无论成功失败都标记为已处理，避免无限重试
                    mark_processed(ckpt.name)
        else:
            # 静默等待（每 5 分钟打印一次状态）
            if int(time.time()) % 300 < POLL_INTERVAL:
                logger.info(f"等待中... 已处理 {len(processed)} 个 checkpoint")

        time.sleep(POLL_INTERVAL)


def main():
    parser = argparse.ArgumentParser(description="Checkpoint 监控 + 多任务推理可视化")
    parser.add_argument("--once", action="store_true",
                        help="只处理最新的 checkpoint 后退出")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="指定要处理的 checkpoint 路径")
    parser.add_argument("--all", action="store_true",
                        help="处理所有未处理的 checkpoint")
    args = parser.parse_args()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        # 处理指定 checkpoint
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            logger.error(f"Checkpoint 不存在: {ckpt}")
            sys.exit(1)
        run_inference_on_checkpoint(ckpt)
        mark_processed(ckpt.name)
    elif args.once:
        # 处理最新 checkpoint
        ckpts = find_checkpoints()
        if not ckpts:
            logger.info("暂无可用 checkpoint")
            sys.exit(0)
        latest = ckpts[-1]
        logger.info(f"处理最新 checkpoint: {latest.name}")
        run_inference_on_checkpoint(latest)
        mark_processed(latest.name)
    elif args.all:
        # 处理所有未处理的
        ckpts = find_checkpoints()
        processed = get_processed_checkpoints()
        new_ckpts = [c for c in ckpts if c.name not in processed]
        if not new_ckpts:
            logger.info("所有 checkpoint 已处理完毕")
            sys.exit(0)
        for ckpt in new_ckpts:
            run_inference_on_checkpoint(ckpt)
            mark_processed(ckpt.name)
    else:
        # 持续监控模式
        monitor_loop()


if __name__ == "__main__":
    main()
