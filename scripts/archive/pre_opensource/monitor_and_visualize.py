#!/usr/bin/env python3
"""Checkpoint monitoring + multi-task inference visualization.

Whenever training saves a new checkpoint, load it automatically, run multi-task inference on the
validation/test sets and write (image | prediction | GT) three-column comparison figures.

Watched directory: output/ssd1b_snvc_lora/checkpoint-*
Data source:       /Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train
Tasks:             depth, normal, segmentation, structure(edge), generation

Usage:
  python scripts/monitor_and_visualize.py                  # keep monitoring
  python scripts/monitor_and_visualize.py --once           # process only the latest checkpoint
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

# ─── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "segmind/SSD-1B"
CKPT_DIR = PROJECT_ROOT / "output" / "ssd1b_v5_vision_only"
DATA_DIR = Path("/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train")
OUTPUT_BASE = PROJECT_ROOT / "output" / "ssd1b_v5_viz"

# SSD-1B is a distilled model: few steps + low CFG (generation task only)
INFERENCE_STEPS = 4
GUIDANCE_SCALE = 1.0
STRENGTH = 0.85
RESOLUTION = 512
SEED = 42

# multi-step img2img inference config (vision tasks)
# from the analysis of thin training-data coverage, single-step inference tends to degrade at extreme noise levels.
# multi-step iteration denoises gradually from high to low noise, closer to standard Euler discrete sampling,
# and clearly improves the prediction quality of the vision tasks.
MULTI_STEP_INFERENCE = True
MULTI_STEP_COUNT = 25  # 25 denoising steps: covers the full noise range, weighted toward the middle noise levels
MULTI_STEP_SCHEDULE = "euler"  # Euler discrete schedule

# single-step epsilon-prediction inference parameters (vision tasks)
# training objective: model(noised_z_input, t) → epsilon
# inference: x0 = noisy - sigma * eps_pred (standard Euler epsilon→x0 extraction)
X0_PREDICTION_TIMESTEP = 551  # sigma≈1.92, a medium noise level
X0_REFINE_TIMESTEP = 200  # refine step: add a little noise to x0 and predict again to sharpen details
USE_REFINEMENT = False  # P0 fix: disabled — t=200 refinement was not trained
# and introduces distribution shift. Enable only after sufficient training.

# take the last N images of each task as validation/test (training used the first 300)
VAL_SAMPLES_PER_TASK = 2
TEST_SAMPLES_PER_TASK = 2

# task config: (subdirectory, task label, prompt)
# v5 training covers all 10 vision tasks (300 samples each, generation skipped)
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

POLL_INTERVAL = 30  # seconds


# ─── Utilities ───────────────────────────────────────────────────────────
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
    """Render a text prompt as an image (shown in the Input column of the generation task)."""
    img = Image.new("RGB", (resolution, resolution), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except (IOError, OSError):
        font = ImageFont.load_default()
    # automatic line wrapping
    import textwrap
    wrapped = textwrap.fill(text, width=38)
    draw.text((10, 10), wrapped, fill=(30, 30, 30), font=font)
    draw.text((10, resolution - 20), "[Text Prompt]", fill=(120, 120, 120), font=font)
    return img


def single_step_x0_predict(generator, input_pil: Image.Image, prompt: str,
                           timestep: int = X0_PREDICTION_TIMESTEP,
                           seed: int = SEED,
                           refine: bool = USE_REFINEMENT) -> Image.Image:
    """Epsilon-prediction inference, fully aligned with the training objective and the SSD-1B pretraining.

    training objective: model(noised_z_input, t) → epsilon
    inference: x0 = noisy - sigma * epsilon_pred (standard Euler epsilon→x0 extraction)

    for img2img cross-image prediction:
      noisy = z_cond + noise * sigma (noise added to the condition latent)
      x0 = noisy - sigma * model_output (extract the target latent from the noise)

    optional refine (refine=True): add a little noise (t≈200) to the extracted x0,
    predict epsilon once more and extract x0 again, which sharpens details.
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
    """Multi-step epsilon→x0 iterative inference, denoising gradually from high to low noise.

    An inference strategy fully aligned with the epsilon-prediction training objective:
    1. encode the input image into a latent (z_cond)
    2. build num_steps timesteps from high to low (covering the strength fraction of the noise range)
    3. at each step:
       a. noise z_cond to t to get noisy
       b. the model predicts epsilon
       c. x0 = noisy - sigma * eps_pred (epsilon→x0 extraction)
       d. the extracted x0 becomes the next step's condition latent (iterative refinement)
    4. VAE-decode the final x0

    With num_steps=1 and timestep_override set, this reduces to single-step inference (backward compatible).
    """
    pipe = generator._get_i2i_pipe()
    device = pipe.unet.device
    dtype = next(pipe.unet.parameters()).dtype

    # 1. encode the input image into latent space
    input_arr = np.array(input_pil, dtype=np.float32) / 255.0
    input_tensor = torch.from_numpy(input_arr).permute(2, 0, 1).unsqueeze(0)
    input_tensor = (input_tensor * 2.0 - 1.0).to(device=device, dtype=dtype)

    with torch.no_grad():
        z_cond = pipe.vae.encode(input_tensor).latent_dist.mode()
        z_cond = z_cond * pipe.vae.config.scaling_factor

    # 2. encode the prompt (reused)
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
    # always initialize the scheduler sigmas (used by the index_for_timestep lookup)
    scheduler.set_timesteps(max(num_steps, 20), device=device)

    generator_rng = torch.Generator(device="cpu").manual_seed(seed)

    # build the timestep sequence
    cfg = getattr(scheduler, "config", scheduler)
    n_train = int(getattr(cfg, "num_train_timesteps", 1000))

    if num_steps == 1 and timestep_override is not None:
        # single-step mode: use the given fixed timestep (backward compatible)
        timesteps_seq = [timestep_override]
    else:
        # multi-step mode: from high to low noise, covering [strength * n_train, low noise]
        max_t = int(n_train * strength)
        # log spacing: denser sampling in the high-noise region
        timesteps_seq = list(np.linspace(max_t, 50, num_steps).astype(int))[::-1]
        # start denoising from high noise
        timesteps_seq = sorted(timesteps_seq, reverse=True)

    # iterative denoising
    z_current = z_cond  # initial condition latent

    for step_idx, t_val in enumerate(timesteps_seq):
        t = torch.tensor([t_val], device=device, dtype=torch.long)

        # noise the current condition latent to timestep t_val
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

        # epsilon → x0 extraction
        x0_pred = noisy - sigma_val * eps_pred

        # the extracted x0 becomes the new condition for the next step (iterative refinement)
        z_current = x0_pred

    # VAE-decode the final x0
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
    """Build an Input | Prediction | GT three-column comparison figure."""
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
    """Build the summary grid of every task's results."""
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
    """Get validation and test samples (from the tail data training did not use).

    Training used the first 300 images (max_per_task=300), so:
    - validation set: images 300~304 (index 300-304)
    - test set: images 305~309 (index 305-309)
    For the generation task (training used 3000 images):
    - validation set: images 3000~3004
    - test set: images 3005~3009
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

    # v5 training: 300 images for each vision task
    train_count = 300
    remaining = all_images[train_count:]

    # if not enough remain, take from the end of all data
    if len(remaining) < VAL_SAMPLES_PER_TASK + TEST_SAMPLES_PER_TASK:
        remaining = all_images[-(VAL_SAMPLES_PER_TASK + TEST_SAMPLES_PER_TASK):]

    val_images = remaining[:VAL_SAMPLES_PER_TASK]
    test_images = remaining[VAL_SAMPLES_PER_TASK:VAL_SAMPLES_PER_TASK + TEST_SAMPLES_PER_TASK]

    def build_samples(img_list):
        samples = []
        for img_path in img_list:
            # find the matching target
            tgt_path = tgt_dir / img_path.name
            if not tgt_path.exists():
                # try matching the stem
                candidates = list(tgt_dir.glob(img_path.stem + "*"))
                tgt_path = candidates[0] if candidates else None
            samples.append({"image": img_path, "target": tgt_path})
        return samples

    return build_samples(val_images), build_samples(test_images)


def get_generation_prompt(task_dir: Path, img_name: str) -> str:
    """Get the generation task prompts from prompts.jsonl."""
    prompts_file = task_dir / "prompts.jsonl"
    if prompts_file.exists():
        with open(prompts_file) as f:
            for line in f:
                item = json.loads(line)
                if item.get("image") == img_name:
                    return item.get("prompt", "a photograph")
    return "a photograph"


# ─── Inference core ──────────────────────────────────────────────────────
def run_inference_on_checkpoint(ckpt_path: Path):
    """Run inference for every task on the given checkpoint and build the visualizations."""
    from open_vision_banana.inference.generator import BaseGenerator

    ckpt_name = ckpt_path.name
    step_num = ckpt_name.split("-")[-1] if "-" in ckpt_name else "final"
    viz_dir = OUTPUT_BASE / ckpt_name
    viz_dir.mkdir(parents=True, exist_ok=True)
    (viz_dir / "triptychs").mkdir(exist_ok=True)

    logger.info(f"\n{'='*70}")
    logger.info(f"Starting inference visualization: {ckpt_name} (step {step_num})")
    logger.info(f"{'='*70}")

    # load the model + LoRA
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Loading SSD-1B + LoRA ({ckpt_name}) on {device}...")
    t0 = time.time()
    generator = BaseGenerator(
        model_name=MODEL_NAME,
        lora_path=str(ckpt_path),
        device=device,
        dtype=torch.bfloat16,
    )
    logger.info(f"Model loaded: {time.time()-t0:.1f}s")

    all_results = []
    metrics_summary = {}

    for task_key, task_cfg in TASKS.items():
        subdir = task_cfg["subdir"]
        label = task_cfg["label"]
        prompt = task_cfg["prompt"]

        val_samples, test_samples = get_val_test_samples(subdir, task_key)
        if not val_samples and not test_samples:
            logger.warning(f"  [{label}] no validation/test samples available, skipping")
            continue

        logger.info(f"\n  [{label}] validation: {len(val_samples)} images, test: {len(test_samples)} images")

        task_dir = DATA_DIR / subdir
        task_metrics = {"val": [], "test": []}

        for split_name, samples in [("val", val_samples), ("test", test_samples)]:
            split_dir = viz_dir / split_name
            split_dir.mkdir(exist_ok=True)

            for i, sample in enumerate(samples):
                img_path = sample["image"]
                tgt_path = sample["target"]
                sample_name = img_path.stem

                # load the image and the GT
                input_pil = load_image(img_path, RESOLUTION)

                # generation task: targets are .txt text prompts, not images
                # the GT is the reference image in images/, and a rendered text image replaces the Input
                if task_key == "generation":
                    target_pil = input_pil  # GT = reference generated image
                elif tgt_path and tgt_path.exists() and tgt_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                    target_pil = load_image(tgt_path, RESOLUTION)
                else:
                    target_pil = input_pil

                # pick the prompt
                if task_key == "generation":
                    task_prompt = get_generation_prompt(task_dir, img_path.name)
                    # if prompts.jsonl has no match, try reading the .txt target
                    if task_prompt == "a photograph" and tgt_path and tgt_path.exists() and tgt_path.suffix == ".txt":
                        task_prompt = tgt_path.read_text().strip() or "a photograph"
                    # build a text-rendered Input image for generation
                    input_pil = _render_text_image(task_prompt, RESOLUTION)
                else:
                    task_prompt = prompt

                # inference
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
                            # multi-step epsilon→x0 iterative inference (or single step, depending on MULTI_STEP_INFERENCE)
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

                    # task-aware metrics
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

                    # save the three-column comparison figure
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
                    logger.error(f"    [{split_name}] {sample_name} inference failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        # summarize this task's metrics
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
                    f"  [{label}/{split_name}] mean: r={avg_r:.3f}, MSE={avg_mse:.4f}, "
                    f"MAE={avg_mae:.4f}{extra_str}"
                )

    # build the summary grid
    if all_results:
        grid_path = make_grid(
            all_results, viz_dir,
            f"SSD-1B LoRA — {ckpt_name} | Steps={INFERENCE_STEPS} CFG={GUIDANCE_SCALE}"
        )
        if grid_path:
            logger.info(f"\nGrid saved: {grid_path}")

    # save the metrics JSON
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
    logger.info(f"Metrics saved: {metrics_path}")

    # free the model memory
    del generator
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    logger.info(f"\n✓ {ckpt_name} visualization done! Output: {viz_dir}")
    return viz_dir


# ─── Monitoring ──────────────────────────────────────────────────────────
def find_checkpoints() -> list[Path]:
    """Find every valid checkpoint directory."""
    if not CKPT_DIR.exists():
        return []
    ckpts = []
    for d in sorted(CKPT_DIR.glob("checkpoint-*")):
        if (d / "adapter_model.safetensors").exists():
            ckpts.append(d)
    return ckpts


def get_processed_checkpoints() -> set[str]:
    """Get the names of the checkpoints already processed."""
    processed_file = OUTPUT_BASE / ".processed_checkpoints.json"
    if processed_file.exists():
        with open(processed_file) as f:
            return set(json.load(f))
    return set()


def mark_processed(ckpt_name: str):
    """Mark a checkpoint as processed."""
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    processed_file = OUTPUT_BASE / ".processed_checkpoints.json"
    processed = get_processed_checkpoints()
    processed.add(ckpt_name)
    with open(processed_file, "w") as f:
        json.dump(sorted(processed), f, indent=2)


def monitor_loop():
    """Keep monitoring for new checkpoints and trigger the inference visualization automatically."""
    logger.info(f"Monitoring checkpoint directory: {CKPT_DIR}")
    logger.info(f"Poll interval: {POLL_INTERVAL}s")
    logger.info(f"Visualization output: {OUTPUT_BASE}")
    logger.info("Waiting for training to save a checkpoint...")

    while True:
        ckpts = find_checkpoints()
        processed = get_processed_checkpoints()
        new_ckpts = [c for c in ckpts if c.name not in processed]

        if new_ckpts:
            for ckpt in new_ckpts:
                logger.info(f"\nNew checkpoint detected: {ckpt.name}")
                try:
                    run_inference_on_checkpoint(ckpt)
                except Exception as e:
                    logger.error(f"Processing {ckpt.name} failed: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # mark it processed whether it succeeded or failed, to avoid retrying forever
                    mark_processed(ckpt.name)
        else:
            # wait quietly (print a status every 5 minutes)
            if int(time.time()) % 300 < POLL_INTERVAL:
                logger.info(f"Waiting... {len(processed)} checkpoints processed")

        time.sleep(POLL_INTERVAL)


def main():
    parser = argparse.ArgumentParser(description="Checkpoint monitoring + multi-task inference visualization")
    parser.add_argument("--once", action="store_true",
                        help="Process only the latest checkpoint and exit")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path of the checkpoint to process")
    parser.add_argument("--all", action="store_true",
                        help="Process every unprocessed checkpoint")
    args = parser.parse_args()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        # process the given checkpoint
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            logger.error(f"Checkpoint not found: {ckpt}")
            sys.exit(1)
        run_inference_on_checkpoint(ckpt)
        mark_processed(ckpt.name)
    elif args.once:
        # process the latest checkpoint
        ckpts = find_checkpoints()
        if not ckpts:
            logger.info("No checkpoint available yet")
            sys.exit(0)
        latest = ckpts[-1]
        logger.info(f"Processing the latest checkpoint: {latest.name}")
        run_inference_on_checkpoint(latest)
        mark_processed(latest.name)
    elif args.all:
        # process every unprocessed one
        ckpts = find_checkpoints()
        processed = get_processed_checkpoints()
        new_ckpts = [c for c in ckpts if c.name not in processed]
        if not new_ckpts:
            logger.info("Every checkpoint has been processed")
            sys.exit(0)
        for ckpt in new_ckpts:
            run_inference_on_checkpoint(ckpt)
            mark_processed(ckpt.name)
    else:
        # keep-monitoring mode
        monitor_loop()


if __name__ == "__main__":
    main()
