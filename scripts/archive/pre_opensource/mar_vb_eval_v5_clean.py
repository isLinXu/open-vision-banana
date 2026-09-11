#!/usr/bin/env python3
"""Evaluate a MAR-VB v5 multitask checkpoint on an explicit clean manifest."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "output/mar_vb_clean_pairs_v1/pairs.jsonl"
TASKS = ("depth", "normal", "seg", "structure")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--n-per-task", type=int, default=16)
    parser.add_argument("--sample-iter", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="DiffLoss sampling temperature; 0 makes denoising deterministic after initial noise.",
    )
    parser.add_argument(
        "--tasks", default="",
        help="Comma-separated tasks to evaluate (default: every task in the manifest).",
    )
    parser.add_argument(
        "--split", choices=("train", "val"), default="val",
        help="Manifest split to evaluate (default: val).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument(
        "--seg-palette", type=Path,
        help="Fixed segmentation palette JSON. Defaults to the manifest metadata when present.",
    )
    return parser.parse_args()


def load_training_module(img_size: int):
    os.environ["VB_IMG_SIZE"] = str(img_size)
    os.environ["VB_NUM_WORKERS"] = "0"
    path = PROJECT_ROOT / "scripts/mar_vb_train_v5_multitask.py"
    spec = importlib.util.spec_from_file_location("mar_vb_v5_clean_train", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_rows(manifest: Path, n_per_task: int, requested_tasks: tuple[str, ...] = (),
              split: str = "val"):
    rows = [json.loads(line) for line in manifest.open() if line.strip()]
    by_task = {task: [] for task in (requested_tasks or TASKS)}
    for row in rows:
        if row.get("split") == split and row.get("task") in by_task:
            by_task[row["task"]].append(row)
    for task, task_rows in tuple(by_task.items()):
        task_rows.sort(key=lambda row: row["input"])
        by_task[task] = task_rows[:n_per_task]
    # A manifest may intentionally omit a task whose original supervision was
    # found to be invalid (for example identity segmentation pairs).  Evaluate
    # every task it does contain, rather than manufacturing a metric for one
    # that has no held-out examples.
    by_task = {task: task_rows for task, task_rows in by_task.items() if task_rows}
    if not by_task:
        raise ValueError("No validation rows found for any supported task")
    return by_task


def parse_task_list(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    aliases = {"segmentation": "seg", "struct": "structure"}
    tasks = tuple(aliases.get(task.strip(), task.strip())
                  for task in value.split(",") if task.strip())
    unknown = sorted(set(tasks) - set(TASKS))
    if unknown:
        raise ValueError(f"unknown --tasks value(s): {', '.join(unknown)}")
    return tuple(dict.fromkeys(tasks))


def load_image(path: str, size: int, *, resample=Image.BICUBIC) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((size, size), resample)
        arr = np.asarray(rgb, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def depth_lut() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.float32)
    for value in range(256):
        t = value / 255.0
        if t < 0.25:
            color = (0, int(t * 4 * 255), 255)
        elif t < 0.5:
            color = (0, 255, int(255 - (t - 0.25) * 4 * 255))
        elif t < 0.75:
            color = (int((t - 0.5) * 4 * 255), 255, 0)
        else:
            color = (255, int(255 - (t - 0.75) * 4 * 255), 0)
        lut[value] = np.array(color, dtype=np.float32) / 255.0
    return lut


def decode_depth_rgb(image: torch.Tensor, lut: np.ndarray) -> np.ndarray:
    """Invert the v5 pseudo-depth rainbow encoding by nearest LUT colour."""
    pixels = image.permute(1, 2, 0).cpu().numpy().reshape(-1, 3)
    values = np.empty(len(pixels), dtype=np.float32)
    for start in range(0, len(pixels), 8192):
        chunk = pixels[start:start + 8192]
        dist = ((chunk[:, None, :] - lut[None, :, :]) ** 2).sum(axis=2)
        values[start:start + len(chunk)] = dist.argmin(axis=1) / 255.0
    return values


def rank_corr(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = np.argsort(np.argsort(left, kind="stable"), kind="stable")
    right_rank = np.argsort(np.argsort(right, kind="stable"), kind="stable")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def normal_angle_deg(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = torch.nn.functional.normalize(pred * 2 - 1, dim=0, eps=1e-6)
    t = torch.nn.functional.normalize(target * 2 - 1, dim=0, eps=1e-6)
    dot = (p * t).sum(dim=0).clamp(-1, 1)
    return float(torch.rad2deg(torch.acos(dot)).mean().item())


def edge_f1_ods(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_gray = pred.mean(dim=0)
    target_edge = target.mean(dim=0) >= 0.5
    best = 0.0
    for threshold in np.linspace(0.1, 0.9, 17):
        pred_edge = pred_gray >= threshold
        tp = (pred_edge & target_edge).sum().item()
        fp = (pred_edge & ~target_edge).sum().item()
        fn = (~pred_edge & target_edge).sum().item()
        score = 2 * tp / max(1, 2 * tp + fp + fn)
        best = max(best, score)
    return float(best)


def load_segmentation_palette(rows: dict[str, list[dict]], explicit: Path | None):
    """Load the palette required to turn decoded segmentation RGB into labels."""
    path = explicit
    if path is None:
        for row in rows.get("seg", []):
            value = row.get("segmentation_palette")
            if value:
                path = Path(value)
                break
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"segmentation palette not found: {path}")
    data = json.loads(path.read_text())
    palette = np.asarray(data.get("class_palette"), dtype=np.float32)
    ignore = np.asarray(data.get("ignore_color"), dtype=np.float32)
    if palette.ndim != 2 or palette.shape[1] != 3 or len(palette) < 2:
        raise ValueError(f"invalid class_palette in {path}")
    if ignore.shape != (3,):
        raise ValueError(f"invalid ignore_color in {path}")
    return {"path": path.resolve(), "palette": palette / 255.0, "ignore": ignore / 255.0}


def quantize_palette(image: torch.Tensor, palette: np.ndarray) -> np.ndarray:
    """Snap RGB pixels to the closest fixed palette color without a SciPy dependency."""
    pixels = image.permute(1, 2, 0).cpu().numpy().reshape(-1, 3)
    labels = np.empty(len(pixels), dtype=np.int32)
    for start in range(0, len(pixels), 8192):
        chunk = pixels[start:start + 8192]
        distances = ((chunk[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
        labels[start:start + len(chunk)] = distances.argmin(axis=1)
    return labels


def segmentation_confusion(pred: torch.Tensor, target: torch.Tensor, palette_info: dict) -> np.ndarray:
    """Return a class confusion matrix while excluding ground-truth ignore pixels."""
    palette, ignore = palette_info["palette"], palette_info["ignore"]
    pred_labels = quantize_palette(pred, palette)
    target_pixels = target.permute(1, 2, 0).cpu().numpy().reshape(-1, 3)
    valid = np.max(np.abs(target_pixels - ignore), axis=1) > (0.5 / 255.0)
    target_labels = quantize_palette(target, palette)
    num_classes = len(palette)
    return np.bincount(
        num_classes * target_labels[valid] + pred_labels[valid],
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def segmentation_metrics(confusion: np.ndarray) -> dict[str, float | int]:
    true_positive = np.diag(confusion).astype(np.float64)
    target_total = confusion.sum(axis=1).astype(np.float64)
    pred_total = confusion.sum(axis=0).astype(np.float64)
    union = target_total + pred_total - true_positive
    present = union > 0
    valid_pixels = int(target_total.sum())
    return {
        "seg_pixel_accuracy": float(true_positive.sum() / max(1, valid_pixels)),
        "seg_miou": float((true_positive[present] / union[present]).mean()) if present.any() else 0.0,
        "seg_present_classes": int(present.sum()),
        "seg_valid_pixels": valid_pixels,
    }


def main():
    args = parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if not args.manifest.exists():
        raise SystemExit(f"manifest not found: {args.manifest}")
    if args.n_per_task < 1 or args.sample_iter < 1:
        raise SystemExit("--n-per-task and --sample-iter must be positive")

    mt = load_training_module(args.img_size)
    device = mt.DEVICE
    base = mt.mar_large(buffer_size=64, diffloss_d=8, diffloss_w=1280,
                        num_sampling_steps="100", img_size=args.img_size,
                        attn_dropout=mt.DROPOUT, proj_dropout=mt.DROPOUT)
    state = mt.load_file(f"{mt.CKPT_DIR}/mar-large.safetensors")
    if args.img_size != 256:
        state = mt.resize_pos_keys(state, "", args.img_size // 16)
    base.load_state_dict(state, strict=True)
    model = mt.MARVBMultiTask(base, num_tasks=len(mt.TASK_LABEL)).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    if not args.no_ema and "ema" in checkpoint:
        ema = mt.EMA(model, decay=checkpoint["ema"].get("decay", 0.999))
        ema.load_state_dict(checkpoint["ema"])
        ema.apply()

    vae = mt.AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4),
                           ckpt_path=f"{mt.CKPT_DIR}/kl16.safetensors").to(device).eval()
    for param in vae.parameters():
        param.requires_grad_(False)

    rows = load_rows(args.manifest, args.n_per_task, parse_task_list(args.tasks), args.split)
    lut = depth_lut()
    seg_palette = load_segmentation_palette(rows, args.seg_palette)
    if "seg" in rows and seg_palette is None:
        print("[seg] no fixed palette metadata; RGB MAE only", flush=True)
    summary, panels = {}, {}
    with torch.no_grad():
        for task, task_rows in rows.items():
            label = mt.TASK_LABEL[task]
            rgb_maes, depth_maes, depth_corrs, normal_angles, edge_f1s = [], [], [], [], []
            seg_confusion = (np.zeros((len(seg_palette["palette"]), len(seg_palette["palette"])), dtype=np.int64)
                             if task == "seg" and seg_palette is not None else None)
            task_panels = []
            for index, row in enumerate(task_rows):
                torch.manual_seed(args.seed + label * 10000 + index)
                np.random.seed(args.seed + label * 10000 + index)
                inp = load_image(row["input"], args.img_size).unsqueeze(0).to(device)
                target_resample = Image.NEAREST if task == "seg" else Image.BICUBIC
                target = load_image(row["target"], args.img_size,
                                    resample=target_resample).unsqueeze(0).to(device)
                context = mt.encode_latent(vae, inp, sample=False)
                labels = torch.tensor([label], device=device)
                latent = model.sample_vb(context, labels, num_iter=args.sample_iter,
                                         cfg=args.cfg, temperature=args.temperature)
                pred = vae.decode(latent / mt.VAE_SCALE).clamp(-1, 1)
                pred01, target01 = (pred[0] + 1) / 2, (target[0] + 1) / 2
                rgb_maes.append(float((pred01 - target01).abs().mean().item()))
                if task == "depth":
                    pred_scalar = decode_depth_rgb(pred01, lut)
                    target_scalar = decode_depth_rgb(target01, lut)
                    depth_maes.append(float(np.abs(pred_scalar - target_scalar).mean()))
                    depth_corrs.append(rank_corr(pred_scalar, target_scalar))
                elif task == "normal":
                    normal_angles.append(normal_angle_deg(pred01, target01))
                elif task == "structure":
                    edge_f1s.append(edge_f1_ods(pred01, target01))
                elif task == "seg" and seg_confusion is not None:
                    seg_confusion += segmentation_confusion(pred01, target01, seg_palette)
                task_panels.append(torch.cat([inp[0].cpu(), target[0].cpu(), pred[0].cpu()], dim=0))

            result = {"n": len(task_rows), "rgb_mae": float(np.mean(rgb_maes))}
            if depth_maes:
                result.update(depth_scalar_mae=float(np.mean(depth_maes)),
                              depth_spearman=float(np.mean(depth_corrs)))
            if normal_angles:
                result["normal_mean_angular_error_deg"] = float(np.mean(normal_angles))
            if edge_f1s:
                result["structure_edge_f1_ods"] = float(np.mean(edge_f1s))
            if seg_confusion is not None:
                result.update(segmentation_metrics(seg_confusion))
            summary[task] = result
            panels[task] = torch.cat(task_panels, dim=0)
            print(f"[{task}] {json.dumps(result, sort_keys=True)}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    for task, panel in panels.items():
        # Each sample panel is [input, target, prediction] with 3 channels.
        save_image(panel.reshape(-1, 3, args.img_size, args.img_size),
                   args.output / f"panel_{task}.png", nrow=3,
                   normalize=True, value_range=(-1, 1))
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": checkpoint.get("step"),
        "manifest": str(args.manifest.resolve()),
        "img_size": args.img_size,
        "n_per_task": args.n_per_task,
        "sample_iter": args.sample_iter,
        "cfg": args.cfg,
        "temperature": args.temperature,
        "split": args.split,
        "ema": not args.no_ema,
        "segmentation_palette": str(seg_palette["path"]) if seg_palette is not None else None,
        "summary": summary,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
