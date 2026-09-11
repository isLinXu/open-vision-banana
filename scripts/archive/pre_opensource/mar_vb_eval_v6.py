#!/usr/bin/env python3
"""MAR-VB v6 Evaluation: Proper turbo colormap decoding + comprehensive metrics.

Fixes the colormap mismatch issue:
  - v3/v5 depth targets are encoded with turbo-like rainbow (Blue->Cyan->Green->Yellow->Red)
    using per-image linear min-max normalization (NOT power transform).
  - v1 eval (mar_vb_eval_metrics.py) wrongly used cube_edge and applied inverse
    power transform -> AbsRel=5.52 was a phantom from decoding mismatch.
  - v5 eval (mar_vb_eval_v5_clean.py) used a hand-built 256-entry LUT -> correct but
    only computes depth_scalar_mae and rank_corr.

This script unifies both:
  1. Uses DepthRGBMapper(colormap="turbo") for analytical segment projection
  2. Decodes depth in NORMALIZED space (rgb_to_normalized), NOT metric space
     (rgb_to_depth), since v3/v5 encoding is linear min-max (no power transform)
  3. Computes relative depth metrics: Pearson, Spearman, position MAE
  4. Computes normal angular error, segmentation mIoU, structure edge F1
  5. Reports per-sample and aggregate statistics

Usage:
  python scripts/mar_vb_eval_v6.py \\
    --checkpoint output/mar_vb_v5_full_clean_v2_3task_refine_384_20260811/ckpt_best.pt \\
    --manifest output/mar_vb_clean_pairs_v2/pairs.jsonl \\
    --output output/mar_vb_v6_eval \\
    --img-size 384 --n-per-task 16 --sample-iter 32 --cfg 1.0

  # Quick smoke test
  python scripts/mar_vb_eval_v6.py \\
    --checkpoint output/mar_vb_v5_full_clean_v2_3task_refine_384_20260811/ckpt_best.pt \\
    --manifest output/mar_vb_clean_pairs_v2/pairs.jsonl \\
    --output output/mar_vb_v6_eval_smoke --n-per-task 4 --sample-iter 16
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.evaluation.depth import evaluate_depth
from open_vision_banana.evaluation.relative_depth import evaluate_relative_depth_rgb

DEFAULT_MANIFEST = PROJECT_ROOT / "output/mar_vb_clean_pairs_v2/pairs.jsonl"
TASKS = ("depth", "normal", "seg", "structure")


def parse_args():
    parser = argparse.ArgumentParser(description="MAR-VB v6 unified evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--n-per-task", type=int, default=16)
    parser.add_argument("--sample-iter", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--tasks", default="",
        help="Comma-separated tasks to evaluate (default: all in manifest).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument(
        "--seg-palette", type=Path,
        help="Fixed segmentation palette JSON.",
    )
    parser.add_argument(
        "--colormap", default="turbo",
        help="Colormap for depth decoding (turbo matches v3/v5 training data).",
    )
    return parser.parse_args()


def load_training_module(img_size: int):
    os.environ["VB_IMG_SIZE"] = str(img_size)
    os.environ["VB_NUM_WORKERS"] = "0"
    path = PROJECT_ROOT / "scripts/mar_vb_train_v5_multitask.py"
    spec = importlib.util.spec_from_file_location("mar_vb_v6_train", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_rows(manifest: Path, n_per_task: int, requested_tasks: tuple[str, ...] = ()):
    rows = [json.loads(line) for line in manifest.open() if line.strip()]
    by_task = {task: [] for task in (requested_tasks or TASKS)}
    for row in rows:
        if row.get("split") == "val" and row.get("task") in by_task:
            by_task[row["task"]].append(row)
    for task, task_rows in tuple(by_task.items()):
        task_rows.sort(key=lambda row: row["input"])
        by_task[task] = task_rows[:n_per_task]
    by_task = {task: task_rows for task, task_rows in by_task.items() if task_rows}
    if not by_task:
        raise ValueError("No validation rows found for any supported task")
    return by_task


def parse_task_list(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    aliases = {"segmentation": "seg", "struct": "structure"}
    tasks = tuple(aliases.get(t.strip(), t.strip()) for t in value.split(",") if t.strip())
    unknown = sorted(set(tasks) - set(TASKS))
    if unknown:
        raise ValueError(f"unknown --tasks value(s): {', '.join(unknown)}")
    return tuple(dict.fromkeys(tasks))


def load_image(path: str, size: int, *, resample=Image.BICUBIC) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((size, size), resample)
        arr = np.asarray(rgb, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


# ── Segmentation helpers ──────────────────────────────────────────────────

def load_segmentation_palette(rows: dict[str, list[dict]], explicit: Path | None):
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
    pixels = image.permute(1, 2, 0).cpu().numpy().reshape(-1, 3)
    labels = np.empty(len(pixels), dtype=np.int32)
    for start in range(0, len(pixels), 8192):
        chunk = pixels[start:start + 8192]
        distances = ((chunk[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
        labels[start:start + len(chunk)] = distances.argmin(axis=1)
    return labels


def segmentation_confusion(pred: torch.Tensor, target: torch.Tensor, palette_info: dict) -> np.ndarray:
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
    tp = np.diag(confusion).astype(np.float64)
    target_total = confusion.sum(axis=1).astype(np.float64)
    pred_total = confusion.sum(axis=0).astype(np.float64)
    union = target_total + pred_total - tp
    present = union > 0
    valid_pixels = int(target_total.sum())
    return {
        "seg_pixel_accuracy": float(tp.sum() / max(1, valid_pixels)),
        "seg_miou": float((tp[present] / union[present]).mean()) if present.any() else 0.0,
        "seg_present_classes": int(present.sum()),
        "seg_valid_pixels": valid_pixels,
    }


# ── Normal & structure helpers ────────────────────────────────────────────

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


# ── Main evaluation ───────────────────────────────────────────────────────

def main():
    args = parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if not args.manifest.exists():
        raise SystemExit(f"manifest not found: {args.manifest}")

    mt = load_training_module(args.img_size)
    device = mt.DEVICE

    # Load base MAR model
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
        print(f"[ema] applied (decay={ema.decay:.4f})", flush=True)

    vae = mt.AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4),
                           ckpt_path=f"{mt.CKPT_DIR}/kl16.safetensors").to(device).eval()
    for param in vae.parameters():
        param.requires_grad_(False)

    # Use turbo colormap (matches v3/v5 depth target encoding)
    depth_mapper = DepthRGBMapper(colormap=args.colormap)
    print(f"[colormap] using '{args.colormap}' for depth decoding", flush=True)

    rows = load_rows(args.manifest, args.n_per_task, parse_task_list(args.tasks))
    seg_palette = load_segmentation_palette(rows, args.seg_palette)
    if "seg" in rows and seg_palette is None:
        print("[seg] no fixed palette metadata; RGB MAE only", flush=True)

    summary, panels = {}, {}
    per_sample_records = []
    t0 = time.time()

    with torch.no_grad():
        for task, task_rows in rows.items():
            label = mt.TASK_LABEL[task]
            rgb_maes, depth_mae_samples, depth_relative = [], [], []
            normal_angles, edge_f1s = [], []
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
                latent = model.sample_vb(context, labels,
                                         num_iter=args.sample_iter,
                                         cfg=args.cfg,
                                         temperature=args.temperature)
                pred = vae.decode(latent / mt.VAE_SCALE).clamp(-1, 1)

                pred01 = (pred[0] + 1) / 2
                target01 = (target[0] + 1) / 2
                rgb_maes.append(float((pred01 - target01).abs().mean().item()))

                sample_record = {"task": task, "index": index, "rgb_mae": rgb_maes[-1]}

                if task == "depth":
                    # IMPORTANT: v3/v5 training data uses per-image linear min-max
                    # normalization (NOT power transform). The depth targets were
                    # generated by depth_to_rgb() which normalizes depth to [0,255]
                    # and maps via turbo colormap. Decoding must not apply inverse
                    # power transform (which would explode values near 1.0).
                    #
                    # Correct decode: RGB -> colormap position (normalized [0,1])
                    # using rgb_to_normalized(), NOT rgb_to_depth().
                    pred_norm = depth_mapper.rgb_to_normalized(pred01.unsqueeze(0))  # [1, 1, H, W]
                    gt_norm = depth_mapper.rgb_to_normalized(target01.unsqueeze(0))

                    # Relative depth metrics (position along curve)
                    rel = evaluate_relative_depth_rgb(
                        pred01.unsqueeze(0), target01.unsqueeze(0),
                        colormap=args.colormap,
                    )
                    depth_relative.append(rel)

                    # Compute depth_scalar_mae in normalized space (matches v5 eval)
                    # This is the proper metric for v3/v5 data since encoding is linear
                    depth_mae = float((pred_norm - gt_norm).abs().mean().item())
                    depth_mae_samples.append(depth_mae)

                    # Rank correlation from evaluate_relative_depth_rgb (pure PyTorch,
                    # no scipy dependency needed)
                    sample_record.update({
                        "depth_mae": depth_mae,
                        "depth_spearman": rel["relative_position_spearman_r"],
                        "pearson": rel["relative_position_pearson_r"],
                        "spearman": rel["relative_position_spearman_r"],
                    })
                    elapsed = (time.time() - t0) / 60
                    print(f"  [{task} {index+1}/{len(task_rows)}] "
                          f"depth_mae={depth_mae:.4f} spearman={rel['relative_position_spearman_r']:.3f} "
                          f"pearson={rel['relative_position_pearson_r']:.3f} "
                          f"({elapsed:.1f}min)", flush=True)

                elif task == "normal":
                    angle = normal_angle_deg(pred01, target01)
                    normal_angles.append(angle)
                    sample_record["normal_angle_deg"] = angle
                    print(f"  [{task} {index+1}/{len(task_rows)}] "
                          f"angle={angle:.2f}deg rgb_mae={rgb_maes[-1]:.4f}", flush=True)

                elif task == "structure":
                    f1 = edge_f1_ods(pred01, target01)
                    edge_f1s.append(f1)
                    sample_record["edge_f1_ods"] = f1
                    print(f"  [{task} {index+1}/{len(task_rows)}] "
                          f"edge_f1={f1:.4f} rgb_mae={rgb_maes[-1]:.4f}", flush=True)

                elif task == "seg" and seg_confusion is not None:
                    seg_confusion += segmentation_confusion(pred01, target01, seg_palette)

                per_sample_records.append(sample_record)
                task_panels.append(torch.cat([inp[0].cpu(), target[0].cpu(), pred[0].cpu()], dim=0))

            # Aggregate
            result = {"n": len(task_rows), "rgb_mae": float(np.mean(rgb_maes))}

            if depth_mae_samples:
                # Normalized-space MAE (matches v5 eval depth_scalar_mae)
                result["depth_mae_normalized"] = float(np.mean(depth_mae_samples))
                result["depth_mae_median"] = float(np.median(depth_mae_samples))
                # Relative metrics from evaluate_relative_depth_rgb (pure PyTorch)
                result["depth_pearson"] = float(np.mean([r["relative_position_pearson_r"] for r in depth_relative]))
                result["depth_spearman"] = float(np.mean([r["relative_position_spearman_r"] for r in depth_relative]))
                result["depth_position_mae"] = float(np.mean([r["relative_position_mae"] for r in depth_relative]))
                result["depth_rgb_mae"] = float(np.mean([r["rgb_mae"] for r in depth_relative]))

            if normal_angles:
                result["normal_mean_angular_error_deg"] = float(np.mean(normal_angles))
                result["normal_median_angular_error_deg"] = float(np.median(normal_angles))

            if edge_f1s:
                result["structure_edge_f1_ods"] = float(np.mean(edge_f1s))

            if seg_confusion is not None:
                result.update(segmentation_metrics(seg_confusion))

            summary[task] = result
            panels[task] = torch.cat(task_panels, dim=0)
            print(f"\n[{task}] SUMMARY: {json.dumps(result, sort_keys=True)}\n", flush=True)

    # Save outputs
    args.output.mkdir(parents=True, exist_ok=True)
    for task, panel in panels.items():
        save_image(panel.reshape(-1, 3, args.img_size, args.img_size),
                   args.output / f"panel_{task}.png", nrow=3,
                   normalize=True, value_range=(-1, 1))

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_val_loss": checkpoint.get("val_loss"),
        "manifest": str(args.manifest.resolve()),
        "img_size": args.img_size,
        "n_per_task": args.n_per_task,
        "sample_iter": args.sample_iter,
        "cfg": args.cfg,
        "temperature": args.temperature,
        "ema": not args.no_ema,
        "colormap": args.colormap,
        "segmentation_palette": str(seg_palette["path"]) if seg_palette else None,
        "summary": summary,
        "per_sample": per_sample_records,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")

    # Print final summary table
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint.name} (step={checkpoint.get('step')})")
    print(f"Config: img={args.img_size} iter={args.sample_iter} cfg={args.cfg} ema={not args.no_ema}")
    print(f"Colormap: {args.colormap}")
    print("-" * 80)
    for task, metrics in summary.items():
        print(f"\n  [{task.upper()}] (n={metrics['n']})")
        for k, v in sorted(metrics.items()):
            if k not in ("n",):
                if isinstance(v, float):
                    print(f"    {k:40s} = {v:.4f}")
                else:
                    print(f"    {k:40s} = {v}")
    print("\n" + "=" * 80)
    print(f"Full report: {args.output / 'metrics.json'}")
    print(f"Panels: {args.output}/panel_*.png")
    print(f"Elapsed: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
