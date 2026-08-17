#!/usr/bin/env python3
"""MAR-VB Sampling Parameter Grid Search.

Sweep cfg x sample_iter combinations to find the best sampling config for each task.
Uses the v6 evaluation pipeline with turbo colormap.

Usage:
  python scripts/mar_vb_grid_search.py \
    --checkpoint output/mar_vb_v5_full_clean_v2_3task_refine_384_20260811/ckpt_best.pt \
    --manifest output/mar_vb_clean_pairs_v2/pairs.jsonl \
    --output output/mar_vb_grid_search \
    --img-size 384 --n-per-task 4 \
    --cfgs 1.0,1.5,2.0,3.0 \
    --iters 16,32,64 \
    --tasks depth

  # Quick depth-only search
  python scripts/mar_vb_grid_search.py \
    --checkpoint output/mar_vb_v5_full_clean_v2_3task_refine_384_20260811/ckpt_best.pt \
    --output output/mar_vb_grid_depth --n-per-task 4 --tasks depth \
    --cfgs 0.5,1.0,1.5,2.0,3.0 --iters 8,16,32
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.evaluation.depth import evaluate_depth
from open_vision_banana.evaluation.relative_depth import evaluate_relative_depth_rgb

DEFAULT_MANIFEST = PROJECT_ROOT / "output/mar_vb_clean_pairs_v2/pairs.jsonl"
TASKS = ("depth", "normal", "seg", "structure")


def parse_args():
    p = argparse.ArgumentParser(description="MAR-VB sampling grid search")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--n-per-task", type=int, default=4)
    p.add_argument("--cfgs", default="1.0,2.0", help="Comma-separated CFG values")
    p.add_argument("--iters", default="16,32", help="Comma-separated sample_iter values")
    p.add_argument("--tasks", default="depth", help="Comma-separated tasks")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    return p.parse_args()


def load_training_module(img_size: int):
    os.environ["VB_IMG_SIZE"] = str(img_size)
    os.environ["VB_NUM_WORKERS"] = "0"
    path = PROJECT_ROOT / "scripts/mar_vb_train_v5_multitask.py"
    spec = importlib.util.spec_from_file_location("mar_vb_grid_train", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_image(path: str, size: int, *, resample=Image.BICUBIC) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((size, size), resample)
        arr = np.asarray(rgb, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def load_rows(manifest: Path, n_per_task: int, tasks: tuple[str, ...]):
    rows = [json.loads(line) for line in manifest.open() if line.strip()]
    by_task = {t: [] for t in tasks}
    for row in rows:
        if row.get("split") == "val" and row.get("task") in by_task:
            by_task[row["task"]].append(row)
    for task_rows in by_task.values():
        task_rows.sort(key=lambda r: r["input"])
    return {t: rs[:n_per_task] for t, rs in by_task.items() if rs}


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
        best = max(best, 2 * tp / max(1, 2 * tp + fp + fn))
    return float(best)


def main():
    args = parse_args()
    cfgs = [float(x) for x in args.cfgs.split(",")]
    iters_list = [int(x) for x in args.iters.split(",")]
    tasks = tuple(t.strip() for t in args.tasks.split(","))
    aliases = {"segmentation": "seg", "struct": "structure"}
    tasks = tuple(aliases.get(t, t) for t in tasks)

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
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    if not args.no_ema and "ema" in ckpt:
        ema = mt.EMA(model, decay=ckpt["ema"].get("decay", 0.999))
        ema.load_state_dict(ckpt["ema"])
        ema.apply()

    vae = mt.AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4),
                           ckpt_path=f"{mt.CKPT_DIR}/kl16.safetensors").to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    depth_mapper = DepthRGBMapper(colormap="turbo")
    rows = load_rows(args.manifest, args.n_per_task, tasks)

    # Pre-encode all contexts (they don't change across cfg/iter)
    contexts = {}
    targets = {}
    for task, task_rows in rows.items():
        ctx_list, tgt_list = [], []
        for row in task_rows:
            inp = load_image(row["input"], args.img_size).unsqueeze(0).to(device)
            tgt_resample = Image.NEAREST if task == "seg" else Image.BICUBIC
            tgt = load_image(row["target"], args.img_size, resample=tgt_resample).unsqueeze(0).to(device)
            ctx = mt.encode_latent(vae, inp, sample=False)
            ctx_list.append(ctx)
            tgt_list.append(tgt)
        contexts[task] = ctx_list
        targets[task] = tgt_list

    results = []
    t0 = time.time()
    total_configs = len(cfgs) * len(iters_list)

    for cfg in cfgs:
        for n_iter in iters_list:
            config_result = {"cfg": cfg, "sample_iter": n_iter, "tasks": {}}
            print(f"\n{'='*60}", flush=True)
            print(f"CFG={cfg} ITER={n_iter}", flush=True)
            print(f"{'='*60}", flush=True)

            for task in tasks:
                label = mt.TASK_LABEL[task]
                task_metrics = []

                for idx in range(len(contexts[task])):
                    torch.manual_seed(args.seed + label * 10000 + idx)
                    np.random.seed(args.seed + label * 10000 + idx)

                    ctx = contexts[task][idx]
                    tgt = targets[task][idx]
                    labels = torch.tensor([label], device=device)

                    with torch.no_grad():
                        latent = model.sample_vb(ctx, labels,
                                                 num_iter=n_iter, cfg=args.cfg,
                                                 temperature=args.temperature)
                        pred = vae.decode(latent / mt.VAE_SCALE).clamp(-1, 1)

                    pred01 = (pred[0] + 1) / 2
                    target01 = (tgt[0] + 1) / 2
                    m = {"rgb_mae": float((pred01 - target01).abs().mean().item())}

                    if task == "depth":
                        pred_depth = depth_mapper.rgb_to_depth(pred01.unsqueeze(0))
                        gt_depth = depth_mapper.rgb_to_depth(target01.unsqueeze(0))
                        valid = (gt_depth < 9.9).squeeze(1)
                        raw = evaluate_depth(pred_depth, gt_depth, valid_mask=valid)
                        if valid.sum() > 0:
                            scale = (gt_depth.squeeze(1)[valid].median() /
                                     pred_depth.squeeze(1)[valid].median().clamp(min=1e-6))
                        else:
                            scale = torch.tensor(1.0, device=device)
                        ali = evaluate_depth(pred_depth * scale, gt_depth, valid_mask=valid)
                        rel = evaluate_relative_depth_rgb(
                            pred01.unsqueeze(0), target01.unsqueeze(0), colormap="turbo")
                        m.update({
                            "AbsRel_raw": raw["AbsRel"],
                            "AbsRel_aligned": ali["AbsRel"],
                            "delta1_aligned": ali["δ<1.25"],
                            "pearson": rel["relative_position_pearson_r"],
                            "spearman": rel["relative_position_spearman_r"],
                        })

                    elif task == "normal":
                        m["normal_angle"] = normal_angle_deg(pred01, target01)

                    elif task == "structure":
                        m["edge_f1"] = edge_f1_ods(pred01, target01)

                    task_metrics.append(m)

                # Aggregate
                agg = {}
                keys = set()
                for m in task_metrics:
                    keys.update(m.keys())
                for k in keys:
                    vals = [m[k] for m in task_metrics if k in m]
                    if vals:
                        agg[k] = float(np.mean(vals))

                config_result["tasks"][task] = agg
                # Print key metric
                if task == "depth":
                    print(f"  depth: AbsRel_al={agg.get('AbsRel_aligned', 0):.4f} "
                          f"d1={agg.get('delta1_aligned', 0):.4f} "
                          f"pearson={agg.get('pearson', 0):.4f}", flush=True)
                elif task == "normal":
                    print(f"  normal: angle={agg.get('normal_angle', 0):.2f}deg", flush=True)
                elif task == "structure":
                    print(f"  structure: edge_f1={agg.get('edge_f1', 0):.4f}", flush=True)

            results.append(config_result)
            elapsed = (time.time() - t0) / 60
            done = len(results)
            print(f"  [{done}/{total_configs}] elapsed={elapsed:.1f}min", flush=True)

    # Find best config per task
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": ckpt.get("step"),
        "img_size": args.img_size,
        "n_per_task": args.n_per_task,
        "colormap": "turbo",
        "grid": {"cfgs": cfgs, "iters": iters_list, "tasks": list(tasks)},
        "results": results,
    }

    # Find best per task
    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS")
    print("=" * 80)
    for task in tasks:
        task_results = [(r["cfg"], r["sample_iter"], r["tasks"].get(task, {}))
                        for r in results if task in r["tasks"]]
        if not task_results:
            continue

        if task == "depth":
            # Lower AbsRel_aligned is better
            best = min(task_results, key=lambda x: x[2].get("AbsRel_aligned", 999))
            metric_name = "AbsRel_aligned"
            metric_val = best[2].get("AbsRel_aligned", 0)
            direction = "min"
        elif task == "normal":
            best = min(task_results, key=lambda x: x[2].get("normal_angle", 999))
            metric_name = "normal_angle"
            metric_val = best[2].get("normal_angle", 0)
            direction = "min"
        elif task == "structure":
            best = max(task_results, key=lambda x: x[2].get("edge_f1", 0))
            metric_name = "edge_f1"
            metric_val = best[2].get("edge_f1", 0)
            direction = "max"
        else:
            best = min(task_results, key=lambda x: x[2].get("rgb_mae", 999))
            metric_name = "rgb_mae"
            metric_val = best[2].get("rgb_mae", 0)
            direction = "min"

        print(f"\n  [{task.upper()}] Best: cfg={best[0]} iter={best[1]} "
              f"{metric_name}={metric_val:.4f} ({direction})")
        print(f"    All metrics: {json.dumps(best[2], sort_keys=True)}")

    (args.output / "grid_search.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nFull report: {args.output / 'grid_search.json'}")
    print(f"Total elapsed: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
