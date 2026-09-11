#!/usr/bin/env python3
"""Multi-checkpoint ensemble inference for TinyVisionBanana V3.

Loads per-task best checkpoints from V3 training, runs inference using
the best checkpoint for each task, and generates a comparison dashboard.

Usage:
    python scripts/infer_tiny_vb_v3.py \
        --ckpt-dir output/tiny_vb_v3/per_task \
        --data-dir /path/to/vb_snvc_train_v3 \
        --output-dir output/tiny_vb_v3/visualizations
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.data.tiny_dataset import (  # noqa: E402
    TINY_TASK_ORDER,
    TinyMultiTaskDataset,
    _load_image,
)
from open_vision_banana.tiny_model_v2 import (  # noqa: E402
    TinyVisionBananaV2,
    TinyVisionBananaV2Config,
    evaluate_tiny_predictions,
)


def load_model(ckpt_path: str, device: str = "cpu") -> TinyVisionBananaV2:
    """Load V2/V3 model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = TinyVisionBananaV2Config(
        base_width=32, num_stages=4, block_depth=2, task_embed_dim=32, resolution=256,
    )
    model = TinyVisionBananaV2(config, tasks=TINY_TASK_ORDER)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_per_task_models(
    ckpt_dir: Path,
    device: torch.device,
) -> dict[str, TinyVisionBananaV2]:
    """Load the best checkpoint for each task."""
    models = {}
    for task in TINY_TASK_ORDER:
        ckpt_path = ckpt_dir / f"best_{task}.pt"
        if ckpt_path.is_file():
            models[task] = load_model(str(ckpt_path), str(device))
            print(f"  Loaded {task} model from {ckpt_path.name}")
        else:
            print(f"  WARNING: No checkpoint for {task} at {ckpt_path}")
    return models


def run_ensemble_inference(
    per_task_models: dict[str, TinyVisionBananaV2],
    dataset: TinyMultiTaskDataset,
    device: torch.device,
    resolution: int = 256,
) -> dict:
    """Run inference using per-task best models."""
    results = {
        task: {"preds": [], "targets": [], "images": [], "metrics": {}}
        for task in TINY_TASK_ORDER
    }

    print(f"Running ensemble inference on {len(dataset)} samples...")

    with torch.no_grad():
        for i, sample in enumerate(dataset.samples):
            if i % 100 == 0:
                print(f"  {i}/{len(dataset)} ...")

            task = sample.task
            if task not in per_task_models:
                continue

            model = per_task_models[task]
            image = _load_image(sample.image_path, resolution, (-1.0, 1.0))
            target = _load_image(sample.target_path, resolution, (0.0, 1.0))

            image_t = image.unsqueeze(0).to(device)
            target_t = target.unsqueeze(0).to(device)
            task_idx = torch.tensor(
                [model.task_index[task]], device=device, dtype=torch.long
            )

            pred = model(image=image_t, task=task_idx)

            results[task]["preds"].append(pred.cpu().squeeze(0))
            results[task]["targets"].append(target.cpu())
            results[task]["images"].append(((image.cpu() + 1.0) / 2.0).clamp(0, 1))

    # Compute per-task metrics
    for task in TINY_TASK_ORDER:
        if results[task]["preds"]:
            preds_cat = torch.stack(results[task]["preds"])
            targets_cat = torch.stack(results[task]["targets"])
            metrics = evaluate_tiny_predictions(preds_cat, targets_cat)
            results[task]["metrics"] = metrics
            print(f"  {task}: MAE={metrics['rgb_mae']:.4f} RMSE={metrics['rgb_rmse']:.4f} "
                  f"Pearson={metrics['pearson']:.4f}")

    return results


def make_grid_image(
    image: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    task: str,
) -> np.ndarray:
    """Create a 1x3 grid: [input | target | prediction]."""
    img_np = (image.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    tgt_np = (target.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    prd_np = (pred.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    h, w = img_np.shape[:2]
    pad = 4
    grid = np.zeros((h, w * 3 + pad * 2, 3), dtype=np.uint8)
    grid[:, :w] = img_np
    grid[:, w + pad:2 * w + pad] = tgt_np
    grid[:, 2 * w + 2 * pad:] = prd_np
    return grid


def make_dashboard(
    results: dict,
    v2_metrics: dict | None,
    per_task_best: dict | None,
    output_dir: Path,
) -> None:
    """Generate HTML dashboard with per-task metrics and sample grids."""
    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>TinyVisionBanana V3 — Ensemble Inference</title>",
        "<style>",
        "body { font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; margin: 20px; }",
        "h1 { color: #e94560; } h2 { color: #0f3460; background: #16213e; padding: 8px; border-radius: 4px; }",
        ".task-grid { display: flex; flex-wrap: wrap; gap: 20px; }",
        ".task-card { background: #16213e; border-radius: 8px; padding: 16px; width: 280px; }",
        ".metric { display: flex; justify-content: space-between; padding: 4px 0; }",
        ".metric-value { font-weight: bold; color: #e94560; }",
        ".sample-grid { margin: 10px 0; }",
        ".sample-grid img { width: 100%; border-radius: 4px; }",
        "table { border-collapse: collapse; width: 100%; margin: 10px 0; }",
        "th, td { border: 1px solid #333; padding: 8px; text-align: center; }",
        "th { background: #0f3460; } .best { color: #00ff88; font-weight: bold; }",
        ".delta-pos { color: #00ff88; } .delta-neg { color: #ff6b6b; }",
        "</style></head><body>",
        "<h1>TinyVisionBanana V3 — Multi-Checkpoint Ensemble</h1>",
        "<p>Per-task best checkpoints | V3 training (8000 steps, weighted sampling, pearson_weight=0.3)</p>",
    ]

    # Per-task best info
    if per_task_best:
        html_parts.append("<h2>Per-Task Best Checkpoints</h2><table>")
        html_parts.append(
            "<tr><th>Task</th><th>Best MAE</th><th>Pearson</th><th>Step</th></tr>"
        )
        for task in TINY_TASK_ORDER:
            if task in per_task_best:
                info = per_task_best[task]
                html_parts.append(
                    f"<tr><td>{task}</td><td>{info['mae']:.4f}</td>"
                    f"<td>{info['pearson']:.4f}</td><td>{info['step']}</td></tr>"
                )
        html_parts.append("</table>")

    # Comparison table: V2 vs V3
    html_parts.append("<h2>Per-Task Metrics: V2 → V3</h2>")
    html_parts.append(
        "<table><tr><th>Task</th><th>V2 MAE</th><th>V3 MAE</th><th>Δ MAE</th>"
        "<th>V2 Pearson</th><th>V3 Pearson</th><th>Δ Pearson</th><th>Samples</th></tr>"
    )

    for task in TINY_TASK_ORDER:
        v3_m = results[task]["metrics"]
        v3_mae = v3_m["rgb_mae"]
        v3_pear = v3_m["pearson"]
        n = len(results[task]["preds"])

        if v2_metrics and task in v2_metrics:
            v2_mae = v2_metrics[task]["rgb_mae"]
            v2_pear = v2_metrics[task]["pearson"]
            d_mae = v3_mae - v2_mae
            d_pear = v3_pear - v2_pear
            mae_cls = "delta-pos" if d_mae < 0 else "delta-neg"
            pear_cls = "delta-pos" if d_pear > 0 else "delta-neg"
            html_parts.append(
                f"<tr><td>{task}</td>"
                f"<td>{v2_mae:.4f}</td><td class='best'>{v3_mae:.4f}</td>"
                f"<td class='{mae_cls}'>{d_mae:+.4f}</td>"
                f"<td>{v2_pear:.4f}</td><td class='best'>{v3_pear:.4f}</td>"
                f"<td class='{pear_cls}'>{d_pear:+.4f}</td>"
                f"<td>{n}</td></tr>"
            )
        else:
            html_parts.append(
                f"<tr><td>{task}</td><td>—</td><td class='best'>{v3_mae:.4f}</td>"
                f"<td>—</td><td>—</td><td class='best'>{v3_pear:.4f}</td>"
                f"<td>—</td><td>{n}</td></tr>"
            )
    html_parts.append("</table>")

    # Per-task sample grids (4 samples each)
    for task in TINY_TASK_ORDER:
        preds = results[task]["preds"]
        if not preds:
            continue
        n_show = min(4, len(preds))
        html_parts.append(f"<h2>{task} — Sample Predictions (ensemble)</h2>")
        html_parts.append("<div class='task-grid'>")
        for i in range(n_show):
            grid = make_grid_image(
                results[task]["images"][i],
                results[task]["targets"][i],
                results[task]["preds"][i],
                task,
            )
            grid_path = output_dir / f"v3_grid_{task}_{i}.png"
            Image.fromarray(grid).save(str(grid_path))
            html_parts.append(
                f"<div class='task-card'><div class='sample-grid'>"
                f"<img src='v3_grid_{task}_{i}.png' /></div>"
                f"<div class='metric'><span>Input | Target | Prediction</span></div></div>"
            )
        html_parts.append("</div>")

    html_parts.append("</body></html>")

    dashboard_path = output_dir / "v3_dashboard.html"
    dashboard_path.write_text("".join(html_parts), encoding="utf-8")
    print(f"Dashboard saved: {dashboard_path}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default="output/tiny_vb_v3/per_task",
                   help="Directory containing best_{task}.pt checkpoints")
    p.add_argument("--aggregate-ckpt", default="",
                   help="Optional aggregate best.pt for fallback")
    p.add_argument("--data-dir",
                   default="/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v3")
    p.add_argument("--v2-metrics", default="",
                   help="Path to v2_metrics.json for comparison")
    p.add_argument("--per-task-best", default="",
                   help="Path to per_task_best.json from V3 training")
    p.add_argument("--output-dir", default="output/tiny_vb_v3/visualizations")
    p.add_argument("--device", default="auto")
    p.add_argument("--resolution", type=int, default=256)
    args = p.parse_args()

    device = torch.device(args.device) if args.device != "auto" else (
        torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    # Load per-task models
    ckpt_dir = Path(args.ckpt_dir)
    print(f"Loading per-task checkpoints from {ckpt_dir}...")
    per_task_models = load_per_task_models(ckpt_dir, device)
    if not per_task_models:
        print("ERROR: No checkpoints found!")
        return
    print(f"Loaded {len(per_task_models)} per-task models")

    # Dataset
    val_ds = TinyMultiTaskDataset(
        args.data_dir, "val", tasks=TINY_TASK_ORDER,
        resolution=args.resolution, balance=False,
    )
    print(f"Val samples: {len(val_ds)}")

    # Run inference
    results = run_ensemble_inference(per_task_models, val_ds, device, args.resolution)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    metrics_json = {
        task: results[task]["metrics"] for task in TINY_TASK_ORDER
        if results[task]["metrics"]
    }
    with open(output_dir / "v3_metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Metrics saved: {output_dir / 'v3_metrics.json'}")

    # Load comparison data
    v2_metrics = None
    if args.v2_metrics:
        v2_path = Path(args.v2_metrics)
        if v2_path.is_file():
            with open(v2_path) as f:
                v2_metrics = json.load(f)

    per_task_best = None
    if args.per_task_best:
        ptb_path = Path(args.per_task_best)
        if ptb_path.is_file():
            with open(ptb_path) as f:
                per_task_best = json.load(f)

    # Generate dashboard
    make_dashboard(results, v2_metrics, per_task_best, output_dir)

    print("\n=== V3 Ensemble Inference Summary ===")
    for task in TINY_TASK_ORDER:
        if results[task]["metrics"]:
            m = results[task]["metrics"]
            print(f"  {task:12s}: MAE={m['rgb_mae']:.4f}  RMSE={m['rgb_rmse']:.4f}  "
                  f"Pearson={m['pearson']:.4f}")


if __name__ == "__main__":
    main()
