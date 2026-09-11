#!/usr/bin/env python3
"""Inference and visualization for TinyVisionBanana V2.

Runs V2 best.pt on all validation samples, computes per-task metrics,
and generates comparison visualizations (V2 vs V1 vs ground truth).
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


def load_v2_model(ckpt_path: str, device: str = "cpu") -> TinyVisionBananaV2:
    """Load V2 model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = TinyVisionBananaV2Config(
        base_width=32, num_stages=4, block_depth=2, task_embed_dim=32, resolution=256,
    )
    model = TinyVisionBananaV2(config, tasks=TINY_TASK_ORDER)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def run_inference(
    model: TinyVisionBananaV2,
    dataset: TinyMultiTaskDataset,
    device: torch.device,
    resolution: int = 256,
) -> dict:
    """Run inference on all val samples, return per-task results."""
    results = {task: {"preds": [], "targets": [], "images": [], "metrics": {}} for task in TINY_TASK_ORDER}

    print(f"Running inference on {len(dataset)} samples...")

    with torch.no_grad():
        for i, sample in enumerate(dataset.samples):
            if i % 100 == 0:
                print(f"  {i}/{len(dataset)} ...")

            task = sample.task
            image = _load_image(sample.image_path, resolution, (-1.0, 1.0))
            target = _load_image(sample.target_path, resolution, (0.0, 1.0))

            image_t = image.unsqueeze(0).to(device)
            target_t = target.unsqueeze(0).to(device)
            task_idx = torch.tensor([model.task_index[task]], device=device, dtype=torch.long)

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
    output_dir: Path,
) -> None:
    """Generate HTML dashboard with per-task metrics and sample grids."""
    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>TinyVisionBanana V2 — Inference Results</title>",
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
        "</style></head><body>",
        "<h1>TinyVisionBanana V2 — Inference Results</h1>",
        f"<p>Model: 12.26M params | Training: 3000 steps, lr=1e-3 | Best: step 2000 (agg MAE=0.1924)</p>",
        "<h2>Per-Task Metrics</h2>",
        "<table><tr><th>Task</th><th>MAE</th><th>RMSE</th><th>Pearson r</th><th>Samples</th></tr>",
    ]

    for task in TINY_TASK_ORDER:
        m = results[task]["metrics"]
        n = len(results[task]["preds"])
        html_parts.append(
            f"<tr><td>{task}</td><td>{m['rgb_mae']:.4f}</td>"
            f"<td>{m['rgb_rmse']:.4f}</td><td>{m['pearson']:.4f}</td><td>{n}</td></tr>"
        )
    html_parts.append("</table>")

    # Per-task sample grids (4 samples each)
    for task in TINY_TASK_ORDER:
        preds = results[task]["preds"]
        if not preds:
            continue
        n_show = min(4, len(preds))
        html_parts.append(f"<h2>{task} — Sample Predictions</h2>")
        html_parts.append("<div class='task-grid'>")
        for i in range(n_show):
            grid = make_grid_image(
                results[task]["images"][i],
                results[task]["targets"][i],
                results[task]["preds"][i],
                task,
            )
            grid_path = output_dir / f"v2_grid_{task}_{i}.png"
            Image.fromarray(grid).save(str(grid_path))
            html_parts.append(
                f"<div class='task-card'><div class='sample-grid'>"
                f"<img src='v2_grid_{task}_{i}.png' /></div>"
                f"<div class='metric'><span>Input | Target | Prediction</span></div></div>"
            )
        html_parts.append("</div>")

    html_parts.append("</body></html>")

    dashboard_path = output_dir / "v2_dashboard.html"
    dashboard_path.write_text("".join(html_parts), encoding="utf-8")
    print(f"Dashboard saved: {dashboard_path}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="output/912b6a07-652a-4b0d-a480-b3b7ee65ea9a/tiny_vb_v2/best.pt")
    p.add_argument("--data-dir", default="/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v3")
    p.add_argument("--output-dir", default="output/912b6a07-652a-4b0d-a480-b3b7ee65ea9a/tiny_vb_v2/visualizations")
    p.add_argument("--device", default="auto")
    p.add_argument("--resolution", type=int, default=256)
    args = p.parse_args()

    device = torch.device(args.device) if args.device != "auto" else (
        torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    model = load_v2_model(args.ckpt, str(device))
    print(f"Model loaded: {model.num_parameters():,} params")

    val_ds = TinyMultiTaskDataset(
        args.data_dir, "val", tasks=TINY_TASK_ORDER,
        resolution=args.resolution, balance=False,
    )
    print(f"Val samples: {len(val_ds)}")

    results = run_inference(model, val_ds, device, args.resolution)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    metrics_json = {
        task: results[task]["metrics"] for task in TINY_TASK_ORDER
    }
    with open(output_dir / "v2_metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Metrics saved: {output_dir / 'v2_metrics.json'}")

    # Generate dashboard
    make_dashboard(results, output_dir)

    print("\n=== V2 Inference Summary ===")
    for task in TINY_TASK_ORDER:
        m = results[task]["metrics"]
        print(f"  {task:12s}: MAE={m['rgb_mae']:.4f}  RMSE={m['rgb_rmse']:.4f}  Pearson={m['pearson']:.4f}")


if __name__ == "__main__":
    main()
