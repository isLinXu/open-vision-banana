#!/usr/bin/env python3
"""Analyze V6 depth probe training results and generate visualizations + report.

Reads metrics.json from the V6 training output, produces:
1. Training dashboard PNG (loss curves, val metrics, comparison table)
2. V4/V5/V6 comparison chart
3. HTML analysis report

Usage:
    python scripts/analyze_probe_v6.py --output-dir output/ssd_probe_v6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def load_metrics(output_dir: Path) -> dict:
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"No metrics.json in {output_dir}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def load_v5_metrics(output_dir: Path) -> dict | None:
    v5_path = output_dir.parent / "ssd_probe_v5" / "metrics.json"
    if v5_path.is_file():
        return json.loads(v5_path.read_text(encoding="utf-8"))
    return None


def load_v4_metrics(output_dir: Path) -> dict | None:
    v4_path = output_dir.parent / "ssd_probe_v4_300ep" / "metrics.json"
    if v4_path.is_file():
        return json.loads(v4_path.read_text(encoding="utf-8"))
    return None


def plot_training_dashboard(
    v6: dict,
    v5: dict | None,
    v4: dict | None,
    save_path: Path,
) -> None:
    """Generate a comprehensive training dashboard."""
    history = v6.get("history", [])
    if not history:
        print("No training history found — cannot generate dashboard")
        return

    epochs = [r["epoch"] for r in history]
    val_mae = [r["val_mae"] for r in history]
    val_pearson = [r["val_pearson"] for r in history]
    val_rmse = [r.get("val_rmse", 0) for r in history]
    train_loss = [r.get("train_loss", 0) for r in history]
    lr_values = [r.get("lr", 0) for r in history]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        f"SSD-1B Depth Probe V6 — Multi-Timestep + Deep Supervision\n"
        f"Best Epoch: {v6['best_epoch']} | "
        f"MAE={v6['best_val_metrics']['mae']:.5f} | "
        f"Pearson={v6['best_val_metrics']['pearson']:.5f} | "
        f"Params={v6['trainable_parameters']/1e6:.2f}M",
        fontsize=14, fontweight="bold",
    )

    # --- 1. Val MAE ---
    ax = axes[0, 0]
    ax.plot(epochs, val_mae, "b-o", markersize=2, label="V6 Val MAE")
    if v5:
        v5_hist = v5.get("history", [])
        if v5_hist:
            ax.plot([r["epoch"] for r in v5_hist], [r["val_mae"] for r in v5_hist],
                    "r--", alpha=0.5, label="V5 Val MAE")
    if v4:
        v4_hist = v4.get("history", [])
        if v4_hist:
            ax.plot([r["epoch"] for r in v4_hist], [r["val_mae"] for r in v4_hist],
                    "g--", alpha=0.5, label="V4 Val MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    ax.set_title("Validation MAE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    best_mae = min(val_mae)
    best_mae_epoch = epochs[val_mae.index(best_mae)]
    ax.axhline(y=best_mae, color="b", linestyle=":", alpha=0.3)
    ax.annotate(f"Best: {best_mae:.5f} @ ep{best_mae_epoch}",
                xy=(best_mae_epoch, best_mae), fontsize=9, color="blue")

    # --- 2. Val Pearson ---
    ax = axes[0, 1]
    ax.plot(epochs, val_pearson, "b-o", markersize=2, label="V6 Val Pearson")
    if v5:
        v5_hist = v5.get("history", [])
        if v5_hist:
            ax.plot([r["epoch"] for r in v5_hist], [r["val_pearson"] for r in v5_hist],
                    "r--", alpha=0.5, label="V5 Val Pearson")
    if v4:
        v4_hist = v4.get("history", [])
        if v4_hist:
            ax.plot([r["epoch"] for r in v4_hist], [r["val_pearson"] for r in v4_hist],
                    "g--", alpha=0.5, label="V4 Val Pearson")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pearson r")
    ax.set_title("Validation Pearson Correlation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 3. Train Loss ---
    ax = axes[0, 2]
    ax.plot(epochs, train_loss, "orange", label="V6 Train Loss", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss (composite)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # --- 4. Learning Rate ---
    ax = axes[1, 0]
    ax.plot(epochs, lr_values, "g-", label="LR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # --- 5. Val RMSE ---
    ax = axes[1, 1]
    ax.plot(epochs, val_rmse, "m-o", markersize=2, label="V6 Val RMSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("Validation RMSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 6. Comparison Table ---
    ax = axes[1, 2]
    ax.axis("off")
    rows = []
    if v4:
        m = v4.get("best_val_metrics", v4.get("final_val_metrics", {}))
        rows.append(["V4", f"{m.get('mae', 0):.5f}", f"{m.get('pearson', 0):.5f}",
                      f"{v4.get('trainable_parameters', 0)/1e6:.2f}M"])
    if v5:
        m = v5.get("best_val_metrics", v5.get("final_val_metrics", {}))
        rows.append(["V5", f"{m.get('mae', 0):.5f}", f"{m.get('pearson', 0):.5f}",
                      f"{v5.get('trainable_parameters', 0)/1e6:.2f}M"])
    m6 = v6["best_val_metrics"]
    rows.append(["V6", f"{m6['mae']:.5f}", f"{m6['pearson']:.5f}",
                  f"{v6['trainable_parameters']/1e6:.2f}M"])

    table = ax.table(
        cellText=rows,
        colLabels=["Version", "Val MAE", "Val Pearson", "Params"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 2)
    # Highlight V6 row
    for col in range(4):
        table[len(rows)][col].set_facecolor("#90EE90")
    ax.set_title("Version Comparison", fontsize=13, fontweight="bold")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Dashboard saved to {save_path}")


def generate_html_report(
    v6: dict,
    v5: dict | None,
    v4: dict | None,
    save_path: Path,
) -> None:
    """Generate a self-contained HTML analysis report."""

    def fmt_metrics(m: dict, prefix: str = "") -> str:
        return (
            f"<b>MAE:</b> {m.get('mae', 0):.5f} &nbsp; "
            f"<b>RMSE:</b> {m.get('rmse', 0):.5f} &nbsp; "
            f"<b>Pearson:</b> {m.get('pearson', 0):.5f} &nbsp; "
            f"<b>Spearman:</b> {m.get('spearman', 0):.5f}"
        )

    # Build comparison rows
    comp_rows = []
    if v4:
        m = v4.get("best_val_metrics", v4.get("final_val_metrics", {}))
        comp_rows.append(
            f"<tr><td>V4 (single-t, frozen)</td><td>{m.get('mae', 0):.5f}</td>"
            f"<td>{m.get('pearson', 0):.5f}</td>"
            f"<td>{m.get('rmse', 0):.5f}</td>"
            f"<td>{v4.get('trainable_parameters', 0)/1e6:.2f}M</td>"
            f"<td>{v4.get('best_epoch', '?')}</td></tr>"
        )
    if v5:
        m = v5.get("best_val_metrics", v5.get("final_val_metrics", {}))
        comp_rows.append(
            f"<tr><td>V5 (single-t, deep-supervision)</td><td>{m.get('mae', 0):.5f}</td>"
            f"<td>{m.get('pearson', 0):.5f}</td>"
            f"<td>{m.get('rmse', 0):.5f}</td>"
            f"<td>{v5.get('trainable_parameters', 0)/1e6:.2f}M</td>"
            f"<td>{v5.get('best_epoch', '?')}</td></tr>"
        )
    m6 = v6["best_val_metrics"]
    comp_rows.append(
        f"<tr style='background:#90EE90'><td><b>V6 (multi-t, deep-supervision)</b></td>"
        f"<td>{m6['mae']:.5f}</td>"
        f"<td>{m6['pearson']:.5f}</td>"
        f"<td>{m6['rmse']:.5f}</td>"
        f"<td>{v6['trainable_parameters']/1e6:.2f}M</td>"
        f"<td>{v6['best_epoch']}</td></tr>"
    )

    train_m = v6.get("train_metrics", {})
    overfit_gap = train_m.get("mae", 0) - m6.get("mae", 0)
    overfit_status = (
        "<span style='color:green'>✓ Minimal overfitting</span>"
        if overfit_gap < 0.01
        else f"<span style='color:orange'>⚠ Overfit gap: {overfit_gap:.5f}</span>"
        if overfit_gap < 0.03
        else f"<span style='color:red'>⚠ Severe overfitting: gap={overfit_gap:.5f}</span>"
    )

    history = v6.get("history", [])
    early_stopped = v6.get("early_stopped", False)
    total_epochs = v6.get("total_epochs_run", 0)

    tc = v6.get("training_config", {})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SSD-1B Depth Probe V6 Analysis Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; background: #fafafa; }}
  h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
  h2 {{ color: #555; margin-top: 30px; }}
  .card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  .metric {{ display: inline-block; margin: 5px 15px; padding: 8px 16px; background: #f0f0f0; border-radius: 4px; font-size: 14px; }}
  .metric b {{ color: #333; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
  th {{ background: #4CAF50; color: white; }}
  .highlight {{ background: #90EE90 !important; }}
  .status-ok {{ color: green; font-weight: bold; }}
  .status-warn {{ color: orange; font-weight: bold; }}
  .status-bad {{ color: red; font-weight: bold; }}
  .footer {{ margin-top: 40px; color: #999; font-size: 12px; border-top: 1px solid #ddd; padding-top: 10px; }}
</style>
</head>
<body>
<h1>SSD-1B Depth Probe V6 — Analysis Report</h1>
<p style="color:#666">Multi-Timestep Feature Extraction + Deep Supervision</p>

<div class="card">
<h2>Best Validation Metrics</h2>
<div class="metric"><b>MAE:</b> {m6.get('mae', 0):.5f}</div>
<div class="metric"><b>RMSE:</b> {m6.get('rmse', 0):.5f}</div>
<div class="metric"><b>Pearson:</b> {m6.get('pearson', 0):.5f}</div>
<div class="metric"><b>Spearman:</b> {m6.get('spearman', 0):.5f}</div>
<p style="margin-top:10px">Best epoch: <b>{v6['best_epoch']}</b> &nbsp;|&nbsp;
Early stopped: {"Yes" if early_stopped else "No"} &nbsp;|&nbsp;
Total epochs run: {total_epochs}</p>
</div>

<div class="card">
<h2>Training Configuration</h2>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td>Timesteps</td><td>{v6.get('timesteps', 'N/A')}</td></tr>
<tr><td>Resolution</td><td>{v6.get('resolution', 'N/A')}px</td></tr>
<tr><td>Probe Width</td><td>{tc.get('width', 'N/A')} channels</td></tr>
<tr><td>Upsample Stages</td><td>{tc.get('upsample_stages', 'N/A')}</td></tr>
<tr><td>Deep Supervision</td><td>{tc.get('deep_supervision', 'N/A')}</td></tr>
<tr><td>Learning Rate</td><td>{tc.get('lr', 'N/A')}</td></tr>
<tr><td>Batch Size</td><td>{tc.get('batch_size', 'N/A')}</td></tr>
<tr><td>Augmentation</td><td>flip_h={tc.get('aug_flip_prob', 0)}, flip_v={tc.get('aug_flip_v_prob', 0)}, brightness={tc.get('aug_brightness', 0)}, mixup={tc.get('mixup_alpha', 0)}</td></tr>
<tr><td>EMA Decay</td><td>{tc.get('ema_decay', 'N/A')}</td></tr>
<tr><td>Patience</td><td>{tc.get('patience', 'N/A')}</td></tr>
<tr><td>Loss Weights</td><td>grad={tc.get('gradient_weight', 0)}, pearson={tc.get('pearson_weight', 0)}, ssim={tc.get('ssim_weight', 0)}, edge={tc.get('edge_weight', 0)}, aux={tc.get('aux_weight', 0)}</td></tr>
</table>
</div>

<div class="card">
<h2>Version Comparison (V4 → V5 → V6)</h2>
<table>
<tr><th>Version</th><th>Val MAE ↓</th><th>Val Pearson ↑</th><th>Val RMSE ↓</th><th>Params</th><th>Best Epoch</th></tr>
{''.join(comp_rows)}
</table>
</div>

<div class="card">
<h2>Overfitting Analysis</h2>
<div class="metric"><b>Train MAE:</b> {train_m.get('mae', 0):.5f}</div>
<div class="metric"><b>Val MAE:</b> {m6.get('mae', 0):.5f}</div>
<div class="metric"><b>Gap:</b> {overfit_gap:.5f}</div>
<p style="margin-top:10px">{overfit_status}</p>
<p style="color:#666">V5 had Train MAE=0.00000 (total memorization). V6 introduces mixup, stronger augmentation,
and multi-timestep features to combat this.</p>
</div>

<div class="card">
<h2>Training History Summary</h2>
<p>Total epochs logged: {len(history)}</p>
"""
    if history:
        first = history[0]
        last = history[-1]
        html += f"""
<table>
<tr><th>Metric</th><th>First Epoch ({first['epoch']})</th><th>Last Epoch ({last['epoch']})</th><th>Change</th></tr>
<tr><td>Val MAE</td><td>{first.get('val_mae', 0):.5f}</td><td>{last.get('val_mae', 0):.5f}</td><td>{last.get('val_mae', 0) - first.get('val_mae', 0):.5f}</td></tr>
<tr><td>Val Pearson</td><td>{first.get('val_pearson', 0):.5f}</td><td>{last.get('val_pearson', 0):.5f}</td><td>{last.get('val_pearson', 0) - first.get('val_pearson', 0):.5f}</td></tr>
<tr><td>Train Loss</td><td>{first.get('train_loss', 0):.5f}</td><td>{last.get('train_loss', 0):.5f}</td><td>{last.get('train_loss', 0) - first.get('train_loss', 0):.5f}</td></tr>
</table>
"""
    html += f"""
</div>

<div class="card">
<h2>Feature Contract</h2>
<p>Feature shapes: {v6.get('feature_shapes', 'N/A')}</p>
<p>Contract: {v6.get('feature_contract', 'N/A')}</p>
</div>

<div class="card">
<h2>Key Improvements (V5 → V6)</h2>
<ul>
<li><b>Multi-timestep features</b>: 4 timesteps (t=0, 250, 500, 750) → 4× richer temporal information</li>
<li><b>Deep supervision</b>: auxiliary decoder heads for better gradient flow</li>
<li><b>Progressive 4-stage decoder</b>: 512px output from 16px features</li>
<li><b>Cosine-scheduled auxiliary loss</b>: deep supervision weight decays over training</li>
<li><b>Stronger regularization</b>: higher dropout, mixup, label smoothing to combat V5 overfitting</li>
</ul>
</div>

<div class="footer">
<p>Generated by analyze_probe_v6.py | Data: {v6.get('data_dir', 'N/A')}</p>
<p>Runtime: {v6.get('runtime_seconds', 0):.1f}s ({v6.get('runtime_seconds', 0)/60:.1f} min)</p>
</div>

</body>
</html>"""

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(html, encoding="utf-8")
    print(f"HTML report saved to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="V6 training output directory (contains metrics.json)")
    args = parser.parse_args()

    v6 = load_metrics(args.output_dir)
    v5 = load_v5_metrics(args.output_dir)
    v4 = load_v4_metrics(args.output_dir)

    print(f"V6: best_epoch={v6['best_epoch']}, MAE={v6['best_val_metrics']['mae']:.5f}, "
          f"Pearson={v6['best_val_metrics']['pearson']:.5f}")
    if v5:
        m = v5.get("best_val_metrics", {})
        print(f"V5: best_epoch={v5.get('best_epoch')}, MAE={m.get('mae', 0):.5f}, "
              f"Pearson={m.get('pearson', 0):.5f}")
    if v4:
        m = v4.get("best_val_metrics", {})
        print(f"V4: best_epoch={v4.get('best_epoch')}, MAE={m.get('mae', 0):.5f}, "
              f"Pearson={m.get('pearson', 0):.5f}")

    # Dashboard
    dashboard_path = args.output_dir / "probe_v6_training_dashboard.png"
    plot_training_dashboard(v6, v5, v4, dashboard_path)

    # HTML report
    report_path = args.output_dir / "probe_v6_analysis_report.html"
    generate_html_report(v6, v5, v4, report_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
