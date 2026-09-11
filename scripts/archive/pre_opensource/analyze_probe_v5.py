#!/usr/bin/env python3
"""Analyze SSD-1B Probe V5 training results and generate comprehensive comparison with V4."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Configuration ──────────────────────────────────────────────
V5_DIR = PROJECT_ROOT / "output" / "ssd_probe_v5"
V4_DIR = PROJECT_ROOT / "output" / "ssd_probe_v4_300ep"
REPORT_DIR = PROJECT_ROOT / "output" / "912b6a07-652a-4b0d-a480-b3b7ee65ea9a"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics(path: Path) -> dict:
    if not path.is_file():
        print(f"Warning: {path} not found")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def extract_history(metrics: dict) -> dict[str, list]:
    history = metrics.get("history", [])
    if not history:
        return {}
    epochs = [h["epoch"] for h in history]
    return {
        "epochs": epochs,
        "train_loss": [h["train_loss"] for h in history],
        "val_mae": [h["val_mae"] for h in history],
        "val_pearson": [h["val_pearson"] for h in history],
        "val_spearman": [h.get("val_spearman", 0) for h in history],
        "val_rmse": [h.get("val_rmse", 0) for h in history],
        "lr": [h["lr"] for h in history],
    }


def generate_dashboard(v5: dict, v4: dict) -> Path:
    """Generate a comprehensive training dashboard comparing V5 and V4."""
    v5_hist = extract_history(v5)
    v4_hist = extract_history(v4)

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        "SSD-1B Probe V5 vs V4 — Training Dashboard",
        fontsize=18, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ── 1. Val MAE Over Epochs ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    if v4_hist:
        ax1.plot(v4_hist["epochs"], v4_hist["val_mae"], label="V4", alpha=0.6, color="steelblue")
    if v5_hist:
        ax5_epochs = v5_hist["epochs"]
        ax1.plot(ax5_epochs, v5_hist["val_mae"], label="V5", alpha=0.8, color="crimson")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val MAE ↓")
    ax1.set_title("Validation MAE")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── 2. Val Pearson Over Epochs ─────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if v4_hist:
        ax2.plot(v4_hist["epochs"], v4_hist["val_pearson"], label="V4", alpha=0.6, color="steelblue")
    if v5_hist:
        ax2.plot(v5_hist["epochs"], v5_hist["val_pearson"], label="V5", alpha=0.8, color="crimson")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Pearson ↑")
    ax2.set_title("Validation Pearson Correlation")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ── 3. Train Loss Over Epochs ──────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    if v4_hist:
        ax3.plot(v4_hist["epochs"], v4_hist["train_loss"], label="V4", alpha=0.6, color="steelblue")
    if v5_hist:
        ax3.plot(v5_hist["epochs"], v5_hist["train_loss"], label="V5", alpha=0.8, color="crimson")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Train Loss")
    ax3.set_title("Training Loss")
    ax3.set_yscale("log")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ── 4. Val Spearman Over Epochs ────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    if v4_hist:
        ax4.plot(v4_hist["epochs"], v4_hist["val_spearman"], label="V4", alpha=0.6, color="steelblue")
    if v5_hist:
        ax4.plot(v5_hist["epochs"], v5_hist["val_spearman"], label="V5", alpha=0.8, color="crimson")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Val Spearman ↑")
    ax4.set_title("Validation Spearman Correlation")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ── 5. Learning Rate Schedule ──────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    if v4_hist:
        ax5.plot(v4_hist["epochs"], v4_hist["lr"], label="V4", alpha=0.6, color="steelblue")
    if v5_hist:
        ax5.plot(v5_hist["epochs"], v5_hist["lr"], label="V5", alpha=0.8, color="crimson")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Learning Rate")
    ax5.set_title("Learning Rate Schedule")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # ── 6. Best Metrics Comparison (Bar Chart) ─────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    v4_best = v4.get("best_val_metrics", {})
    v5_best = v5.get("best_val_metrics", {})
    metrics_names = ["MAE", "RMSE", "Pearson", "Spearman"]
    v4_vals = [v4_best.get(m.lower(), 0) for m in metrics_names]
    v5_vals = [v5_best.get(m.lower(), 0) for m in metrics_names]
    x = np.arange(len(metrics_names))
    width = 0.35
    bars1 = ax6.bar(x - width / 2, v4_vals, width, label="V4", color="steelblue", alpha=0.7)
    bars2 = ax6.bar(x + width / 2, v5_vals, width, label="V5", color="crimson", alpha=0.7)
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics_names)
    ax6.set_title("Best Val Metrics Comparison")
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis="y")
    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax6.annotate(f"{height:.4f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                      fontsize=7)

    # ── 7. Convergence Speed Analysis ──────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    if v4_hist:
        v4_norm_mae = [1 - m for m in v4_hist["val_mae"]]
        ax7.plot(v4_hist["epochs"], v4_norm_mae, label="V4 (1-MAE)", alpha=0.6, color="steelblue")
    if v5_hist:
        v5_norm_mae = [1 - m for m in v5_hist["val_mae"]]
        ax7.plot(v5_hist["epochs"], v5_norm_mae, label="V5 (1-MAE)", alpha=0.8, color="crimson")
    ax7.set_xlabel("Epoch")
    ax7.set_ylabel("1 - MAE ↑")
    ax7.set_title("Convergence Speed (Normalized)")
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # ── 8. Train-Val Gap (Overfitting Indicator) ───────────────
    ax8 = fig.add_subplot(gs[2, 1])
    v4_train_mae = v4.get("train_metrics", {}).get("mae", 0)
    v4_val_mae = v4.get("best_val_metrics", {}).get("mae", 0)
    v5_train_mae = v5.get("train_metrics", {}).get("mae", 0)
    v5_val_mae = v5.get("best_val_metrics", {}).get("mae", 0)
    versions = ["V4", "V5"]
    train_maes = [v4_train_mae, v5_train_mae]
    val_maes = [v4_val_mae, v5_val_mae]
    x_gap = np.arange(len(versions))
    bars_train = ax8.bar(x_gap - width / 2, train_maes, width, label="Train MAE", color="steelblue", alpha=0.7)
    bars_val = ax8.bar(x_gap + width / 2, val_maes, width, label="Val MAE", color="crimson", alpha=0.7)
    ax8.set_xticks(x_gap)
    ax8.set_xticklabels(versions)
    ax8.set_title("Train-Val Gap (Overfitting)")
    ax8.set_ylabel("MAE")
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis="y")
    for bar in list(bars_train) + list(bars_val):
        height = bar.get_height()
        ax8.annotate(f"{height:.5f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                      fontsize=8)

    # ── 9. Summary Text ────────────────────────────────────────
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")
    summary_lines = [
        "V5 vs V4 Summary",
        "=" * 30,
    ]
    if v4 and v5:
        v4_b = v4.get("best_val_metrics", {})
        v5_b = v5.get("best_val_metrics", {})
        summary_lines.extend([
            f"V4 best epoch:  {v4.get('best_epoch', '?')}",
            f"V5 best epoch:  {v5.get('best_epoch', '?')}",
            f"V5 early stop:  {v5.get('early_stopped', '?')}",
            "",
            f"V4 MAE:     {v4_b.get('mae', 0):.5f}",
            f"V5 MAE:     {v5_b.get('mae', 0):.5f}",
            f"Δ MAE:      {(v5_b.get('mae', 0) - v4_b.get('mae', 0)):.5f}",
            "",
            f"V4 Pearson: {v4_b.get('pearson', 0):.5f}",
            f"V5 Pearson: {v5_b.get('pearson', 0):.5f}",
            f"Δ Pearson:  {(v5_b.get('pearson', 0) - v4_b.get('pearson', 0)):.5f}",
            "",
            f"V4 runtime: {v4.get('runtime_seconds', 0) / 60:.1f} min",
            f"V5 runtime: {v5.get('runtime_seconds', 0) / 60:.1f} min",
            "",
            f"V4 params:  {v4.get('trainable_parameters', 0) / 1e6:.2f}M",
            f"V5 params:  {v5.get('trainable_parameters', 0) / 1e6:.2f}M",
        ])
    ax9.text(
        0.05, 0.95, "\n".join(summary_lines),
        transform=ax9.transAxes, fontsize=10, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    dashboard_path = REPORT_DIR / "probe_v5_training_dashboard.png"
    fig.savefig(dashboard_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dashboard_path


def generate_html_report(v5: dict, v4: dict, dashboard_path: Path) -> Path:
    """Generate a self-contained HTML analysis report."""
    v5_b = v5.get("best_val_metrics", {})
    v4_b = v4.get("best_val_metrics", {})
    v5_train = v5.get("train_metrics", {})
    v4_train = v4.get("train_metrics", {})

    # Determine improvements
    mae_improvement = ((v4_b.get("mae", 0) - v5_b.get("mae", 0)) / v4_b.get("mae", 1)) * 100 if v4_b and v5_b else 0
    pearson_improvement = (v5_b.get("pearson", 0) - v4_b.get("pearson", 0)) * 100 if v4_b and v5_b else 0

    import base64
    dashboard_b64 = ""
    if dashboard_path.is_file():
        dashboard_b64 = base64.b64encode(dashboard_path.read_bytes()).decode("ascii")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SSD-1B Probe V5 Training Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f5f5f5; color: #333; }}
  .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 10px; }}
  h2 {{ color: #16213e; margin-top: 30px; }}
  .metric-card {{ display: inline-block; background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 15px 25px; border-radius: 8px; margin: 10px; min-width: 150px; text-align: center; }}
  .metric-card.green {{ background: linear-gradient(135deg, #11998e, #38ef7d); }}
  .metric-card.red {{ background: linear-gradient(135deg, #eb3349, #f45b69); }}
  .metric-card.blue {{ background: linear-gradient(135deg, #4facfe, #00f2fe); }}
  .metric-value {{ font-size: 28px; font-weight: bold; }}
  .metric-label {{ font-size: 12px; opacity: 0.8; margin-top: 5px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
  th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
  th {{ background: #16213e; color: white; }}
  .improvement {{ color: #11998e; font-weight: bold; }}
  .regression {{ color: #eb3349; font-weight: bold; }}
  .tag {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; margin: 2px; }}
  .tag.new {{ background: #38ef7d; color: #006622; }}
  .tag.improved {{ background: #4facfe; color: #002255; }}
  .tag.optimized {{ background: #f5a623; color: #664400; }}
  .feature-list {{ list-style: none; padding: 0; }}
  .feature-list li {{ padding: 8px 0; border-bottom: 1px solid #eee; }}
  .feature-list li:before {{ content: "▸ "; color: #e94560; font-weight: bold; }}
  img.dashboard {{ width: 100%; border-radius: 8px; margin: 15px 0; }}
  .footer {{ text-align: center; margin-top: 40px; color: #999; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">
<h1>SSD-1B Probe V5 训练综合分析报告</h1>
<p style="color:#666;">生成时间：2026-07-30 | 版本：V5 Enhanced</p>

<h2>核心指标对比</h2>
<div>
  <div class="metric-card green">
    <div class="metric-value">{v5_b.get('mae', 0):.5f}</div>
    <div class="metric-label">V5 Best Val MAE ↓</div>
  </div>
  <div class="metric-card blue">
    <div class="metric-value">{v5_b.get('pearson', 0):.5f}</div>
    <div class="metric-label">V5 Best Pearson ↑</div>
  </div>
  <div class="metric-card {'green' if mae_improvement > 0 else 'red'}">
    <div class="metric-value">{'↑' if mae_improvement > 0 else '↓'} {abs(mae_improvement):.1f}%</div>
    <div class="metric-label">MAE 改善幅度</div>
  </div>
  <div class="metric-card {'green' if pearson_improvement > 0 else 'red'}">
    <div class="metric-value">{'↑' if pearson_improvement > 0 else '↓'} {abs(pearson_improvement):.2f}pp</div>
    <div class="metric-label">Pearson 改善幅度</div>
  </div>
</div>

<h2>V5 新增优化</h2>
<ul class="feature-list">
  <li><span class="tag new">NEW</span> <strong>SSIM 损失</strong> — 结构相似性损失捕获空间结构信息，直接应对 V4 发现的"边缘模糊"问题</li>
  <li><span class="tag new">NEW</span> <strong>边缘感知损失</strong> — Sobel 算子计算边缘权重图，物体边界处误差权重提升至 5×</li>
  <li><span class="tag new">NEW</span> <strong>Skip-Connection Decoder</strong> — 冻结特征在多个解码阶段注入，保留细粒度空间细节</li>
  <li><span class="tag optimized">OPT</span> <strong>Early Stopping</strong> — patience=30，自动停止训练避免过拟合，节省训练时间</li>
  <li><span class="tag optimized">OPT</span> <strong>EMA 权重</strong> — 指数移动平均（decay=0.999）平滑推理，提升泛化</li>
  <li><span class="tag optimized">OPT</span> <strong>增强数据增强</strong> — 水平翻转 + 垂直翻转 + 亮度抖动（0.1 强度）</li>
</ul>

<h2>详细指标对比</h2>
<table>
  <thead>
    <tr><th>指标</th><th>V4 (300ep)</th><th>V5 (Enhanced)</th><th>差异</th><th>评估</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>Best Val MAE ↓</td>
      <td>{v4_b.get('mae', 0):.5f}</td>
      <td>{v5_b.get('mae', 0):.5f}</td>
      <td>{(v5_b.get('mae', 0) - v4_b.get('mae', 0)):+.5f}</td>
      <td class="{'improvement' if v5_b.get('mae', 1) < v4_b.get('mae', 1) else 'regression'}">{'✓ 改善' if v5_b.get('mae', 1) < v4_b.get('mae', 1) else '✗ 退化'}</td>
    </tr>
    <tr>
      <td>Best Val RMSE ↓</td>
      <td>{v4_b.get('rmse', 0):.5f}</td>
      <td>{v5_b.get('rmse', 0):.5f}</td>
      <td>{(v5_b.get('rmse', 0) - v4_b.get('rmse', 0)):+.5f}</td>
      <td class="{'improvement' if v5_b.get('rmse', 1) < v4_b.get('rmse', 1) else 'regression'}">{'✓ 改善' if v5_b.get('rmse', 1) < v4_b.get('rmse', 1) else '✗ 退化'}</td>
    </tr>
    <tr>
      <td>Best Val Pearson ↑</td>
      <td>{v4_b.get('pearson', 0):.5f}</td>
      <td>{v5_b.get('pearson', 0):.5f}</td>
      <td>{(v5_b.get('pearson', 0) - v4_b.get('pearson', 0)):+.5f}</td>
      <td class="{'improvement' if v5_b.get('pearson', 0) > v4_b.get('pearson', 0) else 'regression'}">{'✓ 改善' if v5_b.get('pearson', 0) > v4_b.get('pearson', 0) else '✗ 退化'}</td>
    </tr>
    <tr>
      <td>Best Val Spearman ↑</td>
      <td>{v4_b.get('spearman', 0):.5f}</td>
      <td>{v5_b.get('spearman', 0):.5f}</td>
      <td>{(v5_b.get('spearman', 0) - v4_b.get('spearman', 0)):+.5f}</td>
      <td class="{'improvement' if v5_b.get('spearman', 0) > v4_b.get('spearman', 0) else 'regression'}">{'✓ 改善' if v5_b.get('spearman', 0) > v4_b.get('spearman', 0) else '✗ 退化'}</td>
    </tr>
    <tr>
      <td>Best Epoch</td>
      <td>{v4.get('best_epoch', '?')}</td>
      <td>{v5.get('best_epoch', '?')}</td>
      <td></td>
      <td>{'提前收敛' if v5.get('best_epoch', 999) < v4.get('best_epoch', 0) else '更晚收敛'}</td>
    </tr>
    <tr>
      <td>Early Stopped</td>
      <td>No (固定 300ep)</td>
      <td>{'Yes' if v5.get('early_stopped', False) else 'No'}</td>
      <td></td>
      <td>{'节省训练时间' if v5.get('early_stopped', False) else '训练满全量'}</td>
    </tr>
    <tr>
      <td>Train MAE (过拟合指标)</td>
      <td>{v4_train.get('mae', 0):.5f}</td>
      <td>{v5_train.get('mae', 0):.5f}</td>
      <td>{(v5_train.get('mae', 0) - v4_train.get('mae', 0)):+.5f}</td>
      <td>{'gap 缩小' if abs(v5_train.get('mae', 0) - v5_b.get('mae', 0)) < abs(v4_train.get('mae', 0) - v4_b.get('mae', 0)) else 'gap 未改善'}</td>
    </tr>
    <tr>
      <td>Train-Val MAE Gap</td>
      <td>{abs(v4_train.get('mae', 0) - v4_b.get('mae', 0)):.5f}</td>
      <td>{abs(v5_train.get('mae', 0) - v5_b.get('mae', 0)):.5f}</td>
      <td></td>
      <td class="{'improvement' if abs(v5_train.get('mae', 0) - v5_b.get('mae', 0)) < abs(v4_train.get('mae', 0) - v4_b.get('mae', 0)) else 'regression'}">{'过拟合减轻' if abs(v5_train.get('mae', 0) - v5_b.get('mae', 0)) < abs(v4_train.get('mae', 0) - v4_b.get('mae', 0)) else '未改善'}</td>
    </tr>
    <tr>
      <td>训练参数量</td>
      <td>{v4.get('trainable_parameters', 0) / 1e6:.2f}M</td>
      <td>{v5.get('trainable_parameters', 0) / 1e6:.2f}M</td>
      <td>{(v5.get('trainable_parameters', 0) - v4.get('trainable_parameters', 0)) / 1e6:+.2f}M</td>
      <td>Skip decoder 增加容量</td>
    </tr>
    <tr>
      <td>运行时间</td>
      <td>{v4.get('runtime_seconds', 0) / 60:.1f} min</td>
      <td>{v5.get('runtime_seconds', 0) / 60:.1f} min</td>
      <td>{(v5.get('runtime_seconds', 0) - v4.get('runtime_seconds', 0)) / 60:+.1f} min</td>
      <td>{'更高效' if v5.get('runtime_seconds', 0) < v4.get('runtime_seconds', 0) else '耗时增加'}</td>
    </tr>
  </tbody>
</table>

<h2>训练仪表盘</h2>
{"<img class='dashboard' src='data:image/png;base64," + dashboard_b64 + "' alt='Training Dashboard' />" if dashboard_b64 else "<p>仪表盘将在训练完成后生成</p>"}

<h2>下一步优化方向</h2>
<table>
  <thead>
    <tr><th>优先级</th><th>方向</th><th>预期收益</th><th>风险</th></tr>
  </thead>
  <tbody>
    <tr><td><span class="tag optimized">P2</span></td><td>多 timestep 特征融合</td><td>MAE -5~15%</td><td>高 — 需重新提取特征</td></tr>
    <tr><td><span class="tag optimized">P2</span></td><td>解冻 UNet 层（端到端微调）</td><td>MAE -10~20%</td><td>高 — 显著增加计算量</td></tr>
    <tr><td><span class="tag optimized">P3</span></td><td>多分辨率融合（HRNet 风格）</td><td>MAE -5~10%（细节）</td><td>中</td></tr>
    <tr><td><span class="tag optimized">P3</span></td><td>对抗训练（GAN 辅助）</td><td>感知质量提升</td><td>高 — 训练不稳定</td></tr>
    <tr><td><span class="tag optimized">P3</span></td><td>更大数据集（数据采集）</td><td>全面提升</td><td>低 — 需人工标注</td></tr>
  </tbody>
</table>

<div class="footer">
  Generated by SSD-1B Probe V5 Analysis Pipeline | Open Vision Banana Project
</div>
</div>
</body>
</html>"""
    html_path = REPORT_DIR / "probe_v5_analysis_report.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


def main() -> None:
    v5 = load_metrics(V5_DIR / "metrics.json")
    v4 = load_metrics(V4_DIR / "metrics.json")

    if not v5:
        print("ERROR: V5 metrics not found. Training may not have completed yet.")
        sys.exit(1)
    if not v4:
        print("WARNING: V4 metrics not found. Will generate V5-only report.")

    print("Generating training dashboard...")
    dashboard_path = generate_dashboard(v5, v4)
    print(f"  Dashboard: {dashboard_path}")

    print("Generating HTML report...")
    html_path = generate_html_report(v5, v4, dashboard_path)
    print(f"  Report: {html_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("V5 vs V4 Summary")
    print("=" * 60)
    v4_b = v4.get("best_val_metrics", {})
    v5_b = v5.get("best_val_metrics", {})
    if v4_b and v5_b:
        print(f"  V5 MAE:      {v5_b['mae']:.5f}  (V4: {v4_b['mae']:.5f}, Δ={v5_b['mae']-v4_b['mae']:+.5f})")
        print(f"  V5 Pearson:  {v5_b['pearson']:.5f}  (V4: {v4_b['pearson']:.5f}, Δ={v5_b['pearson']-v4_b['pearson']:+.5f})")
        print(f"  V5 Spearman: {v5_b['spearman']:.5f}  (V4: {v4_b['spearman']:.5f}, Δ={v5_b['spearman']-v4_b['spearman']:+.5f})")
    print(f"  V5 best epoch: {v5.get('best_epoch', '?')}")
    print(f"  V5 early stopped: {v5.get('early_stopped', '?')}")
    print(f"  V5 runtime: {v5.get('runtime_seconds', 0)/60:.1f} min")


if __name__ == "__main__":
    main()
