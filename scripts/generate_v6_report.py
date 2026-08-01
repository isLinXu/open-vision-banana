#!/usr/bin/env python3
"""Generate V6 analysis report from available data (epoch 1 + V4/V5 comparison)."""

import json
from pathlib import Path

OUTPUT_DIR = Path("/Users/gatilin/PycharmProjects/open-vision-banana-git-v260722/output/ssd_probe_v6")
V4_PATH = Path("/Users/gatilin/PycharmProjects/open-vision-banana-git-v260722/output/ssd_probe_v4_300ep/metrics.json")
V5_PATH = Path("/Users/gatilin/PycharmProjects/open-vision-banana-git-v260722/output/ssd_probe_v5/metrics.json")

# Load V4/V5
v4 = json.loads(V4_PATH.read_text())
v5 = json.loads(V5_PATH.read_text())

# V6 epoch 1 data (from log)
v6_epoch1 = {
    "epoch": 1,
    "val_mae": 0.21287,
    "val_rmse": 0.25387,
    "val_pearson": 0.53755,
    "val_spearman": 0.52351,
    "train_loss": 0.62554,
    "lr": 0.001360,
    "time_per_epoch_s": 3067,
}

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
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
  th {{ background: #4CAF50; color: white; }}
  .highlight {{ background: #90EE90 !important; }}
  .warning {{ background: #FFF3CD !important; }}
  .danger {{ background: #F8D7DA !important; }}
  .footer {{ margin-top: 40px; color: #999; font-size: 12px; border-top: 1px solid #ddd; padding-top: 10px; }}
  code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-family: monospace; }}
</style>
</head>
<body>
<h1>SSD-1B Depth Probe V6 — Analysis Report</h1>
<p style="color:#666">Multi-Timestep Feature Extraction + Deep Supervision | Generated: 2026-08-01</p>

<div class="card warning">
<h2>⚠ Training Status: Interrupted (1 epoch completed)</h2>
<p>V6 training completed epoch 1 successfully but was interrupted due to prohibitive training time.</p>
<table>
<tr><th>Metric</th><th>Value</th><th>Notes</th></tr>
<tr><td>Epochs completed</td><td>1 / 300</td><td>Only epoch 1 finished</td></tr>
<tr><td>Time per epoch</td><td>~51 minutes</td><td>MPS (Apple Silicon), ~0.1 step/s</td></tr>
<tr><td>Estimated total time (100 ep)</td><td>~85 hours</td><td>Not feasible on MPS</td></tr>
<tr><td>Feature extraction time</td><td>182 min (train) + 10 min (val)</td><td>One-time cost, cached to disk</td></tr>
<tr><td>Feature cache size</td><td>21.5 GB (train) + 1.1 GB (val)</td><td>Persisted at <code>output/ssd_probe_v6/feature_cache/</code></td></tr>
</table>
</div>

<div class="card">
<h2>V6 Epoch 1 Metrics</h2>
<div class="metric"><b>MAE:</b> {v6_epoch1['val_mae']:.5f}</div>
<div class="metric"><b>RMSE:</b> {v6_epoch1['val_rmse']:.5f}</div>
<div class="metric"><b>Pearson:</b> {v6_epoch1['val_pearson']:.5f}</div>
<div class="metric"><b>Spearman:</b> {v6_epoch1['val_spearman']:.5f}</div>
<div class="metric"><b>Train Loss:</b> {v6_epoch1['train_loss']:.5f}</div>
</div>

<div class="card">
<h2>Epoch 1 Comparison: V4 vs V5 vs V6</h2>
<p>Comparing first-epoch performance to assess if multi-timestep features provide early advantage:</p>
<table>
<tr><th>Version</th><th>Epoch</th><th>Val MAE ↓</th><th>Val Pearson ↑</th><th>Val RMSE ↓</th><th>Train Loss</th></tr>
<tr><td>V4 (single-t, frozen)</td><td>1</td><td>0.20596</td><td>0.54877</td><td>0.25496</td><td>0.34965</td></tr>
<tr><td>V5 (single-t, deep-sup)</td><td>8*</td><td>0.16389</td><td>0.67218</td><td>0.27861</td><td>0.31467</td></tr>
<tr class="highlight"><td><b>V6 (multi-t, deep-sup)</b></td><td>1</td><td>0.21287</td><td>0.53755</td><td>0.25387</td><td>0.62554</td></tr>
</table>
<p style="color:#666;font-size:12px">*V5 first logged epoch is 8 (eval_every_epoch=8 in V5 config).</p>
<p><b>Analysis:</b> V6 epoch 1 performance (MAE=0.213, Pearson=0.538) is comparable to V4 epoch 1
(MAE=0.206, Pearson=0.549). Multi-timestep features do not show immediate advantage at epoch 1.
V6's higher train loss (0.626 vs V4's 0.350) suggests the multi-timestep composite loss is harder
to optimize initially. The real test would be epochs 20-50 where V4/V5 plateau.</p>
</div>

<div class="card">
<h2>Final Best Metrics Comparison (V4 vs V5)</h2>
<table>
<tr><th>Version</th><th>Best Epoch</th><th>Val MAE ↓</th><th>Val Pearson ↑</th><th>Val RMSE ↓</th><th>Params</th></tr>
<tr><td>V4 (300 ep)</td><td>{v4['best_epoch']}</td><td>{v4['best_val_metrics']['mae']:.5f}</td>
<td>{v4['best_val_metrics']['pearson']:.5f}</td><td>{v4['best_val_metrics']['rmse']:.5f}</td>
<td>{v4['trainable_parameters']/1e6:.2f}M</td></tr>
<tr><td>V5 (233 ep)</td><td>{v5['best_epoch']}</td><td>{v5['best_val_metrics']['mae']:.5f}</td>
<td>{v5['best_val_metrics']['pearson']:.5f}</td><td>{v5['best_val_metrics']['rmse']:.5f}</td>
<td>{v5['trainable_parameters']/1e6:.2f}M</td></tr>
<tr class="warning"><td>V6 (1 ep only)</td><td>1</td><td>0.21287</td><td>0.53755</td><td>0.25387</td><td>4.05M</td></tr>
</table>
<p><b>V5 vs V4 improvement:</b> MAE -3.9%, Pearson +0.9% — marginal gains from deep supervision alone.</p>
<p><b>V5 overfitting:</b> Train MAE=0.00000 (total memorization) at best epoch 224.</p>
</div>

<div class="card">
<h2>V6 Architecture Details</h2>
<table>
<tr><th>Component</th><th>Specification</th></tr>
<tr><td>Feature extraction</td><td>4 timesteps (t=0, 250, 500, 750) × 6 UNet features = 24 feature maps</td></tr>
<tr><td>Feature channels (per timestep)</td><td>vae_latent(4), down_0(320), down_1(640), down_2(1280), mid(1280), unet_output(4)</td></tr>
<tr><td>Concatenated channels</td><td>vae_latent(4), down_0(1280), down_1(2560), down_2(5120), mid(5120), unet_output(16) = 14,044 total</td></tr>
<tr><td>Probe parameters</td><td>4,053,252 (4.05M) — 2.4× V5's 1.66M</td></tr>
<tr><td>Decoder</td><td>Progressive 4-stage skip decoder (16px → 32px → 64px → 128px → 256px)</td></tr>
<tr><td>Deep supervision</td><td>3 auxiliary heads at intermediate stages, cosine-scheduled weight</td></tr>
<tr><td>Augmentation</td><td>Horizontal flip (p=0.5), brightness (±0.15), Mixup (α=0.2, p=0.5)</td></tr>
<tr><td>Loss</td><td>Composite: MSE + gradient + Pearson + SSIM + edge + aux (cosine-scheduled)</td></tr>
</table>
</div>

<div class="card danger">
<h2>⚠ Key Bottleneck: Training Speed on MPS</h2>
<table>
<tr><th>Metric</th><th>V4</th><th>V5</th><th>V6</th></tr>
<tr><td>Params</td><td>1.58M</td><td>1.66M</td><td>4.05M</td></tr>
<tr><td>Feature channels</td><td>~3.3K</td><td>~3.3K</td><td>~14K (4× more)</td></tr>
<tr><td>Time/epoch (MPS)</td><td>~3 min</td><td>~3 min</td><td>~51 min (17× slower)</td></tr>
<tr><td>100 epochs</td><td>~5 hours</td><td>~5 hours</td><td>~85 hours</td></tr>
</table>
<p><b>Root cause:</b> V6's 4-timestep channel concatenation creates 14K+ channel feature tensors.
Each forward pass involves 6 projection convolutions (up to 5120→128 channels) plus interpolation
to 64×64, creating massive intermediate tensors on MPS. The MPS backend has lower utilization
for large-channel operations compared to CUDA.</p>
</div>

<div class="card">
<h2>Recommendations</h2>
<h3>Option A: Resume V6 on GPU (Recommended)</h3>
<p>Move training to a CUDA GPU (e.g., A100/V100). Expected speedup: 10-20× over MPS.</p>
<ul>
<li>Feature cache (21.5GB) can be transferred and reused</li>
<li>Estimated time: ~3-5 min/epoch → 100 epochs in ~5-8 hours</li>
<li>Full 300 epochs in ~15-25 hours (feasible overnight)</li>
</ul>

<h3>Option B: Reduce Timesteps to 2 (t=0, 500)</h3>
<p>Halve feature channels from 14K to ~7K. Expected ~2× speedup.</p>
<ul>
<li>Time/epoch: ~25 min → 100 epochs in ~42 hours (still slow)</li>
<li>Trade-off: less temporal richness but faster iteration</li>
<li>Can reuse existing cache by slicing (take timesteps 0 and 2)</li>
</ul>

<h3>Option C: Feature Compression Before Training</h3>
<p>Pre-compress multi-timestep features using PCA or learned 1×1 conv to 256 channels per feature.
Store compressed features to disk, then train lightweight probe on compressed features.</p>
<ul>
<li>One-time compression cost: ~30 min</li>
<li>Training speed: similar to V5 (~3 min/epoch)</li>
<li>Trade-off: possible information loss from compression</li>
</ul>

<h3>Option D: Accept V5 as Final (Baseline)</h3>
<p>V5 achieved MAE=0.069, Pearson=0.896 — already strong results. The V5 overfitting issue
(Train MAE=0.0) may not significantly affect inference quality if the validation metrics are good.</p>
<ul>
<li>No additional training needed</li>
<li>Risk: V5 may not generalize to unseen data due to memorization</li>
<li>Can be mitigated with test-time augmentation or ensemble</li></ul>
</div>

<div class="card">
<h2>V6 Artifacts</h2>
<table>
<tr><th>Artifact</th><th>Path</th><th>Size</th></tr>
<tr><td>Best checkpoint (ep 1)</td><td><code>output/ssd_probe_v6/best_probe.pt</code></td><td>16.2 MB</td></tr>
<tr><td>Feature cache (train)</td><td><code>output/ssd_probe_v6/feature_cache/train_cache.pt</code></td><td>21.5 GB</td></tr>
<tr><td>Feature cache (val)</td><td><code>output/ssd_probe_v6/feature_cache/val_cache.pt</code></td><td>1.1 GB</td></tr>
<tr><td>Training log</td><td><code>output/ssd_probe_v6/train.log</code></td><td>~11 KB</td></tr>
<tr><td>Training script</td><td><code>scripts/train_ssd_probe_v6.py</code></td><td>~1041 lines</td></tr>
<tr><td>V6 Probe architecture</td><td><code>src/open_vision_banana/training/depth_probe.py</code> (line 1122+)</td><td>~220 lines</td></tr>
</table>
</div>

<div class="footer">
<p>Generated by analyze_probe_v6.py | V6 Training: 2026-07-31 21:14 → 2026-08-01 01:13 (epoch 1)</p>
<p>Feature extraction: 172 min (train) + 10 min (val) = 182 min total</p>
<p>Probe training: 51 min (1 epoch) | Total wall time: ~4 hours</p>
</div>

</body>
</html>"""

report_path = OUTPUT_DIR / "probe_v6_analysis_report.html"
report_path.write_text(html, encoding="utf-8")
print(f"Report saved to {report_path}")
print(f"Size: {report_path.stat().st_size / 1024:.1f} KB")
