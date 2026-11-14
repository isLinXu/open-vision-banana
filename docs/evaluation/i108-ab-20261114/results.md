# Vision Banana · A/B 外部评测汇总

GitHub 文件页展示 HTML 源码；请下载 [index.html](index.html) 后用浏览器打开交互页面。仅分享 HTML 即可查看图表与内嵌数据；如需核验本地原始结果，请连同本目录的 `evidence/` 一并下载。

快照时间：2026-11-14T23:00:35+08:00。仅展示论文、训练起点与 I108 A/B。

**Ruilin Yao**：25 个模型身份（起点 + 24 个 checkpoint），145 项完整评测。**khazic**：41 个模型身份（起点 + 40 个 checkpoint），205 项完整评测。两者重复权重仍按来源分别展示。

[离线 HTML](index.html) · [原始数值 CSV](results.csv) · [完整结构化数据](data.json) · [来源与校验和](provenance.json) · [HF 固定快照](https://huggingface.co/khazic/ovb-i120-external-eval/tree/88f43c92048111bd25b4483e0117494937acb27f) · [Vision Banana 论文](https://arxiv.org/html/2604.20329v3)

A：学习率 **3e-5**；B：学习率 **1e-5**。共享起点 **Fusion-C1-KQ0r128-60-40**，即 C1_retest 60% + K-Q0-r128 40%。A/B 本地与 HF 实验采用 512 画布。

百分比均显示为 %；pp 为百分点。**粗体**表示数值优于同来源起点，与 HTML 红色一致。ReasonSeg / RefCOCOg 的本地历史起点协议不同，粗体仅为描述性比较。`—` 表示没有完整结果。

## 主要观察

**B 后期改善 NYUv2 深度，同时出现任务权衡**

Ruilin Yao 的 B·100k_best 达到 NYUv2 AbsRel 0.2664、δ1 48.58%；相对同来源起点，AbsRel 降低约 63.0%，δ1 增加 20.85 个百分点。Cityscapes mIoU 同时由 38.70% 降至 29.23%。ETH3D 的 AbsRel 改善，但 δ1 由 29.10% 降至 22.07%，并非所有深度指标同步提高。

**ReasonSeg 随训练步数呈现不同趋势**

本地 A·80k 的 ReasonSeg gIoU 为 43.71%，高于 A·90k 的 43.02% 与 A·95k 的 42.37%；khazic 对应为 43.87%、43.04%、42.41%。B 从本地 20k 的 44.20% 降至 100k 的 33.29%，khazic 也呈相同趋势。较晚 checkpoint 不一定在全部任务上更优。

**同权重复测支持相近结论**

重叠的 8 个 checkpoint 中，8 个已用 adapter SHA-256 核实权重一致。ReasonSeg gIoU 的 khazic−Ruilin Yao 差值为 -0.224 至 +0.375 pp。这种量级的差异不支持‘某一来源系统性偏低’的判断，也不能视作统计显著优势；两边 ReasonSeg 标签实现仍存在细节差别。

**按目标任务选取候选，避免单一综合排名**

如果侧重 NYUv2，当前本地 B·100k_best 是值得保留的候选；如果侧重 A 系列 ReasonSeg，可保留 A·80k。A·17.5k 的本地 NYUv2 AbsRel 0.2742、δ1 42.12%，也优于 A·95k 的 0.3362、33.04%。这些只是已完成外部评测中的观察，不能替代独立测试集上的最终模型选择。

已补充 [均衡融合方案](#merge-proposals)：以 B17.5k 为主对照，提出六组待验证配方；历史融合经验与详细规则见本文后半部分。

## 论文与训练起点

| 模型 / 来源 | NYUv2 AbsRel ↓ | NYUv2 δ1 ↑ | ETH3D AbsRel ↓ | ETH3D δ1 ↑ | DIODE AbsRel ↓ | DIODE δ1 ↑ | Cityscapes mIoU ↑ | ReasonSeg gIoU ↑ | RefCOCOg cIoU ↑ |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| [Vision Banana · 论文参考](https://arxiv.org/html/2604.20329v3) | 0.0810 | 94.80% | 0.1030 | 93.50% | 0.1080 | 91.70% | 69.90% | 79.30% | 73.80% |
| [起点 · C1:K-Q0-r128 = 60:40 · Ruilin Yao](evidence/ruilin/Fusion-C1-KQ0r128-60-40/nyu/full.json) | 0.7199 | 27.73% | 0.5790 | 29.10% | 3.7658 | 30.01% | 38.70% | 35.55% | 52.18% |
| [起点 · C1:K-Q0-r128 = 60:40 · khazic](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/base-000000/core/base-000000/nyu/full.json) | 0.7211 | 27.84% | 0.5847 | 29.23% | 3.7915 | 30.02% | 38.60% | 41.74% | — |

起点 ReasonSeg：Ruilin Yao 为 **200 queries / 历史提示词**，khazic 为 **344 queries / query_v2**。论文 ReasonSeg 使用 Gemini 2.5 Pro 改写查询。三者分数不可视作严格同协议复测。

## Ruilin Yao · A 组

| 模型 / 来源 | NYUv2 AbsRel ↓ | NYUv2 δ1 ↑ | ETH3D AbsRel ↓ | ETH3D δ1 ↑ | DIODE AbsRel ↓ | DIODE δ1 ↑ | Cityscapes mIoU ↑ | ReasonSeg gIoU ↑ | RefCOCOg cIoU ↑ |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| [A · 2,500_best](evidence/ruilin/I108-A/nyu/full.json) | **0.3568** | 26.70% | 0.5945 | 12.93% | **2.4837** | 23.69% | 36.02% | **35.95%** | 47.11% |
| [A · 5,000_best](evidence/ruilin/I108-A-5000/nyu/full.json) | **0.4319** | **39.41%** | **0.5406** | 21.58% | 19.4940 | **30.22%** | 38.60% | **39.39%** | 50.56% |
| [A · 7,500_best](evidence/ruilin/I108-A-7500/nyu/full.json) | **0.3817** | **38.24%** | **0.5566** | 19.18% | 4.3896 | 23.27% | 35.79% | 29.53% | 35.51% |
| [A · 10,000_best](evidence/ruilin/I108-A-10000/nyu/full.json) | **0.3921** | 15.65% | 0.6046 | 7.56% | **0.4675** | 16.92% | 33.55% | 30.04% | 41.61% |
| [A · 12,500_best](evidence/ruilin/I108-A-12500/nyu/full.json) | **0.3489** | 24.20% | **0.5273** | 16.07% | **0.4812** | 23.08% | 33.42% | **38.66%** | 47.44% |
| [A · 15,000_best](evidence/ruilin/I108-A-15000/nyu/full.json) | **0.3397** | 27.01% | **0.5520** | 12.39% | **0.4698** | 22.72% | 32.68% | **40.71%** | 46.71% |
| [A · 17,500_best](evidence/ruilin/I108-A-17500/nyu/full.json) | **0.2742** | **42.12%** | **0.4903** | 17.79% | **0.4188** | 28.85% | 31.34% | **39.64%** | 47.55% |
| [A · 22,500_best](evidence/ruilin/I108-A-22500/nyu/full.json) | **0.3592** | **30.64%** | **0.5654** | 14.28% | **0.5185** | 25.32% | 30.22% | **37.69%** | 48.60% |
| [A · 80,000_best](evidence/ruilin/I108-A-80000/nyu/full.json) | **0.3783** | **31.09%** | **0.5626** | 18.20% | **0.7643** | 19.48% | 26.64% | **43.71%** | — |
| [A · 90,000_best](evidence/ruilin/I108-A-90000/nyu/full.json) | **0.3381** | **33.00%** | **0.4962** | 18.43% | **1.1912** | 19.43% | 26.38% | **43.02%** | — |
| [A · 95,000_best](evidence/ruilin/I108-A-95000/nyu/full.json) | **0.3362** | **33.04%** | **0.5028** | 18.28% | **0.9026** | 19.01% | 26.30% | **42.37%** | — |

## Ruilin Yao · B 组

| 模型 / 来源 | NYUv2 AbsRel ↓ | NYUv2 δ1 ↑ | ETH3D AbsRel ↓ | ETH3D δ1 ↑ | DIODE AbsRel ↓ | DIODE δ1 ↑ | Cityscapes mIoU ↑ | ReasonSeg gIoU ↑ | RefCOCOg cIoU ↑ |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| [B · 2,500_best](evidence/ruilin/I108-B/nyu/full.json) | **0.3704** | 22.65% | 0.6008 | 11.41% | 52.7103 | 22.43% | 38.04% | **39.89%** | 49.05% |
| [B · 5,000_best](evidence/ruilin/I108-B-5000/nyu/full.json) | **0.3472** | **41.22%** | **0.5145** | 20.34% | **0.5179** | **31.16%** | 37.33% | **40.05%** | 50.39% |
| [B · 7,500_best](evidence/ruilin/I108-B-7500/nyu/full.json) | **0.4448** | **31.69%** | 0.6364 | 17.05% | **1.5955** | 21.92% | 38.06% | **39.43%** | 49.55% |
| [B · 10,000_best](evidence/ruilin/I108-B-10000/nyu/full.json) | **0.4357** | 11.08% | 0.6866 | 5.70% | **0.6554** | 9.25% | 36.23% | **40.99%** | 50.47% |
| [B · 12,500_best](evidence/ruilin/I108-B-12500/nyu/full.json) | **0.3925** | 18.92% | 0.6336 | 9.13% | **0.5504** | 15.54% | 37.72% | **40.18%** | 50.18% |
| [B · 15,000_best](evidence/ruilin/I108-B-15000/nyu/full.json) | **0.3380** | **30.20%** | **0.5698** | 13.62% | **0.4927** | 21.31% | 36.26% | **44.02%** | 51.19% |
| [B · 17,500_best](evidence/ruilin/I108-B-17500/nyu/full.json) | **0.2947** | **40.69%** | **0.5172** | 17.67% | **0.4680** | 27.85% | 35.67% | **43.35%** | 51.82% |
| [B · 20,000_best](evidence/ruilin/I108-B-20000/nyu/full.json) | **0.3617** | **33.95%** | 0.6022 | 16.54% | **0.9104** | 22.77% | 35.48% | **44.20%** | **52.96%** |
| [B · 22,500_best](evidence/ruilin/I108-B-22500/nyu/full.json) | **0.3864** | 24.92% | 0.6197 | 14.24% | **1.2875** | 17.55% | 35.05% | **41.09%** | 51.27% |
| [B · 25,000_best](evidence/ruilin/I108-B-25000/nyu/full.json) | **0.3848** | **28.80%** | **0.5691** | 16.10% | **3.3756** | 21.20% | 35.62% | **38.62%** | 49.19% |
| [B · 27,500_best](evidence/ruilin/I108-B-27500/nyu/full.json) | **0.3742** | **32.02%** | **0.5445** | 18.48% | **2.6546** | 21.16% | 34.81% | **36.79%** | 44.02% |
| [B · 95,000_best](evidence/ruilin/I108-B-95000/nyu/full.json) | **0.2692** | **47.12%** | **0.4245** | 21.72% | **0.5909** | **33.57%** | 29.25% | 33.31% | — |
| [B · 100,000_best](evidence/ruilin/I108-B-100000/nyu/full.json) | **0.2664** | **48.58%** | **0.4182** | 22.07% | **0.4802** | **34.50%** | 29.23% | 33.29% | — |

## khazic · A 组

| 模型 / 来源 | NYUv2 AbsRel ↓ | NYUv2 δ1 ↑ | ETH3D AbsRel ↓ | ETH3D δ1 ↑ | DIODE AbsRel ↓ | DIODE δ1 ↑ | Cityscapes mIoU ↑ | ReasonSeg gIoU ↑ | RefCOCOg cIoU ↑ |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| [A · 10,000_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/A-010000-best/core/A-010000-best/nyu/full.json) | **0.3920** | 15.65% | 0.6047 | 7.46% | **0.4675** | 16.93% | 33.71% | 30.13% | — |
| [A · 30,000_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/A-030000-best/core/A-030000-best/nyu/full.json) | **0.3687** | 19.36% | 0.6027 | 10.39% | **0.4906** | 19.46% | 30.93% | 34.79% | — |
| [A · 40,000_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/A-040000-best/core/A-040000-best/nyu/full.json) | **0.3417** | 25.09% | **0.5540** | 11.23% | **0.4762** | 15.45% | 30.06% | 32.29% | — |
| [A · 50,000_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/A-050000-best/core/A-050000-best/nyu/full.json) | **0.3574** | **37.44%** | **0.5308** | 19.50% | **0.7232** | 25.28% | 28.90% | 41.69% | — |
| [A · 60,000_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/A-060000-best/core/A-060000-best/nyu/full.json) | **0.3975** | **46.88%** | **0.4763** | 25.77% | 168.7695 | **31.62%** | 27.63% | 35.66% | — |
| [A · 62,500_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/A-062500-best/core/A-062500-best/nyu/full.json) | **0.3876** | **43.70%** | **0.4522** | 25.45% | 118.6118 | 29.95% | 27.43% | 38.07% | — |
| [A · 65,000_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/A-065000-best/core/A-065000-best/nyu/full.json) | **0.3566** | **40.97%** | **0.4748** | 21.13% | 143.4395 | 28.44% | 26.96% | 37.93% | — |
| [A · 67,500_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/A-067500-best/core/A-067500-best/nyu/full.json) | **0.3689** | **38.68%** | **0.4874** | 19.97% | 37.1528 | 27.03% | 26.59% | 38.66% | — |
| [A · 70,000_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/A-070000-best/core/A-070000-best/nyu/full.json) | **0.3766** | **35.17%** | **0.5325** | 19.15% | **1.6109** | 24.20% | 26.64% | 39.82% | — |
| [A · 72,500_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/A-072500-best/core/A-072500-best/nyu/full.json) | **0.3910** | **33.39%** | **0.5534** | 17.72% | 101.2104 | 22.69% | 26.67% | 39.71% | — |
| [A · 75,000_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/A-075000-best/core/A-075000-best/nyu/full.json) | **0.4009** | **33.20%** | **0.5633** | 17.80% | **2.0249** | 21.32% | 26.61% | 41.18% | — |
| [A · 77,500_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/A-077500-best/core/A-077500-best/nyu/full.json) | **0.3854** | **32.23%** | **0.5747** | 17.69% | **1.4255** | 20.67% | 26.48% | **42.91%** | — |
| [A · 80,000_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/A-080000-best/core/A-080000-best/nyu/full.json) | **0.3788** | **31.05%** | **0.5566** | 18.16% | **0.8373** | 19.60% | 26.89% | **43.87%** | — |
| [A · 82,500_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/A-082500-best/core/A-082500-best/nyu/full.json) | **0.3662** | **31.18%** | **0.5417** | 18.70% | **0.8976** | 19.46% | 26.98% | **43.45%** | — |
| [A · 85,000_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/A-085000-best/core/A-085000-best/nyu/full.json) | **0.3600** | **30.99%** | **0.5474** | 18.33% | **1.3203** | 18.87% | 27.03% | **43.39%** | — |
| [A · 87,500_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/A-087500-best/core/A-087500-best/nyu/full.json) | **0.3472** | **32.72%** | **0.5028** | 18.57% | **0.9138** | 19.19% | 26.45% | **42.80%** | — |
| [A · 90,000_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/A-090000-best/core/A-090000-best/nyu/full.json) | **0.3392** | **33.01%** | **0.4922** | 18.33% | **0.9594** | 19.61% | 26.42% | **43.04%** | — |
| [A · 92,500_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/A-092500-best/core/A-092500-best/nyu/full.json) | **0.3345** | **33.23%** | **0.4867** | 18.25% | **2.2977** | 19.45% | 26.38% | **42.95%** | — |
| [A · 95,000_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/A-095000-best/core/A-095000-best/nyu/full.json) | **0.3345** | **33.17%** | **0.4938** | 18.32% | **1.0104** | 19.13% | 26.16% | **42.41%** | — |
| [A · 97,500_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/A-097500-best/core/A-097500-best/nyu/full.json) | **0.3336** | **32.84%** | **0.4933** | 18.35% | **0.8265** | 18.53% | 26.15% | **42.46%** | — |
| [A · 100,000_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/A-100000-best/core/A-100000-best/nyu/full.json) | **0.3328** | **32.93%** | **0.4972** | 18.52% | **1.0057** | 18.28% | 26.27% | **42.72%** | — |
| [A · 100,000_final](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/A-100000-final/core/A-100000-final/nyu/full.json) | **0.3352** | **32.59%** | **0.5259** | 18.35% | **0.7825** | 17.32% | 26.09% | **42.25%** | — |

## khazic · B 组

| 模型 / 来源 | NYUv2 AbsRel ↓ | NYUv2 δ1 ↑ | ETH3D AbsRel ↓ | ETH3D δ1 ↑ | DIODE AbsRel ↓ | DIODE δ1 ↑ | Cityscapes mIoU ↑ | ReasonSeg gIoU ↑ | RefCOCOg cIoU ↑ |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| [B · 10,000_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/B-010000-best/core/B-010000-best/nyu/full.json) | **0.4355** | 11.07% | 0.6895 | 5.55% | **0.6152** | 9.29% | 36.24% | 41.37% | — |
| [B · 20,000_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/B-020000-best/core/B-020000-best/nyu/full.json) | **0.3624** | **33.99%** | 0.6002 | 16.64% | **1.0664** | 22.91% | 35.32% | **43.98%** | — |
| [B · 30,000_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/B-030000-best/core/B-030000-best/nyu/full.json) | **0.4101** | **29.51%** | 0.5958 | 18.04% | 58.3188 | 18.27% | 33.66% | 36.59% | — |
| [B · 40,000_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/B-040000-best/core/B-040000-best/nyu/full.json) | **0.4295** | 26.65% | **0.5821** | 15.36% | 16.8338 | 16.66% | 33.73% | 31.43% | — |
| [B · 50,000_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/B-050000-best/core/B-050000-best/nyu/full.json) | **0.3837** | 27.46% | **0.5520** | 17.42% | **0.8841** | 19.20% | 32.83% | 31.45% | — |
| [B · 57,500_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/B-057500-best/core/B-057500-best/nyu/full.json) | **0.3456** | **33.54%** | **0.5139** | 19.23% | **1.8390** | 25.47% | 32.39% | 33.45% | — |
| [B · 60,000_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/B-060000-best/core/B-060000-best/nyu/full.json) | **0.3433** | **33.24%** | **0.5143** | 18.95% | **2.7385** | 25.02% | 32.05% | 34.32% | — |
| [B · 65,000_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/B-065000-best/core/B-065000-best/nyu/full.json) | **0.3405** | **28.34%** | **0.5376** | 17.00% | **2.0658** | 22.23% | 30.22% | 34.93% | — |
| [B · 67,500_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/B-067500-best/core/B-067500-best/nyu/full.json) | **0.3389** | **27.94%** | **0.5382** | 16.40% | **2.0020** | 22.72% | 29.86% | 35.25% | — |
| [B · 70,000_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/B-070000-best/core/B-070000-best/nyu/full.json) | **0.3239** | **31.09%** | **0.5226** | 16.88% | **1.9023** | 24.24% | 29.53% | 35.47% | — |
| [B · 75,000_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/B-075000-best/core/B-075000-best/nyu/full.json) | **0.3074** | **35.94%** | **0.4960** | 18.17% | **0.5066** | 26.45% | 29.12% | 34.86% | — |
| [B · 80,000_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/B-080000-best/core/B-080000-best/nyu/full.json) | **0.2985** | **38.40%** | **0.5028** | 18.67% | **0.4645** | 28.27% | 28.91% | 34.47% | — |
| [B · 85,000_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/B-085000-best/core/B-085000-best/nyu/full.json) | **0.2837** | **41.96%** | **0.4610** | 19.68% | **0.4639** | 29.92% | 29.32% | 33.76% | — |
| [B · 87,500_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/B-087500-best/core/B-087500-best/nyu/full.json) | **0.2802** | **42.78%** | **0.4445** | 20.47% | **0.6452** | **30.41%** | 29.13% | 33.77% | — |
| [B · 92,500_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/B-092500-best/core/B-092500-best/nyu/full.json) | **0.2727** | **45.83%** | **0.4360** | 21.37% | **0.6109** | **32.54%** | 29.35% | 33.32% | — |
| [B · 95,000_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/B-095000-best/core/B-095000-best/nyu/full.json) | **0.2695** | **47.05%** | **0.4245** | 21.59% | **0.6003** | **33.37%** | 29.28% | 33.51% | — |
| [B · 100,000_best](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/B-100000-best/core/B-100000-best/nyu/full.json) | **0.2669** | **48.50%** | **0.4183** | 22.14% | **0.4720** | **34.48%** | 29.26% | 33.28% | — |
| [B · 100,000_final](https://huggingface.co/khazic/ovb-i120-external-eval/blob/88f43c92048111bd25b4483e0117494937acb27f/B-100000-final/core/B-100000-final/nyu/full.json) | **0.2631** | **49.67%** | **0.4123** | 22.50% | **0.4229** | **35.32%** | 29.82% | 33.27% | — |

## 同权重复测：ReasonSeg gIoU

差值为 khazic − Ruilin Yao。这里只比较同一步数、同 kind 且核验 adapter SHA 的权重；两边标签实现存在细节差别。

| checkpoint | Ruilin Yao | khazic | 差值（pp） | adapter SHA 一致 |
| :-- | --: | --: | --: | :-- |
| A · 10,000_best | 30.04% | 30.13% | +0.092 | 是 |
| A · 80,000_best | 43.71% | 43.87% | +0.164 | 是 |
| A · 90,000_best | 43.02% | 43.04% | +0.017 | 是 |
| A · 95,000_best | 42.37% | 42.41% | +0.040 | 是 |
| B · 10,000_best | 40.99% | 41.37% | +0.375 | 是 |
| B · 20,000_best | 44.20% | 43.98% | -0.224 | 是 |
| B · 95,000_best | 33.31% | 33.51% | +0.197 | 是 |
| B · 100,000_best | 33.29% | 33.28% | -0.013 | 是 |

## 补充指标：原始 RMSE 与 referring 明细

RMSE 单位为米；ReasonSeg cIoU 均取各 full.json 顶层，未使用嵌套 legacy_union cIoU 替代。论文未在本汇总来源中提供对应 RMSE，故不填。

| 来源 | checkpoint | NYUv2 RMSE ↓ | ETH3D RMSE ↓ | DIODE RMSE ↓ | ReasonSeg cIoU ↑ | RefCOCOg gIoU ↑ |
| :-- | :-- | --: | --: | --: | --: | --: |
| Ruilin Yao | 起点 · C1:K-Q0-r128 = 60:40 | 2.4549 | 5.3516 | 3989.6552 | 28.31% | 58.96% |
| Ruilin Yao | A · 2,500_best | 1.4784 | 6.7676 | 4413.8504 | 26.36% | 50.30% |
| Ruilin Yao | A · 5,000_best | 1.9024 | 5.7657 | 6505.4367 | 28.86% | 56.72% |
| Ruilin Yao | A · 7,500_best | 1.6913 | 5.8585 | 5709.6909 | 23.45% | 43.03% |
| Ruilin Yao | A · 10,000_best | 1.5513 | 7.2176 | 98.0999 | 22.55% | 42.82% |
| Ruilin Yao | A · 12,500_best | 1.3903 | 6.2771 | 316.6602 | 28.33% | 53.58% |
| Ruilin Yao | A · 15,000_best | 1.3996 | 6.3399 | 469.0446 | 31.40% | 52.67% |
| Ruilin Yao | A · 17,500_best | 1.2127 | 5.9761 | 213.8745 | 29.35% | 53.72% |
| Ruilin Yao | A · 22,500_best | 1.5621 | 6.2549 | 603.0651 | 25.61% | 54.80% |
| Ruilin Yao | A · 80,000_best | 1.7855 | 6.4209 | 1287.1742 | 30.87% | — |
| Ruilin Yao | A · 90,000_best | 1.5716 | 5.9005 | 2235.8960 | 30.01% | — |
| Ruilin Yao | A · 95,000_best | 1.5789 | 6.0115 | 1623.0950 | 29.75% | — |
| Ruilin Yao | B · 2,500_best | 1.4991 | 6.6739 | 2.193e+04 | 30.99% | 53.86% |
| Ruilin Yao | B · 5,000_best | 1.5859 | 5.8567 | 645.8700 | 29.22% | 56.85% |
| Ruilin Yao | B · 7,500_best | 1.9678 | 6.0931 | 1540.3347 | 28.86% | 55.09% |
| Ruilin Yao | B · 10,000_best | 1.6477 | 7.1142 | 653.7645 | 30.59% | 56.23% |
| Ruilin Yao | B · 12,500_best | 1.5381 | 6.5229 | 622.4725 | 31.31% | 56.48% |
| Ruilin Yao | B · 15,000_best | 1.4033 | 6.1501 | 340.6160 | 34.52% | 57.97% |
| Ruilin Yao | B · 17,500_best | 1.2831 | 5.9453 | 384.2293 | 33.02% | 58.60% |
| Ruilin Yao | B · 20,000_best | 1.5955 | 6.1751 | 1175.3230 | 34.08% | 59.39% |
| Ruilin Yao | B · 22,500_best | 1.6124 | 6.3142 | 1582.5328 | 30.70% | 55.09% |
| Ruilin Yao | B · 25,000_best | 1.6733 | 5.8936 | 3520.2846 | 30.62% | 52.53% |
| Ruilin Yao | B · 27,500_best | 1.6821 | 5.7414 | 2951.9524 | 29.98% | 45.22% |
| Ruilin Yao | B · 95,000_best | 1.3146 | 5.2769 | 1030.1391 | 23.20% | — |
| Ruilin Yao | B · 100,000_best | 1.3014 | 5.2337 | 756.7914 | 23.25% | — |
| khazic | 起点 · C1:K-Q0-r128 = 60:40 | 2.4630 | 5.3551 | 4033.6150 | 31.39% | — |
| khazic | A · 10,000_best | 1.5513 | 7.2238 | 98.0998 | 22.54% | — |
| khazic | A · 30,000_best | 1.4456 | 6.9837 | 423.3785 | 24.84% | — |
| khazic | A · 40,000_best | 1.4079 | 6.1293 | 276.2450 | 19.36% | — |
| khazic | A · 50,000_best | 1.6121 | 5.5847 | 1237.9743 | 28.97% | — |
| khazic | A · 60,000_best | 1.8512 | 5.1831 | 2.839e+04 | 25.37% | — |
| khazic | A · 62,500_best | 1.7975 | 5.3566 | 1.947e+04 | 26.12% | — |
| khazic | A · 65,000_best | 1.6437 | 5.4069 | 2.104e+04 | 25.56% | — |
| khazic | A · 67,500_best | 1.6880 | 5.5880 | 1.670e+04 | 26.43% | — |
| khazic | A · 70,000_best | 1.7198 | 5.9516 | 2137.9812 | 27.41% | — |
| khazic | A · 72,500_best | 1.7970 | 6.3783 | 3.743e+04 | 26.59% | — |
| khazic | A · 75,000_best | 1.8702 | 6.3954 | 3190.6061 | 28.31% | — |
| khazic | A · 77,500_best | 1.8299 | 6.3494 | 2536.5914 | 29.74% | — |
| khazic | A · 80,000_best | 1.7865 | 6.3451 | 1466.5285 | 31.05% | — |
| khazic | A · 82,500_best | 1.7136 | 6.1788 | 1618.7131 | 31.13% | — |
| khazic | A · 85,000_best | 1.6840 | 6.2641 | 2505.8122 | 30.38% | — |
| khazic | A · 87,500_best | 1.6275 | 5.9863 | 1670.8936 | 30.06% | — |
| khazic | A · 90,000_best | 1.5797 | 5.8556 | 1725.1110 | 30.02% | — |
| khazic | A · 92,500_best | 1.5665 | 5.7496 | 2182.7977 | 30.21% | — |
| khazic | A · 95,000_best | 1.5668 | 5.8285 | 1899.5690 | 29.82% | — |
| khazic | A · 97,500_best | 1.5585 | 5.8407 | 1453.5711 | 29.82% | — |
| khazic | A · 100,000_best | 1.5565 | 5.9018 | 1862.3844 | 29.96% | — |
| khazic | A · 100,000_final | 1.5753 | 6.2441 | 1375.3028 | 30.07% | — |
| khazic | B · 10,000_best | 1.6470 | 7.1812 | 587.1693 | 30.66% | — |
| khazic | B · 20,000_best | 1.5985 | 6.0933 | 1371.9323 | 34.06% | — |
| khazic | B · 30,000_best | 1.8090 | 6.9211 | 1.675e+04 | 28.32% | — |
| khazic | B · 40,000_best | 1.8804 | 5.9554 | 8748.5904 | 22.01% | — |
| khazic | B · 50,000_best | 1.6742 | 5.9061 | 1245.2108 | 19.53% | — |
| khazic | B · 57,500_best | 1.5832 | 5.7070 | 3606.4526 | 21.67% | — |
| khazic | B · 60,000_best | 1.5742 | 5.7242 | 4789.8940 | 23.03% | — |
| khazic | B · 65,000_best | 1.4760 | 5.8836 | 4051.3464 | 23.34% | — |
| khazic | B · 67,500_best | 1.4705 | 5.9099 | 4003.1066 | 23.87% | — |
| khazic | B · 70,000_best | 1.4135 | 5.9777 | 3853.6921 | 24.40% | — |
| khazic | B · 75,000_best | 1.3759 | 5.8917 | 602.8025 | 23.92% | — |
| khazic | B · 80,000_best | 1.3522 | 6.0466 | 451.6181 | 23.29% | — |
| khazic | B · 85,000_best | 1.3184 | 5.6952 | 445.8850 | 23.16% | — |
| khazic | B · 87,500_best | 1.3187 | 5.3900 | 1224.6485 | 23.21% | — |
| khazic | B · 92,500_best | 1.3220 | 5.4363 | 1103.4902 | 23.19% | — |
| khazic | B · 95,000_best | 1.3155 | 5.2770 | 1109.8913 | 23.23% | — |
| khazic | B · 100,000_best | 1.3021 | 5.2348 | 678.8043 | 23.20% | — |
| khazic | B · 100,000_final | 1.2867 | 5.2002 | 464.4062 | 23.24% | — |

## khazic ReasonSeg 标注口径对照

`legacy_union` 为主表口径；`lisa_v1` 使用 flag / ignore 和区域栅格化规则。两套结果不要混成同一序列。此表 cIoU 均取嵌套 annotation_variants 中的对应口径。

| checkpoint | legacy gIoU ↑ | lisa_v1 gIoU ↑ | legacy cIoU ↑ | lisa_v1 cIoU ↑ |
| :-- | --: | --: | --: | --: |
| 起点 · C1:K-Q0-r128 = 60:40 | 41.74% | 39.36% | 31.27% | 26.54% |
| A · 10,000_best | 30.13% | 27.86% | 22.52% | 18.27% |
| A · 30,000_best | 34.79% | 32.91% | 24.83% | 22.80% |
| A · 40,000_best | 32.29% | 30.77% | 19.36% | 16.69% |
| A · 50,000_best | 41.69% | 40.57% | 28.96% | 25.72% |
| A · 60,000_best | 35.66% | 33.59% | 25.36% | 21.20% |
| A · 62,500_best | 38.07% | 36.93% | 26.11% | 22.30% |
| A · 65,000_best | 37.93% | 37.02% | 25.56% | 21.88% |
| A · 67,500_best | 38.66% | 37.89% | 26.42% | 22.79% |
| A · 70,000_best | 39.82% | 38.98% | 27.41% | 23.81% |
| A · 72,500_best | 39.71% | 38.75% | 26.57% | 22.92% |
| A · 75,000_best | 41.18% | 39.96% | 28.30% | 24.63% |
| A · 77,500_best | 42.91% | 42.00% | 29.72% | 26.30% |
| A · 80,000_best | 43.87% | 42.89% | 31.03% | 27.65% |
| A · 82,500_best | 43.45% | 42.68% | 31.12% | 27.79% |
| A · 85,000_best | 43.39% | 42.54% | 30.38% | 26.97% |
| A · 87,500_best | 42.80% | 41.60% | 30.05% | 26.65% |
| A · 90,000_best | 43.04% | 41.73% | 30.01% | 26.69% |
| A · 92,500_best | 42.95% | 41.64% | 30.20% | 26.88% |
| A · 95,000_best | 42.41% | 41.12% | 29.82% | 26.42% |
| A · 97,500_best | 42.46% | 41.08% | 29.81% | 26.35% |
| A · 100,000_best | 42.72% | 41.19% | 29.95% | 26.46% |
| A · 100,000_final | 42.25% | 40.75% | 30.07% | 26.64% |
| B · 10,000_best | 41.37% | 39.22% | 30.54% | 26.54% |
| B · 20,000_best | 43.98% | 42.31% | 33.90% | 29.01% |
| B · 30,000_best | 36.59% | 35.42% | 28.29% | 25.91% |
| B · 40,000_best | 31.43% | 29.01% | 21.94% | 17.73% |
| B · 50,000_best | 31.45% | 29.35% | 19.50% | 15.27% |
| B · 57,500_best | 33.45% | 31.17% | 21.64% | 17.53% |
| B · 60,000_best | 34.32% | 31.88% | 23.00% | 18.82% |
| B · 65,000_best | 34.93% | 32.55% | 23.30% | 19.39% |
| B · 67,500_best | 35.25% | 32.85% | 23.84% | 19.86% |
| B · 70,000_best | 35.47% | 33.06% | 24.36% | 20.41% |
| B · 75,000_best | 34.86% | 32.44% | 23.88% | 19.95% |
| B · 80,000_best | 34.47% | 32.17% | 23.21% | 19.35% |
| B · 85,000_best | 33.76% | 31.45% | 23.08% | 19.20% |
| B · 87,500_best | 33.77% | 31.45% | 23.14% | 19.25% |
| B · 92,500_best | 33.32% | 31.01% | 23.11% | 19.24% |
| B · 95,000_best | 33.51% | 31.20% | 23.15% | 19.28% |
| B · 100,000_best | 33.28% | 30.97% | 23.12% | 19.25% |
| B · 100,000_final | 33.27% | 30.98% | 23.16% | 19.30% |

## 样本规模与评测口径

| 数据集 | 完整评测规模 | 主指标 |
| :-- | --: | :-- |
| NYUv2 | 654 | AbsRel / δ1 |
| ETH3D | 454 | AbsRel / δ1 |
| DIODE-Indoor | 325 | AbsRel / δ1 |
| Cityscapes val | 500 | mIoU |
| ReasonSeg val | 344 queries；本地历史起点 200 | gIoU |
| RefCOCOg UMD val | 4,896 queries | cIoU |

1. 论文 Vision Banana 是 Nano Banana Pro 基础上的官方模型，当前 A/B 是本项目共享融合起点上的训练实验；论文行仅为参考，不代表相同模型、数据或完整推理协议的复现。论文数字来自 arXiv v3 的 Tables 2、4、5、6。
2. A 学习率 3e-5，B 学习率 1e-5；模型仓库命名为 r64、2×8、seed 42。本汇总仅收完整外部评测，不混入 dev 指标；best 与 final 分开。
3. Ruilin Yao 起点的 ReasonSeg 为历史 200 queries，A/B 为 query_v2 的 344 queries；该起点的 RefCOCOg 也是历史提示词。对这两个任务，红色只表示数值优于同来源历史起点，不可单独归因于训练。khazic 起点与 A/B 的 ReasonSeg 均为 query_v2 / 344 queries。
4. 论文 ReasonSeg gIoU 79.3% 使用 Gemini 2.5 Pro 将推理查询改写成描述性指代；本次双方评测使用原始查询。当前 ReasonSeg 与论文不完全同口径。
5. khazic ReasonSeg 同时报告 legacy_union 和 lisa_v1。主表使用 full.json 顶层 gIoU，与 legacy_union gIoU 相等；lisa_v1 另附表。其 cIoU 顶层和嵌套 legacy_union 略有差别，原始层级均保留在 data.json，不混用。
6. 深度保留原始 metric_power / cube_edge 解码分数及历史有限正预测过滤口径。NYUv2 使用 Eigen crop / 10 m cap；ETH3D 为 80 m cap；DIODE 无 cap。主指标为像素加权，不用逐图均值或中位数尺度对齐分数替代。
7. DIODE 的原始 AbsRel 存在极端值，且有效像素分母可能随预测变化。保留科学计数法和原值，不据此计算跨任务综合分或宣称整体提升。
8. 共同运行设置为 512 canvas、50 inference steps、guidance 4.0、strength 0.85、seed 42，在原生 GT 尺寸评分。相同 adapter SHA 只能证明权重一致，不能消除标签、软件或数值实现差别。
9. 空白“—”表示缺少完整结果，不是 0。HF 快照尚无 RefCOCOg 完整结果；本地 A80k/A90k/A95k、B95k 当批仅测非 RefCOCO，B100k 已另开 RefCOCOg 补测，截至 2026-11-14 22:32 尚无完整验收结果。
10. 本地 A20k_best、A25k_best、A27.5k_best、B90k_best 在对应批次固定仓库版本检查时未发布，故未收录；这不代表它们永久不存在。未收录步数不插值、不用 final 替代 best。
11. 评测条目按“来源 × 模型身份 × 数据集”计数，两个来源的同一权重分别保留。展示红色为优于同来源起点的方向性差值，并非显著性检验。

<a id="merge-proposals"></a>

## 权重融合备选方案：兼顾深度与分割（待验证）

建议从 B17.5k_best 这个已完成六项评测的均衡对照出发，先以 75:25 和 50:50 加入 B100k_best，再比较语义保留与指代保留两条路线。以下六组仅为待验证方案，没有新增融合分数，也未启动融合或 GPU 评测。

主表仍仅收录论文、起点与 A/B；历史融合仅作为本节证据附录，不计入主表的 350 项统计。

**建议优先 P1；先把现成 B17.5k 当作需要超过的强对照。所有新配方均未测，禁止把父模型得分的加权平均当作融合预测分数。**

### 过去的 merge 经验

历史结果均来自 Ruilin Yao、512；ReasonSeg 为 200 queries / fig4_v1，RefCOCOg 也是 fig4_v1。只在历史表内部比较，不与新 A/B 的 query_v2 分数直接排序。CK50/CK75/CK90 分别表示 C1:K128=50:50、75:25、90:10；Q0M0 表示 Q0:M0=50:50。

| 模型 / 配方 | NYUv2 AbsRel ↓ | NYUv2 δ1 ↑ | ETH3D AbsRel ↓ | ETH3D δ1 ↑ | Cityscapes mIoU ↑ | ReasonSeg gIoU ↑ | RefCOCOg cIoU ↑ |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| [C1_retest](evidence/merge-history/C1_retest/nyu/full.json) | 0.5840 | 34.85% | 0.5414 | 25.00% | 34.50% | 30.48% | 42.39% |
| [K-Q0-r128](evidence/merge-history/K-Q0-r128/nyu/full.json) | 0.9238 | 14.57% | 0.6821 | 27.60% | 35.48% | 39.00% | 51.39% |
| [C1:K128 = 50:50](evidence/merge-history/Fusion-C1-KQ0r128/nyu/full.json) | 0.7634 | 23.66% | 0.5836 | 30.28% | 37.91% | 36.71% | 52.67% |
| [C1:K128 = 60:40（W0）](evidence/merge-history/Fusion-C1-KQ0r128-60-40/nyu/full.json) | 0.7199 | 27.73% | 0.5790 | 29.10% | 38.70% | 35.55% | 52.18% |
| [C1:K128 = 75:25](evidence/merge-history/Fusion-C1-KQ0r128-75-25/nyu/full.json) | 0.6817 | 31.10% | 0.5709 | 27.51% | 37.82% | 33.54% | 49.79% |
| [C1:K128 = 90:10](evidence/merge-history/Fusion-C1-KQ0r128-90-10/nyu/full.json) | 0.6164 | 34.04% | 0.5276 | 26.27% | 35.76% | 31.71% | 46.22% |
| [CK50 : M0 = 50:50](evidence/merge-history/Fusion-CK128-M0/nyu/full.json) | 0.7702 | 22.53% | 0.5743 | 28.46% | 37.27% | 35.89% | 53.83% |
| [CK50 : M1 = 50:50](evidence/merge-history/Fusion-CK128-M1/nyu/full.json) | 0.7705 | 27.50% | 0.5882 | 27.79% | 36.55% | 35.24% | 51.90% |
| [Q0:M0 = 50:50](evidence/merge-history/Fusion-Q0-M0/nyu/full.json) | 0.9501 | 19.51% | 0.8537 | 25.23% | 33.73% | 40.73% | 55.97% |
| [Q0:M1 = 50:50](evidence/merge-history/Fusion-Q0-M1/nyu/full.json) | 0.8971 | 23.33% | 0.7021 | 26.25% | 33.36% | 40.95% | 56.02% |
| [CK90 : Q0M0 = 50:50](evidence/merge-history/Fusion-CK128-90-10-Q0M0-50-50/nyu/full.json) | 0.7470 | 26.57% | 0.6328 | 24.46% | 35.68% | 36.83% | 54.80% |
| [CK75 : Q0M0 = 75:25](evidence/merge-history/Fusion-CK128-75-25-Q0M0-75-25/nyu/full.json) | 0.7197 | 27.94% | 0.6101 | 25.55% | 37.09% | 36.04% | 53.15% |

1. 60:40 是任务折中，不是深度最优。C1 占比从 50% 增至 90%，NYU AbsRel 从 0.7634 降至 0.6164、δ1 从 23.66% 升至 34.04%；RefCOCOg 却从 52.67% 降至 46.22%，Reason 从 36.71% 降至 31.71%。City 在这四档中的最高点是 60:40 的 38.70%。
2. 融合能出现互补，但指标不按权重线性变化。60:40 的 City 38.70% 高于 C1 的 34.50% 和 K128 的 35.48%；因此不能把两个父模型的得分加权平均，当作新融合的预期分数。
3. 加入 M0/M1 没有带来全面改善。在 CK50 中加入 50% M0，RefCOCOg 增加约 1.16 pp，但 City、Reason 与 NYU 两项下降。Q0:M0 / Q0:M1 虽有约 56% RefCOCOg，City 仅约 33%。语义分割与指代分割应分别保护。
4. 含 M 系列的部分旧融合涉及不同初始化；C1/K128 则共享 Q 基座。新 A/B 共享 W0，首轮更适合在共享起点的 checkpoint 间扫描比例，不把历史 60:40 当作跨实验通用最优解，也不把旧 M0/M1 自动加入新方案。

### 这次应选择哪些父模型

下表均为 **Ruilin Yao 同来源**实际分数；所有新 A/B 为 query_v2。W0 只作共享起点参考，其 referring 为旧协议。

| 模型 / 配方 | NYUv2 AbsRel ↓ | NYUv2 δ1 ↑ | ETH3D AbsRel ↓ | ETH3D δ1 ↑ | Cityscapes mIoU ↑ | ReasonSeg gIoU ↑ | RefCOCOg cIoU ↑ |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| [B · 5,000_best](evidence/ruilin/I108-B-5000/nyu/full.json) | 0.3472 | 41.22% | 0.5145 | 20.34% | 37.33% | 40.05% | 50.39% |
| [B · 17,500_best](evidence/ruilin/I108-B-17500/nyu/full.json) | 0.2947 | 40.69% | 0.5172 | 17.67% | 35.67% | 43.35% | 51.82% |
| [B · 20,000_best](evidence/ruilin/I108-B-20000/nyu/full.json) | 0.3617 | 33.95% | 0.6022 | 16.54% | 35.48% | 44.20% | 52.96% |
| [B · 100,000_best](evidence/ruilin/I108-B-100000/nyu/full.json) | 0.2664 | 48.58% | 0.4182 | 22.07% | 29.23% | 33.29% | — |
| [A · 80,000_best](evidence/ruilin/I108-A-80000/nyu/full.json) | 0.3783 | 31.09% | 0.5626 | 18.20% | 26.64% | 43.71% | — |
| [起点 · C1:K-Q0-r128 = 60:40](evidence/ruilin/Fusion-C1-KQ0r128-60-40/nyu/full.json) | 0.7199 | 27.73% | 0.5790 | 29.10% | 38.70% | 35.55% | 52.18% |

- A80k + B100k 不作均衡首选：A80k 相比 B17.5k 的 Reason 仅高约 0.36 pp，City 却低约 9.03 pp，且 NYU 较差；两个父模型的 City 都偏弱。若要检验跨 A/B 的互补，可另作有针对性的对照，不能假定合并后 City 自然恢复。
- A5k 的 City 38.60% 很高，但 DIODE AbsRel 19.4940 有异常量级；优先用 B5k 做语义保留路线。B100k_final 在 khazic 的若干深度和 City 指标上优于 best，但尚缺本地与 RefCOCOg 完整对照，应作为后续独立消融，而不是悄悄替换所有 best 端点。
- 这些 B checkpoint 共享训练轨迹，并非独立训练的模型，且 best adapter 已使用 EMA；再次平均未必带来独立模型组合的收益。选择这些父模型的依据是已测任务互补，最终是否获益必须由融合实测决定。

### 六组待验证配方

配方中的比例顺序与父模型一一对应，所有 B checkpoint 都使用 **_best**。W0 为已发布的 60:40 起点。

| 方案 | 顺序 | 完整权重配方 | 目的 | 主要风险 |
| :-- | :-- | :-- | :-- | :-- |
| **P1 · 均衡优先** | 首轮 · 优先 | B17.5k 75% + B100k 25% | 保留较多已具备深度与两类分割能力的 B17.5k，仅加入 25% 后期深度权重。建议先测这一组。 | 即使深度权重占比较小，Cityscapes、ReasonSeg 仍可能退化；它是试验假设，不是已验证的最优比例。 |
| **P2 · 更偏深度** | 首轮 · 比例对照 | B17.5k 50% + B100k 50% | 与 P1 保持相同父模型，只增加后期权重占比，判断深度收益与分割损失是否形成可接受的折中。 | B100k 的 City 与 Reason 均明显低于 B17.5k；若 P1 已触及分割保护线，不继续扩大后期占比。 |
| **P3 · 保语义分割** | 首轮 · 语义路线 | B5k 60% + B100k 40% | B5k 的 City 37.33%、ETH δ1 20.34% 较好；以较多早期权重配合 B100k 的深度能力，检查语义保留路线。 | B5k 的 Reason 40.05% 低于均衡对照，不能因 City 较高就预期指代也能保住。 |
| **P4 · 保指代分割** | 首轮 · 指代路线 | B20k 60% + B100k 40% | B20k 的 Reason 44.20%、RefCOCOg 52.96% 是本地较强端点；用后期深度更新尝试弥补其深度短板。 | B20k 的 NYU、ETH 均弱于 B17.5k；融合后必须证明深度收益，同时保住 City，不能只看 Reason。 |
| **P5 · 三方兼顾** | 次轮 · 条件扩展 | B5k 40% + B17.5k 40% + B100k 20% | 以 80% 早中期权重保留语义与指代，再加入 20% 后期深度权重。仅在二方实验显示互补性后尝试。 | 三模型并不天然优于两个模型；变量更多，可能只产生折中而不优于现成 B17.5k。 |
| **P6 · 加入少量起点** | 次轮 · 回拉对照 | W0 20% + B17.5k 55% + B100k 25% | 共享起点的 City 38.70%、ETH δ1 29.10% 支持尝试减弱训练偏移。该配方等价于 W0 + 0.55ΔB17.5k + 0.25ΔB100k。 | 起点 NYU 较弱，可能稀释深度收益；本地起点 referring 是旧协议，不能据其分数承诺 Reason 或 Ref 回升。 |

### 怎样判定是否足够均衡

以下为建议先固定的工程筛选偏好，不是论文标准、显著性结论或成功保证。允许分割最多约 1 pp 的损失，以换取可观察的深度收益；如果目标是不允许任何分割退步，可把保护线提高到 B17.5k 原值。表中数值为显示精度，实际判定使用 data.json 中未四舍五入的对照值与同样的容忍量，避免把舍入差当成改善。

| 任务 | 建议规则 | 理由 |
| :-- | :-- | :-- |
| NYUv2 深度 | AbsRel ≤ 0.2947，δ1 ≥ 40.69%；至少一项优于 B17.5k，对另一项不接受退步。 | 同时检查平均相对误差和阈值准确率，避免只优化 AbsRel。 |
| Cityscapes 语义 | mIoU ≥ 34.67%（建议保护线：B17.5k 的 35.67% − 1 pp）。 | 先限制语义能力损失，再比较候选深度收益。 |
| ReasonSeg 指代 | gIoU ≥ 42.35%（B17.5k 的 43.35% − 1 pp），固定 query_v2 / 344 queries 与相同标签口径。 | 不能用旧起点 200 queries 或更高的另一口径替代。 |
| ETH3D 泛化 | 对照为 AbsRel 0.5172 / δ1 17.67%；建议初筛容忍度：AbsRel ≤ 0.5272、δ1 ≥ 16.67%。 | 这是工程保护线，防止只在 NYUv2 上变好。 |
| RefCOCOg 决赛 | 完整 UMD val 的 cIoU ≥ 50.82%（B17.5k 的 51.82% − 1 pp）。 | 最终均衡结论必须包含 Ref；没有完整结果时只能称暂定候选。 |
| DIODE 稳定性 | 保留原始 AbsRel、δ1、RMSE 和有效像素数，核对极端预测与评分支持；不据异常降幅决定冠军。 | 历史口径的有效像素支持会随预测变化，不宜设单一降幅目标。 |

### 建议评测顺序（尚未启动）

1. 固定对照与口径：B17.5k 为均衡主对照、B5k 为语义对照、B100k_best 为深度对照。若评测代码或标签版本变化，先在同版本下校准这些对照；P6 如需讨论起点 referring 的恢复，补齐 W0 的 query_v2 评测。
2. 首轮只做 P1–P4，先测完整 NYUv2、Cityscapes、ReasonSeg（654 / 500 / 344），快速识别语义与指代是否受损。首选观察 P1，再用 P2 判断后期权重剂量效应。
3. 通过保护线的候选补完整 ETH3D 与 DIODE，再对候选前两名补 RefCOCOg UMD。当前 B100k 的 Ref 补测结果也应同步纳入父模型对照。
4. 只有二方结果提示值得扩展时才尝试 P5/P6；可围绕已测较佳比例做小范围细化，不默认把深度端加至 75% 或 90%。
5. 最终保留非支配候选：若一个方案深度更好但分割更差，明确列为偏好不同的备选，不计算未经约定的综合总分。若没有融合优于均衡对照，就继续使用 B17.5k。反复用于选比例的外部集会成为开发信号，最终泛化结论仍需独立留出数据确认。

### 正确融合的实现口径

```text
Wi = W0 + Δi
Wmerge = c0·W0 + Σ ci·Wi = W0 + Σ ci·Δi
c0 + Σ ci = 1；所有系数非负
```

1. 这里的 B5k/B17.5k/B20k/B100k 均指本报告实际评测的 _best EMA adapter；不能按 checkpoint 顶层 online 标签换用另一份权重。父文件 SHA 在下表与 data.json 中固定。
2. W0 固定为 Melo111/Fusion-C1-KQ0r128-60-40@9d5b9b8ede04d7d9f4a4ff8a6a75a4b734ab1b4d，直接使用已保存的起点。不要从 C1/K128 重新展开计算，以免改变历史 BF16 舍入。
3. 在共享基座上先恢复每个实际增量 Δi，再计算 Wmerge = c0W0 + ΣciWi = W0 + ΣciΔi，其中 c0 + Σci = 1。所有候选使用非负系数；初轮不做外推。
4. 按真实 adapter_config 恢复 rsLoRA 缩放。已核验 A80k/B100k 的 r64、α64、use_rslora=true，对应 α/√r = 8；实际实施前仍需逐个确认所选父模型的模块、shape 与 scaling。
5. 不要分别平均 LoRA_A 与 LoRA_B：两者相乘会引入跨 adapter 的交叉项，通常不等于所需的加权增量。首轮采用完整有效 Transformer 权重或精确 ΔW 加权，不同时引入 SVD 降秩、TIES/DARE 或分层调权，以便解释比例效果。
6. 延续历史实现：FP32 形成增量并加权，最后只转一次 BF16；冻结的 VAE、文本编码器等沿用共享版本。检查相同 tensor key/shape、有限值、adapter key 完整使用、父文件/输出 SHA，以及保存回读一致性。
7. 嵌套配方需区分数学系数和实际保存权重。例如 CK50:M0=50:50 的名义系数是 C1 25% + K128 25% + M0 50%，不是各 1/3；真正复现应加载保存后的 CK50 父模型。不同初始化的旧 M/Q 模型必须先在各自正确基座恢复完整权重。

实际评测父 adapter 的 SHA-256（W0 是完整融合基座，故不列 adapter SHA）：

| 模型 | adapter SHA-256 |
| :-- | :-- |
| B · 5,000_best | `705dde91d5f6ff67c9a91e47baccf048f5d1967495775d55cdd4de628fca85fa` |
| B · 17,500_best | `df5f97400d9f2924df240d261260898101bd92ebf4174c071d14a935954f9912` |
| B · 20,000_best | `f00541e1568ca9a2ba5026b99e1f5891cc4621b6dddcdc8d26769656897642c5` |
| B · 100,000_best | `fbd0a1a85ad6f42308196298077355ba3bd02fb978355c9f8b5f742b1a38d091` |
| A · 80,000_best | `b9ecfaf9e445e58a80d4d706c438dd8f99ac6b496061336003f28d44a0dd9fa7` |

权重平均的研究支持把共享初始化的微调模型作为候选，但不能保证本项目这些 checkpoint 处于相同低损失区域，也不能保证多任务指标同时提高。 [Model soups](https://proceedings.mlr.press/v162/wortsman22a.html)。

证据：[历史 60:40 实际融合配方](evidence/merge-history/fusion_recipe.json) · [历史融合保存回读核验](evidence/merge-history/historical-fusion-verification.json) · [历史 FP32 融合实现](evidence/merge-history/historical-materialize.py) · [历史指标与配方快照](evidence/merge-history/records.json)


## 可追溯与分享

- khazic 结果仓库固定 revision：`88f43c92048111bd25b4483e0117494937acb27f`。所有 205 个完整 JSON 已保存在 `evidence/khazic/`。
- Ruilin Yao 的 145 个完整 JSON 已复制至 `evidence/ruilin/`，逐一核验与本地导出的 SHA-256 一致。
- 主表模型名称链接到 NYUv2 的原始结果；其他数据集来源在 HTML 数值链接及 `provenance.json` 中逐项列出。
- HTML 中包含全部展示数据、样式与交互，可单独离线分享；需要查看本地原始结果时，请分享整个目录或 ZIP。
- data.json 保留未四舍五入的数值、补充指标、样本数、数据指纹及权重身份。下载 CSV 的百分数指标为 0–1 原值。
- 本次仅整理与导出，未重新运行 GPU 评测，也未修改历史指标。
