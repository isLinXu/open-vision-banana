# Open-Vision-Banana 三条路线边界说明

> 状态：v0.5.0-rc 实验冻结点 · 2026-08-01

本文档明确项目的三条技术路线及其边界、当前状态和实验契约。

---

## 路线 A: Frozen Depth Probe（已验证替代路线）

**定位**：产品基线 / 控制对照组

**核心思想**：
冻结 SSD-1B/VAE 中间特征，仅训练一个 50.97 万参数的 ScalarDepthProbe，
做确定性相对深度预测。不经过扩散去噪循环。

**关键文件**：
- `src/open_vision_banana/depth_probe_checkpoint.py` — 版本化 artifact
- `src/open_vision_banana/training/depth_probe.py` — 特征提取/探针
- `src/open_vision_banana/inference/depth_probe.py` — FrozenDepthEstimator
- `scripts/train_ssd_probe_v6.py` — 当前最佳训练脚本

**当前最强结果**（512px, 2181 train, 115 val, 3 seed）：

| 指标 | 均值 | 最佳 seed44 |
|---|---:|---:|
| MAE | 0.12960 | 0.12690 |
| RMSE | 0.18642 | 0.18335 |
| Pearson | 0.77146 | 0.77875 |
| Spearman | 0.75653 | 0.76137 |

产物：`output/ssd1b_depth_probe_production_refit_512_full_exposure_scaled_20260728/`

**契约**：
- 输出 = 相对深度（非 metric depth）
- 对比对象 = Depth-Anything 零样本 + VAE-only baseline
- 不参与论文 benchmark 宣称

---

## 路线 B: SSD-1B Source-Conditioned Diffusion（论文复现主线）

**定位**：论文生成式范式的开源复现路径

**核心思想**：
用 SSD-1B 替代闭源 NBP，通过 InstructPix2Pix 风格的 source conditioning
（8 通道 conv_in + LoRA）做 RGB 图像条件扩散生成，统一处理多视觉任务。

**关键文件**：
- `src/open_vision_banana/conditioning.py` — SourceConditioningConfig
- `src/open_vision_banana/training/tuner.py` — InstructionTuner
- `src/open_vision_banana/inference/generator.py` — BaseGenerator
- `scripts/train_sdxl_mps.py` — 正式训练入口
- `scripts/train_multitask_lora.py` — 多任务 runner
- `configs/experiments/e1_gate_replica.yaml` — 最小复现实验

**当前状态**：
- 五任务 smoke 闭环（loss 正常下降）
- 正式训练 checkpoint 250/500，但 held-out RGB Pearson ≈ 0
- **尚未通过论文同口径 benchmark**

**实验契约**（P1 阶段）：
1. depth 8 样本过拟合 gate（RGB MAE < 0.10, Pearson > 0.7）
2. held-out 相对深度 gate
3. metric-depth 数据 gate
4. 与 FrozenDepthEstimator 和 Depth-Anything 对比
5. 不含推理后处理的原始输出仍保持结构和色彩正确

**P2 阶段**：扩展到正常→语义→实例→指代，最后做统一混训和 generation retention。

---

## 路线 C: Tiny Model CNN（独立探索对照基线）

**定位**：独立于扩散路线的轻量对照

**核心思想**：
不依赖任何扩散模型，用 12–28M 参数的小型 CNN 直接从 RGB 输入预测
相对深度/法向/分割等任务，作为路线 B 的对照——回答"扩散表征是否做
了额外贡献"。

**关键文件**：
- `src/open_vision_banana/tiny_model_v5.py` — 当前最佳版本（bw=48, tanh）
- `src/open_vision_banana/data/tiny_dataset.py` — 数据集加载
- `scripts/train_tiny_vb_v5.py` — 训练脚本
- `configs/experiments/tiny_vision_banana.yaml`

**当前状态**：
V5 已修复 V2 的 sigmoid/pooling 问题，有初步多任务结果。
不作为论文复现的一部分，仅用于方法内部分析。

---

## 通用实验契约

所有路线的实验必须满足：
1. **固定随机种子**（默认 42）
2. **固定训练/验证 split**（checksum 验证）
3. **独立进程重载**（从磁盘加载 checkpoint 重新跑验证）
4. **记录完整元数据**：数据路径、split 定义、模型配置、超参数、运行时环境
5. **可视化失败样本**
6. **结果保存为 JSON + 可视化图片**

---

## 版本历史

- v0.5.0-rc (2026-08-01): 初始路线边界文档
