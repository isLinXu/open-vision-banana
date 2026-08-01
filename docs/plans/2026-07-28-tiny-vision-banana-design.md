# Tiny Vision Banana — 设计与实现方案

> 创建日期: 2026-07-28
> 状态: 代码已实现，待训练验证
> 目标: 完全独立于 SSD-1B 的轻量视觉模型，<50M 参数，覆盖 5 任务

## 1. 背景与动机

### 当前生产路径
- 架构: 冻结 SSD-1B (1B 参数) 提取特征 → ScalarDepthProbe (509K 参数) 预测深度
- 指标: Pearson 0.77, MAE 0.13 (115 val, 3-seed mean)
- 问题: 推理时需加载完整 SSD-1B，实际推理体积 ~1.5GB
- 仅 depth 任务有生产路径，其余 4 任务无产出

### Tiny 模型目标
- **完全独立**: 不依赖任何 diffusion model，推理时仅加载 tiny 模型
- **多任务**: 单一架构覆盖 depth/normal/edge/semantic/generation
- **轻量**: 默认配置 1.93M 参数 (~7.7MB safetensors)，远超 <20M 理想目标
- **MPS 可训练**: Apple Silicon 上单任务 gate < 5 分钟

## 2. 架构设计

### 2.1 核心思路

所有 5 个任务的输出统一为 `[3, H, W]` in `[0, 1]` 的 RGB 图像。这意味着 tiny 模型只需要一个统一的 image-to-image 架构，通过 task embedding 区分任务。

### 2.2 架构图

```
Input [-1,1] RGB [3, 256, 256]
    │
    ├── [Stem] ConvBlock(3→32, depth=1)
    │       DepthwiseSeparableConv
    │
    ├── [Encoder Stage 0] ConvBlock(32→64) → Downsample(64)
    │   skip_0 = [64, 128, 128]
    │
    ├── [Encoder Stage 1] ConvBlock(64→128) → Downsample(128)
    │   skip_1 = [128, 64, 64]
    │
    ├── [Encoder Stage 2] ConvBlock(128→256) → Downsample(256)
    │   skip_2 = [256, 32, 32]
    │
    ├── [Encoder Stage 3] ConvBlock(256→512) → Downsample(512)
    │   skip_3 = [512, 16, 16]
    │
    ├── [Bottleneck] ConvBlock(512→512)
    │       + task embedding modulation (FiLM: scale + shift)
    │
    ├── [Decoder Stage 0] Upsample(512) → cat(skip_3) → ConvBlock(1024→512)
    │
    ├── [Decoder Stage 1] Upsample(512) → cat(skip_2) → ConvBlock(768→256)
    │
    ├── [Decoder Stage 2] Upsample(256) → cat(skip_1) → ConvBlock(384→128)
    │
    ├── [Decoder Stage 3] Upsample(128) → cat(skip_0) → ConvBlock(192→64)
    │
    ├── [Output Head] Conv(64→64) → GN → SiLU → Conv(64→3)
    │       + residual_scale * input_01
    │       → sigmoid
    │
    Output [0,1] RGB [3, 256, 256]
```

### 2.3 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 卷积类型 | DepthwiseSeparableConv | 参数量仅为标准 Conv 的 1/k² + 1 |
| 归一化 | GroupNorm | 不依赖 batch size，MPS 兼容 |
| 激活 | SiLU | 平滑梯度，适合 dense prediction |
| 任务区分 | FiLM modulation | 仅在 bottleneck 注入，不干扰 encoder/decoder |
| 跳连接 | U-Net style | 多尺度信息保留 |
| 残差 | input → output (可学习 scale) | generation 任务受益于 identity mapping |
| 输出 | sigmoid | target 范围 [0, 1] |

### 2.4 参数量分析

| 配置 | base_width | stages | 参数量 | 大小 |
|------|-----------|--------|--------|------|
| Ultra | 16 | 4 | 509K | ~2MB |
| **Default** | **32** | **4** | **1.93M** | **~7.7MB** |
| Medium | 48 | 4 | 4.27M | ~17MB |
| Large | 64 | 4 | 7.53M | ~30MB |
| XLarge | 64 | 5 | 41.2M | ~165MB |

## 3. 数据流

### 3.1 输入输出契约

```
Input:  image  [B, 3, H, W]  range [-1, 1]  (与 V3FlatDataset 一致)
        task   str | Tensor[B]               (任务名或索引)
Output: pred   [B, 3, H, W]  range [0, 1]   (RGB 可视化)
```

### 3.2 v3 数据集任务映射

| 任务标签 | v3 目录名 | train 数 | val 数 | 输出编码 |
|----------|----------|----------|--------|---------|
| depth | depth | 2,181 | 115 | cube_edge colormap RGB |
| normal | normal | 1,705 | 91 | 线性映射 RGB |
| edge | structure | 2,454 | 144 | 二值白/黑 RGB |
| semantic | segmentation | 7,132 | 351 | 110色色板 RGB |
| generation | generation | 5,644 | 337 | 原始 RGB (target=image) |

### 3.3 任务平衡

训练时默认开启 oversample balancing：将所有少数任务过采样至与最大任务（semantic, 7132 样本）一致。可通过 `--no-balance` 关闭。

## 4. 损失函数

```python
L = L_value + α * L_gradient + β * L_pearson

# L_value: Smooth L1 on RGB [0,1]
# L_gradient: L1 on spatial gradients (dx, dy)
# L_pearson: 1 - Pearson correlation per sample
```

默认权重: α=0.5, β=0.1

## 5. 训练方案

### Phase 1: 过拟合 Gate (验证架构容量)

```bash
PYTHONPATH=src python scripts/train_tiny_vision_banana.py \
    --data-dir /path/to/vb_snvc_train_v3 \
    --tasks depth \
    --num-samples 8 \
    --steps 500 \
    --batch-size 4 --grad-accum 1 \
    --lr 5e-4 --warmup-steps 20 \
    --resolution 256 \
    --eval-every 50 --gate \
    --output-dir output/tiny_vb_gate_depth_256
```

成功判据: `rgb_mae <= 0.05, pearson >= 0.95`

### Phase 2: 单任务全量训练

```bash
PYTHONPATH=src python scripts/train_tiny_vision_banana.py \
    --data-dir /path/to/vb_snvc_train_v3 \
    --tasks depth \
    --steps 3000 \
    --batch-size 4 --grad-accum 4 \
    --lr 1e-3 --warmup-steps 100 \
    --resolution 256 \
    --eval-every 250 --save-every 500 \
    --output-dir output/tiny_vb_depth_full_256
```

目标: `rgb_mae < 0.20, pearson > 0.60` (vs frozen probe 0.13/0.77)

### Phase 3: 多任务联合训练

```bash
PYTHONPATH=src python scripts/train_tiny_vision_banana.py \
    --data-dir /path/to/vb_snvc_train_v3 \
    --tasks depth normal edge semantic generation \
    --steps 5000 \
    --batch-size 4 --grad-accum 4 \
    --lr 1e-3 --warmup-steps 200 \
    --resolution 256 \
    --eval-every 250 --save-every 500 \
    --output-dir output/tiny_vb_multitask_full_256
```

## 6. 评估对比

```bash
PYTHONPATH=src python scripts/evaluate_tiny_vs_probe.py \
    --tiny-checkpoint output/tiny_vb_depth_full_256/final_model \
    --data-dir /path/to/vb_snvc_train_v3 \
    --tasks depth \
    --output output/tiny_vs_probe_report.json
```

输出包含与 frozen-feature probe 的对比表:

```
Task         RGB MAE      Pearson     Count    vs Probe
depth        0.xxxx       0.xxxx      115      +0.xxxx
```

## 7. 文件清单

| 文件 | 用途 |
|------|------|
| `src/open_vision_banana/tiny_model.py` | 模型定义 + 损失 + 评估 |
| `src/open_vision_banana/data/tiny_dataset.py` | 多任务数据加载器 |
| `scripts/train_tiny_vision_banana.py` | 训练脚本 |
| `scripts/evaluate_tiny_vs_probe.py` | 评估脚本 |
| `configs/experiments/tiny_vision_banana.yaml` | 实验配置参考 |

## 8. 预期精度对比

| 指标 | Frozen Probe (SSD-1B) | Tiny (预期) | 差距 |
|------|----------------------|-------------|------|
| 推理体积 | ~1.5GB | ~8MB | 187x 压缩 |
| 参数量 | 1B + 509K | 1.93M | 500x+ |
| depth MAE | 0.1296 | 0.15-0.20 | +15-55% |
| depth Pearson | 0.7715 | 0.55-0.70 | -7-22% |
| 多任务 | 仅 depth | 5 任务 | +4 任务 |
| MPS 训练 | 不适用 | 可行 | - |

**注意**: tiny 模型精度会低于 frozen probe（后者有 SSD-1B 的强大表征力），但以 187x 体积压缩 + 5 任务覆盖换取合理精度损失是核心价值主张。

## 9. 后续扩展

1. **知识蒸馏**: 用 frozen probe 的输出作为 soft label，加速 tiny 模型收敛
2. **ONNX 导出**: 模型架构全为标准 Conv/GN/SiLU，ONNX 兼容性好
3. **12 任务扩展**: 只需在 `task_embed` 中注册新任务名，架构无需修改
4. **分辨率提升**: 训练 256px → 推理 512px（全卷积架构天然支持）
5. **量化**: INT8 量化可将 8MB 进一步压缩到 2MB
