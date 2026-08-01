# P1-1 Paper-Style Gate 负结果分析

**日期**: 2026-08-01
**实验**: `output/paper_style_gate_rank64_256_8x300/`
**耗时**: 11 分 4 秒
**结论**: ❌ **NOT PASSED**（负结果，符合预期）

---

## 实验设计

完全对齐论文 Vision Banana 范式：

| 维度 | 论文做法 | 本实验做法 |
|---|---|---|
| 基座 | Nano Banana Pro（多模态） | SSD-1B（text-to-image） |
| 架构改动 | 无 | 无（不扩 conv_in） |
| 条件通道 | 4ch 标准 | 4ch 标准 |
| 训练模式 | 少量 vision + 大量 gen 混合 | 50% t2i-depth + 50% img2img-photo |
| 推理方式 | text + image → image | SDEdit img2img（strength=0.85） |

## 实验结果

| 指标 | 值 | 目标 | 状态 |
|---|---:|---:|:---:|
| 训练 loss | 0.364 → 0.256 ↓ | 下降 | ✅ |
| rgb_mae | 0.408 | <0.15 | ❌ |
| pearson | nan | >0.70 | ❌ |
| 视觉评估 | 模型生成 RGB 图像而非 viridis 色图 | viridis 深度 | ❌ |

可视化证据（panel_00.png）：
- **左**：输入照片（暗光房间场景）
- **中**：模型输出 — 是带艺术化扭曲的 RGB 图像（场景有窗户、椅子等结构，但与目标深度域完全不一致）
- **右**：目标 viridis 深度图（黄→紫渐变）

## 负结果的根本原因

SSD-1B **不是多模态模型** —— 它只能接受文本输入，没有"看图"的架构通道。在 `_training_step_sdxl` 不加 source conditioning 的情况下：

1. **img2img 模式**：模型收到 prompt="generate depth" + 加噪照片，被训练去**重建输入照片**（不是深度图）。prompt 只参与 cross-attention 调制，但噪声去除的目标就是照片本身。

2. **t2i 模式**：模型被训练去从零生成深度图（学到"深度图长什么样"），但**完全忽略输入照片**。

3. **推理时 SDEdit**：input photo + strength=0.85 + vision prompt，模型用 prompt 引导去噪方向，但因为训练时 prompt 与目标域的关联很弱，输出仍然是 RGB 图像特征（基础模型先验），不是 viridis 色图。

**论文中 NBP 是多模态模型**（同时接受 text + image 输入），所以论文范式对它直接有效。对 SSD-1B 来说，**必须加 source conditioning（IP2P 风格）或类似机制**才能完成"输入图→深度图"任务。

## 与 Source Conditioning Gate 的对比

之前用 IP2P 风格 source conditioning 跑过类似的过拟合 gate（`output/quick_validate_ssd_v3/`），
最低能跑到 RGB MAE < 0.10、Pearson > 0.85（split-clean 数据），
证明 **架构修改是 SSD-1B 范式的必要代价**。

## 推论

对当前 open-vision-banana 项目的真实定位：

1. **"纯 prompt 复现论文"** 在 SSD-1B 上**理论不可能**，因为架构不同
2. **"加 source conditioning 复现论文思想"** 已验证可行（Pearson 0.85+ overfit gate 通过）
3. **"真完整复现论文"** 需要换基座：
   - 闭源 NBP（无）
   - 或多模态开源模型（Qwen-VL, LLaVA, CogVLM, InternVL 等）作为生成器

## 后续路径

1. 短期：把 source conditioning 路线作为 SSD-1B 唯一可行路径，清理假说
2. 中期：尝试多模态开源模型作为基座（用 IP2P-style 注入或模型原生图像输入）
3. 长期：与 NBP 团队申请 API 复现官方 benchmark

产物位置：`output/paper_style_gate_rank64_256_8x300/`
- `gate_result.json` — 数值结果
- `panel_00.png` / `panel_01.png` — 三联对比
- `lora_step150/` / `lora_step300/` / `lora_final/` — 训练 checkpoint
