# P1 多任务分离机制探索分析

**日期**: 2026-08-01
**实验**: `output/mps_multitask_gate_rank64_256/` (v1) 和 `output/mps_multitask_gate_rank64_256_v2/` (v2)

---

## 实验设计

对齐 Vision Banana 论文的多任务统一范式：

| 维度 | 论文做法 | 本实验做法 |
|---|---|---|
| 基座 | Nano Banana Pro（多模态） | SSD-1B + source conditioning |
| 任务 | 5 视觉任务 + generation | 5 视觉任务 + generation |
| 混合比例 | 很低（未公布） | 15% (v1) → 30% (v2) |
| 训练方式 | 单一模型 + prompt 切换 | 单一模型 + prompt 切换 |
| 步数 | 未公布 | 600 (v1) → 2000 (v2) |

## 实验结果

### v1 (600 步, 15% vision ratio, 11.4 分钟)

| 任务 | rgb_mae | 状态 |
|---|---:|:---:|
| depth | 0.1517 | ✅ 接近目标 |
| normal | 0.3040 | ❌ |
| instance | 0.4068 | ❌ |
| referring | 0.4397 | ❌ |
| keypoint | 0.4230 | ❌ |
| **overall** | **0.3450** | ❌ |

### v2 (2000 步, 30% vision ratio, 38.3 分钟)

| 任务 | rgb_mae | vs v1 | 状态 |
|---|---:|:---:|:---:|
| depth | 0.1541 | +0.002 | ✅ |
| normal | 0.2904 | -0.014 | ❌ |
| instance | 0.4073 | +0.001 | ❌ |
| referring | 0.4369 | -0.003 | ❌ |
| keypoint | 0.4211 | -0.002 | ❌ |
| **overall** | **0.3420** | **-0.003** | ❌ |

## 核心发现

### ✅ 成功：连续任务可学

**depth** (viridis 连续色图)：
- rgb_mae 0.15，接近目标
- 视觉质量好：床黄近/背景紫远
- source conditioning 有效

**normal** (连续 RGB 映射)：
- rgb_mae 0.29，部分可学
- 输出更像 depth 而非 normal
- 说明模型偏向 depth 域

### ❌ 失败：离散任务不可学

**instance** (离散彩色分割)：
- rgb_mae 0.41，完全失败
- 输出完全是 depth 色图，非实例分割
- 模型无法学习离散颜色映射

**referring** (单色分割)：
- rgb_mae 0.44，完全失败
- 输出完全是 depth 色图

**keypoint** (离散点检测)：
- rgb_mae 0.42，完全失败
- 输出完全是 depth 色图

### ❌ 失败：更多训练不能解决

- 600 → 2000 步：overall 仅从 0.3450 → 0.3420
- 15% → 30% vision ratio：基本无改善
- **任务混淆是架构问题，不是训练不足**

## 根本原因

### 1. SSD-1B 不是多模态模型

- 缺乏任务理解能力
- prompt 不足以区分任务
- 模型对所有任务默认输出 depth 域

### 2. 离散任务需要更强的归纳偏置

- instance/referring/keypoint 使用离散颜色 palette
- diffusion 模型擅长连续输出，不擅长离散输出
- 需要额外的结构约束（如 codebook、quantization）

### 3. 论文 NBP 是多模态模型

- 能理解图像内容和任务差异
- 输出层也适配视觉任务
- 单一模型多任务在 NBP 上可行，在 SSD-1B 上不可行

## 推论

### 短期（当前可行）

1. **depth-only 完整复现** — 既然 depth 能学，先做 depth 的标准 benchmark + 生成保真
2. **normal 作为次要任务** — 部分可学，但需要更多工作
3. **放弃离散任务** — instance/referring/keypoint 在 SSD-1B 上不可行

### 中期（需要更多工作）

1. **task embedding** — 学习任务特定的 embedding 向量注入模型
2. **per-task adapter** — 每任务独立 adapter（破坏"单一模型"但可能有效）
3. **codebook 量化** — 离散任务需要 codebook 约束输出空间

### 长期（需要换基座）

1. **多模态基座** — Qwen2-VL, LLaVA-1.5, InternVL 等
2. **全微调** — 不是 LoRA，而是全模型微调
3. **与 NBP 合作** — 申请 API 复现官方 benchmark

## 产物

- v1: `output/mps_multitask_gate_rank64_256/` (11.4 分钟)
- v2: `output/mps_multitask_gate_rank64_256_v2/` (38.3 分钟)
- 5 任务 × 2 样本 panel 可视化
- lora_step500/1000/1500/2000 + lora_final checkpoints

## 关键结论

**Vision Banana 论文的多任务统一范式在开源 diffusion 模型（SSD-1B + source conditioning）上不可行**。

原因：
1. SSD-1B 不是多模态模型，缺乏任务理解能力
2. 离散任务需要更强的归纳偏置（codebook/quantization）
3. 单一模型多任务需要多模态基座（如 NBP）

**建议**：
- 短期：depth-only 完整复现（已验证可行）
- 中期：探索 task embedding 或 codebook
- 长期：换多模态基座
