# Vision Banana 论文细粒度拆解与 open-vision-banana 对齐分析

> 论文: **Image Generators are Generalist Vision Learners** (2604.20329v3, 2026-06-05)
> 作者: Google Research (Valentin Gabeur★, Shangbang Long★, Songyou Peng★ 等)
> 基座模型: Nano Banana Pro (NBP)
> 项目: vision-banana.github.io

---

## TL;DR

论文核心发现：对 Nano Banana Pro 做轻量级 instruction-tuning（视觉任务数据仅占总训练数据的极小比例），就能在 2D/3D 视觉理解任务上达到或超越 SOTA 专用模型。关键实现点：

1. **所有视觉任务输出统一为 RGB 图像** — 图像生成即视觉理解
2. **instruction-tuning 策略与 LLM 对齐** — 类似 LLM 的 pretrain→instruction-tune 范式
3. **极低的视觉数据比例** — 保证不遗忘生成能力
4. **不开源的闭源系统** — 基座 NBP、训练数据、推理 pipeline 均闭源，复现需要自行推断

---

## 一、论文方法论完整拆解

### 1.1 核心范式

```
Generative Pretraining (Nano Banana Pro)
         ↓
Lightweight Instruction-Tuning (视觉任务数据 + 原始生成数据混合)
         ↓
Vision Banana (统一生成+理解模型)
```

类比：图像生成预训练 ≈ LLM 预训练，instruction-tuning ≈ LLM 的 instruction fine-tuning（SFT）。

**与以往工作的关键区别**：
- 不做 full fine-tuning，不添加专门模块（如 segmentation head）
- 不牺牲生成能力
- 通过 prompt 切换任务，权重完全共享

### 1.2 训练数据与策略

| 维度 | 论文描述 | 详细参数 |
|------|---------|---------|
| **基座模型** | Nano Banana Pro (NBP) | Google 闭源，图像/视频生成模型 |
| **训练方式** | 视觉任务数据混入 NBP 原始训练数据 | "at a very low ratio" |
| **视觉任务数据来源** | 2D: in-house 模型标注的 web-crawled 图像; 3D: rendering engine 合成数据 | **不含任何评估 benchmark 的训练数据** |
| **深度数据** | entirely synthetic depth data from simulation engines | Zero real-world depth data |
| **指令格式** | 自然语言 + JSON mapping + RGB tuple | prompt 风格多样性 |

**关键约束**：
- ⚠️ 论文未公开混合比例的具体数值（只说 "very low ratio"、"lightweight"）
- ⚠️ 训练数据完全闭源（in-house annotations, synthetic rendering）
- ⚠️ 训练超参数未公开

### 1.3 已实现但需要验证的细节（对应我们的代码）

以下是论文中明确给出、且我们的代码已经实现的定量参数：

## 二、各任务实现细节逐项拆解

### 2.1 Metric Depth Estimation（度量深度估计）

#### 深度→RGB 编码（Eq.1）

论文公式：
```
f(d, λ, c) = 1 - (1 - d/(λc))^(λ+1)    [λ=-3, c=10/3]
```

其中 d ∈ [0, ∞) 为度量距离（米），f(d) ∈ [0, 1) 为归一化后的值。

#### 色图映射（Color Cube Edge Traversal）

论文描述：归一化后的 f(d) 沿 RGB 颜色立方体的边缘插值，遍历路径类似 3D Hilbert curve 第一阶：
```
black → red → yellow → green → cyan → blue → magenta → white
（沿立方体边缘：黑→红→黄→绿→青→蓝→品红→白）
```

注：早期审计根据文字转录漏掉了品红顶点并反转了前半路径。上面的八顶点顺序来自论文 Figure 5；七段均严格位于 RGB 立方体边缘。

这是本文**最核心且独特的编码方案**，不同于常见的 Turbo/Viridis 色图。

#### 解码（逆映射）

- RGB → 投影到最近立方体边缘线段 → 逆线性插值 → f(d)
- f(d) → 逆幂变换 → d（度量深度，米）

#### 训练增强

论文提到训练时混入替代色图：Plasma, Inferno, Viridis, grayscale — 增强模型对不同颜色表示的鲁棒性。

#### 关键特点

- **不需要相机内参** — training 和 inference 均不使用 intrinsics
- 幂变换 λ=-3：近处精度高，远处精度低（符合实际应用需求）

#### 对齐我们的代码

| 论文细节 | 代码实现 | 对齐状态 |
|---------|---------|---------|
| Eq.1 幂变换 λ=-3, c=10/3 | `depth_mapper.py:power_transform()` | ✅ 已实现 |
| Cube edge colormap | `depth_mapper.py:CUBE_EDGE_COLORMAP` | ✅ 已实现 |
| 替代色图训练增强 | `depth_mapper.py` supports plasma/inferno/viridis/grayscale | ✅ 已实现 |
| 逆映射 | `depth_mapper.py:inverse()` | ✅ 已实现 |
| 不使用相机内参 | `pipeline.py` 深度推理无 intrinsics 输入 | ✅ 已实现 |

#### 评估数据集与指标

论文在 6 个 benchmark 上评估：

| 数据集 | δ1 | AbsRel | 备注 |
|--------|-----|--------|------|
| NYU v2 | 0.948 | 0.081 | 室内 |
| iBims-1 | 0.934 | 0.078 | 室内 |
| ETH3D | 0.935 | 0.103 | 室内/外 |
| DIODE-Indoor | 0.917 | 0.108 | 室内 |
| KITTI | 0.915 | 0.107 | 自动驾驶 |
| nuScenes | 0.643 | 0.219 | 自动驾驶 |

**论文平均 δ1 = 0.882**（6 datasets），**四个数据集(NYU+ETH3D+DIODE+KITTI) δ1=0.929 vs Depth Anything V3 的 0.918**。

我们的 eval 代码：`evaluation/depth.py` — 已实现 AbsRel, SqRel, RMSE, δ1/δ2/δ3 ✅

#### 与 SOTA 对比的关键对齐点

- DepthLM-7B: 训练用了 nuScenes，不是 zero-shot
- Depth Anything V3: 4 数据集评估，δ1=0.918，Vision Banana=0.929
- **论文声称 Vision Banana 在这些数据集上是 zero-shot**（未使用其训练数据）

### 2.2 Surface Normal Estimation（表面法线估计）

#### 法线→RGB 编码

论文公式（右手坐标系, +x右, +y上, +z朝向相机）:
```
R = trunc((1 - x)/2, min=0, max=1) × 255
G = trunc((1 + y)/2, min=0, max=1) × 255
B = trunc((1 + z)/2, min=0, max=1) × 255
```

**注意 y 方向的正负号差异**：G 通道是 `(1 + y)/2`，而我们的代码是 `G = (1 + y) / 2`。

关键标志色：
- 面向左 (-1,0,0) → Pinkish Red
- 面向上 (0,1,0) → Light Green
- 面向相机 (0,0,1) → Light Blue

#### 对齐我们的代码

| 论文细节 | 代码实现 | 对齐状态 |
|---------|---------|---------|
| R=(1-x)/2, G=(1+y)/2, B=(1+z)/2 | `normal_mapper.py` | ✅ 已实现 |
| 右手坐标系 | `normal_mapper.py` | ✅ 已实现 |

#### 评估数据集与指标

> 勘误（2026-10-09）：论文 Table 7 的列顺序是「室内均值 / NYUv2 / DIODE-indoor / ScanNet / VKITTI」，
> 此前版本把第一对数（室内均值）误当成 NYUv2，导致后面所有逐数据集列整体错位。下表已按原文重新核对，
> 并用算术验证：Vision Banana 室内均值 (17.778+13.818+15.052)/3 = 15.549 ✓。

| 方法 | 室内均值 mean/median | NYUv2 | DIODE-indoor | ScanNet | VKITTI（室外） |
|------|------|------|------|------|------|
| Marigold | 19.606/11.828 | 20.864/11.134 | 16.671/12.084 | 21.284/12.268 | – |
| DSINE | 17.017/10.190 | 16.4/8.4 | 18.453/13.871 | 16.2/8.3 | 28.9/9.9 |
| StableNormal | 17.168/10.028 | 19.707/10.527 | 13.701/9.46 | 18.098/10.097 | – |
| Lotus-2-Normal | 16.558/– | 16.9/N/A | 18.575/N/A | 14.2/N/A | 28.894/9.677 |
| **Vision Banana** | **15.549/9.300** | 17.778/8.876 | 13.818/11.556 | 15.052/7.468 | 29.063/10.699 |

**Table 1 的 4 数据集均值 18.928**：(17.778+13.818+15.052+29.063)/4 = 18.928 ✓；Lotus-2 为 (16.9+18.575+14.2+28.894)/4 = 19.642 ✓。

**要点**：
- 室内均值 15.549 / 9.300，是表中最低的 mean 和 median。
- NYUv2 上 Vision Banana (17.778) 并非最优（DSINE 16.4、Lotus-2 16.9 更低）；ScanNet 的 mean 13.818 接近 StableNormal (13.701)。
- VKITTI 上 29.063 / 10.699 与 DSINE (28.9 / 9.9)、Lotus-2 (28.894) 基本持平，论文称其为 "on par"；Lotus-2 训练时用了 Virtual KITTI 2。

我们的 eval 代码：`evaluation/normal.py` — mean/median angle error ✅

### 2.3 Semantic Segmentation（语义分割）

#### Prompt 格式

论文支持三种 prompt 风格：
1. **自然语言**: "the macaron cakes are represented by (255,255,0)"
2. **JSON mapping**: `{"cat": "red", "lock": "pink", ...}`
3. **Hex codes**: "The menu is #80C000"

**关键特性**：open-vocabulary — 类别集合不限固定集，prompt 中动态指定 class-to-color mapping。

**评估时**：对所有 19 个 Cityscapes 类别（含图中不存在的类别）使用同一 prompt。

#### 解码（RGB → 语义 mask）

简单最近邻分配：每个像素分配给距离最近的目标颜色的类别。

#### 评估结果 (Cityscapes val, mIoU)

| 方法 | mIoU | 设置 |
|------|------|------|
| SegMan-L (Fu 2025) | 84.2 | Non-zero-shot (trained on Cityscapes) |
| APE-D (Shen 2024) | 44.2 | Zero-shot |
| OpenSeeD (Zhang 2023) | 47.8 | Zero-shot |
| X-Decoder (Zou 2023) | 52.0 | Zero-shot |
| SAM 3 (Carion 2025) | 65.2 | Zero-shot |
| **Vision Banana** | **69.9** | **Zero-shot** |

**Vision Banana 超越 SAM 3 4.7 点 mIoU**。

#### 对齐我们的代码

| 论文细节 | 代码实现 | 对齐状态 |
|---------|---------|---------|
| JSON color mapping | `prompt.py:build_semantic_prompt()` | ✅ 已实现 |
| RGB tuple mapping | `prompt.py` | ✅ 已实现 |
| Hex code mapping | `prompt.py` | ✅ 已实现 |
| 最近邻颜色解码 | `segmentation/decoder.py` | ✅ 已实现 |
| Cityscapes eval | `evaluation/semantic.py` | ✅ 已实现 |

### 2.4 Instance Segmentation（实例分割）

#### Prompt 格式

实例数未知 → 不预先指定颜色，让模型动态分配不同颜色给不同实例：
```
"Generate an instance segmentation visualization of this image.
 Each piece of garlic is colored differently."
```

#### 后处理算法（Appendix A — 5阶段聚类算法）

这是论文给出的**最详细的后处理算法**，参数全部公开：

**Stage 1: Background Initialization**
```
||C(p) - C_bg||² < τ²     τ = 14
```

**Stage 2: Color Similarity Grouping**
- 16-connectivity seed-based floodfill
- ||C(pi) - C(p_seed)||² ≤ τ² (τ = 14)

**Stage 3: Noise Pruning**
```
|S| < θ_size × (H × W)     θ_size = 2 × 10⁻⁴ (0.02% of image)
```

**Stage 4: Boundary Artifact Elimination**
- 3×3 方形结构元 binary erosion
- 保留面积比 < θ_erosion = 0.1 → 丢弃

**Stage 5: Spatially Constrained Merging**
```
||μ_a - μ_b||² ≤ τ² = 14² = 196
A(S_a ∪ S_b) ≤ γ × (A(S_a) + A(S_b))     γ = 5.0
```

#### 评估数据集 SA-Co/Gold

**关键 pipeline**: Vision Banana + Gemini 3.1 Flash-Lite（用于正/负样本分类）

流程：
1. Gemini 判断 NP 是否在图中（降低 false positive）
2. Vision Banana 仅为 positive 预测的图像生成 masks
3. 使用 SA-Co 官方评估代码计算指标

**指标**（SA-Co 定义的三种）：
- `pmF1`: Positive Micro-F1 — 10 个 IoU thresholds [0.5:0.05:0.95]
- `IL_MCC`: Image-Level Matthews Correlation Coefficient — 二分类准确率
- `cgF1 = 100 × pmF1 × IL_MCC` — 统一指标（论文 Eq. 7）

#### 评估结果

| 方法 | cgF1 | IL_MCC | pmF1 |
|------|------|--------|------|
| SAM 3 (trained on SA-Co) | 54.1 | 0.82 | 66.1 |
| SAM 3 + Llama3.2 (ft) | 61.2 | 0.86 | 70.8 |
| Vision Banana + Gemini | **47.5** | **0.84** | **56.0** |

**Zero-shot 中最强**，但 lag behind SAM 3（SAM 3 训练用了 SA-Co 数据集）。

#### 对齐我们的代码

| 论文细节 | 代码实现 | 对齐状态 |
|---------|---------|---------|
| 5阶段聚类 (Stage 1-5) | `segmentation/decoder.py` | ✅ 已实现 |
| τ=14 | `segmentation/postprocess.py` | ✅ 已实现 |
| θ_size=2e-4 | `segmentation/postprocess.py` | ✅ 已实现 |
| 16-connectivity | `segmentation/grouping.py` | ✅ 已实现 |
| 3x3 erosion | `segmentation/postprocess.py` | ✅ 已实现 |
| θ_erosion=0.1 | `segmentation/postprocess.py` | ✅ 已实现 |
| γ=5.0 (合并系数) | `segmentation/postprocess.py` | ✅ 已实现 |
| Gemini-based NP filter | `evaluation/gemini.py`, `scripts/run_paper_judges.py` | ✅ 可选 REST 集成；需运行时 API key |

### 2.5 Referring Expression Segmentation（指代表达分割）

#### Prompt 格式

论文对 ReasonSeg 使用**两阶段 pipeline**：
1. Gemini 2.5 Pro 将推理查询翻译为描述性引用
2. 描述性引用作为 Vision Banana 的 prompt

#### 评估结果

**RefCOCOg UMD val (cIoU)**:

| 方法 | cIoU | 设置 |
|------|------|------|
| HyperSeg-Phi2-2.7B | 79.4 | Non-zero-shot |
| X-SAM-Phi3-3.8B | 83.8 | Non-zero-shot |
| SAM3 + Gemini 2.5 Pro | 73.4 | Zero-shot |
| **Vision Banana** | **73.8** | **Zero-shot** |

**ReasonSeg val (gIoU)**:

| 方法 | gIoU | 设置 |
|------|------|------|
| X-SAM-Phi-3-3.8B | 56.6 | Non-zero-shot |
| LISA-13B | 65.0 | Non-zero-shot |
| SAM3 + Gemini 2.5 Pro | 77.0 | Zero-shot |
| **Vision Banana + Gemini 2.5 Pro** | **79.3** | **Zero-shot** |

**关键**: Vision Banana+Gemini 2.5 Pro 在 ReasonSeg 上超越了直接在 ReasonSeg 上训练的 LISA 和 X-SAM。

#### 对齐我们的代码

| 论文细节 | 代码实现 | 对齐状态 |
|---------|---------|---------|
| Referring segmentation prompt | `prompt.py:build_referring_prompt()` | ✅ 已实现 |
| RefCOCOg eval | `evaluation/referring.py` | ✅ 数据集级 cIoU/gIoU |
| Gemini-based reasoning pipeline | `evaluation/gemini.py`, `scripts/run_paper_judges.py` | ✅ 可选 REST 集成；需运行时 API key |

### 2.6 Image Generation Preservation（生成能力保留）

#### 评估

| Benchmark | Vision Banana win rate | 说明 |
|-----------|----------------------|------|
| GenAI-Bench (T2I) | **53.5%** vs NBP 46.5% | Text-to-image 人类评估 |
| ImgEdit (Editing) | 47.8% vs NBP 52.2% | Image editing 人类评估 |

**结论**: Vision Banana 在 T2I 上甚至略优于基座 NBP，在图像编辑上接近持平。

#### 对齐我们的代码

我们的代码没有 human evaluation 机制 — 使用 FID-proxy + CLIP score 替代 ✅

---

## 三、论文中未公开/模糊的关键实现细节

### 3.1 需要推断的部分（高优先级）

| # | 细节 | 论文状态 | 推断/建议 |
|---|------|---------|----------|
| 1 | **视觉数据:生成数据混合比例** | 仅说 "at a very low ratio" | 我们默认 1:9 (10% vision)，需实验验证 |
| 2 | **LoRA rank 和 alpha** | 完全未提及 | 我们默认 rank=64, alpha=128 (rsLoRA)，合理但不精确 |
| 3 | **学习率** | 完全未提及 | 我们默认 5e-5 |
| 4 | **具体训练步数/epochs** | 完全未提及 | 需实验确定 |
| 5 | **NBP 使用哪种 pipeline** | NBP 是闭源的 | 我们用 FLUX.1-schnell 替代，这是合理的开源选择 |
| 6 | **图像分辨率** | 未明确 | 需实验确定（可能 1024×1024 或更高） |
| 7 | **batch size** | 完全未提及 | 需实验确定 |
| 8 | **instruction-tuning 的 loss 函数** | 未提及 | 推断为标准 diffusion loss (MSE) |
| 9 | **是否使用 CFG (classifier-free guidance)** | 未提及 | FLUX.1-schnell 通常 guidance_scale=0 |
| 10 | **深度数据的 camera intrinsics 处理** | 明确说不需要 | ✅ 确认对齐 |

### 3.2 架构层面的未知

| # | 细节 | 重要性 |
|---|------|--------|
| 1 | NBP 的具体架构（多少参数？Transformer? UNet?） | 🔴 P0 — 影响复现精度 |
| 2 | NBP 是否支持 img2img（image-conditioned generation） | 🔴 P0 — 论文 Fig.1 暗示输入→输出是 img2img 范式 |
| 3 | 推理时 sampling steps | 🟡 P1 |
| 4 | 是否使用 VAE latent space 训练 | 🟡 P1 |

---

## 四、与论文对齐的差距分析（代码 vs 论文）

### 4.1 ✅ 已完全对齐

| 组件 | 位置 |
|------|------|
| 深度 Power Transform (Eq.1, λ=-3, c=10/3) | `core/depth_mapper.py` |
| Cube edge colormap | `core/depth_mapper.py` |
| 法线 RGB 编码 (右手坐标系) | `core/normal_mapper.py` |
| 5阶段实例分割聚类 (Appendix A) | `core/segmentation/` |
| 语义分割最近邻解码 | `core/segmentation/decoder.py` |
| 深度评估指标 (δ1, AbsRel, etc.) | `evaluation/depth.py` |
| 法线评估指标 (mean/median angle error) | `evaluation/normal.py` |
| 分割评估指标 (mIoU, cIoU, gIoU) | `evaluation/semantic.py, referring.py` |
| 实例分割评估 (cgF1, pmF1) | `evaluation/instance.py` |
| 多任务 prompt 构建 | `inference/prompt.py` |
| LoRA 混合训练 (视觉:生成=1:9) | `training/tuner.py` |
| rsLoRA (rank=64, alpha=128) | `config.py` / `training/tuner.py` |

### 4.2 ⚠️ 需要修正或新增

| 问题 | 严重性 | 说明 |
|------|--------|------|
| **NBP→FLUX 替换的影响** | 🔴 P0 | 论文核心是 NBP 预训练的内隐视觉理解能力。FLUX.1 虽然也是 SOTA 生成模型，但不一定有相同的内部表征质量。这是复现的最大不确定因素 |
| **训练数据不匹配** | 🔴 P0 | 论文用 Google 内部标注和渲染数据，我们用公开数据集 + 合成数据 |
| **Gemini 结果尚未执行** | 🟡 P1 | 已提供 SA-Co 过滤与 ReasonSeg query→description 工具；仍需 API key、官方数据和实际运行结果 |
| **深度基准评估对齐** | 🟡 P1 | 论文 δ1=0.929 (4 datasets average)。需要用我们的 FLUX LoRA checkpoint 在相同数据集上跑一次完整评估，验证能否接近这个数字 |
| **色图训练增强** | 🟢 P2 | 论文提到训练时混入 plasma/inferno/viridis/grayscale，我们的 depth_mapper 支持但需确认训练 pipeline 是否实际启用 |
| **图片分辨率** | 🟢 P2 | 需确定最佳推理分辨率 |
| **img2img vs t2i 范式** | 🟡 P1 | 论文 Fig.1 显示模型接收输入图像+prompt 生成 RGB 可视化。这是 img2img（image-conditioned generation），我们的 pipeline 应使用 FluxImg2ImgPipeline |

### 4.3 论文独有的任务（我们未实现）

| 任务 | 状态 |
|------|------|
| **Image Generation (GenAI-Bench)** | ✅ `evaluation/generation_preservation.py` |
| **Image Editing (ImgEdit)** | ✅ `evaluation/generation_preservation.py` |

注意：我们的代码已经实现了论文未提及的额外任务：
- Edge Detection, Optical Flow, Keypoint Detection, Object Detection, Depth Edge — **这些是 open-vision-banana 的扩展**，不在论文范围内

---

## 五、复现 Roadmap（按优先级）

### 🔴 P0 — 必须完成才能声称"复现"

1. **确认 img2img pipeline 范式** — 用 `FluxImg2ImgPipeline` 而非纯 t2i
2. **训练 depth LoRA** — 用 synthetic depth data（Hypersim + VirtualKITTI），色图增强启用
3. **训练 normal LoRA** — 用 ScanNet + OASIS 或类似数据
4. **训练 segmentation LoRA** — semantic + instance + referring（或统一一个）
5. **在 NYU/ETH3D/DIODE/KITTI 上评估 depth** — 目标接近 δ1=0.929 (4 datasets avg)
6. **在 NYU/DIODE/ScanNet 上评估 normal** — 目标 mean angle error < 19°
7. **在 Cityscapes val 上评估 semantic seg** — 目标 mIoU > 65

### 🟡 P1 — 提升复现精度

8. **实现 Gemini-based filtering pipeline**（或替代方案）用于 SA-Co 评估
9. **实现 ReasonSeg reasoning pipeline**
10. **混合多任务联合训练** — 论文暗示同时训练所有视觉任务
11. **ablation study**: 不同视觉:生成数据比例的影响

### 🟢 P2 — 锦上添花

12. **扩展到论文未涉及的边缘/光流/关键点/检测任务**
13. **多 backbone 对比**（FLUX.1 vs SDXL vs Qwen-Image 等）
14. **推理加速** — 论文提到 "significantly higher computational overhead" 是当前限制

---

## 六、关键数据速查表

### 6.1 所有硬编码超参数

| 参数 | 值 | 位置 | 来源 |
|------|-----|------|------|
| 深度幂变换 λ | -3 | Eq.1 | 论文 |
| 深度幂变换 c | 10/3 | Eq.1 | 论文 |
| 实例分割颜色阈值 τ | 14 | Eq.2 | Appendix A |
| 噪声修剪 θ_size | 2×10⁻⁴ | Eq.4 | Appendix A |
| 侵蚀保留比 θ_erosion | 0.1 | Eq.5 | Appendix A |
| 合并系数 γ | 5.0 | Eq.6 | Appendix A |
| 连通性 | 16-connectivity | Stage 2 | Appendix A |
| 结构元大小 | 3×3 | Stage 4 | Appendix A |
| LoRA rank | 64 | config.yaml | 我们的默认值 |
| LoRA alpha | 128 | config.yaml | 我们的默认值 |
| 视觉:生成数据比 | 1:9 | training/tuner.py | 我们的推断 |
| 学习率 | 5e-5 | config.yaml | 我们的默认值 |

### 6.2 所有评估 benchmark 汇总

| 任务 | Benchmark | 指标 | Vision Banana 得分 | SOTA 对比 |
|------|-----------|------|-------------------|-----------|
| Depth | NYU v2 | δ1 | 0.948 | Depth Anything V3: 0.963（VB 落后） |
| Depth | ETH3D | δ1 | 0.935 | Depth Anything V3: 0.917 |
| Depth | DIODE | δ1 | 0.917 | Depth Anything V3: 0.838（按脚注 0.918 均值反推，见附录第 2 条） |
| Depth | KITTI | δ1 | 0.915 | Depth Anything V3: 0.953（同上；此处 VB 落后） |
| Depth | iBims-1 | δ1 | 0.934 | Depth Anything V3: 0.913 |
| Depth | nuScenes | δ1 | 0.643 | DepthLM-7B: 0.865 (non-zero-shot) |
| Normal | NYU v2 | mean↓ | 17.778 | DSINE: 16.4 / Lotus-2: 16.9 |
| Normal | DIODE-indoor | mean↓ | 13.818 | StableNormal: 13.701 |
| Normal | ScanNet | mean↓ | 15.052 | DSINE: 16.2 / Lotus-2: 14.2 |
| Normal | VKITTI（室外） | mean↓ | 29.063 | Lotus-2: 28.894 (训练含 VKITTI，非零样本) |
| Normal | 室内均值 | mean↓ | 15.549 | Lotus-2: 16.558 |
| Normal | 4 数据集均值（Table 1） | mean↓ | 18.928 | Lotus-2: 19.642 |
| Semantic | Cityscapes val | mIoU | 69.9 | SAM 3: 65.2 |
| Instance | SA-Co/Gold | cgF1 | 47.5 | SAM 3: 54.1 (trained on SA-Co) |
| Referring | RefCOCOg UMD | cIoU | 73.8 | SAM3+Gemini2.5: 73.4 |
| Referring | ReasonSeg | gIoU | 79.3 | SAM3+Gemini2.5: 77.0 |
| T2I Gen | GenAI-Bench | win rate | 53.5% | vs NBP: 46.5% |
| Editing | ImgEdit | win rate | 47.8% | vs NBP: 52.2% |

### 6.3 论文中提到的对比方法一览

| 模型 | 论文中角色 | 备注 |
|------|-----------|------|
| SAM 3 | 主要分割对比 | Carion et al., 2025 |
| Depth Anything V3 | 主要深度对比 | Lin et al., 2025 |
| Lotus-2 | 主要法线对比 | He et al., 2025 |
| Marigold / DSINE / StableNormal | 法线 baselines | |
| Unik3D / MoGe-2 / Depth Pro / DepthLM | 深度 baselines | |
| APE-D / OpenSeeD / X-Decoder | 语义分割 baselines | |
| OWLv2 / DINO-X / Gemini 2.5 | 实例分割 baselines | |
| HyperSeg / X-SAM / LISA / SegZero / RSVP | 指代分割 baselines | |

---

## 七、复现策略建议

### 7.1 最务实路径

由于 Nano Banana Pro 完全闭源，我们的复现本质上是 **"验证范式"而非"精确复现"**：

1. **用 FLUX.1-schnell 作为基座**（已实现）— 这是目前最接近 SOTA 的开源图像生成模型
2. **首攻 Depth** — 论文的深度编码方案最独特，如果 FLUX LoRA 能在 NYU/ETH3D 上接近论文数字，就能证明范式有效性
3. **再攻 Normal** — 相对简单（直接线性映射）
4. **最后 Segmentation** — 最复杂（prompt engineering + 后处理 pipeline）

### 7.2 论文 vs 我们代码的核心差异

| 维度 | 论文 (Vision Banana) | 我们 (open-vision-banana) |
|------|---------------------|--------------------------|
| 基座模型 | Nano Banana Pro (闭源) | FLUX.1-schnell (开源) |
| 训练数据 | Google 内部标注 + 渲染 | 公开数据集 + 合成数据 |
| 模型规模 | ~12B? (Transformer) | 12B (FLUX.1 Transformer) |
| 推理步数 | 未知 | 4 steps (schnell) |
| Gemini 辅助 | ✅ 用于 NP 过滤和 reasoning | ❌ 未实现 |
| 额外任务 | 5 个核心任务 | 10 个任务（含论文外的 5 个） |
| 发布形式 | 闭源 | 完全开源 |

### 7.3 最重要的一条建议

> **论文的核心论点不是 "NBP 多强"，而是 "图像生成预训练 = 通用视觉理解"。**
> 如果我们能用 FLUX.1-schnell + LoRA 在多个视觉任务上达到 competitive 的结果，
> 就成功验证了这个范式。不需要精确匹配论文的所有数字。
>
> 当前最应该做的是：**实际训练一个 multi-task LoRA checkpoint，在 depth/normal/seg 上做完整评估**。

---

---

## 附录：数值核验说明

### ⚠️ 需要人工复核的数值

1. **Surface Normal 4-dataset average (18.928)**：已解决。此前的"无法复现"源于 Table 7 列错位（见 §2.2 勘误）。按正确列 (17.778+13.818+15.052+29.063)/4 = 18.928 ✓，Lotus-2 同理得 19.642 ✓。

2. **Depth Anything 3 逐数据集 δ1**：Table 6 在若干单元格缺项，列归属不唯一。论文脚注给出 DA3 在 NYU+ETH3D+DIODE+KITTI 上均值为 0.918。取 NYU 0.963、ETH3D 0.917、DIODE 0.838、KITTI 0.953 时 (3.671/4 = 0.918) 恰好吻合，故采用该分配；此前文档中的 DIODE 0.671 / KITTI 0.843 应属其他列。结论：Vision Banana 在 ETH3D、DIODE 上领先 DA3，在 NYU、KITTI 上落后，四项均值 0.929 vs 0.918 的领先幅度很小。建议最终以论文原表人工复核。

3. **Depth 6-dataset avg δ1=0.882**: (0.948+0.934+0.935+0.917+0.915+0.643)/6 = 0.882，✅ 验证通过。

4. **Depth 4-dataset avg δ1=0.929**: (0.948+0.935+0.917+0.915)/4 = 0.929，✅ 验证通过。

---

*文档生成日期: 2026-07-12*
*对应论文: arXiv:2604.20329v3*
*对应代码仓库: open-vision-banana v0.4.2*

## 附录 B：深度评测协议核对（2026-10-09）

对照论文 Table 6 与 `evaluation/benchmark.py`、`scripts/evaluate_paper_benchmarks.py`：

| 项 | 论文 | 仓库 | 状态 |
|---|---|---|---|
| 指标 δ1 (<1.25)、AbsRel | 给出 | 实现一致 | ✅ |
| 不使用相机内参 | 明确 | 无内参输入 | ✅ |
| 是否做尺度对齐 | 度量深度，无对齐 | 主指标无对齐；`*_aligned` 仅作诊断 | ✅ |
| crop / 深度上限 | **未披露** | NYU Eigen + 10m，KITTI Garg + 80m，ETH3D 80m（沿用社区常规） | ⚠️ 假设 |
| 聚合方式 | **未披露** | 像素加权（主指标）+ 逐图平均（`*_per_image`） | ⚠️ 假设，两种都报 |
| 评测分辨率 | **未披露**，推测原图分辨率 | 默认方形 512×512；`--native-resolution` 在原始 GT 分辨率打分（深度 NYU/iBims/DIODE/KITTI/ETH3D；法向量 NYU/DIODE-Indoor/ScanNet/VKITTI；RefCOCOg 掩码） | ⚠️ 部分解决 |

**已修复的协议错误**：NYU 的 "Eigen crop" 原先用的是 KITTI Garg crop 的比例（上 40.8%、下 99.2%、左右 3.5%），即评测时丢掉图像上方约 40%；KITTI 的 Garg crop 又因为要求画布宽 ≥1000，在默认 512×512 下从不触发。现已改为：NYU 用 Eigen 的 `[45:471, 41:601]`（480×640 帧），KITTI 用 Garg 比例，二者按比例在任意画布尺寸生效，并新增单测。

**对已有结果的影响**：`docs/plans/2026-08-28*`、`2026-08-30*`、`2026-08-31*` 中的 NYU δ1（0.403 / 0.357 / 0.286 / 0.253 / 0.018 等）、`output/acceptance_stage/*.json` 都是在旧 crop 下得到的。**绝对数值不可与修复后的结果直接比较**，需要用同一 checkpoint 重跑；方法之间的相对排序大概率不变，但没有验证。KITTI 相关结果（`sana_kitti_metric_depth_*`）同样受影响，因为之前 Garg crop 根本没生效。

**偏离的处理进展**：
1. 聚合方式：`DepthAccumulator` 现在同时输出像素加权（主指标，键名不变）和逐图平均（`AbsRel_per_image`、`RMSE_per_image`、`delta_1.25_per_image`、`images`）。论文未说明用哪种，建议两种都报。
2. 评测分辨率：`scripts/evaluate_paper_benchmarks.py --native-resolution` 把解码后的深度双线性上采样到原始 GT 分辨率再计分，GT 不再被缩放。支持 NYU、iBims、DIODE、KITTI、ETH3D（数据集 `native_gt=True` 返回 `gt_depth_native` / `valid_mask_native`，有效像素规则与缩放后一致）；其他数据集会明确报错而不是静默退回方形画布。`depth_trivial_baselines.py` 同样支持 `--native-resolution`，保证基线与模型可比。
   - 模型本身仍在方形 512 画布上生成，压扁长宽比的问题只是在"打分阶段"消除，没有在"生成阶段"消除。

**平凡基线**：`scripts/depth_trivial_baselines.py`，在同一协议下给出常数深度、数据集中位数常数（oracle，上界）、逐图中位数（oracle，"只知道尺度"的上界）和合成训练集的像素均值图（合法的零样本先验基线）。判断一个 δ1 是否体现了几何学习，应以这些基线为参照，尤其是 `prior_mean_map`。

## 附录 C：语义分割必须是「prompt 定义的开放词表」（2026-10-09）

**论文**：类别→颜色映射写在 prompt 里，支持命名颜色 JSON、RGB 元组、十六进制和自然语言四种写法（Fig. 2）；Cityscapes 评测时对每张图使用同一个 19 类 prompt，包含图中不存在的类别，解码为最近色。

**仓库此前的做法**（会让模型学成"背调色板"而不是"读 prompt"）：
- `configs/encoding_contract.yaml`：SANA-1600M 训练线用 SegFormer 伪标签 + 固定 ADE20K-150 调色板 + 单一固定 prompt。
- `task_prompts.normalize_task_prompt`：旧训练路径（`training/tuner.py`、`data/datasets.py::V3FlatDataset`）会把**任何**语义样本的 prompt 强制改写为固定的 Cityscapes 图例 prompt——即使目标图用的是别的调色板，prompt 与目标也自相矛盾。
- `PromptBuilder.build_semantic_prompt` 在类别数 >19 且未指定模板时退化为不含图例的通用 prompt，开放词表推理因此无法直接使用。

**新增**：`data/openvocab_semantic.py` + `scripts/build_openvocab_semantic_manifest.py`。对每个 (图, variant) 重新采样：提到的类别（可见类 + 不在图中的干扰类）、每类颜色（两两 RGB 距离 ≥ 90，含背景色）、背景色、prompt 写法（4 种）。输出 `pairs.jsonl`，每行自带 `prompt`，由统一训练器的 `ManifestVisionDataset` 原样读取（不会被 `normalize_task_prompt` 改写）。

```bash
python scripts/build_openvocab_semantic_manifest.py \
  --images data/ade20k/images/training --labels data/ade20k/annotations/training \
  --classes ade20k_classes.json --source-name ade20k --label-offset 1 --ignore-index -1 \
  --out-dir data/openvocab_semantic_ade20k --variants 2
```

**使用注意**：
1. 只能走统一训练器（`scripts/train.py`）。旧 `InstructionTuner` / `V3FlatDataset` 路径会覆盖 prompt，不要用于这份数据。
2. 脚本默认拒绝 Cityscapes 等评测集作为训练源（`--allow-benchmark-source` 才放行），以免破坏零样本口径；用于"能否学会"的域内 gate 时，结果不得按零样本汇报。
3. 推理/评测时用 `PromptBuilder.build_semantic_prompt(mapping, template_idx=0|1|2)` 显式指定模板；Cityscapes 评测沿用 `SEMANTIC_SEGMENTATION_PROMPT` 即可（它就是一种合法的 19 类映射）。
4. 标签需是单通道索引 PNG；ADE20K SceneParse150 的 0 表示 other，用 `--label-offset 1` 处理。
5. 未验证项：这套数据在真实模型上能否提升 Cityscapes 零样本 mIoU——需要 GPU 训练与 Cityscapes val 评测，本机无法验证。


## 附录 D：编辑型基座实验、prompt 开关与 DIODE 适配器（2026-10-09）

### D.1 为什么要换条件范式：FLUX.1 Kontext 的 in-context 条件

此前所有训练线（SANA、FLUX.1-schnell/dev）都是文生图骨干，输入图只能通过 `img2img_flow` 注入：把"源图隐变量→目标隐变量"当作确定性桥（`flow_sigma=0`，无噪声）。论文的 NBP 是原生"图像+指令→图像"的编辑模型，输出从噪声生成。FLUX.1 Kontext 是最接近的开源对应物：源图的 latent token 拼在目标 token 之后、位置编码第 0 轴置 1，只对目标 token 的输出算损失。

**已实现并验证（CPU，随机初始化的小型 `FluxTransformer2DModel`；约定已对照 diffusers `FluxKontextPipeline` 源码）**：
- 注册表新增 `flux-kontext`（`in_context_conditioning=True`）。
- `ModelBackend.forward(..., context_latents=...)`：拼接 token 与位置 id、FLUX 原生 timestep σ=1−t、只取目标 token、输出取负以保持仓库内的 flow 约定（`v = target − noise`），因此损失、辅助损失、验证代码完全不用改。
- 训练器对 in-context 模型自动关闭 `img2img_flow`；其他模型的调用方式逐字节不变。
- 推理：`BaseGenerator.generate(image=...)` 对 in-context 模型走官方 `FluxKontextPipeline`（`_auto_resize=False`，保持输入尺寸，**绝不单步**）。

**预注册的通过标准**（`evaluation/gates.py`，实验前固定）：
1. 度量闸门：NYU δ1 ≥ 最佳非 oracle 平凡基线 + 0.10，且 AbsRel 不劣于该基线；
2. 结构闸门：逐图中位数对齐后的 δ1 > `oracle_image_median`（完美尺度但零结构）。

**运行**：`NYU_ROOT=/path/to/nyu bash scripts/run_kontext_depth_gate.sh [N]`（平凡基线 → 训练 → 逐 checkpoint 原分辨率多步评测 → 闸门）。配置 `configs/experiments/flux_kontext_depth_gate.yaml`，需 ≥24GB CUDA 显存；Kontext-dev 为非商用许可。

**未验证**：真实 12B 权重上的训练是否收敛、闸门是否通过——本机无 GPU 与权重。这是一个**可证伪**的实验：若 Kontext 同样过不了结构闸门，则此前的负结果主要源于数据/规模而非条件范式。

### D.2 `normalize_task_prompt` 的开关

默认行为不变（仍强制 Cityscapes 图例，`test_semantic_dataset_uses_complete_cityscapes_legend` 把这个约定固化了）。新增显式开关 `keep_custom`（`VisionTaskDataset` / `V3FlatDataset` 的 `keep_semantic_prompts`），仅对非空的语义 prompt 生效，供开放词表数据在旧路径上使用；统一训练器路径本来就不改写 prompt。

### D.3 DIODE 适配器此前读不了官方数据

旧 `DIODEDataset` 只扫描 `*_rgb.png`，而官方文件名是 `<stem>.png` + `<stem>_depth.npy` + `<stem>_depth_mask.npy`，且深度数组形状为 `[H, W, 1]`（旧代码对其做 `unsqueeze` 后送入 `interpolate` 会报维度错误）。另外它会把室内外场景混在一起，而论文评测的是 **DIODE-Indoor**。现已：兼容官方与旧命名、正确读取 `[H,W,1]`、新增 `scene_type`（`indoors`/`outdoor`），`evaluate_paper_benchmarks.py` 对 DIODE 默认 `scene_type="indoors"`。
**此项依据官方 DIODE 的公开布局与单元测试（合成的官方布局文件），没有在真实 DIODE 数据上验证。**

## 附录 E：法向量评测与 RefCOCOg cIoU（2026-10-09）

论文 Table 7 的法向量是 NYU、DIODE-Indoor、ScanNet、VKITTI 四项。仓库原先有适配器，但没有原分辨率、有效掩码、DIODE 室内过滤，也没有和深度同一思路的闸门。指代分割的 cIoU 指标在 `evaluation/referring.py`，评测脚本却只跑 depth / normal / semantic。

**法向量真值从哪来**（适配器不从深度估计法向量）：

| 数据集 | 官方发布 | 适配器读取 |
|---|---|---|
| NYU | H5 只有 RGB 和深度 | 社区 `normals/`（`.npy` 相机系向量，或论文编码 PNG） |
| DIODE | `<stem>_normal.npy`，可选 `*_depth_mask.npy` / `*_normal_mask.npy` | 官方文件；`scene_type="indoors"` |
| ScanNet | 网格，不是逐帧法向量 | `images/`+`normals/`，或路径里 `color`→`normal` |
| VKITTI2 | 不带法向量 | `rgb/` 旁的 `normal/` 或 `normals/` |

零向量（长度 < 1e-3）和 DIODE mask 为 0 的像素不计入。`native_gt=True` 额外返回 `gt_normal_native` / `valid_mask_native`。`VisionEvaluator` 把预测双线性放大到该分辨率后再归一化。`NormalAccumulator` 同时给出像素加权均值和逐图均值（`mean_angle_deg_per_image`）。

**预注册闸门**（`evaluation/gates.py::evaluate_normal_gate`，实验前固定，角度越低越好）：

1. 度量闸门：平均角误差 ≤ 最佳非 oracle 平凡基线 − 5°。非 oracle 是朝相机 `(0,0,1)`、朝上 `(0,1,0)`，以及可选的训练集平均方向。
2. 结构闸门：平均角误差 < `oracle_image_mean`（每张图一个最佳常数方向）。通过说明预测带有空间上的法向量场。

```bash
python scripts/normal_trivial_baselines.py \
  --dataset-key diode_normal --root /data/diode \
  --dataset-kwargs '{"scene_type": "indoors"}' --native-resolution \
  --output-json output/eval/diode_normal_trivial.json

python scripts/evaluate_paper_benchmarks.py \
  --benchmark normal --normal-root /data/benchmarks --native-resolution \
  --model-name <base> --lora-path <ckpt> \
  --output-json output/eval/normals.json

python scripts/check_normal_gate.py \
  --model-json output/eval/normals.json --benchmark normal/diode \
  --baseline-json output/eval/diode_normal_trivial.json
```

`evaluate_paper_benchmarks.py` 对 `diode_normal` 默认 `scene_type="indoors"`。

**RefCOCOg**：`--benchmark referring --refcocog-root`。优先 `refs(umd).p`（论文 UMD val），没有则用 `refs(g).p`；带 `split` 字段的样本按 `--split` 过滤（默认 `val`）。每条句子单独计分。掩码来自 `instances.json` 的多边形或 RLE，画布尺寸用 COCO `images` 的高宽或 RLE `size`，**不用 bbox 填成区域**。没有 segmentation 的标注不进入样本。`native_gt` 返回 `gt_mask_native`，预测掩码最近邻放大后算数据集级 cIoU。

**未验证**：本机没有 NYU / DIODE / ScanNet / VKITTI / RefCOCOg 的真实文件，上述行为由合成布局的单元测试覆盖。社区法向量的坐标约定（尤其 VKITTI、ScanNet 的 y 轴朝向）必须已经是论文的「+x 右、+y 上、+z 朝相机」；适配器不会猜测并翻转。真实数据上的平均角误差和 cIoU 都还没有跑出来。
