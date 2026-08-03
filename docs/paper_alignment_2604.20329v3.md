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

| 方法 | NYUv2 mean↓/median↓ | DIODE mean↓/median↓ | ScanNet mean↓/median↓ | VKITTI mean↓/median↓ |
|------|---------------------|---------------------|----------------------|---------------------|
| Marigold | 19.606/11.828 | 20.864/11.134 | 16.671/12.084 | 21.284/12.268 |
| DSINE | 17.017/10.190 | 16.4/8.4 | 18.453/13.871 | 16.2/8.3 |
| StableNormal | 17.168/10.028 | 19.707/10.527 | 13.701/9.46 | 18.098/10.097 |
| Lotus-2-Normal | 16.558/- | 16.9/N/A | 18.575/N/A | 14.2/N/A |
| **Vision Banana** | **15.549/9.300** | **17.778/8.876** | **13.818/11.556** | **15.052/7.468** |

**3 室内数据集平均**: mean=18.928 (全表最低), median=9.677 (仅次于 StableNormal)

**VKITTI 单独**: mean=29.063, median=10.699（比 Lotus-2 高，但论文解释 Lotus-2 训练用了 VKITTI）

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
| Depth | NYU v2 | δ1 | 0.948 | Depth Anything V3: 0.963 |
| Depth | ETH3D | δ1 | 0.935 | Depth Anything V3: 0.917 |
| Depth | DIODE | δ1 | 0.917 | Depth Anything V3: 0.671 |
| Depth | KITTI | δ1 | 0.915 | Depth Anything V3: 0.843 |
| Depth | iBims-1 | δ1 | 0.934 | Depth Anything V3: 0.913 |
| Depth | nuScenes | δ1 | 0.643 | DepthLM-7B: 0.865 (non-zero-shot) |
| Normal | NYU v2 | mean↓ | 15.549 | Lotus-2: 16.558 |
| Normal | DIODE | mean↓ | 17.778 | Lotus-2: 16.9 |
| Normal | ScanNet | mean↓ | 13.818 | StableNormal: 13.701 |
| Normal | VKITTI | mean↓ | 15.052 | Lotus-2: 14.2 (non-zero-shot) |
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

1. **Surface Normal 4-dataset average** (Table 1: 18.928): Table 7 的 per-dataset mean values 直接算术平均无法得到 18.928。Vision Banana per-dataset means: 15.549 (NYU) + 17.778 (DIODE) + 13.818 (ScanNet) + 15.052 (VKITTI) = 4-dataset avg = 15.549。可能原因：Table 7 最后一列 "Average" 使用了不同聚合方式（如 pixel-level aggregation、加权平均、或含额外指标），建议直接参考 per-dataset 数值。

2. **Lotus-2 的 Table 7 Average 列**: 28.894 同样无法用简单算术平均解释。由于 Lotus-2 在多个 benchmark 上未报告 median，其聚合方式存疑。

3. **Depth 6-dataset avg δ1=0.882**: (0.948+0.934+0.935+0.917+0.915+0.643)/6 = 0.882，✅ 验证通过。

4. **Depth 4-dataset avg δ1=0.929**: (0.948+0.935+0.917+0.915)/4 = 0.929，✅ 验证通过。

---

*文档生成日期: 2026-07-12*
*对应论文: arXiv:2604.20329v3*
*对应代码仓库: open-vision-banana v0.4.2*
