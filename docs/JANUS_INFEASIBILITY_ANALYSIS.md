# 为什么 Janus 无法用于验证 open-vision-banana —— 深度分析

> 项目：open-vision-banana（论文 "Image Generators are Generalist Vision Learners" 复现）
> 分析日期：2026-08-02 ｜ 证据来源：scripts/janus_*.py（19 个实验目录）、output/janus_*/ 全部 metrics、train.log、git log

## TL;DR

**Janus 路线不是"验证失败"，而是"从实验设计上就不构成对论文命题的验证"。** 三代架构迭代（VQ token 原生生成 → continuous RGB head → SigLIP 回归 probe）全部实测失败后，v7 为了出结果不得不彻底绕开生成路径——此时实验验证的命题已从"图像生成器是通用视觉学习器"退化为"SigLIP 特征含少量几何信息"，与论文命题无关。根因分五层：①预训练目标不匹配（离散 AR ≠ 连续生成）；②理解/生成解耦架构使"生成器理解"这一待验命题从未被触及；③VQ tokenizer 的 16× 降采样 + 16384 码本量化在信号层面摧毁了 colormap 的逐像素精度；④1.3B + MPS + 500 样本的规模与论文涌现机制的尺度前提相差数个数量级；⑤实测指标（held-out δ1=0.11~0.21 vs 论文 0.88）在数量级上确认此路不通。

---

## 一、实测证据：三代路线全部失败

### 路线 A：Janus 原生 VQ token 生成路径（janus_vision_banana_full）

做法：目标 RGB → VQ encode → 576 tokens，teacher forcing + CrossEntropy，最"忠实"于统一架构的路线。

| 指标 | 实测（held-out） | 论文目标 | 判定 |
|---|---|---|---|
| δ1 | **0.215** | 0.882 | 🔴 差 4 倍 |
| AbsRel | **6.99** | ~0.10 | 🔴 数值无意义（尺度完全错乱） |
| rgb_mae | 0.442 | — | 🔴 |

来源：`output/janus_depth_complete/complete_result.json`

### 路线 B：continuous RGB head（janus_continuous_rgb v1–v6）

做法：替换 gen_head 为 RGB 回归头（LLM hidden → RGB 值，MSE loss），绕开 VQ。

| 指标 | 实测（held-out） | 判定 |
|---|---|---|
| δ1 | **0.110**（比路线 A 更差） | 🔴 |
| AbsRel | 3.33 | 🔴 |
| rgb_mae | 0.354 | 🔴 |

来源：`output/janus_depth_complete_continuous/complete_result.json`。git 提交记录自我标注：`e6652ba` "held-out benchmark（仍过拟合）"、`3b9e2ba` "depth-only 完整复现（held-out 失败）"。

### 路线 C：v7 回归 probe（janus_continuous_rgb_v7）

v3~v6 失败后，v7 脚本头注释自述：**"直接预测 RGB 效果差，因为 SigLIP 提取语义特征而非几何特征"**，于是彻底放弃生成：SigLIP 特征 → CNN depth decoder → 确定性 colormap 后处理。

```
v7 诊断摘要（output/janus_continuous_rgb_v7/train.log）：
avg rgb_mae = 0.371 ｜ avg psnr = 7.4 dB
avg pred_std = 0.4475  vs  avg tgt_std = 0.1785   ← 输出塌缩为过度平滑的均值预测
```

🔴 **关键定性**：路线 C 已经不是"图像生成"，而是标准的冻结编码器 + 回归探针（等价于 DINOv2 + DPT probe）。即使它成功，验证的也是"SigLIP 特征含几何信息"，**与论文命题"生成预训练涌现理解"零相关**——而它连这个也失败了。

### 对照：同数据上其他架构证明任务本身可学

| 模型 | depth | normal | 结论 |
|---|---|---|---|
| tiny_vb_v3（自研小模型） | Pearson **0.721** | **0.844** | ✅ 任务可学 |
| SSD-1B img2img（扩散） | Pearson 0.511 (val) | 0.364 | 🟡 扩散路线至少有信号 |
| **Janus-1.3B（全部路线）** | δ1 ≤ 0.21 | — | 🔴 全面垫底 |

结论：失败不是数据或任务的问题，是 Janus 架构与验证目标的根本错配。

---

## 二、根因分析（按层级）

### 🔴 P0-1 预训练目标错配：离散 AR token 预测 ≠ 连续图像生成

论文命题的独立变量是**"图像生成预训练"**——NBP 的生成机制是连续潜空间上的扩散/流式生成。Janus 的生成路径是**对 16384 码本的离散 next-token 分类**：

- 换独立变量 = 换实验。即使 Janus 成功，也只能发表"AR VQ 生成器是通用视觉学习器"，不能引用为 NBP 论文的复现证据；
- 离散 CE loss 对"颜色接近但码本 ID 不同"不做连续度量惩罚——而 depth colormap 解码恰恰要求逐像素连续精度。目标函数与评估指标在数学上错位。

### 🔴 P0-2 解耦架构使待验命题从未被触及

Vision Banana 范式要求**输入图像与输出可视化走同一条生成通路**（生成器即理解器）。Janus 恰好是为反对这一点而设计的：

- 输入图像 → **SigLIP 理解编码器**（带显式理解监督预训练）→ LLM；
- 生成路径（VQ encode/decode）**从不接触输入图像**。

因此在 Janus 上，"生成器是否理解"这个问题在结构上不存在——最多只能问"LLM 能否把 SigLIP 特征路由成 VQ token"。且 Janus 的理解能力可能完全来自其理解侧预训练，**混杂变量（confounder）使任何正面结果都无法归因于生成预训练**。

### 🔴 P0-3 VQ tokenizer 在信号层面摧毁 colormap 精度

- 384×384 → 24×24 = 576 token，**16× 空间降采样**：深度边界的像素级锐度（δ1 指标的核心区分点）被结构性抹除；
- Janus 的 VQ decoder 为自然图像感知质量训练，不保证度量颜色值保真——路线 A 的 AbsRel=6.99 即"colormap 经 VQ 往返后数值语义已毁"的直接证据；
- 光栅序 AR 生成无 2D 空间归纳偏置，teacher forcing 训练 vs 自由推理的 exposure bias 在 576 步上累积误差。

### 🟡 P1-4 规模与训练量级不满足涌现前提

- NBP 为闭源 SOTA（参数量与生成训练数据量推测 ≫10B/亿级）；论文的"涌现"声明是**尺度条件命题**；
- 实测条件：1.3B 模型 + LoRA(r=64) + MPS 单卡 + depth 仅 500–2,297 样本 + 2,000 步。train loss ↓0.15 而 val_loss 平台期 ~0.21——教科书式**记忆化区间**，连 LoRA 容量内拟合都未达成，谈不上检验涌现；
- 数据分布失衡：depth/normal 是 53,678 样本中最小的两片（2,297 / 1,797），最难的几何任务喂得最少。

### 🟢 P2-5 工程层面加速失败但不改变结论

多任务 benchmark 每任务仅 2 样本（仅够冒烟）；部分实验目录无 metrics.json；评估口径不统一。这些影响迭代效率，但即使全部修好，P0 三层结构问题依然成立。

---

## 三、逻辑总结：为什么"无法用于验证"

```
论文命题：生成预训练 →（指令微调，仅改 prompt/RGB 参数化）→ 通用视觉理解

Janus 实验实际检验的：
  路线 A：SigLIP理解特征 → LLM → VQ离散token分类 能否回归 colormap   ← 变量全换
  路线 B：SigLIP理解特征 → LLM → 线性回归头 能否回归 RGB             ← 无生成
  路线 C：SigLIP理解特征 → CNN probe 能否回归 depth                   ← 与命题无关

三条路线中没有一条保持"生成预训练是唯一能力来源"这一实验控制。
```

**一句话：Janus 的理解走 SigLIP、生成走 VQ-AR，两头都不满足论文范式的前提——前者污染归因，后者毁掉信号；当工程上被逼到用回归 probe 出数时，实验已经悄然换了命题。**

---

## 四、建议（验证命题的正确姿势）

1. **回到扩散主线**：FLUX.1 / Z-Image / Qwen-Image 等连续潜空间生成骨干才是命题的合法检验对象（repo 原生路径即为此设计）；SSD-1B img2img 已有 0.51 Pearson 的初步信号，值得按论文配比（9:1 生成:视觉）加大投入。
2. **若坚持统一 AR 架构**：需满足三条件才可称为"验证"——连续图像 token（非 VQ）、输入输出共享生成通路、与 understanding-only 预训练做消融对照。Janus 系（含 Pro）均不满足前两条。
3. **规模底线**：GPU 训练、骨干 ≥3B、单任务样本 ≥10k、held-out 每任务 ≥100 样本再下结论。
4. **归档 Janus 路线**：19 个 janus_* 目录建议保留 3 个代表性结果（A/B/C 各一）+ 本报告，其余移入 archive，避免 output/ 继续膨胀。

## 产物路径

- 本报告：`docs/JANUS_INFEASIBILITY_ANALYSIS.md`
- 关键证据：`output/janus_depth_complete/complete_result.json`（路线 A）、`output/janus_depth_complete_continuous/complete_result.json`（路线 B）、`output/janus_continuous_rgb_v7/train.log`（路线 C 诊断）、`output/janus_multitask_benchmark/benchmark_result.json`（多任务冒烟）
- 背景报告：`output/afb0f3c9-43f6-4986-83bd-4f2ce76f7c0a/reports/训练实验现状分析.md`
