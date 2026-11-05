# OVB Qwen-Image-2.1 实验复盘（2026-10）

> 范围：E1（30k）、E2/E2b（10k）系列 run 的 benchmark 未达标问题。
> 状态：根因已定位到"容量 × 曝光"，辅助因素已逐项实证或排除。
> 本文档是分析结论 + 排查方法 + 后续训练计划的工作手册，不是论文复现声明。

---

## 0. 导读（先看这里）

### TL;DR · 30 秒版

- **做了什么**：用 Qwen-Image-2.1 + LoRA 复现"图像生成模型学会做视觉任务"（Vision Banana）范式，训练了 E1（30k 步）和 E2/E2b（10k 步）多组实验
- **结果**：范式跑通但数值远未达标 —— 复合分约为官方论文值的 1/3；分割/引用分割部分达标，**深度指标直接崩了**（比"全图输出 3 米"的常数基线还差）
- **根因（已实证）**：不是数据错、不是评测错 —— 是 **LoRA 容量太小（r32）× 实际训练量只有计划的 1/8**（配置里 grad_accum 实为 4 而非 32），模型还没学会"绝对深度尺度"就被 2D 任务挤占了容量
- **下一步**：先零成本读已有的消融实验结果（rank/批量/单任务对照），再按结论定主跑配方（详见 §4）

### 这个项目在做什么（背景 30 秒）

Open-Vision-Banana 是论文《Image Generators are Generalist Vision Learners》的**非官方开源复现**。论文的思路是：把深度图、法线图、分割掩码等视觉任务的答案**编码成 RGB 彩色图**（例如深度用一条特定色带，颜色越亮代表越远），然后用"看图说话"的方式微调图像生成模型 —— 模型看到一张照片，输出一张对应的彩色"答案图"，解码回来就得到深度/分割结果。好处是一个生成模型可以零样本迁移到多个视觉任务上。

本项目用国产的 Qwen-Image-2.1 作为基座（论文官方用的是 FLUX），通过 **LoRA 低秩微调**（只训练一小部分参数）在 6 类视觉任务 + 文本生成数据的混合集上训练。

### 术语速查表

| 术语 | 含义 |
|---|---|
| δ1.25 | 深度预测的核心指标：预测值与真值比值在 1.25 倍以内的像素占比，越接近 1 越好 |
| raw vs aligned | raw = 直接比对绝对深度（协议指标）；aligned = 先按中位数缩放再比对（只反映"形状对不对"，不代表"距离准不准"，仅作诊断） |
| mIoU / cIoU / gIoU | 分割类指标：预测区域与真值区域的交并比，越接近 1 越好 |
| colormap / 色板 | 把深度数值映射成 RGB 颜色的配色方案（cube_edge、plasma 等）。训练和评测必须用同一块色板，否则解不对 |
| LoRA / r32 | 低秩微调：冻结大模型主干，只训练少量旁路参数；r32 指旁路的秩，秩越小容量越小、越省显存 |
| grad_accum | 梯度累积：攒多少个样本才更新一次参数，直接决定每个 step 看到的样本量 |
| EMA / _best | 参数滑动平均，用于平滑训练噪声；本项目 checkpoint 的 "_best" 实为按验证 loss 选的末位权重 |
| oracle 解码 | 诊断手段：对每个输出图尝试多种色板解码取最优，用来分辨"模型画错了色板"还是"输出本身就是坏的" |

### 怎么读这份文档

- **只看结论**：本节 TL;DR + §1 判定列 + §4 计划
- **想知道为什么**：§2 六假设判定表 + 失效链条图 + 附录可视化面板
- **要接手训练**：§3 排查清单 + §4 配方表 + §5 流程改进

> **引用/转发注意**：本项目是**非官方复现**，不代表论文作者的实现；官方对比数字仅作量级参考，协议差异见各节说明。文中 aligned / oracle 数值均为诊断口径，不能当作论文指标对外引用。

---

## 1. 结果复盘：实测 vs 预期

| 基准 | 本次最佳 | 官方 OVB 参考 | 预注册标准 | 判定 |
|---|---|---|---|---|
| Cityscapes mIoU | 0.348 | ~0.69（单次） | ≥0.30 | ✅ 达标但差距大 |
| RefCOCOg cIoU | 0.498 | ~0.685 | ≥0.35 | ✅ 达标 |
| ReasonSeg gIoU | 0.328 | ~0.736 | — | ❌ 差距最大（-40 点） |
| NYUv2 δ1.25 raw | 0.135（E2b@10k） | — | >0.45 | ❌ 低于常数基线 0.352 |
| NYUv2 δ1.25 aligned | 0.707 | ~0.96 | — | 诊断值，非协议指标 |
| ETH3D / DIODE | 0.72 / 0.72（aligned） | — | — | raw 口径同样崩 |
| 法线 / SA-Co | 未评测 | — | — | 覆盖缺失 |

关键观测：
- **E1 15k→30k：NYU raw δ1 从 0.164 掉到 0.046，同时 RefCOCOg 0.409→0.423 在涨** —— 越训越差，且只发生在深度
- `_best` checkpoint 实际是末位 EMA（val_loss 一路下降），评测的就是漂移最严重点
- 单 seed 方差极大：同配置 ETH3D 差 77%、ReasonSeg 差 50%

> **一句话看懂上表**：模型**不是没学会看深度** —— 缩放校正后它判断"哪里近哪里远"有 ~71% 正确率；它没学会的是"具体几米"这个绝对尺度，且这个能力在继续训练中被其他任务挤掉了，导致越训越差。打个比方：学生学会了"比较两个物体谁更远"，但一直没能记住标尺的刻度；而老师（验证 loss）却因为他别的科目进步把他评为了"最佳"。

## 2. 假设原因分析：六项假设的判定

| # | 假设 | 验证手段 | 判定 |
|---|---|---|---|
| 1 | 训练数据是相对深度、评测期望 metric（数据错位） | `qwen_baseline_pairs.audit.json`：`rejected_non_metric_depth_rows=0`，47,585 行全过 cube-edge 校验 | ❌ 排除 |
| 2 | canonical prompt 不绑色板（歧义入口） | 30 样本 oracle 选板：cube_edge 23 / grayscale 4 / inferno 3 | 部分成立（~23% 漂移），非主因 |
| 3 | 输出是"换色板"的一致深度 | oracle 解码未提升（0.124 ≈ raw 0.135） | ❌ 非 cube_edge 输出是坏输出，不是"换板" |
| 4 | metric 绝对尺度未学到 | raw δ1=0.135 < 常数 0.352；aligned=0.707；AbsRel 1.62（~3 倍系统性偏差） | ✅ **直接证据** |
| 5 | 容量 × 曝光不足 | `effective_config.json`：`grad_accum=4`（计划 32）→ 每 vision 任务 ~12k 样本（计划 96k）；LoRA r32/α64 冻结主干 | ✅ **根因** |
| 6 | 训练侧 "using the X colormap" prompt 是 OOD | 双 prompt 可视化：named "cube_edge" prompt 的 oracle 选板 **6/6 均为 plasma** —— 旧代码下 cube_edge 从未被命名，该 prompt 不在训练分布 | ✅ 实证（修复必要性成立） |

**假设 6 完整配对数据**（同图同 seed，canon vs named，6/6 样本）：

| 样本 | canon δ1.25 | canon aligned | canon 选板 | named δ1.25 | named aligned | named 选板 |
|---|---|---|---|---|---|---|
| 00001 | 0.153 | 0.717 | cube_edge | 0.075 | 0.882 | plasma |
| 00002 | 0.000 | 0.997 | cube_edge | 0.348 | 0.788 | plasma |
| 00009 | 0.704 | 0.979 | cube_edge | 0.505 | 0.431 | plasma |
| 00014 | 0.000 | nan | grayscale | 0.020 | 0.972 | plasma |
| 00015 | 0.031 | 0.876 | cube_edge | 0.172 | 0.353 | plasma |
| 00016 | 0.000 | 0.589 | cube_edge | 0.000 | 0.097 | plasma |

结论：canon→cube_edge 5/6（1 张 grayscale = 23% 漂移的实例）；named→plasma 6/6 全 OOD，且 named 对齐均分（~0.49）反而比 canon（~0.83）差 —— OOD prompt 不仅不产 cube_edge，整体输出质量也劣化。

**可视化失效模式**（`~/ovb_eval/viz/panel_*.png`）：canon 输出是正确色系的 cube_edge 带、物体轮廓清晰，但**整体色相系统性偏黄**（亮度≈距离，黄=高估 → AbsRel 1.62 的视觉对应）；named 输出是近乎平坦的黄色块，无明确色板，属欠训练坏输出而非"换色板"。

### 失效链条

```
LoRA r32 共享容量 + eff_batch=4（计划 1/8 曝光）
  └→ 6 任务挤占，2D 任务信号稠密胜出
      └→ 深度最脆弱的"全局绝对尺度"绑定先被覆写
          （aligned 的局部排序稳健 → aligned 0.71 而 raw 0.14）
      └→ Cityscapes 细类图例绑定同理（6/19 类 IoU≈0，road 0.95）
      └→ ~23% 样本 palette 漂移 → off-manifold 颜色 → 解码 wrap（DIODE RMSE 38km）
  └→ val_loss 选末位 ckpt → 评测到的是漂移最严重点
```

**白话版**：模型里真正在学的参数很少（LoRA r32），6 个任务共用这点容量；同时配置失误让每个训练步实际只见到计划的 1/8 样本。两个因素叠加 → 相对容易的 2D 任务（分割、引用）把容量用满了，而"绝对深度尺度"这种最难、最脆弱的绑定最先被顶掉。现象就是：任务内部"谁近谁远"的排序一直对（aligned ~0.71），但"离相机几米"的刻度始终没绑上（raw 0.14），色板绑定也约 1/4 概率漂移。

### 为什么"越训越差"的结构解释

- raw δ1 依赖全局尺度（整条色带），是 LoRA 里最薄的一层表示 —— 容量被挤时最先覆写
- aligned δ1 只依赖像素间单调序，编码在共享低级特征里 —— 动不了
- 多彩图增强 × 欠训练：cube_edge 显式绑定样本 ~20% × 12k ≈ 2.4k，从未巩固；继续训练拉向"泛色板平均"，产出 off-manifold 混合色

## 3. 排查方案（已执行与待执行）

*本节说明：上面每个结论是怎么验证出来的（已执行的工具链），以及还有哪些便宜的检查没做完 —— 接手或复核时从这里开始。*

### 已执行的排查工具链

1. **数据审计**：`build_qwen_baseline_manifest.py` 的 strict audit（排除假设 1）
2. **oracle 多色板解码**：`--depth-oracle-colormaps cube_edge,plasma,...` —— roundtrip 误差最小者当选，直方图揭示 palette 漂移比例（排除假设 3、量化假设 2）
3. **双 prompt 配对可视化**：同图同 seed 跑 canonical vs named，面板对比 + 逐图指标（实证假设 6）
4. **配置对账**：`effective_config.json` vs `exp_setting.md` —— 抓到 grad_accum 偏差（证实假设 5）
5. **per-class IoU 分解**：Cityscapes 的失败定位为图例绑定而非分割能力（road 0.95 / 细类 0）

### 待执行的排查（按性价比排序）

| 步骤 | 目的 | 成本 |
|---|---|---|
| ✅ 已读 `ovb-fix_*` 消融结果 → 见 §4.2.1 | 判定完成：权重平均>rank>minsnr-off | ~~0~~ |
| `run_depth_colormap_sweep.sh` 对 e1@30k 全量扫描 | named/oracle 能否救回旧 ckpt | ~半天 GPU |
| E-0 无训练 base control | 区分"LoRA 学到的"和"基座自带" | ~半天 |
| 旧 ckpt 多拐点评测（15k/20k/25k/30k） | 定位漂移拐点、验证"选点放大" | 每 ckpt ~15min |
| 补 normal / SA-Co 评测 | 补齐任务覆盖 | ~1 天 |

## 4. 后续训练计划

*本节说明：分三个阶段推进 —— Phase A 几乎零成本（读已有消融结果 + 补评测），Phase B 单变量小实验锁定根因方向，Phase C 才是主训练。配方调整表和协议纪律是给执行人看的 checklist。*

### 4.1 已落地的代码修复（本轮）

- `trainer.py`：`depth_colormap_prompt()` —— 多彩图增强下所有色板（含 cube_edge）的 prompt 都显式写色板名；val split 同步
- `task_prompt_stats` 进 metadata —— semantic 等多 prompt 任务可审计；Cityscapes 评测守卫升级（ADE-150 训练拒绝静默评测）
- `evaluate_paper_benchmarks.py --depth-oracle-colormaps` —— oracle 解码为诊断字段，不污染协议指标
- `scripts/run_depth_colormap_sweep.sh` —— baseline/named/oracle 一键对比 + summary.csv

### 4.2 实验序（按依赖排）

#### 4.2.1 ovb-fix_* 消融结果已读出（2026-10-29 更新）

fix 系列仓库现已公开可读。30k final EMA 在**训练分布 val 集**上的指标（非零样本 benchmark，与 §1 不可直接比较，仅用于消融相对比较）：

| run | depth δ1 | semantic mIoU | normal 角度↓ |
|---|---|---|---|
| **best3_s42（3 ckpt 权重平均）** | **0.691** | **0.252** | **19.0** |
| no_minsnr | 0.649 | 0.218 | 20.7 |
| rank64 | 0.621 | 0.187 | 23.1 |
| main_s42（基线） | 0.605 | 0.172 | 23.9 |
| eff_batch8 | 0.591 | 0.166 | 24.9 |
| rank16 | 0.575 | 0.158 | 27.5 |
| no_rslora | 0.560 | 0.145 | 30.4 |
| ctrl_depth（单任务） | 0.602 | — | — |

- **checkpoint 权重平均（best3）是最大单项收益**：δ1 +14%、mIoU +47%、normal 角度 -4.9° —— 末段尺度漂移确实能被平均掉，且几乎零训练成本。Phase C 主跑必带
- **rank 单调有效**：16→32→64（δ1 0.575→0.605→0.621，mIoU 0.158→0.172→0.187）—— 容量假设确认但幅度温和，rank128 值得一试
- **ctrl_depth 单任务 δ1=0.602 ≈ main 0.605** —— **任务挤占不是深度失败的主因**：单任务不比多任务强，深度差是"任务本身没学好"而非"被 2D 抢走"。§2 白话版中"容量挤占"的表述应修正为：容量限制的是**整体能力上限**，而非任务间互抢
- **eff_batch8 不升反降（0.591）** —— 同 step 下增大批量无益，曝光量不是当前瓶颈；no_rslora 最差（0.560）→ rsLoRA 的 √r 缩放是必要的
- 结论排序：**权重平均 > rank64 > minsnr-off ≈ rank32基线 > eff_batch8**。Phase B 的 B1/B2 已部分被回答：下一步主跑 = rank128 + best3 + minsnr-off + 计划内 accum + 多 seed

#### 4.2.2 详细执行计划（2026-10-29 细化）

原则：**先零训练补齐证据，再按已验证配方主跑，每步有预注册成功标准**。

##### Phase A · 零训练补齐（~2 天，本地 MPS 即可）

| 步骤 | 内容 | 成功标准 | 成本 |
|---|---|---|---|
| A1 ✅ | 读 ovb-fix_* 消融结果 | 已完成 → §4.2.1 | done |
| A2 | e1@30k 色板全量扫描：`bash scripts/run_depth_colormap_sweep.sh --lora-path <ckpt> --nyu-root data/raw/nyu_depth_v2` | named/oracle δ1.25 > baseline 0.046+0.1 → prompt 绑定可部分救回 | ~半天 |
| A3 | E-0 无训练对照：e0 repo 已有 smoke evals（steps20/50），补 200 样本全任务 | 所有结果表含 base 行 | ~半天 |
| A4 | 补 normal 评测（DIODE normals）+ SA-Co（无 Gemini key 则建 COCO-val fallback） | 6/6 任务覆盖 | ~1 天 |
| A5 | e1 多拐点评测：15k/20k/25k/30k × NYU 100 样本 | 定位漂移拐点，验证"评测选点"机制 | ~1h/ckpt |
| A6 | e2b_cap10k 补 seed 43/44（各 10k） | cap 结论经方差检验 | 2×10k GPU |

##### Phase B · 主跑训练（~5-7 天 GPU）

**B0 配方冻结（0.5 天）**：基于 `qwen_image_21_paper_aligned.yaml` 新建 `fix_main_v2.yaml`：

- `lora_rank: 128, lora_alpha: 256`（rank 单调已验证，64→128 再上一档）
- `min_snr: off`（no_minsnr > 基线）
- `grad_accum: 16`（eff_batch8 无收益说明曝光非当前瓶颈，但 16 恢复大部分计划曝光且省成本；lr 1e-4 保持或微降至 7e-5）
- `vision_ratio: 0.5 + cap10k 均匀采样`（沿用 e2b 配方）
- named colormap prompt + task_prompt_stats（本轮已落地的代码修复）
- `ema 0.9999`，每 2.5k step 存 ckpt（供 best3 平均用）
- **3 seeds**（42/43/44），方差已实测 77% 不可省

**B1 训练**：3 seeds × 30k steps（成本 ≈ E1 同级）。

**B2 评测协议（与训练并行）**：每 5k 拐点自动 mini-eval（NYU 100 + RefCOCOg 200 + Cityscapes 100）；最终候选 = best3（≥15k 的最后 3 个 ckpt 权重平均）vs final EMA，benchmark 取优。

**B3 预注册成功标准**：NYU raw δ1.25 > 0.30（且对齐口径 >0.75）；Cityscapes mIoU > 0.45；RefCOCOg cIoU > 0.50；复合 paper_pct > 45%。未达标 → Phase D。

##### Phase C · 验证与发布（~2 天）

- **C1 全量 benchmark**：6 任务官方样本量 × 3 seed 取中位数
- **C2 TTS**：Cityscapes 8~20 pass 换色板投票（官方 +8.3 mIoU，零训练成本）
- **C3 结果表**：E-0 base / e2b / 新配方 三行 + paper 参考列 + inference protocol 标注（steps/guidance/seed/native scoring）
- **C4 复盘 v2**：与本文档对照更新结论

##### Phase D · 备选路径（Phase B 未达标时触发）

- **D1** 解冻 transformer 后 1/3 + LoRA 组合（论文明确指出局部微调不足）
- **D2** depth+normal 独立 LoRA 组 / `task_loss_weights` 给 depth ×2 权重
- **D3** 训练分辨率 512→768（分割边界收益，显存换精度）
- **D4** Cityscapes 立场决策：保 zero-shot（接受差距）vs 加 2,975 张 train 转 in-domain —— **需人工决策，不可默认**

##### 并行落地的流程机制（Phase B 前完成）

- **M1 config 对账**：train.py 启动时 diff effective_config vs yaml，累积/批量类参数偏差即告警（本次 8× 曝光缩水的复发保险）
- **M2 任务公平性看板**：wandb per-task loss + 样本计数，挤占在曲线上可见
- **M3 ckpt 钩子**：on_checkpoint 自动 mini-eval 并落盘 benchmark 曲线，_best 语义修正为"benchmark 最优"或改名 _latest
- **M4 产物规范**：--save-predictions + task_prompt_stats + oracle 字段 + inference protocol 全部进 metadata

> **时间线**：Phase A ~2 天 → Phase B 5~7 天 → Phase C ~2 天 ≈ 两周一轮。Phase B 的 B0-B2 依赖本轮代码修复（已落地 + ~560 测试全绿）。
  A2. e1@30k 全量色板扫描 + 多拐点评测
  A3. E-0 base control + 补 normal/SA-Co eval

Phase B（单变量验证，各 10-30k step）
  B1. rank64 + accum16（同 e2b 配方对照）
  B2. ctrl_depth 单任务（任务干扰确认）
  B3. best3 EMA 平均（末段漂移是否可平均掉）

Phase C（主跑，按 B 结果择路）
  C1. 若容量是瓶颈：rank128（或解冻后 1/3 主干）+ accum32 + 80k + 3 seed
  C2. 若干扰是瓶颈：task_loss_weights 给 depth 加权 / depth+normal 独立 LoRA 组
  C3. 全程用新 prompt 显式化 + 评测驱动选 ckpt
```

### 4.3 配方调整清单

| 项 | 旧值 | 新值 | 理由 |
|---|---|---|---|
| `grad_accum` | 4（实际，非计划） | 16~32 | 恢复计划曝光量 |
| `lora_rank` | 32 | 64/128（待 B1 判定） | 容量 |
| `depth_colormaps` prompt | 仅非 cube_edge 命名 | 全部显式命名 | 消除 OOD 入口 + eval 对齐 |
| ckpt 选择 | val_loss 末位 | mini-eval δ1.25 最优 + EMA 平均 | 止住"越训越差被固化" |
| seed | 1 | ≥3 | 方差 77% 不可信 |
| task_loss_weights | 无 | depth 加权（待 B2 判定） | 抗挤占 |
| TTS | 无 | 8~20 pass 换色板投票 | 官方 +8.3 mIoU |

### 4.4 协议纪律（不变）

- raw δ1 是协议指标如实记录；aligned 只作诊断列
- Cityscapes 保 zero-shot（不加 train 集），否则明确转 in-domain 声明
- 每 run 记录 inference protocol（steps/guidance/seed/native scoring）
- oracle/aligned 等诊断字段不入结果表

## 5. 优化流程（防复发）

*本节说明：防止同类问题再发生的流程改进 —— 核心是"启动前对账配置、训练中监控任务公平性、评测驱动选 checkpoint、协议口径不混淆"四件事。*

1. **配置对账前置**：训练启动时 diff `effective_config.json` vs 计划文档，accum/batch 类参数偏差直接报警 —— 本次 8× 曝光缩水如果能开机时发现，整个 E1 不会白跑
2. **评测驱动 checkpoint**：`on_checkpoint` 挂钩 mini-eval（30 张 NYU + 50 张 RefCOCOg），`_best` 改按 benchmark 而非 val_loss；附"每 ckpt 指标曲线"进 wandb
3. **任务公平性监控**：wandb 按 task 分 loss/样本计数看板 —— 挤占问题在曲线上可见（depth loss 平台期 + 他任务下降）
4. **样本预算表**：每 run 启动前打印"任务 × 预计样本数"（steps × accum × bs × ratio / n_tasks），低于阈值（如 <20k）告警
5. **预测图落盘**：`--save-predictions` 常驻 —— 本次所有失败模式推断都是间接的，有图即可直读
6. **E-0 control 常驻**：每张结果表第一行必须是无训练基座
7. **协议指标 + 诊断指标分列**：raw/aligned/oracle 永不混列
