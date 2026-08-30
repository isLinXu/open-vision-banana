# Open-Vision-Banana 论文范式复现深度分析与优化方案 v1.3.0

> 分析日期：2026-08-17 · 审计基线：git HEAD `8ca08b3`（工作树干净）
> 论文：*Image Generators are Generalist Vision Learners*（arXiv:2604.20329v3，Google Research，基座 Nano Banana Pro）
> 前置文档：`paper_alignment_2604.20329v3.md`（算法对齐）、`reproducibility_assessment_2026-08-11.md`（复现性边界）、`OPEN_VISION_BANANA_DEEP_ANALYSIS_v120.md`（v1.2.0 工程分析）、`ROUTE_BOUNDARIES.md`（三路线契约）
> 本版增量：① 纳入 08-13 ~ 08-17 SANA 路线的**生成式深度 gate 通过**这一关键新证据，修订 v1.2.0「RGB diffusion 路径仍失败」的结论；② 汇总四条路线的最新实测矩阵；③ 给出以「论文口径 benchmark 闭环」为目标的优化方案。

---

## 0. 一句话结论

**范式验证已取得决定性进展，但论文口径复现尚未开始**：SANA-600M + LoRA 生成式路线在 512px / 20 步推理下达到 val 相对深度 **Pearson 0.79（n=128）**，首次追平冻结探针基线（0.771），推翻了 7 月底「生成式路线在开源基座上不可行」的判断；然而语义分割（mIoU≈0.0001）、metric depth（缺真值数据）、13 个论文 benchmark（零在位）三条战线仍全线未通。**当前唯一阻塞从「范式可行性」转移为「benchmark 数据与评估闭环」。**

---

## 1. 项目总览与四维评分

| 维度 | v1.2.0 (07-26) | v1.3.0 (08-17) | 变化依据 |
|------|:---:|:---:|---------|
| 学术对齐度（编码/解码/指标代码） | A- | **A** | 算法层 ~98% 对齐维持；实例分割默认参数偏离问题被 08-11 评估精确指认 |
| 工程完成度 | A- | **A-** | 三路线契约固化（ROUTE_BOUNDARIES），launchd 实验编排成熟 |
| 实证进展 | C- | **C+ ↑** | 生成式深度 gate 通过（Pearson 0.79 @ n=128）；structure F1 0.816 |
| 论文口径复现度 | — | **D+** | 0/13 benchmark 在位；0/8 端到端评估项跑通 |
| 生产就绪度 | C+ | C+ | 不变 |

**关键瓶颈排序更新**：
~~① 数据/split~~（已解决）→ ~~② 标定口径~~（已解决）→ ~~③ 训练容量~~（已解决：SANA rank+512px+20 步推理组合有效）→ ~~④ 生成式范式可行性~~（**08-16 已证伪失败论**）→ **⑤ benchmark 数据零在位 + metric depth 真值缺失（当前唯一 P0 集群）** → ⑥ 语义/法线任务质量 → ⑦ MPS 吞吐。

---

## 2. 论文范式对齐矩阵（代码层，继承并核验既有结论）

| 论文要素 | 项目实现 | 状态 |
|---------|---------|------|
| 深度幂变换 Eq.1（λ=-3, c=10/3） | `core/depth_mapper.py` | ✅ 逐参数对齐，roundtrip 0.34% |
| Cube-edge 色图（Fig.5 八顶点） | `core/depth_mapper.py` + 替代色图增强 | ✅ |
| 法线 RGB 编码（右手系） | `core/normal_mapper.py` | ✅（trunc 省略，无害偏差） |
| 实例分割 5 阶段聚类（τ=14, θ_size=2e-4, θ_erosion=0.1, γ=5.0） | `core/segmentation/` | ⚠️ 参数全对齐，但**默认 `grouping_mode="adjacent"` + 自适应 τ 偏离论文**，需显式 `seed` + 固定 τ |
| Prompt 体系（NL/JSON/Hex） | `inference/prompt.py` | ✅ 覆盖 + 5 个论文外扩展任务 |
| 评测指标（δ1/AbsRel、角度误差、mIoU、cgF1、cIoU/gIoU） | `evaluation/` | ✅ 代码完整，**无数据可跑** |
| img2img 生成范式 | `conditioning.py`（IP2P 风格 source conditioning） | ✅ SSD/SANA 路径；⚠️ 论文 NBP 原生多模态，开源 UNet 必须扩 conv_in，属必要架构代价（08-01 gate 已证） |
| 生成:视觉数据 9:1 混合 | `data/mixed.py` | ⚠️ 比例可实现，但生成数据来自开源 FLUX 合成（5,981 张），非基座原生分布，「保持生成能力」的前提不严格成立 |
| Gemini judge（SA-Co / ReasonSeg） | `evaluation/gemini.py` + `scripts/run_paper_judges.py` | ⚠️ 已实现未接线（需 API key + manifest） |

---

## 3. 实证进展时间线（07-02 → 08-17，本报告核心新增）

| 日期 | 路线 | 结果 | 判定 |
|------|------|------|------|
| 07-02 | SSD-1B v6 exp-b | val corr 0.321 | 方法正确性初证 |
| 07-26 | SSD-1B capacity ladder | rank8/128px 连 8 样本都无法过拟合（train r=-0.09） | 负结果→定位容量问题 |
| 07-28 | **路线 A** 冻结探针 | official-val Pearson **0.771**（3 seed, 512px） | ✅ 生产基线确立 |
| 08-01 | paper-style gate | SSD-1B 纯 prompt 范式理论不可能（无图像输入通道） | 负结果→排除纯 prompt 路线 |
| 08-03 | FLUX.1 主线 | depth Pearson 0.0005，flow matching 推理不收敛 | ❌ FLUX 路线冻结 |
| 08-11 | 复现性审计 | 13 benchmark 零在位，端到端评估 3/8 | 工程差距清单化 |
| 08-13 | SANA structure/semantic | structure edge **F1=0.816**；semantic mIoU≈**0.0001** | structure ✅ / semantic ❌ |
| 08-16 | **SANA depth full refine** | ckpt-1000_best：val n=128，512px，20 步推理，**Pearson 0.7907 / Spearman 0.777 / 相对位置 MAE 0.1444**；final-6000 同水平（0.789） | ✅ **生成式路线首次追平探针基线** |
| 08-17 | SANA depth-edge reweight 系列 | n=32 复测 Pearson 0.793（两个独立配置交叉验证一致） | ✅ 可复现 |
| 08-17 | `sana_full_multitask_v01`（进行中） | depth 5813 + normal 4740 + semantic 9215 + structure 4424 + gen 1968 = 26,160 样本，10K steps，step 50 loss=0.1901 | ⏳ 首个论文式多任务联合训练 |

**关键科学判断**：SANA（DiT，591.7M）+ LoRA（10.1M 可训练参数，1.68%）+ img2img flow 的成功，与 FLUX（12B）的失败形成对照——**瓶颈不是基座规模而是「VAE/调度器/推理步数」组合的可训练性**。这直接支持论文的核心论点（生成预训练 → 视觉理解迁移）在小基座上的定性成立。

---

## 4. 当前实测 vs 论文（诚实对照）

| 任务 | 论文 Vision Banana | 本项目当前最佳 | 口径差异 |
|------|-------------------|--------------|---------|
| 深度（相对排序） | δ1=0.882（6 集，metric） | Pearson 0.79 / Spearman 0.78（内部 val，**相对深度**） | ⚠️ 指标不可直接比；缺 metric 真值 |
| 深度（metric） | NYU δ1=0.948 | 无（v5 数据无 metric scale） | ❌ 数据缺失 |
| 法线 | NYU mean 15.5° | 无效（v5 normal targets 已塌陷，仅审计口径） | ❌ 目标数据需重建 |
| 语义分割 | Cityscapes mIoU 69.9 | 内部 val mIoU ≈ 0.0001 | ❌ 任务未学出 |
| 结构/边缘 | （论文外扩展） | edge F1-ODS 0.816 | ✅ 超出论文范围的正结果 |
| 实例分割 | SA-Co cgF1 47.5 | 未跑（数据集 + Gemini 未接线） | ❌ |
| 生成保持 | GenAI-Bench 胜率 53.5% | 仅 proxy 指标 | ⚠️ 人工评测不可复现 |

---

## 5. 问题清单（P0 / P1 / P2）

### 🔴 P0 — 阻塞「论文口径复现」声明

| # | 问题 | 证据 | 方案 |
|---|------|------|------|
| P0-1 | **13 个论文 benchmark 零在位**，`evaluate_paper_benchmarks.py` 仅覆盖 depth/normal/semantic 3/8 项 | 08-11 审计 §4/§2 | 下载 NYU-v2 / DIODE-Indoor / Cityscapes 三个最轻量集先行；扩展评估脚本至 referring/instance/generation |
| P0-2 | **metric depth 真值缺失**：v5 训练数据为 turbo 相对深度，无法产出 AbsRel/δ1 | metrics.json notes 自述 | Hypersim metric 深度 pipeline（`hypersim_depth_data_report.md` 已有调研），用 Eq.1 幂变换编码后重训 SANA depth LoRA |
| P0-3 | **语义分割未学出**：mIoU≈0.0001，117 类调色板下模型输出与目标颜色无对应关系 | 08-13 unified eval | 降级 palette 难度（先 19 类 Cityscapes 固定调色板对齐论文评估协议）+ 提升 seg 任务在多任务混合中的权重；复用 depth 已验证的 edge-reweight 损失 |
| P0-4 | **v5 normal targets 塌陷**：法线任务无有效训练目标 | 08-13 eval notes | 重建 normal 目标（Hypersim GT 法线 → `normal_mapper` 编码），替换 v5 数据 |
| P0-5 | **实例分割解码默认值偏离论文**：默认 adjacent 分组 + 自适应 τ | 08-11 审计 §2 | 复现配置中显式 `grouping_mode="seed", adaptive_color_tolerance=False`；或改默认值并加回归测试 |

### 🟡 P1 — 显著

| # | 问题 | 方案 |
|---|------|------|
| P1-1 | 多任务联合训练效果未知（10K steps run 进行中） | 完成后逐任务 val 评估 + 与单任务 specialist 对照，回答「联合训练是否稀释单任务质量」 |
| P1-2 | Gemini judge 未接线（SA-Co presence filter / ReasonSeg 改写） | 优先落地 Qwen2.5-VL 替代方案（roadmap P3-1 已有设计），避免外部 API 依赖 |
| P1-3 | 生成保持只有 proxy；生成数据非基座原生分布 | 用 SANA 自身重生成 5K 样本替换 FLUX 合成数据，闭合「基座原生分布」假设 |
| P1-4 | `tuner.py` / `datasets.py` / `model_registry.py` 三个 2000+ 行单文件 | Phase 3 按 family/职责拆分（v1.2.0 §4 方案沿用） |
| P1-5 | 推理步数与分辨率消融缺失：20 步是否必要？4 步 schnell 式推理掉多少点？ | steps ∈ {4, 8, 20, 50} × res ∈ {256, 512} 网格，产出效率-质量曲线 |

### 🟢 P2 — 优化项

| # | 问题 |
|---|------|
| P2-1 | MPS 吞吐瓶颈（eval 分 chunk 起新进程，n=128 需 ~45 分钟）；H20/A100 迁移配置已备未执行 |
| P2-2 | pixel L1 在 MPS bf16 不稳定，需在 CUDA 重估后决定是否设为默认 |
| P2-3 | 首个 LoRA 权重发布（HF Hub + MODEL_CARD）——SANA depth ckpt-1000_best 已具备发布条件 |
| P2-4 | 作者署名/README 引用矛盾（v1.2.0 遗留） |
| P2-5 | 本次会话未能运行测试套件（工具审批超时）；最近已知全绿为 06-17 的 140/140，tests/ 现 72 个文件，建议下次会话先跑 `pytest tests/ -q` 回归 |

---

## 6. 跨路线交互风险

- **口径混杂风险**：路线 A（探针，MAE/Pearson）、路线 B/C（生成式，RGB MAE + 相对位置 Pearson）、论文（δ1/AbsRel）三套指标并存，报告引用时必须携带口径标签，已在 ROUTE_BOUNDARIES 固化，需在所有对外文档执行。
- **多任务稀释风险**：FLUX 时代已观察到 11 任务稀释 LoRA 容量（AbsRel 2404）；SANA 多任务 run 若复现该现象，应回到「单任务 specialist + 任务路由」而非统一 LoRA。
- **数据契约漂移**：v5 normal targets 塌陷说明目标生成 pipeline 缺乏回归测试；每次数据重生成必须跑 roundtrip + 分布统计 gate。
- **负结果遗忘风险**：SSD 纯 prompt 不可能、FLUX 推理不收敛、rank8/128px 容量不足——三个负结果均已文档化，新实验设计应先查此清单避免重复消耗。

---

## 7. 优化路线图

### Phase 1（本周）— Benchmark 闭环最小集
1. **等待并评估 `sana_full_multitask_v01`**（进行中）：完成后跑 depth/normal/semantic/structure 四任务 val n=128 统一评估，与单任务 specialist 对照。
2. **下载 NYU-v2 + DIODE-Indoor + Cityscapes val**（三集约 30GB，覆盖 depth/normal/semantic 三大任务的最轻组合），修复 `SACoInstanceDataset` 为真实格式。
3. **P0-5 修复**：复现配置显式对齐论文实例分割参数 + 加默认值回归测试。
4. **跑首次论文口径评估**：SANA depth ckpt-1000_best → NYU-v2 δ1（即便是相对深度映射 + 标定，也给出一个可引用的零样本数字）。

### Phase 2（两周）— 数据与任务质量
1. **Hypersim metric depth pipeline**：下载 → Eq.1 编码 → SANA depth LoRA 重训 → NYU/ETH3D/DIODE 评估，目标 4 集平均 δ1 > 0.80（开源基座现实预期，论文 0.929）。
2. **normal targets 重建 + 重训**；**语义分割降级方案**（19 类固定调色板）验证 mIoU 能否脱离 0.0001 量级。
3. **推理步数 × 分辨率消融**（P1-5），确定生产推理配置。
4. **Gemini → Qwen2.5-VL judge 替代**落地。

### Phase 3（月度）— 规模化与发布
1. 迁移 H20/A100（配置已备），数据扩至 50K+，训练 10K-50K steps。
2. 发布首个 HF 权重（SANA depth specialist）+ MODEL_CARD。
3. 拆分三个 2000+ 行单文件；补 v5 数据 pipeline 回归测试。
4. 撰写论文对齐评估报告 v2.0（含与 Table 1 的逐项对照与差距归因）。

---

## 8. 结论与行动项

项目处于**方法论验证已立、论文口径复现未启**的拐点。8 月的 SANA 突破把问题从「能不能」变成「跑不跑 benchmark」。顺序不可调换的三件事：

1. **多任务 run 评估**（等待 `sana_full_multitask_v01` 完成 → 统一 val n=128 评估）；
2. **三 benchmark 数据在位 + 首次论文口径评估**（P0-1，这是从「内部指标」走向「可对外声明」的唯一路径）；
3. **metric depth 数据 pipeline**（P0-2，决定 δ1/AbsRel 能否出现在报告里）。

> 备注：本报告基于文档审计 + 实验产物 JSON 核验；测试套件因工具审批超时未能本次运行，列为 P2-5 跟进项。

---

## 9. 增补（2026-08-21）：多任务稀释负结果 + P0-5 修复

### 9.1 多任务稀释风险已实证（P1-1 部分回答）

`sana_full_multitask_v01_lr5e5_20260817` 在 step 2100 中断（08-18 09:07），best checkpoint = `checkpoint-2000_best`（val_loss=0.1165）。其四任务 val n=8 评估（`eval_checkpoint-2000_best_val_n8/metrics.json`）与单任务 specialist 对照：

| 任务 | 单任务 specialist（n=32/128） | 多任务 2000_best（n=8） | 判定 |
|------|---------------------------:|----------------------:|------|
| depth（Pearson r） | **0.79** | **0.19** | ❌ 灾难性稀释 |
| normal（mean angle） | targets 塌陷，仅审计 | 90.1°（≈随机） | ❌（目标数据问题未解） |
| semantic（mIoU） | 0.0001 | 2e-6 | ❌ 两路线均未学出 |
| structure（edge F1-ODS） | **0.816** | **0.737** | ⚠️ 显著退化 |

**结论**：§6 预警的「多任务稀释」风险在 SANA 路线上复现——统一 LoRA（10.1M 参数）在 4 任务混合下无法保住单任务质量，印证 FLUX 时代的同类观察。**路线建议修订为「单任务 specialist + 任务路由」作为生产主线，多任务联合仅作为论文范式验证的学术对照**；若坚持统一模型，需提升 LoRA rank（64→256+）并配合任务加权采样。n=8 样本量偏小，建议 GPU 空闲后补 n=128 统一评估坐实结论。注：另一会话正在运行 `sana_multitask_v03_no_normal_weighted_20260821`（任务加权、去 normal、500 步），已在验证同一假设。

### 9.2 P0-5 已修复：实例分割解码默认值对齐论文

**改动**：6 个定义面、10 处默认值由 `grouping_mode="adjacent"` + `adaptive_color_tolerance=True` 翻转为论文 Appendix A 协议（`"seed"` + 固定 τ=14）：

- `core/segmentation/decoder.py` — `InstanceDecodingConfig` 与 `SegmentationDecoder.__init__`
- `config.py` — `SegmentationConfig`
- `core/rgb_mapper.py` — `RGBMapper.__init__`
- `inference/pipeline.py` — `VisionBanana.__init__` / `from_pretrained`
- `inference/decoder.py` — `VisualizationDecoder.__init__`
- `evaluation/evaluator.py` 的裸 `SegmentationDecoder()` 调用随之自动对齐

adjacent/adaptive 保留为显式可选（消融用途）；`PAPER_INSTANCE_DECODING_CONFIG` 与 `paper_reproduction()` 语义不变。

**回归测试**：新增 `tests/test_paper_default_alignment.py`（6 项断言），锁定所有公开构造面的默认值恒等于论文协议。

**验证状态**：⚠️ 改动已通过 Grep 全量静态核验（10/10 处一致）+ 受影响测试静态审查（`test_optimizations_v021` / `test_p0_fixes` / `test_modules` / `test_segmentation_v040` 均用「背景+大色块」合成图或直接调用私有方法，seed+固定 τ 行为等价，破坏风险低），但 pytest 执行因工具审批超时未能运行（08-21 深夜 7 次尝试均 expired）——下次会话需先跑 `pytest tests/test_paper_default_alignment.py tests/test_seed_based_segmentation.py tests/test_segmentation_v040.py tests/test_modules.py tests/test_p0_fixes.py tests/test_data_config.py -q`，随后全量回归。改动未提交，建议测试通过后 commit。

### 9.3 增补（2026-08-22）：多任务 v02/v03 对照 + P0-1/P0-2 数据进展

**多任务稀释三线对照（均 val n=8）**：

| 实验 | 配置 | depth Pearson | structure F1 | semantic mIoU | 判定 |
|------|------|---:|---:|---:|------|
| v01 2000_best | 4 任务均匀 | 0.19 | 0.737 | 2e-6 | ❌ |
| v02 2000_best | visualcore，lr=1e-5 | **-0.04** | 0.779 | 6e-4 | ❌ 更差 |
| v03 step-500 | 去 normal + 任务加权 | 0.37 | 0.659 | 1e-3 | ⚠️ 改善但远不及 specialist 0.79 |

结论强化：三个多任务变体均无法接近单任务 specialist 水平，**「specialist + 任务路由」应正式确立为生产主线**（需 n=128 复测坐实，待 GPU 空闲）。

**P0-1/P0-2 数据进展（并行会话产出，已核验 manifest）**：

- ✅ Cityscapes 在位：`data/benchmarks/cityscapes/`（426 对：376 train + 50 val，19 类官方调色板），同步产出 `data/cityscapes_19class_512/` 训练对（P0-3 降级方案数据基础）
- ✅ Hypersim metric depth pipeline 打通：`data/hypersim_depth_512/`（200 对，2 scenes，metric_depth/*.npy float32，cube_edge λ=-3/c=10/3 编码，roundtrip 误差 0.33%）——规模尚小（200 对），足够 smoke，不足训练
- ⏳ `sana_cityscapes19_specialist_lr1e5_20260822` 训练中（step 150/2000，~29 min/50 步，预计 08-22 晚间完成）——P0-3 的首次实证检验
- ❌ NYU-v2 / DIODE 仍未在位（P0-1 剩余项）

---

*报告路径：`docs/OPEN_VISION_BANANA_DEEP_ANALYSIS_v130.md`*
*对应代码版本：git HEAD `8ca08b3` + 未提交 P0-5 默认值修复（2026-08-22）*
