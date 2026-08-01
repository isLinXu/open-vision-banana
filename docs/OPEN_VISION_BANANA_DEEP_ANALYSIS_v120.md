# Open-Vision-Banana 深度分析报告 v1.2.0（优化方案专项）

> 分析日期：2026-07-26 · 基于 v0.4.2 + 未提交工作区（26 文件 / +2266 行）
> 前置基线：`OPEN_VISION_BANANA_DEEP_ANALYSIS_v110.md`（同日 17:25）、`DEEP_ANALYSIS_INDEPENDENT_20260726.md`
> 本版增量：① 核验 17:25 之后新增的 **capacity ladder 实验数据**（micro 4×20 / stage3 32×100 / capacity 8×200，全部产出 split-clean 评估 JSON）；② 审查未提交的 source-conditioning 主干化改动；③ 给出按「精度 / 收敛 / 效率 / 参数效率」四维量化的优化路线图。

---

## 0. 一句话结论

**数据与评估闭环的问题已经修完，当前唯一的阻塞点是「训练配置容量」本身**：v1.1.0 报告的 P0/P1 中，split 泄漏、标签语义、标定统计口径、prompt 一致性均已在未提交代码中系统性解决；但今晚三组 capacity ladder 实验证明——**rank=8 + 128px 的配置连 8 个训练样本都无法过拟合（train Pearson r = -0.09）**，与成功 gate（rank=64、r=0.8154）形成决定性对照。下一步不是修代码，而是**把成功 gate 的配置在可复用训练路径上 1:1 复刻**。

---

## 1. 新增实证：capacity ladder 实验核验（本报告核心）

### 1.1 三组实验的完整证据链

| 实验 | 配置 | 步数 | 末步 loss | train r | val r | 判定 |
|------|------|------|-----------|---------|-------|------|
| micro_4x20_fresh | rank 8, 128px, 4 样本 | 20 | — | — | -0.06 量级（raw MAE 0.248） | 未学出 |
| stage3_32x100 | rank 8, 128px, 32 样本 | 100 | — | — | **-0.059**（calibrated -0.062） | 未学出 |
| capacity_8x200 | rank 8, 128px, 8 样本 | 200 | **0.638** | **-0.087** | **-0.119** | **过拟合失败** |
| （对照）standalone gate | **rank 64**, 4-step 推理调度, 单样本 | ~300 | — | **+0.8154** | — | 几何已学出 |

**三个关键判读：**

1. **这不是泛化问题，是容量问题。** capacity_8x200 在 **train split 上** r = -0.087——模型连自己反复看过 25 个 epoch 的 8 张图都拟合不了，泛化无从谈起。val r 与 train r 同为负值，说明 split 无关、泄漏无关。
2. **收敛远未完成。** 200 步后 loss 仍 0.638（起点约 1.08），而成功 gate 的 loss 量级更低。rank 8 的 rsLoRA（α=8，scaling=α/√r≈2.83）+ 128px 的组合，有效参数与输入信息量都不足以承载「RGB 色彩精确重绘」这一目标——注意评估显示 raw prediction curve MAE=0.16 vs target curve MAE=0.006，**模型输出的颜色曲线形状本身就是错的**，不是标定能救的。
3. **实验设计本身是对的，值得肯定。** checkpoint-200 的 `conditioning_config.json` 完整记录了 mode/colormap/标定统计/prompt/推理步数（version=1），评估 JSON 同时给出 raw 与 calibrated 两套指标、16 样本逐样本明细——**可复现实验协议已经达标**，缺的是配置扫描的维度（rank、分辨率）。

### 1.2 成功 gate 与可复用路径的差异矩阵（待逐项对齐）

| 维度 | 成功 standalone gate | 当前可复用 CLI 路径 | 差异风险 |
|------|---------------------|--------------------|---------|
| LoRA rank / α | 64 / 128 | **8 / 8** | ⬛ 极高（首要嫌疑） |
| 分辨率 | 512px 量级 | **128px** | ⬛ 极高（深度色图细节全丢） |
| 时间步采样 | 4-step 推理调度 | 已对齐（`timestep_sampling=inference` ✓） | ✅ 已修 |
| x0 辅助损失 | — | 已加 `x0_loss_weight=1.0` ✓ | ✅ 已加 |
| pixel L1 | — | 已加但默认关闭（MPS 上不稳定） | ⚠️ 可选 |
| 标定 | 16 样本前缀（不具代表性） | 全量 2162 train 统计 ✓ | ✅ 已修 |
| 数据 | viridis 检测 + 匹配 prompt | 已对齐 ✓ | ✅ 已修 |

**结论：v3 数据管线的问题全部清零后，剩下的唯一显著差异就是 rank 与分辨率。这是一个干净的、单变量可验证的假设。**

---

## 2. 未提交代码改动审查（26 文件，+2266/-507 行）

### 2.1 改动质量评估

**主干化方向完全正确。** v1.1.0 时 source-conditioning 还是 standalone gate 脚本里的「特设路径」，本次改动把它提升为 `InstructionTuner` 的一等能力：

- **新增 `conditioning.py`（311 行）**：`SourceConditioningConfig` / `expand_unet_conv_in` / `sample_source_conditioned_timesteps` / `set_scheduler_prediction_type`——抽象干净，单一职责，是本次改动中最好的部分。
- **`tuner.py` +306 行**：source conditioning、x0 loss、pixel loss、逐样本 prompt（`prompts/{stem}.txt` 覆盖默认 prompt）、输出标定全部参数化，且全部带输入校验（负权重、非法采样模式、缺统计量时直接 raise）——防御性到位。
- **checkpoint 元数据 v1**：`conditioning_config.json` 随 checkpoint 落盘，fresh-process reload 像素级一致已验证（stage2 设计文档记录）。
- **`datasets.py` +591 行 / `generator.py` +266 行 / `onnx_export.py` +264 行**：规模偏大，需复查（见 P1）。

### 2.2 必须立即处理的事项

| # | 事项 | 级别 |
|---|------|------|
| 1 | **这批改动未提交**。26 个脏文件横跨 core/data/training/inference/export/tests，一旦回滚或冲突，今晚的实验基础设施全部丢失。应立即提交为 `v0.5.0-rc`（source-conditioning 主干化） | 🔴 P0 |
| 2 | `tuner.py` 已涨到 **2183 行**，`datasets.py` **2711 行**，加上 `model_registry.py` 2221 行——三个 2000+ 行单文件已成结构性债务 | 🟡 P1 |
| 3 | capacity ladder 的实验脚本/配置未见入库（`output/` 内有结果但无对应 configs 或 scripts 变更），实验不可复现于他人机器 | 🟡 P1 |

---

## 3. 问题清单更新（相对 v1.1.0 的增量）

### 🔴 P0 — 阻塞性

| # | 问题 | 证据 | 优化方案 |
|---|------|------|---------|
| N1 | **训练容量不足，连过拟合都未达成** | capacity_8x200：train r=-0.087，loss 0.638@200 步 | 见 §4 Phase 1 的「gate 复刻实验」 |
| N2 | **source-conditioning 主干化改动未提交** | `git status` 26 个脏文件 | 立即 commit + tag v0.5.0-rc |
| 3 | （沿用）基座替换未实证、无已发布权重、作者引用矛盾 | v1.1.0 §四 | 不变 |

### 🟡 P1 — 显著

| # | 问题 | 优化方案 |
|---|------|---------|
| N4 | ~~LoRA 配置默认 rank=8~~ **复查更正（07-27）**：框架默认 rank 实为 64（tuner）/ 128（train_sdxl_mps.py），rank=8 是 capacity ladder 实验的**故意下探参数**，非默认值缺陷 | 无需修改默认值；E1 实验配置已入库 `configs/experiments/e1_gate_replica.yaml` |
| N5 | **128px 训练分辨率对 RGB 色彩映射任务过低**：深度色图在 128px 下相邻色的空间梯度被 VAE 下采样抹平 | source-conditioning 模式下默认 256px 起步，显存允许时 512px |
| 6 | （沿用）MPS 74 s/step、model_registry 拆分、Gemini filtering、ONNX 兜底 | 不变 |
| N7 | tuner/datasets/registry 三个 2000+ 行单文件 | 按 family / 按职责拆分（见 §4 Phase 3） |

### 🟢 P2 — 优化项

| # | 问题 |
|---|------|
| N8 | pixel L1 在 MPS 上「可行但不稳定」（1 步改善、8 步回退）——值得在 H20 上重新评估，MPS 的 bf16 原子性可能是元凶 |
| N9 | 评估指标里 raw_prediction_curve_mae (0.16) 与 target_curve_mae (0.006) 相差 25×，建议把「曲线形状误差」提升为一等指标（它比 Pearson r 更早暴露色彩系统性错误） |
| 10-13 | （沿用）speculative 任务基准、README 声明位置、INT8 量化评估、weights_only 统一 |

---

## 4. 优化路线图（按四维量化目标组织）

### Phase 1（本周）— Gate 复刻实验：单变量验证容量假设

> 目标：**在可复用 CLI 路径上复现 r>0.8 的过拟合 gate**，把「范式能否在 SSD-1B 上学出深度色图」变成已回答的问题。

| 实验 | 配置 | 成功判据 | 预计耗时（MPS） |
|------|------|---------|----------------|
| E1 gate 复刻 | rank=64, α=128, 512px, 8 样本, 4-step 推理调度采样, x0_loss=1.0 | train r ≥ 0.8，RGB MAE ≤ 0.14 | ~10h（512px 更慢） |
| E2 rank 扫描 | rank ∈ {16, 32, 64} @ 256px, 8 样本 200 步 | 找到 r 由负转正的 rank 阈值 | 3 × 3h |
| E3 分辨率扫描 | rank=64 @ {128, 256, 512}px | 分辨率对 curve MAE 的边际收益曲线 | 3 × 4h |

**工程要求**：三个实验的配置 YAML 全部入库 `configs/experiments/`，结果 JSON 追加到 `experiments/` 报告——延续本仓库的审计文化。

### Phase 2（两周）— 规模化与对标

1. **迁移 H20/A100**（配置已备 `training_h20_*.yaml`）：MPS 74 s/step → 目标 ≥10×。Phase 1 的 E1 在 H20 上约 1 小时，扫描实验当天可完成。
2. **split-clean 泛化验证**：E1 配置 × 2162 train 样本，在 135 官方 val 上给出第一个真实泛化数字（工具已就绪）。
3. **pixel L1 在 CUDA 上重估**（P2-N8）：若稳定则设为 source-conditioning 默认——它直接监督「目标色彩」，正是当前 curve MAE 0.16 的解药。
4. **论文对标路径决定**：v3 数据只有 viridis 相对深度，**无法**产出论文 AbsRel/δ1。要么引入 Hypersim/NYU 带 metric 真值的数据（`hypersim_depth_data_report.md` 已有调研），要么明确项目主张为「相对深度范式验证」。建议两者都做：相对深度先出成果，metric depth 作为论文对标的第二阶段。

### Phase 3（月度）— 结构与生产

1. **拆分三个 2000+ 行单文件**：
   - `model_registry.py` → `registry/{sd,flux,dit,sana,zimage}.py` + 公共 `ModelConfig`
   - `tuner.py` → `training/{tuner,source_conditioning_hooks,losses,dataset}.py`
   - `datasets.py` → `data/{vision_pairs,mixed,generation,transforms}.py`
2. **发布第一个权重**：E1 通过 gate 后，哪怕只是 8 样本过拟合的 depth LoRA，也发布到 HF Hub 并配 MODEL_CARD（模板已备）——让 checkpoint registry 出现第一个 `available`。
3. **统一作者/引用**（P0-3 沿用）：README 署名与 `paper_alignment_2604.20329v3.md` 的矛盾必须在发布权重前解决。

---

## 5. 四维评估更新

| 维度 | v1.1.0 | v1.2.0 | 变化依据 |
|------|--------|--------|---------|
| 学术对齐度 | A- | A- | 不变 |
| 工程完成度 | B+ | **A-** ↑ | source-conditioning 主干化质量高、checkpoint 元数据 v1、评估协议完整 |
| 实证进展 | C- ↑ | C-（持平） | 新实验证明「未学出」，但这是**有信息量的负结果**——它排除了数据/split/标定嫌疑，把问题收敛到容量单变量 |
| 文档可信度 | B | B | 不变（作者矛盾未修） |
| 生产就绪度 | C+ | C+ | 不变 |

**关键瓶颈排序更新**：
~~① 数据与 split~~（已解决）→ ~~② 标定统计口径~~（已解决）→ **③ 训练容量（rank/分辨率）——当前唯一阻塞** → ④ MPS 吞吐 → ⑤ 基座表征质量上限（科学问题，Phase 2 后回答）。

---

## 6. 结论

与 v1.1.0 相比，项目在 6 小时内完成了两件重要的事：把特设 gate 能力主干化为工程化训练路径（+2266 行，质量在线），并用三组 split-clean 实验**证伪了「当前配置可学出深度」的假设**。这是典型的良性迭代——负结果精确地指认了下一个实验。

**当前最高优先级的三件事，顺序不可调换：**
1. `git commit` 这批改动（否则一切免谈）；
2. 跑 E1 gate 复刻（rank=64 / 512px / 8 样本过拟合，配置已入库 `configs/experiments/e1_gate_replica.yaml`）——预计一晚上，通过则范式验证进入新阶段，不通过则排错空间已收敛到极小；
3. ~~提交前顺手把 source-conditioning 模式的默认 rank 从 8 提到 32+~~（07-27 复查：默认已是 64/128，rank=8 为实验故意下探，此项取消）。

---

### 6.1 后续 Stage 8 更新（2026-07-27）

在本报告原始截止时间之后，冻结判别式路径已完成多 seed、dev-selected 的
split-clean 对照：128 train-fit / 32 dev / 32 official-val，seed 42/43/44，
128px、300 steps。三 seed official-val 平均增量为 Pearson `+0.0844`、
Spearman `+0.0834`、MAE `-0.0060`，且三 seed 均同时改善两项相关性；Stage 8
gate 通过。该结果将当前实证瓶颈从「SSD 中间特征是否有泛化贡献」推进到
「更大规模、跨数据集与更高绝对质量验证」，但不改变原 RGB diffusion 路径
仍失败、relative-depth 不等于 metric depth 的边界。

Stage 8 的完整实现、split checksum、per-seed metrics 与 checkpoint 位于
`scripts/compare_frozen_depth_features.py` 和
`output/ssd1b_stage8_multiseed_128fit32dev32val_300_128/result.json`；本节作为
v1.2.0 的增补，不回写 07-26 当时的历史判断。

### 6.2 全量数据更新（2026-07-27）

全量 `vb_snvc_train_v3` 实验已完成：2,182 train（1,926 fit + 256 dev）和
115 official val，128px、4,000 steps、三 seed。VAE+SSD official-val 聚合
达到 MAE `0.1733`、Pearson `0.6527`、Spearman `0.6345`，相对 VAE-only 的
平均增量为 Pearson `+0.0812`、Spearman `+0.0835`、MAE `-0.0169`，三 seed
全部通过。完整产物位于
`output/ssd1b_full_stage8_1926fit256dev115val_4000_128/result.json`。下一步
应在协议固定后用全部 2,182 train 做一次最终部署 refit。

### 6.3 五任务 LoRA runner 闭环（2026-07-27）

基于 `vb_snvc_train_v3` 的五任务审计与短 SSD-1B smoke 已通过：depth、normal、
semantic、edge/structure、generation 均完成训练加载，5 optimizer steps 的 loss
保持有限，并保存 `lora_final.pt`。新增的 manifest-backed held-out 验证在 smoke
checkpoint 中逐任务记录：depth `0.8562`、normal `0.6454`、semantic `0.6146`、
edge `0.7027`、generation `0.6398`；checkpoint metadata 同时记录 train/val
task sizes、schedule 和审计配置。产物位于
`output/multitask_smoke_5task_5step_validation_v2/`。

这只证明五任务数据闭环、split-clean 验证与 checkpoint 审计可运行，不代表质量
提升或生产权重已经完成。随后已启动正式全量 run：
`output/multitask_lora_v3_full_256/`，256px、rank 128、最多 3,000 optimizer
steps、五任务 train split 无放回遍历、每 250 步逐任务 validation。启动阶段前 20
steps loss 为 `0.5881`，仍保持有限；最终判断须等待 checkpoint validation 与
base/frozen control 对照，且 instance/referring/keypoint/object_detection 等
任务契约仍未纳入生产范围。

### 6.4 冻结深度估计器生产验证完成（2026-07-28）

后续证据推翻了本报告主体中“继续扩大 source-conditioned diffusion 容量”作为
生产主线的建议：该路径在 8 张图、4,800 步的 raw geometry gate 中仍无法过拟合，
因此所有 SSD/LoRA 生成式训练均已停止。生产路线改为冻结 SSD-1B/VAE、仅训练
`ScalarDepthProbe` 的确定性标量深度估计器。

匹配的 VAE-only 与 VAE+SSD contribution gate 在 256px 和原生 512px 均通过；
512px gate 的 VAE+SSD 三 seed 均值为 MAE `0.1824`、Pearson `0.5786`、
Spearman `0.5782`，相对 VAE-only 分别改善 Pearson `+0.1641`、Spearman
`+0.1678`。随后按 dev-selected 样本暴露量在全部 2,181 张当前训练图上完成
512px production refit，并仅在端点确定后评估 115 张 official val：

| Seed | MAE | RMSE | Pearson | Spearman |
|---:|---:|---:|---:|---:|
| 42 | 0.13177 | 0.18823 | 0.76460 | 0.74936 |
| 43 | 0.13012 | 0.18769 | 0.77103 | 0.75886 |
| 44 | **0.12690** | **0.18335** | **0.77875** | **0.76137** |
| 三 seed 均值 | **0.12960** | **0.18642** | **0.77146** | **0.75653** |

seed 44 是当前部署候选。三个 artifact 均包含 checksum-verified
`model.safetensors`、完整 `config.json` 和四张训练期可视化；另以独立进程从磁盘
重载所有权重，对固定 12 张 official-val 子集重新提取 SSD 特征并生成每个 seed
的 12 张推理图与 contact sheet。独立子集 Pearson 为 seed 42 `0.83140`、seed 43
`0.84264`、seed 44 `0.83663`，未发现常量输出、色图反转或 checkpoint/feature
contract 不匹配。子集指标只用于重载验证，不替代上述 115 张 official-val 指标。

最终产物位于
`output/ssd1b_depth_probe_production_refit_512_full_exposure_scaled_20260728/`。
历史 2,182 train 计数包含后来被隔离的全黑样本 `depth_hyp_real_00285`；当前有效
数据契约为 2,181 train + 115 official val，split 无重叠。

*本报告主体基于 2026-07-26 23:40 前的工作区源码；§6.1 纳入 2026-07-27 Stage 8
实测更新；§6.4 记录 2026-07-28 的最终生产验证。*
