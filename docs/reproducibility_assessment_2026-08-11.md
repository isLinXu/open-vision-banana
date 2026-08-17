# Open-Vision-Banana 复现性评估报告

**日期**: 2026-08-11
**审计基线**: commit `ab9d822` (main, 工作树含未提交的 v5 数据/训练脚本)
**论文**: *Image Generators are Generalist Vision Learners* (Gabeur et al., Google, 2026-06-05, 30 页 PDF 全文核对)

---

## 1. 结论（TL;DR）

**不能完整复现论文效果，且存在两类不可逾越的硬障碍**。项目复现的是"方法"而非"结果"：

1. **基座模型不可得**：论文的 Vision Banana 是对 **Nano Banana Pro**（Google 闭源图像生成模型）做 instruction-tuning。该模型权重、训练数据均未公开。本项目用 FLUX.1 / ZImage / Qwen-Image / SANA / SSD-1B 等开源模型替代，基座能力差距直接决定上限。
2. **训练数据不可得**：论文使用内部模型标注 + 渲染引擎合成数据（深度任务明确"全部合成、零真实数据"），未公开。本项目用 Hypersim/VKITTI/COCO 等公开数据做功能替代。

在上述约束下，项目自身定位（README §Training）也明确承认："The paper does not disclose its base weights, exact data mixture, LoRA rank, or optimizer settings"，因此只能做 **method reproduction（方法复现）**，数字层面的 Table 1 复现**原则上不可能**。

更关键的是工程现状：项目当前实测最好成绩与论文相差 **1 个数量级以上**（详见 §5），端到端评测链路未打通。

---

## 2. 复现矩阵

| 层级 | 论文要素 | 项目状态 | 判定 |
|------|---------|---------|------|
| 基座模型 | Nano Banana Pro（闭源） | FLUX.1-schnell / SSD-1B / SANA / MAR 等开源替代 | ❌ 不可得（硬障碍） |
| 训练数据 | 内部标注 + 渲染合成，混合比例未公开 | Hypersim/VKITTI 等公开替代，9:1 自选比例 | ⚠️ 替代实现 |
| 训练配方 | 未公开（rank/LR/优化器均无） | rsLoRA (peft `use_rslora=True`)，rank 64/alpha 128 为项目自选 | ⚠️ 自创配方 |
| 深度编码 | power transform λ=-3, c=10/3 + RGB 立方体棱边插值 | `core/depth_mapper.py` 逐参数对齐，测试断言 λ/c 及 Fig.5 顶点 | ✅ 保真 |
| 法向量编码 | R=(1-x)/2, G=(1+y)/2, B=(1+z)/2 | `core/normal_mapper.py` 一致（浮点无 trunc，属无害偏差） | ✅ 保真 |
| 实例分割解码 | 5 阶段聚类，τ=14, θ_size=2e-4, θ_erosion=0.1, γ=5.0 | 参数全对齐，但**默认路径偏离**：`grouping_mode="adjacent"`（论文为种子 floodfill）+ `adaptive_color_tolerance=True`（自适应 τ，论文固定 14）；需显式关闭才对齐 | ⚠️ 保真但默认值偏离 |
| Prompt 体系 | 语义/实例/指代/深度/法向量 5 类 | `inference/prompt.py` 全覆盖 + 额外自创 5 类（edge/flow/keypoint 等） | ✅ 覆盖 |
| 评测指标 | depth δ1/AbsRel、normal 角度误差、mIoU、SA-Co pmF1/IL_MCC/cgF1、cIoU/gIoU | `evaluation/` 全部实现（含 SA-Co eq.7 的 cgF1=100·pmF1·IL_MCC） | ✅ 代码完整 |
| Gemini 依赖 | SA-Co presence 用 Gemini 3.1 Flash-Lite；ReasonSeg 改写用 Gemini 2.5 Pro | `run_paper_judges.py` + `evaluation/gemini.py` 实现，需 API key，仓库无 manifest | ⚠️ 可用但未接线 |
| 评测基准数据 | 13+ 个公开基准 | `data/` 下**零评测集**（仅 3 个训练数据符号链接）；`SACoInstanceDataset._scan` 为启发式简化，非真实标注格式 | ❌ 缺失 |
| 端到端评测 | Table 1 全表 8 项 | `evaluate_paper_benchmarks.py` 仅覆盖 depth/normal/semantic 3 项，无 referring/SA-Co/GenAI-Bench/ImgEdit | ❌ 不存在 |
| 生成能力保持 | GenAI-Bench 胜率 53.5%、ImgEdit 47.8%（人工评测） | 仅有离线健康代理（FID/LPIPS/CLIP 代理），README 明确标注"不得当作 GenAI-Bench 胜率" | ❌ 不可复现（人工评测） |

## 3. 训练管线工程状态

- `scripts/train.py` + `training/tuner.py` 是完整可用的 LoRA 训练器（rsLoRA、9:1 混合、DDP/FSDP/DeepSpeed 分派均有真实实现）。
- 但 **FLUX 主线（paper_reproduction_baseline.yaml）从未跑通**：配置指向的 `data/paper_reproduction/` 不存在；`wandb/` 下 7 个 run 全部 `runtime: 0`（初始化即退出）。
- 实际产出 checkpoint 的是替代路径：`train_sdxl_mps.py`（SSD-1B）、`train_sana_512.py`、以及 `mar_vb_*` 系列（MAR 掩码扩散架构，属论文方法的架构替代实验，非复现）。
- 模型注册表 31 条目，但训练验证过的只有 flux / sdxl / sd3 / sana 四个 family，其余"注册多于验证"。

## 4. 数据现状

- 训练数据：`data/vb_snvc_train_v3`（符号链接，48k+ 视觉样本），`docs/v3_availability_report.md` 报告源图可用率 73.5%，scannetpp/51world/IDD 等为 0%。
- 生成数据：`build_generation_dataset.py` 用开源 FLUX 合成的 5,981 张图——非 Nano Banana Pro 原生分布，9:1 混合的"保持生成能力"前提不成立。
- 评测数据：13 个基准**无一在位**；数据集类已注册但无从加载。

## 5. 实测成绩 vs 论文（数量级差距）

| 指标 | 论文 Vision Banana | 本项目实测最佳 | 证据 |
|------|-------------------|--------------|------|
| 深度 δ1 ↑ | 0.882（6 集平均） | **0.214**（FLUX ckpt-300）；0.077（SSD-1B） | `docs/FLUX_REPLICATION_FINAL_REPORT.md`；`output/mps_depth_benchmark_best/benchmark_result.json` |
| 深度 Pearson r | — | **≈0.0005**（空间结构未学到） | 同上 |
| 语义分割 mIoU ↑ | 0.699（Cityscapes） | **0.00017**（MAR-VB，近零） | `output/mar_vb_seg_fixed_palette_eval_*/metrics.json` |
| 法向量 mean angle ↓ | 15.05° | **81.9°**（≈随机） | 内部评测记录 |

注：以上为本项目在小模型（SSD-1B 1.3B / MAR 479M）、小数据、MPS 单卡上的阶段性结果，并非管线理论上限；但足以证明"当前状态 ≠ 论文效果"。

## 6. 差距分类与补齐路径

**A. 原则性不可复现（任何团队都无法做到）**
- Nano Banana Pro 权重与其原始训练数据 → Table 1 的精确数字不可复现。
- GenAI-Bench / ImgEdit 人工胜率评测 → 只能做代理指标。

**B. 项目可补齐的工程差距**
1. 跑通 FLUX 主线 baseline（先备齐 `data/paper_reproduction/`，GPU 环境，而非 MPS 上的 256px tiny 实验）。
2. 下载 13 个评测基准，修复 `SACoInstanceDataset` 为真实标注格式，补 ReasonSeg 数据集类。
3. 扩 `evaluate_paper_benchmarks.py` 至 8 项全覆盖；接入 Gemini judge manifest 生成。
4. 实例分割解码默认值对齐论文（`grouping_mode="seed"` + `adaptive_color_tolerance=False`），或至少在复现配置中显式设置。
5. 深度/分割目前连映射本身都没学会（Pearson≈0、mIoU≈0）——优先排查训练目标与单步 x0 推理是否合理，再谈 scale-up。

## 7. 最终判断

> 该项目是 Vision Banana **方法学的忠实开源实现**（核心编码/解码算法逐参数对齐论文，训练/评测框架完整），但**不具备、也无法具备完整复现论文效果的条件**：基座与数据是闭源的，这是论文层面就封死的上限；项目自身的工程状态（评测数据零在位、主线训练未跑通、实测指标差 1 个数量级）距离"在开源基座上复现方法趋势"也还有明显距离。合理的预期目标是：在开源基座上验证"生成预训练 → 视觉理解迁移"这一**定性结论**，而非复现 Table 1 数字。
