# 实验方案：Full-FT vs LoRA 消融（merged e2b 起点）

> 版本：2026-10-31 · 状态：待评审
> 目的：在固定数据配比与训练协议下，对比**全参数微调**与 **LoRA 微调**的差异 —— 直接检验复盘结论"LoRA r32 容量是主瓶颈"。

## 1. 实验设计

### 1.1 变量矩阵

| 组 | 起点权重 | 训练方式 | 可训练参数 |
|---|---|---|---|
| **E-0**（对照） | `gatilin/ovb-e2b_cap10k_s42_10000_merged1` | 不训练，仅起点评测 | — |
| **F1** | 同上 merged 权重 | **全参数微调** | ~20B transformer |
| **L1** | 同上 merged 权重 | **LoRA r32/α64（rsLoRA）** | ~224 层 adapter |

> **设计说明（需确认的一处歧义）**：建议两组都从 **merged 权重**出发 —— F1 直接续训全参，L1 在 merged 上叠加**新的** LoRA（不加载旧 adapter）。这样两组起点完全一致，唯一变量是"可训练容量/训练方式"，差值可归因于训练范式而非起点。
> 若 L1 意图是"qwen-2.1 原始权重 + LoRA"（复现 e2b 配方），则比较对象变成"起点 × 方式"两个变量，解释力弱 —— 不推荐。

### 1.2 数据配比（沿用 T1 = `ablation_no_normal` + C1 的深度编码）

| 项 | 取值 | 来源 |
|---|---|---|
| tasks | `depth, semantic, instance, referring, generation`（**无 normal**） | T1 |
| vision_ratio | `0.9` | T1 |
| depth_colormaps | `[cube_edge]`（单板，不增强） | **C1** |
| depth_curve | `metric_power` | C1 |
| manifest | `data/qwen_main_depth_reasoning/pairs.jsonl`（与 i88 消融同源） | T1/C1 共用 |
| semantic_prompt_styles | `json_named, json_tuple, natural, hex` + `color_remap_prob 0.5` | 沿用 |
| aspect_preserving_crop | `true` | 沿用 |

### 1.3 训练配置（沿用 C1 框架，仅 lr 按模式区分）

| 项 | F1（全参） | L1（LoRA） | 依据 |
|---|---|---|---|
| **lr** | **1e-5** | 3e-5 | 全参见 §3 调研；LoRA 沿 C1 |
| lr scheduler | cosine → 0，warmup 500 | cosine → 0 | C1 |
| optimizer | AdamW 8bit（或 Adafactor+fused_backward，见 §3.3） | AdamW | — |
| precision | bf16 + gradient_checkpointing | bf16 | — |
| batch | bs1 × accum4 × 8 GPU = **global 32** | 同左 | C1 |
| steps | 10,000 | 10,000 | 方案 |
| ckpt | 每 2,000 + final（6 个：0/2k/4k/6k/8k/10k） | 同左 | 方案 |
| EMA | 0.9999 | 0.9999 | 沿用 |
| seed | 42（建议补 43/44 视算力） | 42 | — |
| min_snr | off | off | fix 消融结论 |
| resolution | 512 | 512 | 沿用 |

### 1.4 评测协议（与既有 run 完全一致）

- **中间步采样评测**：每个 ckpt 跑 mini-eval —— NYU 100 + RefCOCOg 200 + Cityscapes 100（同 seed=42，50 步推理，guidance 4.0，native resolution，canonical prompt + cube_edge 解码）
- **起点（merged, step 0）与终点（10k）完整评测**：6 任务全量（NYU 654 / ETH3D 454 / DIODE 325 / Cityscapes 500 / RefCOCOg 4896 / ReasonSeg 200）
- raw δ1.25 为协议指标；aligned/oracle 仅诊断列
- 结果表标注 inference protocol；E-0 行必含

## 2. 预注册成功标准

| 指标 | 判定阈值 | 当前基线（e2b LoRA） |
|---|---|---|
| NYU raw δ1.25 | F1 > 0.35（LoRA 上限已证明 ~0.14） | 0.135 |
| NYU aligned δ1.25 | > 0.75 | 0.71 |
| Cityscapes mIoU | F1 > 0.45 | 0.327 |
| RefCOCOg cIoU | > 0.50 | 0.423 |
| ReasonSeg gIoU | > 0.45 | 0.328 |
| 生成保留 | 不明显劣化（proxy FID/LPIPS） | — |

**核心判读**：
- F1 ≫ L1 → 容量瓶颈实锤，论文"全量微调"路径确认 → 后续主跑走 full-FT/partial-unfreeze
- F1 ≈ L1 → 瓶颈不在参数量而在数据/时长/协议 → 回 §4.2.1 重新归因
- F1 深度上去了但 2D 掉了 → 任务干扰而非容量 → 走 task_loss_weights/分组方案

## 3. 全参微调超参调研（已核实来源）

| 来源 | lr | 备注 |
|---|---|---|
| DiffSynth-Studio 官方 Qwen-Image full-FT 示例 | **1e-5** | ZeRO-2 offload，AdamW，dit 全量 |
| Musubi Trainer 官方（DreamBooth/FT） | **5e-5** | 文档明确从 1e-4 下调至 5e-5；Adafactor+fused_backward 用于 ≤48GB |
| 社区 finetune 配置（tsiendragon/qwen-image-finetune） | 1e-4（LoRA 向） | LoRA 侧不是全参口径 |

**建议取值 lr=1e-5**：起点是已含 LoRA 增益的 merged 权重，大 lr 会快速洗掉已学能力；1e-5 是全参最保守且被官方示例采用的值。可加 warmup 500 step 防早期冲击。

**显存前提（必须确认）**：20B 全参 bf16 + AdamW ≈ 权重 40G + 梯度 40G + 优化器态 160G ≈ **240GB+**，需 **8×80G ZeRO-3/FSDP** 或 DiffSynth 式 ZeRO-offload；单机 M3 不可行。备选：musubi `full_bf16` + Adafactor + fused_backward（48GB 可行但换精度/需 stochastic rounding）。

## 4. 待确认细节清单

1. **L1 起点**：merged + 新 LoRA（推荐）还是 qwen-2.1 原始权重 + LoRA？
2. **训练框架**：已支持。`training.tune_mode: full`（配置见 `configs/experiments/qwen_image_21_fullft.yaml`）跳过 LoRA，冻结 VAE/text encoder，训练 backbone。优化器仍只收 `requires_grad` 参数。多卡全参显存见 §3，单机默认仍建议 LoRA。
3. **硬件**：8×80G 可用否？决定 ZeRO-3 vs offload vs musubi 方案
4. **task_prompts**：单板训练下 depth prompt 用命名 cube_edge 版（本轮已落地的代码修复保持启用）
5. **ckpt 平均**：10k 后是否追加 best3 平均（fix 消融显示 +14% δ1）—— 建议终点评测同时报 final EMA 和 best3 两列
