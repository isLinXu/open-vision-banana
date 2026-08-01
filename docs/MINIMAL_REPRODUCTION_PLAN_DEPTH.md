# 最小复现实验方案（先攻 Depth）

> 目标：用真实 benchmark 数字验证「Vision Banana 范式」在本开源实现（FLUX.1 基座 + 单任务 LoRA）上是否成立。
> 这是项目立身之本——目前仓库**没有任何在真实 benchmark 上复现的权重**（checkpoint_registry.yaml 全为 `placeholder`）。
>
> 本文档中的所有命令均复用仓库已有脚本/配置，不新增训练框架。

> **✅ Config 已落地（2026-07-26）**：`configs/training_config.yaml` 已改为 depth-only 单任务版，并新增 `training.flow_sigma=1.0`（修复验证门 2 的推理外推问题）。`TrainingConfig` 加了 `flow_sigma` 字段，`train.py` 改为从 config 读取（CLI `--flow-sigma` 仍可覆盖）。运行：
> ```bash
> python scripts/train.py --config configs/training_config.yaml \
>                         --data-path data/prepared/hypersim
> ```

---

## 0. 为什么先做 Depth（且只做 Depth）

1. Depth 是论文里精度最高、最易量化的任务（δ1=0.929 / AbsRel 量级），证据最强。
2. 数据管线最成熟：`prepare_hypersim.py` + `core/depth_mapper.py` 已打通；`evaluate_depth_standard.py` 已实现标准指标（AbsRel/SqRel/RMSE/δ<1.25³）。
3. 单任务 LoRA 把变量降到最低：**一次只验证一个假设**——"FLUX-img2img-LoRA 能否学会把输入图编码成深度 RGB 可视化并解码回深度"。
4. 完成 depth 后，normal / semantic 是同构的（换 mapper + 换数据集即可），可快速复制。

---

## 1. 关键架构决策（写死，避免漂移）

| 决策项 | 取值 | 理由 |
|---|---|---|
| 基座模型 | **FLUX.1-dev**（非 schnell） | schnell 是 1/4-step 蒸馏，不适合 LoRA 微调；dev 与 `checkpoint_registry.yaml` 中 `base_model` 一致。 |
| 映射 colormap | **cube_edge**（两端强制一致） | `DepthRGBMapper` 默认即 cube_edge；**训练编码与推理解码必须用同一 colormap，否则数字无意义**（见 §4 验证门 1）。 |
| 任务配置 | 单任务 depth（仅 `--data-path` 指向 depth 目录） | **注意：train.py 忽略 yaml 的 `tasks:` 块**，任务由 `--data-path` 目录内容自动检测（`build_dataset` 遍历 depth/normal/semantic…）。隔离 depth = `--data-path` 指向仅含 depth 的预处理目录（如 `prepare_hypersim.py` 产物）。 |
| LoRA | rank=64, alpha=128, rsLoRA=True, 完整 target_modules | 与仓库 `training_config.yaml` 原值及 `LoRAConfig` 默认八项一致；把原 yaml 里过窄的 `[to_k,to_v]` 扩回完整范围。 |
| 分辨率 | 先 512（配置默认），显存够则升 768/1024 | FLUX 原生 1024，512 有信息损失但能跑通；见风险 R3。 |
| generation_ratio | 第一轮 **0**（纯 depth 隔离信号）；第二轮 **0.1** 验证生成保持 | 论文主张保留 90% 生成能力，但最小复现先证深度可行。 |

---

## 2. 端到端步骤（命令可直接复制）

### 步骤 1：准备训练数据（Hypersim → 深度 RGB）
复用 `scripts/prepare_hypersim.py`，它已用 `DepthRGBMapper` 把 GT depth 编码成 `targets/` 下的 RGB。

```bash
# 1) 下载 Hypersim（约 70k 训练图），按仓库 download 脚本
python scripts/download_datasets.py --dataset hypersim

# 2) 转换为训练布局：data/hypersim_depth/{images/,targets/}
python scripts/prepare_hypersim.py \
    --source /path/to/hypersim \
    --output data/hypersim_depth/ \
    --max-samples 10000 \      # 最小复现先取 1万，跑通后再全量
    --validate

# ⚠️ 验证门 1：确认脚本使用的是 colormap="cube_edge"（见 §4）
```

### 步骤 2：配置单任务训练
编辑 `configs/training_config.yaml`：
- `model`: 指向 `black-forest-labs/FLUX.1-dev`
- `lora`: `r=32, alpha=32, rs_lora=true`
- `data.vision_ratio / generation_ratio`: 第一轮 `vision_ratio=1.0, generation_ratio=0.0`
- `data.tasks`: 仅保留 `depth_estimation`，其 `datasets=[hypersim_depth]`；其余任务块清空
- `data.resolution`: 512（首轮）

### 步骤 3：先跑冒烟测试（MPS/CPU，200 图）
仓库已有 `scripts/train_smoke.py` 与 `train_lora_mps.py`。先用极小数据确认**训练/推理 img2img 机制对齐**（见 §4 验证门 2），再上 GPU。

```bash
python scripts/train_smoke.py --config configs/training_config.yaml \
    --data-path data/hypersim_depth --max-samples 200 --epochs 2
```

### 步骤 4：正式训练（单卡 A100-40G 为例）
复用已有启动脚本 `scripts/launch_training_a100_40g_deepspeed.sh`（或 `train.py` 直接跑）：

```bash
python scripts/train.py --config configs/training_config.yaml \
    --epochs 10 --output-dir checkpoints/depth_lora_v1
# 多卡：bash scripts/launch_training_a100_40g_deepspeed.sh
```

### 步骤 5：标准指标评估（论文协议）
`scripts/evaluate_depth_standard.py` 已完整实现：img2img 推理 → 同 colormap 解码 → AbsRel/SqRel/RMSE/δ1。

```bash
# 准备一个 held-out 评测集（NYUv2 或 iBims-1），用同样的 DepthRGBMapper 编码 GT
python scripts/prepare_hypersim.py --source /path/to/ibims1 --output data/ibims1_depth --validate

python scripts/evaluate_depth_standard.py \
    --lora-path checkpoints/depth_lora_v1 \
    --data-dir data/ibims1_depth \
    --num-samples 50 \
    --output results/depth_eval_v1.json
```

### 步骤 6：与基线对比（证明"范式成立"的关键）
用 `src/open_vision_banana/evaluation/baselines.py` / `scripts/evaluate_paper_benchmarks.py` 把 FLUX-LoRA 与 **zero-shot 基线**（MiDaS / Depth-Anything）同榜比较。
> 这是定义"复现是否成功"的核心：若 FLUX-LoRA depth 连 Depth-Anything zero-shot 都打不过，则证明基座替换（FLUX vs NBP）存在天花板问题，需要重新评估路线。

### 步骤 7：发布 checkpoint
训练验证后，把 `configs/checkpoint_registry.yaml` 里 `vision-banana-depth` 的 `status` 从 `placeholder` 改为 `available`，并填入 `metrics.depth_abs_rel` 等真实数字（参考文件底部模板注释）。

---

## 3. 成功标准（Definition of Done）

- [ ] 在 iBims-1 / NYUv2 上得到 **AbsRel 与 δ1** 真实数字（非 null）。
- [ ] FLUX-LoRA depth **不低于** Depth-Anything zero-shot 基线（或至少在同一数量级，并给出合理解释）。
- [ ] 推理用 `evaluate_depth_standard.py` 跑通，且训练/推理 colormap 一致（验证门 1 通过）。
- [ ] checkpoint 发布到 registry（`status: available` + 指标），而非停留在 placeholder。
- [ ] 在 README 明确标注：**此数字由本仓库 FLUX 复现得到，非原论文 NBP 数字**。

---

## 4. 必过的验证门（决定实验是否可信）

**验证门 1 — colormap 一致性（P0）**
`prepare_hypersim.py` 编码与 `evaluate_depth_standard.py` 解码**必须**用同一 `DepthRGBMapper(colormap="cube_edge")`。
检查：grep 两个脚本确认都显式传入 `colormap="cube_edge"`，否则生成的 RGB 与解码对不上，指标全废。

**验证门 2 — 训练/推理 img2img 机制对齐（P0，已代码核验 ✅）**
**结论：训练确为图像条件（img2img）机制，不是纯 t2i。** 代码路径（`tuner.py` `_flow_training_step`，1599–1685 行）：
- depth 样本 `__getitem__` 返回 `image`（输入图，[-1,1]），`_collate_fn` 透传（全 None 才走空）；
- 训练步构造：`noisy = (1-t)·input_lat + t·target_lat + noise_term`，预测 `velocity = target_lat − input_lat`；
- `use_img2img_flow=True`（默认，且 `train.py` 仅 `--no-img2img-flow` 才关）、`use_source_conditioning=False`（FLUX 不走 SDXL concat 分支）。
→ 与推理 `generator.generate(image=...)` 走 `FluxImg2ImgPipeline` 的"图像条件重绘"范式**方向一致**，不存在"t2i vs img2img"的致命错配。**这一项可勾掉。**

**但发现一个需实证验证的精细偏差（新增风险 R7，P1）：**
默认 `flow_sigma=0.0` → 训练是**无噪声的 latent 凸混合回归**（输入/目标皆为干净 latent，模型从未见过带噪 latent）；
而推理 `FluxImg2ImgPipeline(strength=0.85)` 是从 **input + 噪声** 起步做多步去噪——即推理处于"带噪 latent"区间，训练未显式覆盖。
→ 模型在带噪起点上的行为属于**外推**，不保证成立。这是冒烟可视化的头号检查项（输入/目标/重建三图对照，确认不是退化解或纯噪声）。
- **缓解**：训练时设 `flow_sigma>0`（如 1.0），让混合里带噪，使训练分布向推理的带噪起点靠拢；并确认推理对 FLUX.1-dev 用足步数（非 1-step schnell 默认值）。
- 核对命令：`python scripts/train.py --no-img2img-flow` 会**关掉** img2img 退回纯 t2i——务必不传此 flag。

**验证门 3 — 指标口径（P1）**
确认 `evaluation/depth.evaluate_depth` 的尺度处理（metric depth vs relative depth）与论文一致；Hypersim→iBims 跨数据集需对齐尺度（median scaling），否则 AbsRel 不可比。

---

## 5. 风险登记

| ID | 风险 | 等级 | 缓解 |
|---|---|---|---|
| R1 | **基座替换**（FLUX vs 论文 NBP 闭源）天花板未知 | P0 | 步骤 6 与 zero-shot 基线对比；若显著落后，记录并评估换更大基座/更多数据 |
| R2 | 训练/推理 img2img 机制未对齐 | P0 | 验证门 2，冒烟可视化 |
| R3 | 分辨率 512 与 FLUX 原生 1024 不匹配 | P1 | 显存允许升 768/1024；记录分辨率对指标的影响 |
| R4 | Hypersim 1万图 vs 论文训练量偏小，可能欠拟合 | P1 | 先 1万跑通，再扩到全量 + 加 iBims/DA-2K |
| R5 | colormap 两端不一致 | P0 | 验证门 1 |
| R6 | 显存：FLUX.1-dev bf16 ≈24GB | P1 | cpu_offload / DeepSpeed ZeRO-2 / 降 rank |
| R7 | 训练/推理噪声区间错位：默认 `flow_sigma=0`（无噪混合回归）vs 推理带噪起点 | P1 | 设 `flow_sigma>0` 缩小错位；**冒烟可视化**必做 |

---

## 6. 工作量与时序（粗估）

| 阶段 | 动作 | 预估 |
|---|---|---|
| 1 | 数据准备 + 验证门 1/2（冒烟） | 0.5–1 天 |
| 2 | 单卡训练 1万图 ×10 epoch | 数小时–1 天（A100） |
| 3 | 评估 + 基线对比 | 0.5 天 |
| 4 | 发布 checkpoint + 文档修正 | 0.5 天 |
| **合计** | 最小可信复现 | **约 2–3 天（含一次全量扩展）** |

---

## 7. 完成 depth 后的同构扩展

normal → 复用 `prepare_*.py` + `NormalRGBMapper`；semantic → `prepare_*.py` + 调色板；其余任务均为"换 mapper + 换数据集"的同构操作。
**多任务统一 LoRA**（`vision-banana` 总 checkpoint）应在各单任务验证通过后再训练，避免变量耦合。
