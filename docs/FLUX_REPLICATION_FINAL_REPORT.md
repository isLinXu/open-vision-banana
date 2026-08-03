# Vision Banana FLUX 复现 — 最终实验报告

**日期**: 2026-08-03
**硬件**: Apple M3 Max, 128GB 统一内存, MPS 后端
**模型**: FLUX.1-schnell (12B), rsLoRA

---

## 1. 实验总结

### Phase 1: 管线验证 ✅
- 9 step 完成, Loss 从初始值稳定下降
- 确认 FLUX.1 + LoRA + img2img flow matching 训练管线正常
- 修复 transformers 版本 + HF 认证问题

### Phase 2: 单任务深度实验 (depth-only)
| Checkpoint | Steps | Train Loss | AbsRel | Pearson r | δ<1.25 |
|-----------|-------|-----------|--------|-----------|--------|
| ckpt-100 | 100 | ~0.50 | — | — | — |
| ckpt-200 | 200 | 0.16 | — | — | — |
| **ckpt-300** | **300** | **0.10** | **2.28** | **0.0005** | **0.214** |
| ckpt-400 | 400 | 0.10 | — | — | — |
| ckpt-500 | 500 | 0.09 | 15.45 | 0.000004 | 0.062 |

**最佳结果**: ckpt-300, AbsRel=2.28
**问题**: Pearson r≈0（空间结构未学习），ckpt-500 过拟合

### Phase 3: 多任务训练 (11 tasks + generation)
- 数据: 48,362 视觉 + 5,625 generation = 53,987 样本
- 训练: 410 steps, Loss 1.03 (plateau)
- 推理方法: single-step x0

| Checkpoint | Train Loss | AbsRel | Pearson r | δ<1.25 |
|-----------|-----------|--------|-----------|--------|
| ckpt-200 | 1.33 | 2,404 | 0.021 | 0.004 |
| ckpt-400 | 1.03 | 14,304 | 0.036 | 0.004 |

**结论**: 多任务训练无法学会深度映射（LoRA 容量被 11 任务稀释）

### Phase 3b: 高容量 LoRA (rank=256)
- 数据: 2,181 depth-only 样本
- LoRA: rank=256, alpha=512, 12 target modules, dropout=0.01
- 训练: 240 steps, Loss=0.25
- 推理: single-step x0, AbsRel=14~437 (7/10 samples)

**结论**: 更高 rank 仍无法解决根本问题

---

## 2. 根本原因分析

### 2.1 核心问题: Flow Matching 推理不收敛

训练使用 img2img flow matching:
```
noisy = (1-t) * input_latents + t * target_latents
velocity_target = target_latents - input_latents
```

推理使用 single-step x0:
```
v_pred = model(input_latents, t=0, prompt)
x0_pred = input_latents + v_pred
```

**诊断发现** (Phase 2):
- 训练样本点速度预测: cos_sim=0.95 (t=0.5) → 模型记住了训练数据
- Euler 积分收敛性: cos_sim=0.38~0.45 → 速度场无法积分
- 单步 x0 推理: 不稳定，结果高度依赖 checkpoint 质量

### 2.2 LoRA 容量瓶颈

| 配置 | 可训练参数 | 占比 | 结果 |
|------|-----------|------|------|
| rank=64, to_k+to_v | ~157M | 1.3% | AbsRel=2.28 (最佳) |
| rank=64, 12 modules | ~629M | 5.2% | AbsRel=2404 (多任务) |
| rank=256, 12 modules | ~2.5B | 20.8% | AbsRel=14~437 |

增加 LoRA 容量并未改善推理质量，反而加剧过拟合。

### 2.3 数据量不足

- 2,181 depth 样本 → 模型记忆而非泛化
- 48K 多任务样本 → 每任务分摊不足
- 论文使用 10M+ 样本 → 我们差 3 个数量级

### 2.4 推理方法不匹配

论文使用完整多步生成 (50 steps Euler):
```
z_0 = encode(input_image)
for t in [0, 0.02, 0.04, ..., 1.0]:
    v = model(z_t, t, prompt)
    z_{t+1} = z_t + dt * v
z_target = z_1
```

我们在 MPS 上无法负担 50 步推理（每步 ~5s，总计 ~250s/样本）。
4 步 Euler 已验证不收敛 (cos_sim=0.38)。

---

## 3. 与论文对比

| 指标 | 论文 (FLUX+LoRA) | 我们的最佳结果 | 差距 |
|------|-----------------|--------------|------|
| 深度 AbsRel | ~0.05 (估计) | 2.28 | 45x |
| Pearson r | >0.9 (估计) | 0.0005 | 1800x |
| 训练样本 | 10M+ | 2,181 | 4600x |
| 推理步数 | 50 | 1 (x0) | 50x |
| 可训练参数 | 1.3% | 1.3%~20.8% | 类似 |

---

## 4. 结论与建议

### 结论
在 MPS (Apple M3 Max) 硬件约束下，FLUX.1+LoRA 的 img2img flow matching 方法无法实现有效的深度估计。核心瓶颈是:
1. **推理不收敛**: 速度场无法通过 Euler 积分得到正确目标
2. **过拟合**: 有限数据下模型记忆训练样本而非学习泛化映射
3. **计算约束**: 无法负担论文级别的 50 步推理

### 建议方向
1. **直接潜空间回归**: 绕过 flow matching，直接训练 input_latents → target_latents 映射
2. **更多数据**: 需要 10x~100x 更多深度训练数据
3. **更小模型**: 使用 SDXL/SD3 等更小模型在 MPS 上实验
4. **GPU 集群**: 在 CUDA GPU 上进行完整多步推理评估
5. **不同范式**: 考虑 ControlNet 或 adapter-based 方法

---

## 5. 已保存资产

### Checkpoints
- `output/flux_depth_gate/checkpoint-{100,200,300,400,500}/` — Phase 2 depth-only
- `output/flux_multitask/checkpoint-{200,400}/` — Phase 3 multi-task
- `output/flux_depth_hirank/checkpoint-200/` — Phase 3b high-rank

### 配置文件
- `configs/training_mps_depth_gate.yaml` — Phase 2 配置
- `configs/training_mps_multitask.yaml` — Phase 3 配置
- `configs/training_mps_depth_hirank.yaml` — Phase 3b 配置

### 评估脚本
- `scripts/eval_depth_gate_x0.py` — 单步 x0 推理评估
- `scripts/eval_depth_gate_flow.py` — 多步 Euler 推理评估
