# Open-Vision-Banana 论文对齐执行方案 v1.0

> **目标**: 将 open-vision-banana 从"代码对齐"推进到"训练+评估验证对齐"，验证论文核心论点。  
> **论文**: Image Generators are Generalist Vision Learners (arXiv:2604.20329v3)  
> **当前成熟度**: 8.5/10（代码层），训练验证层 ~40%  
> **制定日期**: 2026-07-13

---

## 一、现状速览

| 维度 | 当前状态 | 目标状态 | 差距 |
|------|---------|---------|------|
| 核心算法编码 | ✅ 100% | 100% | 无 |
| 推理范式 (img2img) | ✅ 100% | 100% | 无 |
| 训练基础设施 | ✅ 90% | 100% | 色图增强未生效 |
| 训练深度 (FLUX) | ⚠️ ~100 steps | 5K-10K+ steps | 严重不足 |
| Benchmark 评估 | ❌ 未执行 | 6 depth + 4 normal + Cityscapes | 核心差距 |
| LLM 辅助 pipeline | ❌ 未实现 | Qwen2.5-VL 替代 | 中等差距 |
| 生成能力保留 | ⚠️ Proxy 指标 | FID/LPIPS | 次要差距 |

**核心瓶颈**: 代码已全部就绪，但缺少"大规模训练 → 标准 benchmark 评估 → 论文指标对比"的闭环。

---

## 二、阶段规划总览

```
Phase 0 快速修复          (1-2 天)
    ├── P0-1 修复色图训练增强
    ├── P0-2 统一双轨训练脚本
    └── P0-3 训练配置校准

Phase 1 数据准备与单任务训练  (2-3 周)
    ├── P1-1 深度数据集准备 (Hypersim + VirtualKITTI)
    ├── P1-2 Depth LoRA 训练 (目标: 5K+ steps, loss 收敛)
    ├── P1-3 Normal 数据集准备 (ScanNet + OASIS)
    ├── P1-4 Normal LoRA 训练
    └── P1-5 生成数据混合比例消融 (1:9 / 1:19 / 1:4)

Phase 2 多任务联合训练与评估  (2-3 周)
    ├── P2-1 多任务联合训练 (depth + normal + semantic)
    ├── P2-2 Depth Benchmark 评估 (NYU/ETH3D/DIODE/KITTI)
    ├── P2-3 Normal Benchmark 评估 (NYU/DIODE/ScanNet/VKITTI)
    ├── P2-4 Semantic Seg 评估 (Cityscapes val)
    └── P2-5 指标对比与论文对齐报告

Phase 3 完善与扩展         (1-2 周)
    ├── P3-1 Gemini 替代方案 (Qwen2.5-VL NP filter)
    ├── P3-2 Reasoning Pipeline (Qwen2.5 翻译)
    ├── P3-3 FID/LPIPS 真实指标集成
    └── P3-4 生成能力保留验证 (GenAI-Bench proxy)

Phase 4 发布与文档         (1 周)
    ├── P4-1 HuggingFace 预训练权重发布
    ├── P4-2 论文对齐报告 v2.0
    └── P4-3 PyPI 发布准备
```

---

## 三、Phase 0 — 快速修复（1-2 天）

### P0-1: 修复色图训练增强 🔴 P0

**问题**: `MixedVisionDataset.__getitem__` 为 depth 样本附加了 `colormap` 字段，但 `InstructionTuner._training_step` 完全不消费该字段，导致色图增强仅停留在数据层。

**修复方案**:
```python
# 在 tuner.py _training_step() 的 augmentation 段落后插入:
colormap = batch.get("colormap")
if colormap is not None and any(c != "cube_edge" for c in colormap):
    # 动态重编码 depth → RGB
    from ..core.depth_mapper import DepthRGBMapper
    mapper = DepthRGBMapper()
    for i, cmap in enumerate(colormap):
        if cmap != "cube_edge":
            # target_rgb[i] 需要先从 RGB 解码为深度，再用新色图编码
            # 或者：如果 dataset 已经提供原始深度值，直接重新编码
            # 更优方案：dataset 层直接提供原始深度值，训练时动态编码
            pass
```

**更优方案**: 修改 `VisionTaskDataset` 和 `MixedVisionDataset`，让 depth 样本携带原始深度值 `depth_gt`（而非预编码的 RGB），在 `_training_step` 中根据 `colormap` 动态编码。

**验收标准**:
- [ ] `tuner.py` 中 `batch.get("colormap")` 被实际消费
- [ ] 训练日志中记录当前 step 使用的色图类型
- [ ] 新增单元测试验证动态编码逻辑

**预计时间**: 4-6 小时

---

### P0-2: 统一双轨训练脚本 🟡 P1

**问题**: `scripts/train.py`（主脚本）和归档的 `scripts/archive/train_vb_v7.py` 存在功能分叉。v7 有 12 项改进（Min-SNR、logit-normal、CFG dropout、per-sample 验证等）尚未完全反向移植到 `InstructionTuner`。

**检查清单**:
| v7 特性 | InstructionTuner 状态 | 行动 |
|---------|----------------------|------|
| Min-SNR | ✅ `compute_min_snr_weight()` 已集成 | 无需修改 |
| logit-normal timesteps | ✅ `sample_timesteps_logit_normal()` 已集成 | 无需修改 |
| CFG dropout | ✅ `cfg_dropout_rate` 已集成 | 无需修改 |
| 多数据集支持 | ⚠️ `MixedVisionDataset` 支持但需验证 | 验证 |
| per-sample 验证 | ❌ 未实现 | 评估是否必要 |
| 学习率 warmup ratio | ✅ `warmup_ratio` 参数已存在 | 无需修改 |

**结论**: 经检查，`InstructionTuner` 已集成 v7 的核心改进。差异主要在配置层面，无需大规模重构。

**验收标准**:
- [ ] 确认 `train.py` 可通过 configs 复现 v7 的全部训练行为
- [ ] 删除或明确标记 archive 目录中真正过时的脚本

**预计时间**: 2-4 小时

---

### P0-3: 训练配置校准 🔴 P0

**当前配置问题**:
- `training_flux_depth.yaml`: `vision_ratio=1.0`（纯视觉数据），但论文要求混入生成数据（"at a very low ratio"）
- `training_config.yaml`: `vision_ratio=0.1`（10% 视觉），但缺少生成数据集路径
- `batch_size=1`（MPS 配置）过小，A100 上应设为 4-8

**推荐配置**:
```yaml
# configs/training_flux_depth_aligned.yaml
model:
  name: "black-forest-labs/FLUX.1-schnell"
  dtype: "bfloat16"

lora:
  rank: 64
  alpha: 128
  rs_lora: true
  target_modules: ["to_k", "to_v"]
  dropout: 0.0

optimizer:
  lr: 5.0e-5          # 论文未公开，保持默认
  weight_decay: 0.01
  scheduler: "cosine"
  warmup_steps: 500
  max_grad_norm: 1.0

training:
  epochs: 5           # 约 5K-10K steps 取决于数据集大小
  gradient_accumulation_steps: 4
  use_amp: true
  use_ema: true
  ema_decay: 0.9999
  save_every: 500
  eval_every: 500
  max_keep: 5
  seed: 42
  log_every: 10
  early_stopping_patience: null

data:
  vision_ratio: 0.1       # 论文 "very low ratio" — 10% 视觉数据
  generation_ratio: 0.9
  resolution: 512
  batch_size: 4           # A100 40G 可承受
  num_workers: 4
  augmentation: true
  val_split: 0.05
  generation_path: null   # TODO: 设置生成数据集路径

  tasks:
    depth_estimation:
      datasets:
        - name: "Hypersim"
          split: "train"
        - name: "Virtual_KITTI_2"
          split: "train"
```

**验收标准**:
- [ ] 创建 `training_flux_depth_aligned.yaml`
- [ ] 创建 `training_flux_multitask_aligned.yaml`
- [ ] 确认 batch_size=4 在 A100 40G 上不 OOM

**预计时间**: 2 小时

---

## 四、Phase 1 — 数据准备与单任务训练（2-3 周）

### P1-1: 深度数据集准备 🔴 P0

**论文要求**:
- 深度数据: "entirely synthetic depth data from simulation engines"
- 不使用真实深度数据（如 NYU 用于训练）
- 论文 zero-shot 在 NYU/ETH3D/DIODE/KITTI 上评估

**推荐数据集**:
| 数据集 | 规模 | 类型 | 用途 |
|--------|------|------|------|
| **Hypersim** | ~50K | 室内合成 | 深度 + 法线训练 |
| **Virtual KITTI 2** | ~10K | 室外合成 | 深度训练 |
| **TartanAir** | ~100K | 合成 | 深度补充 |

**预处理流程**:
1. 下载原始数据
2. 使用 `scripts/prepare_depth_dataset.py` 将深度图编码为 RGB（Cube edge colormap）
3. 组织为 `images/` + `targets/` 的 `VisionTaskDataset` 格式
4. 验证 roundtrip 误差 < 0.5%

**验收标准**:
- [ ] Hypersim 深度训练集 > 30K 样本
- [ ] Virtual KITTI 2 深度训练集 > 5K 样本
- [ ] roundtrip 误差验证通过

**预计时间**: 3-5 天（含下载时间）

---

### P1-2: Depth LoRA 训练 🔴 P0

**训练参数**:
| 参数 | 值 | 说明 |
|------|-----|------|
| 基座模型 | FLUX.1-schnell | 12B 参数 |
| LoRA | rank=64, α=128, rsLoRA | 论文推断 |
| 学习率 | 5e-5 | 默认值 |
| 视觉:生成比例 | 1:9 | 论文 "very low ratio" |
| 分辨率 | 512×512 | 保守设置 |
| batch_size | 4 | A100 40G |
| gradient_accumulation | 4 | 等效 batch=16 |
| 训练步数 | 5K-10K | 目标收敛 |

**收敛判断标准**:
- Loss 从 ~1.8 下降到 < 0.3 并稳定
- 每 500 steps 保存 checkpoint
- 使用 EMA 权重进行推理评估

**关键监控指标**:
```
wandb/TensorBoard:
  - train/loss
  - train/lr
  - val/loss (如果启用验证)
  - epoch/train_loss
```

**验收标准**:
- [ ] 完成 5K+ steps 训练
- [ ] final loss < 0.3
- [ ] 保存 EMA checkpoint

**预计时间**: 5-7 天（A100 上约 1-2 天/1K steps）

---

### P1-3: Normal 数据集准备 🔴 P0

**推荐数据集**:
| 数据集 | 规模 | 类型 |
|--------|------|------|
| **Hypersim** | ~50K | 含法线 GT |
| **ScanNet** | ~1.2M | 室内真实（但论文用合成）|
| **OASIS** | ~50K | 室外真实 |

**注意**: 论文未明确法线训练数据来源。由于论文说 "entirely synthetic depth data"，法线数据可能同样来自合成引擎。Hypersim 是最稳妥的选择。

**验收标准**:
- [ ] Hypersim 法线训练集 > 30K 样本
- [ ] 法线 RGB 编码验证（面向左=红，面向上=绿，面向相机=蓝）

**预计时间**: 2-3 天

---

### P1-4: Normal LoRA 训练 🔴 P0

**可与 Depth 训练并行（如果有多个 GPU）或串行。**

**验收标准**:
- [ ] 完成 5K+ steps 训练
- [ ] final loss < 0.3

**预计时间**: 5-7 天

---

### P1-5: 混合比例消融 🟡 P1

**论文只说 "at a very low ratio"，我们需要验证哪个比例最优。**

**实验设计**:
| 实验 | 视觉:生成 | 训练步数 | 评估 |
|------|----------|---------|------|
| A | 1:19 (5%) | 5K | NYU δ1 |
| B | 1:9 (10%) | 5K | NYU δ1 |
| C | 1:4 (20%) | 5K | NYU δ1 |
| D | 1:1 (50%) | 5K | NYU δ1 |

**预期**: 比例过低 → 视觉能力不够；比例过高 → 遗忘生成能力。论文的 "very low" 暗示 5-10% 是 sweet spot。

**验收标准**:
- [ ] 至少完成 3 个比例的对比实验
- [ ] 报告最优比例

**预计时间**: 1-2 周（可与其他训练并行）

---

## 五、Phase 2 — 多任务联合训练与评估（2-3 周）

### P2-1: 多任务联合训练 🔴 P0

**论文暗示同时训练所有视觉任务**（instruction-tuning on all vision tasks simultaneously）。

**配置**:
```yaml
# configs/training_flux_multitask_aligned.yaml
data:
  vision_ratio: 0.1
  tasks:
    depth_estimation:
      datasets: [Hypersim, Virtual_KITTI_2]
    normal_estimation:
      datasets: [Hypersim]
    semantic_segmentation:
      datasets: [COCO-Stuff, ADE20K]
```

**验收标准**:
- [ ] 多任务混合数据集正确采样
- [ ] 各任务 loss 曲线正常

**预计时间**: 5-7 天

---

### P2-2: Depth Benchmark 评估 🔴 P0

**评估数据集**（论文 Table 1）:
| 数据集 | 论文 δ1 | 论文 AbsRel | 我们的目标 |
|--------|--------|------------|-----------|
| NYU v2 | 0.948 | 0.081 | δ1 > 0.90 |
| ETH3D | 0.935 | 0.103 | δ1 > 0.90 |
| DIODE-Indoor | 0.917 | 0.108 | δ1 > 0.88 |
| KITTI | 0.915 | 0.107 | δ1 > 0.88 |
| iBims-1 | 0.934 | 0.078 | δ1 > 0.90 |
| nuScenes | 0.643 | 0.219 | δ1 > 0.55 |

**评估脚本**:
```bash
python scripts/evaluate.py \
  --task depth \
  --checkpoint output/checkpoints_flux_depth/checkpoint-best \
  --datasets nyu,eth3d,diode,kitti,ibims,nuscenes \
  --output_dir output/eval_depth
```

**验收标准**:
- [ ] 6 个数据集全部跑完
- [ ] 4-dataset avg (NYU+ETH3D+DIODE+KITTI) δ1 > 0.88
- [ ] 生成与论文 Table 1 的对比表格

**预计时间**: 2-3 天

---

### P2-3: Normal Benchmark 评估 🔴 P0

**评估数据集**（论文 Table 7）:
| 数据集 | 论文 mean↓ | 目标 mean↓ |
|--------|-----------|-----------|
| NYU v2 | 15.549 | < 20 |
| DIODE | 17.778 | < 22 |
| ScanNet | 13.818 | < 18 |
| VKITTI | 15.052 | < 20 |

**验收标准**:
- [ ] 4 个数据集全部跑完
- [ ] 3 室内数据集 avg (NYU+DIODE+ScanNet) mean < 20

**预计时间**: 2-3 天

---

### P2-4: Semantic Segmentation 评估 🔴 P0

**评估数据集**: Cityscapes val (19 classes)

**论文指标**: mIoU = 69.9 (zero-shot)

**验收标准**:
- [ ] Cityscapes val mIoU > 60（考虑到 FLUX vs NBP 基座差异）

**预计时间**: 1-2 天

---

### P2-5: 论文对齐报告 v1.5

**输出**: `docs/paper_alignment_evaluation_report.md`

**内容**:
- 训练配置详情
- 每个 benchmark 的详细结果
- 与论文数字的对比表格
- 差距分析与原因推断
- 基座模型差异的影响评估

**验收标准**:
- [ ] 所有评估结果汇总在一个报告中
- [ ] 明确指出哪些指标达到/接近/低于论文

**预计时间**: 1 天

---

## 六、Phase 3 — 完善与扩展（1-2 周）

### P3-1: Gemini NP Filter 替代方案 🟡 P1

**论文**: 实例分割 SA-Co 评估需要 Gemini 3.1 Flash-Lite 判断 negative phrase 是否在图中。

**替代方案**: Qwen2.5-VL-7B-Instruct

**实现**:
```python
# src/open_vision_banana/evaluation/np_filter.py
class NPFilter:
    """Negative Phrase filter using open-source VLM."""
    
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        self.model = AutoModelForVision2Seq.from_pretrained(...)
    
    def is_positive(self, image: PIL.Image, phrase: str) -> bool:
        prompt = f"Does the image contain '{phrase}'? Answer yes or no only."
        # VLM inference
        return "yes" in response.lower()
```

**验收标准**:
- [ ] 在 SA-Co val 子集上验证 filter 准确率 > 80%
- [ ] 集成到 `evaluation/instance.py`

**预计时间**: 3-5 天

---

### P3-2: Reasoning Pipeline 替代方案 🟡 P1

**论文**: ReasonSeg 需要 Gemini 2.5 Pro 将推理查询翻译为描述性引用。

**替代方案**: Qwen2.5-72B-Instruct 或 DeepSeek-V3

**实现**:
```python
# src/open_vision_banana/inference/reasoning.py
def translate_reasoning_query(query: str) -> str:
    """将推理查询翻译为描述性引用。"""
    system = "You are a visual reasoning assistant..."
    prompt = f"Translate this reasoning query into a descriptive referring expression..."
    # LLM API call
    return description
```

**验收标准**:
- [ ] 在 ReasonSeg val 子集上验证翻译质量
- [ ] 集成到 `inference/prompt.py`

**预计时间**: 2-3 天

---

### P3-3: FID/LPIPS 真实指标 🟡 P1

**当前**: `evaluation/generation_preservation.py` 使用 proxy 指标（CLIP score 等）。

**集成方案**:
- `pytorch-fid` 计算 FID
- `lpips` 计算感知距离

**验收标准**:
- [ ] T2I 生成 FID < 15（COCO 30K）
- [ ] 图像编辑 LPIPS < 0.15

**预计时间**: 2-3 天

---

### P3-4: 生成能力保留验证 🟢 P2

**论文**: GenAI-Bench win rate 53.5% vs NBP 46.5%

**我们无法做人类评估，用自动化指标替代**:
- CLIP score（文本-图像对齐）
- FID（图像质量）
- PickScore（人类偏好 proxy）

**验收标准**:
- [ ] 训练前后的 T2I 指标对比
- [ ] 报告指标退化率 < 10%

**预计时间**: 1-2 天

---

## 七、Phase 4 — 发布与文档（1 周）

### P4-1: HuggingFace 权重发布

- 上传最佳 depth/normal/multitask LoRA 权重
- 提供 Model Card（包含训练配置、评估结果、使用示例）
- 已有 `MODEL_CARD.md` 模板

### P4-2: 论文对齐报告 v2.0

- 完整的训练-评估-对比报告
- 明确声明哪些部分完全复现、哪些是推断、哪些是扩展

### P4-3: PyPI 发布准备

- 版本号升级到 0.5.0
- `pip install open-vision-bana` 可用

---

## 八、资源需求

### 计算资源

| 阶段 | GPU 需求 | 估计 GPU 小时 | 说明 |
|------|---------|--------------|------|
| Phase 1 (Depth 训练) | 1× A100 40G | ~200h | 5K steps @ 512² |
| Phase 1 (Normal 训练) | 1× A100 40G | ~200h | 可与 depth 并行 |
| Phase 1 (消融实验) | 1× A100 40G | ~300h | 3 个比例各 5K steps |
| Phase 2 (多任务训练) | 1× A100 40G | ~300h | 10K steps |
| Phase 2 (评估) | 1× A100 40G | ~50h | 6+4+1 个 benchmark |
| **总计** | | **~1050 GPU 小时** | 约 44 天单卡 / 15 天 3 卡并行 |

### 存储需求

| 项目 | 大小 |
|------|------|
| Hypersim | ~100GB |
| Virtual KITTI 2 | ~20GB |
| ScanNet (v2) | ~1.5TB |
| FLUX.1-schnell 模型 | ~23GB |
| 训练输出 (checkpoints) | ~50GB |
| **总计** | **~200GB+** |

---

## 九、风险与应对

| 风险 | 可能性 | 影响 | 应对 |
|------|--------|------|------|
| A100 资源不足 | 中 | 训练延期 | 优先 depth 单任务；用 H20 替代 |
| FLUX 基座与 NBP 表征差异大 | 中 | 指标远低于论文 | 调整预期为"验证范式"而非"精确复现" |
| 训练不收敛 | 低 | 需调参 | 已有 v5/v6/v7 收敛经验 |
| 数据集下载困难 | 中 | 准备延期 | 用轻量级子集先跑通 pipeline |
| diffusers 版本兼容性 | 中 | 训练崩溃 | 锁定 diffusers 版本；已有 MPS 兼容性处理经验 |

---

## 十、推荐的最短路径（如果资源有限）

### 场景 A: 只有 MPS (Mac Studio)

**可做**: Phase 0 修复 + Phase 1 小规模数据集训练（~500 样本）
**不可做**: 大规模 benchmark 评估（太慢）
**建议**: 用 MPS 验证代码正确性，然后迁移到 A100/H20

### 场景 B: 1× A100 40G

**最短闭环**:
1. Phase 0 修复（1 天）
2. Hypersim 5K 子集 + depth 训练 5K steps（5 天）
3. NYU val 快速评估（1 天）
4. 如果 NYU δ1 > 0.85 → 继续全量训练；否则调参

### 场景 C: 3× A100 80G

**并行策略**:
- GPU 0: Depth 训练 (Hypersim + VKITTI)
- GPU 1: Normal 训练 (Hypersim)
- GPU 2: 混合比例消融 (1:9 vs 1:19)

预计 2 周完成 Phase 1+2。

---

## 十一、即刻可执行的下一步

**今天就可以开始**:

1. **修复色图增强**（4 小时）
   ```bash
   # 修改 tuner.py _training_step，消费 batch["colormap"]
   # 修改 VisionTaskDataset，让 depth 样本携带原始深度值
   ```

2. **准备 Hypersim 数据**（启动下载，后台运行）
   ```bash
   python scripts/download_data.py --dataset hypersim --output_dir data/hypersim
   ```

3. **跑一个快速 smoke test**（验证训练 pipeline 通畅）
   ```bash
   python scripts/train.py --config configs/training_flux_quick.yaml
   ```

4. **创建对齐训练配置**（1 小时）
   ```bash
   # 基于 training_flux_depth.yaml 创建 aligned 版本
   # 设置 vision_ratio=0.1, generation_path=...
   ```

---

*方案制定: 2026-07-13*  
*对应代码版本: open-vision-banana v0.4.2 (git HEAD 1d89605)*
