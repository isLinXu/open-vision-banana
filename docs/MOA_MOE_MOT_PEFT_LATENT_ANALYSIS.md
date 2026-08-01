# Open-Vision-Banana: MOA / MOE / MOT / PEFT / Latent MOE 深度架构分析

> 分析日期: 2026-07-27
> 项目版本: v0.4.2
> 分析范围: Mixture-of-Adapters (MOA), Mixture-of-Experts (MOE), Mixture-of-Tasks/Tokens (MOT), Parameter-Efficient Fine-Tuning (PEFT) 及 Planner, Latent MOE
> 代码库: `/Users/gatilin/PycharmProjects/open-vision-banana-git-v260722`

---

## 0. 一句话结论

Open-Vision-Banana 项目的核心训练范式是 **PEFT (rsLoRA)**, 其实现深度和工程成熟度远超一般开源项目。然而, 用户关注的 **MOA、MOT、Latent MOE 均未实现**, **MOE 仅作为外部模型骨架的描述性引用存在** (NucleusMoE-Image / HiDream-I1), 项目内未自建任何 Mixture 路由架构。本文档逐一分析这五个维度的现状、代码证据、设计空间和推荐路径。

---

## 1. PEFT (Parameter-Efficient Fine-Tuning) — 核心训练范式

### 1.1 架构概述

PEFT 是本项目最深度集成的架构概念, 是整个训练管线的基石。项目使用 HuggingFace `peft` 库的 LoRA 实现, 并在此基础上做了大量工程加固。

**核心论文对齐参数:**

| 参数 | 论文值 | 代码默认值 | 文件位置 |
|------|--------|-----------|---------|
| LoRA rank | 64 | 64 (tuner) / 128 (train_sdxl_mps) | `config.py:111`, `tuner.py:217` |
| LoRA alpha | 128 | 128 (tuner) / 256 (train_sdxl_mps) | `config.py:112`, `tuner.py:218` |
| rsLoRA scaling | alpha / sqrt(rank) | True | `config.py:113`, `tuner.py:220` |
| 可训练参数比例 | 0.5%-1% | ~0.5% (rank=64 on FLUX 12B) | `tuner.py:745` |
| 数据混合比例 | 9:1 (gen:vision) | 0.1 vision_ratio | `config.py:175` |

### 1.2 核心实现: `InstructionTuner`

**文件**: `src/open_vision_banana/training/tuner.py` (2394 行)

```python
class InstructionTuner:
    """LoRA-based instruction alignment for Vision Banana.

    Applies rsLoRA scaling (rank=64, alpha=128) to prevent learning rate
    decay at high ranks. Trains only 0.5%-1% of parameters while preserving
    90%+ generation capability.
    """
```

**关键方法链路:**

```
InstructionTuner.__init__()
  ├── _setup_lora()              # 创建 LoraConfig + get_peft_model
  │   ├── expand_unet_conv_in()  # (可选) 扩展 conv_in 4→8 通道
  │   ├── get_peft_model()       # 包装 backbone
  │   └── cast LoRA params → base dtype  # 修复 MPS 混精度
  ├── _probe_forward_signature() # 初始化时探测 forward 签名
  ├── _detect_peft_misroute()    # 检测 PEFT 包装器路由错误
  └── create_source_spatial_adapter()  # (可选) T2I-Adapter
```

#### 1.2.1 LoRA 配置与目标模块

**文件**: `src/open_vision_banana/config.py:105-136`

```python
class LoRAConfig:
    def __init__(self, rank=64, alpha=128, rs_lora=True,
                 target_modules=None, dropout=0.0):
```

**模型专属 LoRA 目标模块** (29 个注册模型, 每个有独立的 `lora_target_modules`):

| 模型族 | LoRA 目标模块 | 特点 |
|--------|-------------|------|
| FLUX | `to_q, to_k, to_v, to_out.0, proj_in, proj_out, ff.net.0.proj, ff.net.2` | Transformer attention + FFN |
| SDXL/SSD | 同上 | UNet attention + FFN |
| NucleusMoE | 上述 + `gate_up_proj, down_proj` | **MoE 门控层** (唯一涉及 MoE 的 LoRA 目标) |
| Qwen-Image | `to_q, to_k, to_v, to_out.0, linear_1, linear_2` | 无 FFN proj |
| Lumina | `to_q, to_k, to_v, to_out.0, linear_1, linear_2, adaLN_modulation` | 含 adaLN |

#### 1.2.2 rsLoRA 缩放

**文件**: `tuner.py:705-761`

```python
def _setup_lora(self):
    lora_config = LoraConfig(
        r=self.lora_rank,           # 64
        lora_alpha=self.lora_alpha,  # 128
        target_modules=self.target_modules,
        use_rslora=self.use_rslora,  # True
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    self.peft_model = get_peft_model(backbone, lora_config)
```

rsLoRA 的缩放公式: `scale = alpha / sqrt(rank)` (而非标准 LoRA 的 `alpha / rank`), 这防止了高 rank 下学习率衰减, 是论文的关键贡献之一。

#### 1.2.3 PEFT Mis-Route 检测 (P0-7 修复)

**文件**: `tuner.py:586-671`

这是项目中工程量最大的 PEFT 适配, 解决了 diffusers PEFT 集成的一个已知 bug:

```python
def _detect_peft_misroute(self) -> bool:
    """Detect whether the PEFT wrapper mis-routes forward() calls.

    diffusers' PEFT integration (peft_utils.py) sometimes captures the
    wrong forward_fn — e.g. T5EncoderModel.forward instead of
    FluxTransformer2DModel.forward. When this happens, calling the
    PEFT-wrapped model with FLUX kwargs raises TypeError mentioning 'input_ids'.
    """
```

**检测逻辑:**
1. 初始化时通过 `inspect.signature` 探测 PEFT 包装器的 forward 签名
2. 如果 forward 接受 `input_ids` 但不接受 `hidden_states` → 判定 mis-route
3. 如果包装器有 `**kwargs` (不透明), 做一次 trial call 验证

**绕过策略:**

```python
def _direct_forward(self, forward_kwargs: dict) -> torch.Tensor:
    """Call the underlying transformer's forward directly, bypassing PEFT wrapper.

    Manually applies LoRA scale/unscale around the raw forward call.
    """
    transformer = self.peft_model.base_model.model
    scale_lora_layers(transformer, 1.0)
    output = transformer(**forward_kwargs)
    unscale_lora_layers(transformer, 1.0)
    return output
```

#### 1.2.4 LoRA 手动合并 (推理)

**文件**: `src/open_vision_banana/inference/generator.py:891-1016`

当 `pipe.load_lora_weights()` 触发 PEFT mis-route 时, 项目实现了一套完整的 LoRA 手动合并 fallback:

```python
def _load_and_merge_lora(self, pipe, lora_path):
    # 直接读取 adapter_model.safetensors
    # 计算 scale = alpha / sqrt(rank) (rsLoRA) 或 alpha / rank (标准)
    # 对每个目标模块执行: W += scale * (B @ A)
    # 支持 2D (Linear) 和 4D (Conv2d 1x1) 两种 LoRA 因子
```

#### 1.2.5 PEFT 检查点管理

**文件**: `src/open_vision_banana/training/checkpoint.py`

- 保存: `adapter_model.safetensors` + `adapter_config.json` (PEFT 标准格式)
- 加载: 通过 `peft.PeftModel` 或直接 safetensors 重映射
- 架构签名: `architecture_signature()` 确保 resume 时模型结构兼容

### 1.3 训练后端 (5 种)

**文件**: `src/open_vision_banana/training/trainer_interface.py` (665 行)

| 后端 | 类 | 适用场景 |
|------|---|--------|
| Native | `NativeTrainer` | 单 GPU/CPU + AMP |
| DDP | `DDPTrainer(NativeTrainer)` | 多 GPU 数据并行 |
| FSDP | `FSDPTrainer(NativeTrainer)` | 多 GPU 全分片 |
| DeepSpeed | `DeepSpeedTrainer(NativeTrainer)` | ZeRO-3 |
| Accelerate | `AccelerateTrainer(TrainerInterface)` | HF Accelerate |

所有后端共享同一个 LoRA 训练循环 (`_training_step`), 仅在 backward / optimizer_step / grad scaling 层面不同。

### 1.4 Source-Conditioning (可选增强)

**文件**: `src/open_vision_banana/conditioning.py` (339 行)

这是 PEFT 之上的一个可选架构扩展, 采用 InstructPix2Pix 风格的通道拼接:

```python
class SourceConditioningConfig:
    mode: str = "source_latent_concat"
    in_channels: int = 8        # 4 noisy target + 4 source image
    condition_channels: int = 4
    prediction_type: str = "epsilon"
```

**关键函数 `expand_unet_conv_in()`:**
- 将 UNet 的 `conv_in` 从 4 通道扩展到 8 通道
- 新增的 source 通道零初始化 (ControlNet 风格), 保证扩展后初始输出不变
- 仅对 SDXL/SSD 家族可用

### 1.5 T2I-Adapter (空间适配器)

**文件**: `src/open_vision_banana/spatial_adapter.py` (83 行)

```python
class SourceSpatialAdapter(nn.Module):
    """T2I-Adapter-XL with ControlNet-style zero output projections."""
```

- 多尺度空间适配器: channels = (320, 640, 1280, 1280)
- 零初始化输出投影 (ControlNet 风格)
- 下采样因子 16 (pixel → adapter input)
- 残差通过 `down_intrablock_additional_residuals` 注入 UNet

**这是一个单一空间适配器, 不是 Mixture-of-Adapters。**

### 1.6 多任务训练 (数据层面)

**文件**: `src/open_vision_banana/data/mixed.py` (190 行)

```python
class MixedVisionDataset(Dataset):
    """Mixed dataset implementing a configurable vision-to-generation ratio."""
```

- 论文 9:1 混合比例 (generation : vision)
- 任务级加权采样: 支持 `"sqrt"` (1/sqrt(N)) 和 `"uniform"` (等权) 模式
- 多色图增强 (paper Section 3.2): 深度任务随机切换 colormap

**文件**: `scripts/train_sdxl_mps.py:53-100`

```python
class TaskScheduleSampler(torch.utils.data.Sampler[int]):
    """Deterministic sampler that guarantees task coverage in each cycle."""
```

轮询采样器确保每个周期覆盖所有任务, 避免 epoch 内任务遗漏。

### 1.7 PEFT 小结

| 维度 | 状态 | 成熟度 |
|------|------|--------|
| rsLoRA 核心 | 已实现 | 生产级 |
| 多模型 LoRA 目标 | 29 个注册模型 | 生产级 |
| PEFT mis-route 检测/绕过 | 已实现 | 工程级 (解决 diffusers bug) |
| LoRA 手动合并 | 已实现 | 工程级 (推理 fallback) |
| 检查点管理 | 已实现 | 生产级 |
| 5 种训练后端 | 已实现 | 生产级 |
| Source-conditioning | 已实现 | 实验级 (SSD only, depth only) |
| T2I-Adapter | 已实现 | 实验级 (零初始化, 需 --no-ema) |

---

## 2. MOE (Mixture of Experts) — 外部模型引用

### 2.1 现状: 描述性引用, 非自建架构

项目中 **未实现任何 MOE 架构**。MOE 仅出现在两个外部模型的注册描述中:

#### 2.1.1 NucleusMoE-Image

**文件**: `src/open_vision_banana/model_registry.py:808-835`

```python
_register_model("nucleusmoe-image", lambda: ModelConfig(
    name="nucleusmoe-image",
    model_id="NucleusAI/NucleusMoE-Image",
    family="nucleusmoe-image",
    description="NucleusMoE-Image — NucleusAI, Qwen3-VL encoder, MoE DiT",
    lora_target_modules=[
        "to_q", "to_k", "to_v", "to_out.0",
        "proj_in", "proj_out",
        "linear_1", "linear_2",
        "gate_up_proj", "down_proj",  # ← MoE 门控/专家投影层
    ],
))
```

#### 2.1.2 HiDream-I1

**文件**: `src/open_vision_banana/model_registry.py:866-894`

```python
_register_model("hidream-image", lambda: ModelConfig(
    description="HiDream-I1 — HiDream-ai, 4-encoder (CLIP×2+T5+Llama3), MoE DiT",
))
```

### 2.2 LoRA 对 MoE 层的适配

当使用 NucleusMoE-Image 作为骨干时, LoRA 会作用于 `gate_up_proj` 和 `down_proj`:

```
gate_up_proj → MoE 门控上投影 (gating + up projection)
down_proj    → MoE 专家下投影 (expert down projection)
```

这意味着项目的 LoRA 微调 **可以** 影响 MoE 模型的路由决策和专家输出, 但 MoE 架构本身由外部 diffusers pipeline 提供, 本项目不控制路由策略、专家数量或负载均衡。

### 2.3 NucleusMoE 特殊处理

**文件**: `model_registry.py:1125, 1254, 1541-1559, 1997-2000`

项目对 NucleusMoE 做了若干模型专属适配:

| 适配点 | 代码位置 | 内容 |
|--------|---------|------|
| Processor 替代 Tokenizer | `1125` | `NucleusMoE/LongCat use processor instead of tokenizer` |
| Pipeline 映射 | `1178` | `"NucleusMoEImagePipeline": "diffusers.pipelines.nucleusmoe_image"` |
| 文本编码 | `1541-1559` | Qwen3-VL processor 编码, `hidden_states` 在特定索引 |
| Backbone forward | `1997-2000` | `transformer(hidden_states, img_shapes, encoder_hidden_states, encoder_mask, timestep)` |

### 2.4 MOE 小结

| 维度 | 状态 |
|------|------|
| 自建 MoE 架构 | 未实现 |
| MoE 路由器/门控网络 | 未实现 |
| MoE 负载均衡损失 | 未实现 |
| 外部 MoE 模型支持 | 已注册 (NucleusMoE, HiDream) |
| LoRA 作用于 MoE 层 | 已实现 (gate_up_proj, down_proj) |

---

## 3. MOA (Mixture of Adapters) — 未实现

### 3.1 现状

项目中 **不存在任何 Mixture-of-Adapters 架构**。全文搜索 `moa`、`mixture_of_adapter`、`mixture_of_attention` 返回零匹配。

### 3.2 当前 Adapter 架构

项目使用 **单一统一 LoRA adapter** 策略:

- 每个训练任务 (或多个任务联合) 训练一个 LoRA checkpoint
- 推理时加载单个 LoRA adapter (或手动合并到基座)
- 不存在多个 adapter 的动态路由/混合机制

**已实现的 adapter 类型:**

| 类型 | 文件 | 性质 |
|------|------|------|
| LoRA adapter | `tuner.py` | PEFT 参数高效适配 (核心) |
| Source Spatial Adapter | `spatial_adapter.py` | T2I-Adapter-XL 空间条件 (单一, 非混合) |
| Source-Latent Conditioning | `conditioning.py` | InstructPix2Pix 通道拼接 (非 adapter) |

### 3.3 与 MOA 的差距

MOA 的典型实现需要:
1. 多个并行 adapter (每个任务/域一个)
2. 路由网络 (基于输入 token / 任务 ID 选择 adapter)
3. 加权组合多个 adapter 输出

项目当前缺少上述全部组件。多任务能力通过 **数据混合** 而非 **adapter 路由** 实现。

---

## 4. MOT (Mixture of Tasks / Mixture of Tokens) — 未实现

### 4.1 现状

项目中 **不存在任何 Mixture-of-Tasks 或 Mixture-of-Tokens 架构模块**。没有任务路由器、token 级别路由或任务条件化注意力。

### 4.2 当前的多任务实现

多任务通过 **数据层面** 而非 **架构层面** 实现:

#### 4.2.1 数据混合

**文件**: `src/open_vision_banana/data/mixed.py`

```python
class MixedVisionDataset(Dataset):
    # 9:1 generation:vision 混合
    # 任务级加权: "sqrt" / "uniform" / 自定义权重
    # 多色图增强: 深度任务随机切换 colormap
```

#### 4.2.2 任务调度采样

**文件**: `scripts/train_sdxl_mps.py:53-100`

```python
class TaskScheduleSampler(torch.utils.data.Sampler[int]):
    """Deterministic sampler that guarantees task coverage in each cycle."""
    # 轮询遍历所有任务, 确保每个周期覆盖全部任务
```

#### 4.2.3 多任务 LoRA 训练脚本

**文件**: `scripts/train_multitask_lora.py` — 审计过的五任务 LoRA runner

### 4.3 与 MOT 的差距

| MOT 特征 | 项目是否具备 |
|---------|------------|
| 任务路由网络 | 否 |
| Token 级路由 | 否 |
| 任务条件化 attention | 否 (仅通过 prompt 文本条件化) |
| 任务专用 FFN/MLP | 否 (LoRA 是共享的, 非任务专属) |
| 负载均衡损失 | 否 |

当前的多任务能力完全依赖 **prompt 条件化** (文本指令区分任务) + **数据混合** (9:1 比例), 这是论文 Vision Banana 的原始设计, 而非 MOT 架构。

---

## 5. Latent MOE (Latent Mixture of Experts) — 未实现

### 5.1 现状

项目中 **不存在任何 Latent MOE 架构**。全文搜索 `latent_moe`、`latent_expert` 返回零匹配。

### 5.2 "Latent" 在项目中的含义

"Latent" 在代码中出现 105+ 次, 全部指 **VAE 潜空间** 操作, 与 Latent MOE 无关:

| 文件 | 用途 |
|------|------|
| `training/latent_cache.py` | `LatentCacheDataset` — VAE 编码后的潜变量缓存 |
| `conditioning.py` | `build_source_conditioned_input()` — 潜空间噪声/拼接 |
| `tuner.py` | `_encode_latents()`, `_pack_latents()`, `_unpack_latents()` — FLUX 潜空间打包 |
| `inference/generator.py` | `_generate_source_conditioned()` — 从纯噪声去噪 |

### 5.3 Latent MOE 的设计空间

Latent MOE (如 MoLE、L-MoE) 通常指在 VAE 潜空间中部署专家路由, 而非在 token/attention 层面。其核心思路:

1. 将潜变量分块 (如空间网格)
2. 每个块通过路由器分配到不同专家
3. 专家在潜空间中处理, 输出合并后送入下一层

项目当前的 VAE 潜空间操作仅包含编码/解码/加噪/拼接, 不存在专家分治。

---

## 6. Planner — 未实现

### 6.1 现状

全文搜索 `planner` 返回零匹配。项目不存在任何名为 "planner" 的模块。

### 6.2 相关的"规划"概念

虽然没有名为 planner 的模块, 项目中存在若干与"规划"相关的概念:

| 概念 | 文件 | 性质 |
|------|------|------|
| Overfit gate | `scripts/overfit_gate_depth.py` | 验证门控 (SSIM > 0.80, L1 < 0.10) |
| Geometry gate | `docs/adr/0001` | 深度几何学习门控 |
| Coverage smoke gate | `train_sdxl_mps.py:53` | 任务覆盖采样器 |
| Promotion gate | `docs/plans/frozen-feature-*` | 冻结特征路径晋升门控 |
| Task schedule | `train_sdxl_mps.py:53` | 确定性任务轮询 |

这些是 **验证/调度** 机制, 非架构层面的 planner。

---

## 7. 架构关系图

```
┌─────────────────────────────────────────────────────────────────┐
│                    Open-Vision-Banana 架构                       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    PEFT (rsLoRA)  ← 已深度实现             │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐    │   │
│  │  │ LoRA Config │  │ Mis-Route    │  │ Manual Merge   │    │   │
│  │  │ rank=64     │  │ Detection    │  │ (Inference)    │    │   │
│  │  │ alpha=128   │  │ & Bypass     │  │ W += s*(B@A)  │    │   │
│  │  │ rsLoRA      │  └──────────────┘  └────────────────┘    │   │
│  │  └─────────────┘                                         │   │
│  │  ┌─────────────────────────────────────────────────┐      │   │
│  │  │        5 Training Backends                        │      │   │
│  │  │  Native | DDP | FSDP | DeepSpeed | Accelerate  │      │   │
│  │  └─────────────────────────────────────────────────┘      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────────────────────┐     │
│  │ Source-Condition │  │ T2I-Adapter-XL                   │     │
│  │ (InstructPix2Pix)│  │ (Zero-init Spatial Adapter)      │     │
│  │ conv_in 4→8 ch   │  │ Multi-scale residual injection   │     │
│  └──────────────────┘  └──────────────────────────────────┘     │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              多任务 (数据层面, 非架构 MOT)                  │   │
│  │  MixedVisionDataset (9:1 gen:vision)                     │   │
│  │  TaskScheduleSampler (round-robin)                       │   │
│  │  Task weights: sqrt / uniform / custom                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              MOE (仅外部引用, 非自建)                     │   │
│  │  NucleusMoE-Image → LoRA targets: gate_up_proj, down_proj│   │
│  │  HiDream-I1 → "MoE DiT" (描述性)                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐     │
│  │ MOA        │  │ Latent MOE │  │ Planner                │     │
│  │ ❌ 未实现  │  │ ❌ 未实现  │  │ ❌ 未实现              │     │
│  └────────────┘  └────────────┘  └────────────────────────┘     │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Latent (VAE 潜空间, 非 Latent MOE)            │   │
│  │  LatentCacheDataset · _encode_latents · _pack_latents    │   │
│  │  build_source_conditioned_input · VAE encode/decode      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. 核心代码文件索引

| 概念 | 核心文件 | 行数 | 关键类/函数 |
|------|---------|------|------------|
| PEFT 核心 | `training/tuner.py` | 2394 | `InstructionTuner._setup_lora()`, `_detect_peft_misroute()`, `_direct_forward()` |
| PEFT 配置 | `config.py` | 401 | `LoRAConfig`, `VisionBananaConfig` |
| PEFT 检查点 | `training/checkpoint.py` | - | `CheckpointManager` |
| PEFT 推理合并 | `inference/generator.py` | 1026 | `BaseGenerator._load_and_merge_lora()` |
| 训练后端 | `training/trainer_interface.py` | 665 | `TrainerInterface`, `NativeTrainer`, `FSDPTrainer`, `DeepSpeedTrainer` |
| Source Conditioning | `conditioning.py` | 339 | `SourceConditioningConfig`, `expand_unet_conv_in()`, `build_source_conditioned_input()` |
| T2I-Adapter | `spatial_adapter.py` | 83 | `SourceSpatialAdapter` |
| 多任务数据 | `data/mixed.py` | 190 | `MixedVisionDataset` |
| 任务采样 | `scripts/train_sdxl_mps.py` | 962 | `TaskScheduleSampler` |
| MOE (外部) | `model_registry.py` | 2226 | `_register_model("nucleusmoe-image")`, `_register_model("hidream-image")` |
| VAE Latent | `training/latent_cache.py` | 123 | `LatentCacheDataset` |
| 深度探测 | `training/depth_probe.py` | 514 | `ScalarDepthProbe`, `FrozenSDXLDepthFeatureExtractor` |
| 结构损失 | `training/structure_loss.py` | 247 | `decoded_x0_structure_loss()`, `decoded_x0_scalar_geometry_loss()` |
| 优化器 | `training/optimization.py` | 520 | `SchedulerFactory`, `MemoryManager`, `compute_min_snr_weight()` |

---

## 9. 差距分析与推荐路径

### 9.1 现状总结

| 概念 | 实现状态 | 等效替代 |
|------|---------|---------|
| **PEFT** | 深度实现 (rsLoRA + 5 backends + mis-route fix + manual merge) | — (即核心) |
| **MOE** | 仅外部模型引用 (NucleusMoE, HiDream) | LoRA 作用于 MoE 门控层 |
| **MOA** | 未实现 | 单一 LoRA adapter + 数据混合 |
| **MOT** | 未实现 | Prompt 条件化 + MixedVisionDataset |
| **Latent MOE** | 未实现 | VAE latent space encode/decode (非 MoE) |
| **Planner** | 未实现 | Gate 验证机制 (overfit/geometry/promotion) |

### 9.2 推荐演进路径

#### Phase 1: MOA (Mixture of Adapters) — 低风险增量

**目标**: 每个视觉任务训练独立 LoRA adapter, 推理时动态路由。

**实现要点**:
1. 复用现有 `InstructionTuner` 训练 per-task LoRA (已有基础设施)
2. 新增 `AdapterRouter` 模块: 基于 prompt embedding 做软路由
3. 推理时加权合并: `W_effective = sum(w_i * W_i)` (已有手动合并代码可复用)
4. 评估: 对比单 adapter vs MOA 在多任务 bench 上的精度/生成保留

**代码改动量**: 中等 (~500 行新增)
**风险**: 低 (不改变训练流程, 仅推理路由)

#### Phase 2: MOT (Mixture of Tasks) — 中风险架构

**目标**: 在 Transformer 内部引入任务路由, 而非仅靠 prompt 区分。

**实现要点**:
1. 任务 token 嵌入: 在 prompt embedding 后追加 task embedding
2. 任务条件化 FFN: 每个任务有独立 FFN (或 LoRA 分支)
3. 路由网络: 轻量 MLP 基于全局 context 选择 FFN
4. 负载均衡损失: 确保各任务被均匀路由

**代码改动量**: 高 (~1500 行新增)
**风险**: 中 (改变模型前向, 需重新训练)
**参考**: TaskMoE, MoE-LLaVA

#### Phase 3: Latent MOE — 高风险研究

**目标**: 在 VAE 潜空间中部署专家分治, 而非全图统一处理。

**实现要点**:
1. 潜变量空间分块 (spatial patches)
2. 每个块通过路由器分配到 K 个专家
3. 专家是轻量 Conv/MLP (非完整 Transformer)
4. 输出合并后送入主 Transformer

**代码改动量**: 高 (~2000 行新增)
**风险**: 高 (改变 latent 表示, 可能破坏 VAE 解码)
**参考**: Latent Diffusion MoE, Switch-Diffusion

#### Phase 4: MOE 自建 — 高风险架构

**目标**: 在项目骨干 Transformer 中自建 MoE 层, 而非依赖外部模型。

**实现要点**:
1. 将 FFN 层替换为 MoE FFN (top-k routing)
2. 路由器: token-level gating network
3. 专家: N 个并行 FFN
4. 负载均衡: auxiliary loss
5. 训练: LoRA 作用于所有专家的 attention + FFN

**代码改动量**: 极高 (~3000 行新增)
**风险**: 高 (从头训练 MoE Transformer, 需大量算力)
**替代**: 继续使用已注册的 NucleusMoE-Image / HiDream 作为 MoE 骨干

---

## 10. 关键设计决策记录

### 10.1 为什么选择单一 LoRA 而非 MOA?

论文 Vision Banana 的核心论点是 **不需要任务专属架构**: 通过 RGB 参数化 + prompt 条件化, 一个统一模型即可处理所有视觉任务。因此:
- 单一 LoRA adapter 足以实现多任务 (通过 prompt 区分)
- 多 adapter 路由 (MOA) 增加了推理复杂度, 与"轻量指令微调"理念冲突
- 数据混合 (9:1) 保留了生成能力, 不需要架构层面的任务隔离

### 10.2 为什么不自建 MOE?

- 项目定位是 **开源实现论文**, 而非自研新架构
- MoE Transformer 需要大量预训练算力, 超出微调项目范围
- 外部 MoE 模型 (NucleusMoE, HiDream) 已可通过 LoRA 适配, 无需自建

### 10.3 为什么用数据混合而非 MOT 架构?

- 论文 Vision Banana 的方法就是 **9:1 数据混合 + LoRA**, 不是 MOT
- Prompt 条件化已提供了任务区分能力
- MOT 架构增加参数量和训练复杂度, 且与论文方法不一致

---

## 附录 A: rsLoRA 缩放数学推导

标准 LoRA:

```
W' = W + (alpha / rank) * B @ A
```

rsLoRA (rank-stabilized LoRA):

```
W' = W + (alpha / sqrt(rank)) * B @ A
```

**为什么 rsLoRA 更好?**

在高 rank (如 rank=64) 时:
- 标准 LoRA: scale = 128/64 = 2.0
- rsLoRA: scale = 128/sqrt(64) = 16.0

rsLoRA 的缩放因子随 rank 增长更慢, 防止了高 rank 下梯度爆炸和学习率有效衰减。代码中 (`tuner.py:712`):

```python
lora_config = LoraConfig(
    r=self.lora_rank,
    lora_alpha=self.lora_alpha,
    use_rslora=self.use_rslora,  # True → alpha/sqrt(rank)
    ...
)
```

推理手动合并 (`generator.py:925`):

```python
scale = lora_alpha / lora_rank ** 0.5 if use_rslora else lora_alpha / lora_rank
```

---

## 附录 B: PEFT Mis-Route 问题详解

### 问题描述

diffusers 的 PEFT 集成 (`peft_utils.py`) 在某些情况下会捕获错误的 `forward_fn`:

```
预期: peft_model.forward → FluxTransformer2DModel.forward(hidden_states, timestep, ...)
实际: peft_model.forward → T5EncoderModel.forward(input_ids, ...)
```

当用 FLUX 参数 (`hidden_states=...`) 调用时, T5 的 forward 报错: `TypeError: ... 'input_ids'`

### 项目解决方案

1. **初始化时探测** (`_detect_peft_misroute`): 通过 `inspect.signature` 检查 forward 是否接受 `hidden_states`
2. **绕过 PEFT 包装器** (`_direct_forward`): 直接调用底层 transformer, 手动 scale/unscale LoRA
3. **运行时 fallback**: 即使初始化未检测到, 运行时遇到 `input_ids` 错误也自动切换

```python
# tuner.py:2118-2135
if self._use_direct_forward:
    predicted_velocity = self._direct_forward(forward_kwargs)
else:
    try:
        predicted_velocity = self.peft_model(**forward_kwargs)
    except TypeError as e:
        if "input_ids" in str(e):
            self._use_direct_forward = True
            predicted_velocity = self._direct_forward(forward_kwargs)
```

---

## 附录 C: 训练损失函数体系

### Flow-Matching (FLUX 家族)

```
loss = MSE(v_theta(z_t, t, c), z_0 - epsilon)
```

其中:
- `z_0 = VAE.encode(target_rgb)`
- `z_t = (1-t)*z_0 + t*epsilon`
- `v* = z_0 - epsilon`
- `t ~ LogitNormal(0, 1)`

**Min-SNR 加权**: `weight = min(SNR, gamma) / (SNR + 1)`, gamma=5.0

### SDXL Epsilon Prediction

```
target = noise (epsilon prediction)
loss = MSE(pred, noise) * min_snr_weight
```

可选辅助损失:
- **x0 loss**: `weight * MSE(sigma * pred, z_0 - noisy)` (implied x0)
- **pixel L1**: 解码 x0 后做像素级 L1
- **structure loss**: 多尺度值/梯度 Charbonnier loss
- **scalar geometry loss**: 色图投影后的标量深度位置/梯度/序数 loss

---

*本文档基于 v0.4.2 代码库完整分析生成, 涵盖 15,500+ 行 Python 源码和 40+ 文档文件。*
