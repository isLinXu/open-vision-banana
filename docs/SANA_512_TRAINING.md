# Sana_600M_512px Training — Open-Vision-Banana

使用 `Sana_600M_512px_diffusers` 在 `vb_snvc_train_v5_full` 上进行 LoRA 指令微调，复刻 Vision Banana 的方法范式：所有视觉任务都编码为 RGB 图像，由同一生成器和文本指令共同条件化，不使用任务专属预测头。

## 文件清单

| 文件 | 说明 |
|------|------|
| `scripts/train_sana_512.py` | 主训练脚本 |
| `configs/sana_512_train.yaml` | v5_full 多任务复现实验配置（CLI 参数优先） |
| `output/mar_vb_clean_pairs_v4/pairs.jsonl` | 标注驱动、无 source/target 泄漏的 v5 训练/验证配对清单 |
| `src/open_vision_banana/model_registry.py` | 已新增 `sana-600m-512px` 模型注册 |

## 模型架构

| 组件 | 规格 |
|------|------|
| Backbone | SanaTransformer (Linear DiT, ~590M params) |
| VAE | AutoencoderDC (DC-AE f32c32, 32x downsample, 32 latent channels) |
| Text Encoder | Gemma-2-2B-IT (max_length=300, dim=2048) |
| Resolution | 512x512 |
| Scheduler | Flow matching (linear_flow, flow_shift=3.0) |
| Latent shape | [B, 32, 16, 16] (512/32=16) |

## 快速开始

### 前置条件

```bash
# 确保安装了 diffusers (支持 SanaPipeline) 和 peft
pip install diffusers peft transformers accelerate
```

### 基本训练

```bash
# v5_full 统一多任务方法复现（默认 10% vision / 90% generation retention）
python scripts/train_sana_512.py --config configs/sana_512_train.yaml
```

`v5_full` 是扁平化图片目录，不能按同名文件猜测 source/target 关系。默认配置使用 `output/mar_vb_clean_pairs_v4/pairs.jsonl`，其中包含 25,476 对已审计样本：depth 6,114、normal 4,991、semantic 9,709、structure 4,662；划分按输入图片分组，避免跨任务泄漏。

生成保持集位于 `data/vb_generation_5k`，可读取仓库已有的 `captions.json`。训练中视觉样本走 `source image -> task RGB target` 的 img2img flow，生成保持样本走 `noise -> original image` 的 text-to-image flow；后者不会错误退化为原图到原图的零速度目标。

### 单任务调试

```bash
# 仅训练 depth 任务
python scripts/train_sana_512.py \
    --data-dir /path/to/vb_snvc_train_v5_full \
    --pair-manifest output/mar_vb_clean_pairs_v4/pairs.jsonl \
    --tasks depth \
    --steps 2000 \
    --output-dir output/sana_512_depth_debug
```

### 带 Generation 数据混合（vision:generation = 1:9）

```bash
python scripts/train_sana_512.py \
    --data-dir /path/to/vb_snvc_train_v5_full \
    --pair-manifest output/mar_vb_clean_pairs_v4/pairs.jsonl \
    --generation-dir /path/to/vb_generation_5k \
    --vision-ratio 0.1 \
    --output-dir output/sana_512_mixed
```

### 断点续训

```bash
python scripts/train_sana_512.py \
    --data-dir /path/to/vb_snvc_train_v5_full \
    --pair-manifest output/mar_vb_clean_pairs_v4/pairs.jsonl \
    --resume output/sana_512_vb/checkpoint-5000/lora.pt \
    --steps 10000 \
    --output-dir output/sana_512_vb
```

## 关键参数

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-name` | `sana-600m-512px` | 模型注册名 |
| `--lora-rank` | 32 | LoRA 秩 |
| `--lora-alpha` | 64 | LoRA alpha |
| `--no-rslora` | false | 禁用 rsLoRA scaling |
| `--dtype` | float16 | transformer 精度（text_encoder/VAE 自动转 bf16） |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--steps` | 10000 | 总训练步数 |
| `--lr` | 5e-5 | 学习率 |
| `--batch-size` | 1 | 每步 batch size |
| `--grad-accum` | 8 | 梯度累积步数（effective batch = 8） |
| `--resolution` | 512 | 训练分辨率 |
| `--task-weights` | sqrt | 任务采样权重（sqrt/uniform/逗号分隔权重） |

### Flow Matching 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--no-img2img-flow` | false | 禁用 img2img flow（使用 noise->target 代替 input->target） |
| `--flow-sigma` | 0.0 | Flow 噪声 sigma（0 = 纯 flow） |
| `--no-logit-normal-timesteps` | false | 使用均匀分布采样时间步 |
| `--no-min-snr` | false | 禁用 Min-SNR 损失加权 |
| `--cfg-dropout-rate` | 0.1 | CFG dropout 率 |

### 设备适配

脚本自动检测设备（CUDA > MPS > CPU），并根据设备调整：

- **CUDA**: 支持 AMP 混合精度，`pin_memory=True`
- **MPS**: bfloat16 autocast，`num_workers=0`
- **CPU**: 禁用 AMP，`num_workers=0`

## 数据集

默认使用 `vb_snvc_train_v5_full` 及其 annotation-backed v4 配对清单：

| 任务 | 训练样本 | 验证样本 |
|------|----------|----------|
| depth | 5,813 | 301 |
| normal | 4,740 | 251 |
| semantic segmentation | 9,215 | 494 |
| structure | 4,424 | 238 |
| **总计** | **24,192** | **1,284** |

生成保持集单独提供 1,968 张有 caption 的图像。v5 输入/目标配对仅由 JSONL manifest 提供，不允许从 `images/` 与 `targets/` 目录按文件名自行匹配。

## 训练流程

1. **模型加载**: 从 HuggingFace 下载 `Sana_600M_512px_diffusers`，提取 transformer/VAE/text_encoder
2. **LoRA 注入**: 对 transformer 的 `linear`、`proj_out`、`conv_inverted`、`conv_point` 层注入 LoRA 适配器
3. **数据加载**: 以统一权重采样 4 类 RGB 视觉任务；外层按 1:9 混入生成保持样本
4. **Flow Matching**:
   - VAE 编码 target 和 vision input 图像到 latent 空间
   - 采样 logit-normal 时间步
   - 视觉样本构建 source-image 到 task-target 的 flow；生成样本构建 noise 到原图的 flow
   - 两种样本共享 SANA transformer、LoRA 和 prompt 条件
5. **损失计算**: MSE loss + 可选 Min-SNR 加权
6. **优化**: AdamW + cosine LR scheduler + 梯度裁剪 + EMA

## Checkpoint 输出

```
output/sana_v5_vision_banana/
├── training_config.json          # 训练配置
├── checkpoint-1000/
│   ├── lora.pt                   # LoRA 权重 + optimizer/scheduler state
│   ├── lora_adapter/             # PEFT 格式 LoRA adapter
│   └── metadata.json             # 步数、loss 等元信息
├── checkpoint-2000/
│   ...
├── checkpoint-10000_best/        # 最优验证 loss 的 checkpoint
└── checkpoint-10000_final/       # 最终 checkpoint
```

## 注意事项

1. **网络要求**: 首次运行需从 HuggingFace 下载模型（~2.5GB），设置了 HF 镜像 `hf-mirror.com`
2. **显存要求**: 建议至少 8GB VRAM（fp16 + gradient checkpointing + LoRA rank=16）
3. **DC-AE VAE**: Sana 使用 32x 下采样的 DC-AE（非标准 SD VAE 的 8x），latent shape 为 [B, 32, 16, 16]
4. **Gemma-2B 文本编码器**: 文本编码器较大（~2B 参数），编码时临时转为 float32 保证数值稳定性
5. **gradient checkpointing**: 默认开启以节省显存，可通过 `--no-gradient-checkpointing` 关闭（实际该脚本默认 True）
