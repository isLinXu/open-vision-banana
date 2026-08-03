# Sana_600M_512px Training — Open-Vision-Banana

使用 `Sana_600M_512px_diffusers` 模型在 `vb_snvc_train_v3` 数据集上进行 LoRA 指令微调。

## 文件清单

| 文件 | 说明 |
|------|------|
| `scripts/train_sana_512.py` | 主训练脚本 |
| `configs/sana_512_train.yaml` | 训练配置文件（参考用，CLI 参数优先） |
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
# 全任务训练（12个视觉任务）
python scripts/train_sana_512.py \
    --data-dir /Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v3 \
    --steps 10000 \
    --lr 5e-5 \
    --batch-size 4 \
    --grad-accum 4 \
    --output-dir output/sana_512_vb
```

### 单任务调试

```bash
# 仅训练 depth 任务
python scripts/train_sana_512.py \
    --data-dir /path/to/vb_snvc_train_v3 \
    --tasks depth \
    --steps 2000 \
    --output-dir output/sana_512_depth_debug
```

### 带 Generation 数据混合（9:1）

```bash
python scripts/train_sana_512.py \
    --data-dir /path/to/vb_snvc_train_v3 \
    --vision-ratio 0.9 \
    --output-dir output/sana_512_mixed
```

### 断点续训

```bash
python scripts/train_sana_512.py \
    --data-dir /path/to/vb_snvc_train_v3 \
    --resume output/sana_512_vb/checkpoint-5000/lora.pt \
    --steps 10000 \
    --output-dir output/sana_512_vb
```

## 关键参数

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-name` | `sana-600m-512px` | 模型注册名 |
| `--lora-rank` | 16 | LoRA 秩 |
| `--lora-alpha` | 32 | LoRA alpha |
| `--no-rslora` | false | 禁用 rsLoRA scaling |
| `--dtype` | float16 | transformer 精度（text_encoder/VAE 自动转 bf16） |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--steps` | 10000 | 总训练步数 |
| `--lr` | 5e-5 | 学习率 |
| `--batch-size` | 4 | 每步 batch size |
| `--grad-accum` | 4 | 梯度累积步数（effective batch = 4*4 = 16） |
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

使用 `vb_snvc_train_v3` 数据集（v3 格式）：

| 任务 | 训练样本 | 验证样本 |
|------|----------|----------|
| depth | 2,181 | 115 |
| normal | 1,705 | 91 |
| segmentation | 7,132 | 351 |
| generation | 5,644 | 337 |
| structure (edge) | 2,454 | 144 |
| instance | 5,291 | 290 |
| keypoint | 8,969 | 438 |
| object_detection | 8,956 | 477 |
| depth_edge | 8,606 | 439 |
| referring | 1,616 | 74 |
| multiview | 918 | 47 |
| specular_removal | 534 | 30 |
| **总计** | **48,362** | **2,496** |

数据格式：每个任务包含 `images/`（输入图像）、`targets/`（目标可视化）、`prompts/`（文本提示）。

## 训练流程

1. **模型加载**: 从 HuggingFace 下载 `Sana_600M_512px_diffusers`，提取 transformer/VAE/text_encoder
2. **LoRA 注入**: 对 transformer 的 `linear`、`proj_out`、`conv_inverted`、`conv_point` 层注入 LoRA 适配器
3. **数据加载**: 按任务权重（默认 sqrt 平衡）从多任务数据集中采样
4. **Flow Matching**: 
   - VAE 编码 target 和 input 图像到 latent 空间
   - 采样 logit-normal 时间步
   - 构建 noisy latent: `(1-t)*input + t*target`（img2img flow）
   - 预测 velocity: `target - input`
5. **损失计算**: MSE loss + 可选 Min-SNR 加权
6. **优化**: AdamW + cosine LR scheduler + 梯度裁剪 + EMA

## Checkpoint 输出

```
output/sana_512_vb/
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
