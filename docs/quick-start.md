# Open-Vision-Banana GPU 训练快速启动指南

> **适用硬件**: NVIDIA H20 (96GB) / A100-80GB / A100-40GB  
> **更新日期**: 2026-07-08  
> **配套版本**: v0.5.0+

---

## 一、环境准备（一次性）

### 1.1 基础依赖检查

```bash
# 检查 CUDA 和 GPU
nvidia-smi                          # 确认 GPU 可见
nvcc --version                      # 确认 CUDA 版本 ≥ 11.8
python3 --version                   # Python 3.10 - 3.12

# 推荐环境
# CUDA >= 11.8, PyTorch >= 2.1 (需匹配 CUDA 版本)
```

### 1.2 项目克隆与安装

```bash
# 1. 克隆项目
git clone https://github.com/isLinXu/open-vision-banana.git
cd open-vision-banana

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装项目（推荐 all 模式，包含 training + inference + evaluation + deepspeed）
pip install -e ".[all]"

# 4. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -m pytest tests/test_model_registry.py -q
```

---

## 二、数据与模型准备

### 2.1 下载 Base 模型权重（FLUX.1-schnell，~23GB）

```bash
# 方式 1: 通用下载脚本（推荐，支持断点续传、内存筛选）
python scripts/download_model.py --model flux-schnell

# 方式 2: 使用环境变量传入 HF token
HF_TOKEN=hf_xxx python scripts/download_model.py --model flux-schnell

# 验证本地 cache 完整性
python scripts/download_model.py --model flux-schnell --local-files-only
```

> 首次下载约 **23GB**，需 10-30 分钟。脚本内置 `hf-mirror.com` 镜像加速。

### 2.2 下载训练数据集

```bash
# 查看支持的数据集
python scripts/download_data.py --list

# 方式 1: 下载指定数据集（推荐）
python scripts/download_data.py \
    --dataset hypersim-depth \
    --output data/hypersim/ \
    --train-split 0.9

# 方式 2: 批量下载全部数据集
python scripts/download_data.py \
    --all \
    --output data/

# 方式 3: Dry-run 预览（不实际下载）
python scripts/download_data.py \
    --dataset hypersim-depth \
    --output data/hypersim/ \
    --dry-run
```

---

## 三、H20 (96GB) 快速启动

### 3.1 配置文件

| 文件 | 场景 | 每卡 batch | 有效全局 batch |
|------|------|-----------|---------------|
| `configs/training_h20_single.yaml` | 单卡 H20 | 4 × 2 accum = **8** | 8 |
| `configs/training_h20_ddp_8x.yaml` | 8卡 DDP | 2 × 4 accum = **8** | 64 (8×8) |
| `configs/training_h20_fsdp_8x.yaml` | 8卡 FSDP | 4 × 2 accum = **8** | 64 (8×8) |
| `configs/deepspeed_zero3_h20.json` | DeepSpeed ZeRO-3 | auto | auto |

### 3.2 启动命令

```bash
# --- 单卡 H20（快速验证 / 调试）---
bash scripts/launch_h20_single.sh --data-path /data/vb_depth_pseudo_5k

# --- 8×H20 DDP（单节点生产，最常用）---
bash scripts/launch_h20_ddp.sh --data-path /shared/data

# --- 8×H20 FSDP（最大吞吐，1024² 分辨率）---
bash scripts/launch_h20_fsdp.sh --data-path /shared/data --gradient-checkpointing

# --- 多节点 2×8 H20 ---
# Node 0 (Master):
bash scripts/launch_h20_ddp.sh \
    --nodes 2 --node-rank 0 \
    --master-addr 10.0.0.1 \
    --data-path /shared/data

# Node 1 (Worker):
bash scripts/launch_h20_ddp.sh \
    --nodes 2 --node-rank 1 \
    --master-addr 10.0.0.1 \
    --data-path /shared/data
```

### 3.3 H20 特有优化

| 优化项 | 配置值 | 说明 |
|--------|--------|------|
| `dtype` | `bfloat16` | Hopper 原生 BF16，无损动态范围 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | HBM3 显存碎片管理 |
| `NCCL_ASYNC_ERROR_HANDLING` | `1` | NVSwitch 稳定性 |
| `num_workers` | 4~8 | CUDA 多进程 DataLoader |
| `use_amp` | `true` | CUDA GradScaler 启用 |

---

## 四、A100-80GB 快速启动

### 4.1 配置文件

| 文件 | 场景 | 每卡 batch | 说明 |
|------|------|-----------|------|
| `configs/training_a100_80g.yaml` | 单任务/标准任务 | 4 × 2 accum = **8** | 生产级默认 |
| `configs/training_a100_80g_multitask.yaml` | 多任务 (6 类视觉任务) | 4 × 2 accum = **8** | 语义/深度/法线/边缘/光流/关键点 |

### 4.2 启动命令

```bash
# --- 统一入口（自动检测 GPU 数并推荐 DDP/FSDP）---
bash scripts/launch_training_a100_80g.sh --data-path /shared/data

# --- 8×A100 DDP（最常用）---
bash scripts/launch_training_a100_80g_ddp.sh \
    --config configs/training_a100_80g.yaml

# --- 8×A100 FSDP + gradient-checkpointing ---
bash scripts/launch_training_a100_80g_fsdp.sh --gradient-checkpointing

# --- 多任务训练 ---
bash scripts/launch_training_a100_80g_ddp.sh \
    --config configs/training_a100_80g_multitask.yaml \
    --data-path /shared/data
```

### 4.3 A100-80GB 特有优化

| 优化项 | 配置值 | 说明 |
|--------|--------|------|
| `dtype` | `bfloat16` | A100 Tensor Core 原生 BF16 |
| `NVIDIA_TF32_OVERRIDE` | `1` | TF32 加速 (~1.5-2×) |
| `use_amp` | `true` | 零额外开销 |
| `num_workers` | 8 | 充分预取保持 GPU 饱和 |

---

## 五、A100-40GB 快速启动

> A100-40GB 显存较紧，**必须启用 DeepSpeed ZeRO-3**。

### 5.1 配置文件

| 文件 | 场景 | 每卡 batch | 适用 |
|------|------|-----------|------|
| `configs/training_a100_40g_deepspeed.yaml` | 训练超参配置 | 2 × 4 accum = **8** | 通用 |
| `configs/deepspeed_a100_40g_zero3.json` | ZeRO-3 无 offload | 2 | 8×A100 推荐 |
| `configs/deepspeed_a100_40g_zero3_offload.json` | ZeRO-3 CPU offload | 2 | 1-4×A100 fallback |

### 5.2 启动命令

```bash
# --- 8×A100 生产训练（推荐）---
bash scripts/train_a100_40g_deepspeed.sh \
    --mode multi \
    --data-path /shared/data \
    --wandb \
    --wandb-run-name "a100_40g_depth_v1"

# --- 单卡调试（自动启用 CPU offload）---
bash scripts/train_a100_40g_deepspeed.sh \
    --mode single \
    --data-path /data/small

# --- 2-4 卡（NVLink 不足时启用 offload）---
bash scripts/train_a100_40g_deepspeed.sh \
    --mode multi-offload \
    --data-path /shared/data
```

### 5.3 A100-40GB 显存估算

- FLUX.1-schnell 12B BF16 权重 ≈ 24 GB → ZeRO-3 分片后每卡 **3 GB**
- LoRA 可训练参数 ≈ 60M → BF16 **120 MB**
- 激活值 (batch=2, 512², gradient-checkpointing) ≈ **8-12 GB**
- 总计每卡 ≈ **14-18 GB < 40 GB** ✅

---

## 六、统一启动脚本（推荐新手使用）

```bash
# auto 模式：自动检测 H20/A100/MPS/CPU 并选择最佳策略
bash scripts/unified_launch.sh --config configs/unified_training_config.yaml

# 强制指定模式
bash scripts/unified_launch.sh --mode ddp     --config configs/training_config.yaml
bash scripts/unified_launch.sh --mode fsdp    --config configs/training_config.yaml --gradient-checkpointing
bash scripts/unified_launch.sh --mode deepspeed --config configs/training_config.yaml

# Dry-run 预览命令（不执行）
bash scripts/unified_launch.sh --config configs/training_config.yaml --dry-run

# 环境变量覆盖
OVB_BATCH_SIZE=4 OVB_LR=1e-4 bash scripts/unified_launch.sh --config configs/training_config.yaml
```

---

## 七、断点续训与监控

### 7.1 断点续训

```bash
# 从 checkpoint 恢复
bash scripts/launch_h20_ddp.sh \
    --resume output/checkpoints/checkpoint-5000 \
    --data-path /shared/data

# 统一脚本续训
bash scripts/unified_launch.sh \
    --mode fsdp \
    --resume output/checkpoints/checkpoint-5000
```

### 7.2 WandB 监控

```bash
# 首次使用需登录
wandb login

# 训练时开启
bash scripts/launch_training_a100_80g.sh \
    --data-path /shared/data \
    --wandb \
    --wandb-project "vision-banana-prod" \
    --wandb-run-name "depth-flux-v1"
```

### 7.3 TensorBoard

```bash
# 训练时开启
bash scripts/launch_training_a100_80g.sh --tensorboard

# 查看（服务器上绑定所有 IP）
tensorboard --logdir /shared/output/runs --bind_all
```

---

## 八、显存策略速查表

| GPU 类型 | 显存 | batch_size | 推荐策略 | 关键参数 |
|---------|------|-----------|---------|---------|
| H20 | 96GB | 4-8 | DDP / FSDP | bf16, amp=true, num_workers=8 |
| A100-80GB | 80GB | 4-8 | DDP / FSDP | bf16, amp=true, checkpointing可选 |
| A100-40GB | 40GB | 2 | **DeepSpeed ZeRO-3** | bf16, grad-checkpointing, zero3 |
| RTX 4090 | 24GB | 1-2 | DeepSpeed ZeRO-3 + offload | bf16, grad-checkpointing, offload |
| RTX 3090 | 24GB | 1 | DeepSpeed ZeRO-3 + offload | bf16, grad-checkpointing, offload |

---

## 九、常见问题速查

| 问题 | 原因 | 解决 |
|------|------|------|
| `CUDA out of memory` | batch_size 太大 | `--batch-size 1 --gradient-checkpointing` |
| `NCCL error` | 多卡通信失败 | `NCCL_IB_DISABLE=1` 或检查网络 |
| `dataset not found` | 数据路径错误 | 确认 `--data-path` 指向 `images/` + `targets/` 目录 |
| wandb 报错 | 未登录 | `wandb login` 或去掉 `--wandb` |
| 训练极慢 | num_workers=0 | 设置 `--num-workers 4+` |
| MPS/CPU 被选中 | CUDA 未检测到 | 检查 `torch.cuda.is_available()` |

---

## 十、文件路径索引

```
configs/
├── training_h20_single.yaml          # H20 单卡
├── training_h20_ddp_8x.yaml          # H20 8卡 DDP
├── training_h20_fsdp_8x.yaml         # H20 8卡 FSDP
├── deepspeed_zero3_h20.json          # H20 DeepSpeed
├── training_a100_80g.yaml            # A100-80G 标准
├── training_a100_80g_multitask.yaml  # A100-80G 多任务
├── training_a100_40g_deepspeed.yaml  # A100-40G 训练
├── deepspeed_a100_40g_zero3.json     # A100-40G ZeRO-3
├── deepspeed_a100_40g_zero3_offload.json  # A100-40G offload
└── unified_training_config.yaml      # 通用统一配置

scripts/
├── launch_h20_single.sh              # H20 单卡
├── launch_h20_ddp.sh                 # H20 DDP
├── launch_h20_fsdp.sh                # H20 FSDP
├── launch_h20_deepspeed.sh           # H20 DeepSpeed
├── launch_training_a100_80g.sh       # A100-80G 统一入口
├── launch_training_a100_80g_ddp.sh   # A100-80G DDP
├── launch_training_a100_80g_fsdp.sh  # A100-80G FSDP
├── launch_training_a100_80g_deepspeed.sh  # A100-80G DeepSpeed
├── train_a100_40g_deepspeed.sh       # A100-40G DeepSpeed
└── unified_launch.sh                 # 6 模式通用启动
```

---

## 十一、最快启动路径（5 分钟）

```bash
# 1. 环境（2 min）
git clone https://github.com/isLinXu/open-vision-banana.git && cd open-vision-banana
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# 2. 模型（已下载则跳过）
python scripts/download_model.py --model flux-schnell

# 3. 数据（已准备则跳过）
python scripts/download_data.py --dataset hypersim-depth --output data/hypersim/

# 4. 训练（1 min 配置）
# H20:
torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) \
    scripts/train.py --config configs/training_h20_ddp_8x.yaml \
    --ddp --data-path data/nyu --wandb

# A100-80G:
bash scripts/launch_training_a100_80g.sh --data-path data/nyu --wandb

# A100-40G:
bash scripts/train_a100_40g_deepspeed.sh --mode multi --data-path data/nyu --wandb
```
