# TinyVisionBanana 本机 GPU 训练指南

## 背景

沙箱环境 (3GB cgroup) 限制了模型规模上限：w=32/256px 的训练因 OOM 不可行。本指南提供了在本地 Mac (Apple Silicon MPS) 上训练更大模型的脚本和说明。

## 沙箱实验结果

| 配置 | 参数量 | 分辨率 | MAE | Pearson | 训练时间 | 备注 |
|------|--------|--------|-----|---------|---------|------|
| w=16 r=128 | 2.99M | 128px | 0.1873 | 0.6198 | ~1h | 基线 |
| w=24 r=192 | 6.69M | 192px | 0.1771 | 0.6645 | ~5.4h | 沙箱最优 |
| w=32 r=256 | 11.86M | 256px | - | - | - | 沙箱 OOM |
| **Probe (SSD-1B)** | **~1B** | **512px** | **0.1296** | **0.7715** | - | **目标基线** |

## 本机训练配置

### 环境准备

```bash
# 确认 PyTorch MPS 可用
python3 -c "import torch; print(torch.backends.mps.is_available())"

# 安装依赖
pip install torch numpy pillow
```

### 配置选项

| 配置名 | 任务 | base_width | 分辨率 | 参数量 | 预计时间(Mac) |
|--------|------|-----------|--------|--------|-------------|
| `large` | depth | 32 | 256px | 11.86M | 1-2h |
| `xlarge` | depth | 48 | 256px | 26.6M | 2-4h |
| `multitask` | 5任务 | 32 | 256px | 11.86M | 3-5h |
| `multitask_xlarge` | 5任务 | 48 | 256px | 26.6M | 5-10h |

### 启动训练

```bash
# 单任务 depth (推荐先跑这个)
bash scripts/train_local_gpu.sh large 5000

# 超大模型
bash scripts/train_local_gpu.sh xlarge 5000

# 5 任务联合训练
bash scripts/train_local_gpu.sh multitask 5000
```

### 评估

```bash
# 评估单个 checkpoint
PYTHONPATH=src python3 scripts/evaluate_local_gpu.py \
    --checkpoint output/local_gpu/depth_w32_256/best.pt \
    --data-dir /Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v3 \
    --base-width 32 --resolution 256

# 对比所有 checkpoint
PYTHONPATH=src python3 scripts/evaluate_local_gpu.py \
    --checkpoint-dir output/local_gpu/depth_w32_256 \
    --data-dir /Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v3 \
    --base-width 32 --resolution 256 \
    --compare
```

## 关键说明

1. **MPS fallback**: 脚本设置了 `PYTORCH_ENABLE_MPS_FALLBACK=1`，MPS 不支持的算子会自动回退到 CPU
2. **Batch size**: `large` 和 `multitask` 配置使用 batch_size=2，MPS 通常有 16GB+ 统一内存
3. **学习率**: w=48 使用更低的 lr=3e-4，避免大模型训练不稳定
4. **检查点**: 每 500 步自动保存，崩溃后可用 `--resume` 恢复
