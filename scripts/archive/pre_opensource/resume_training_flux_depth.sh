#!/bin/bash
# 恢复纯深度估计训练 — 从 checkpoint-50 热启动
# ================================================
# 训练配置: configs/training_flux_depth.yaml
# 数据: data/vb_depth_pseudo_5k (5839 samples, pure depth)
# LoRA: rank=64, alpha=128, rsLoRA, target=to_k/to_v
# Resume: checkpoint-50 (loss=8.34, step=50)
#
# 修复点:
#   1. CheckpointManager.load 直接加载 safetensors，避免 PeftModel.from_pretrained double-wrapping
#   2. Key 映射: .lora_A.weight → .lora_A.default.weight
#   3. tuner.train() 新增 start_step/start_epoch 参数，LR scheduler 正确恢复
#   4. _direct_forward 不再解包 __wrapped__，保留 LoRA 梯度路径

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

cd "$PROJECT_DIR"

exec python scripts/train.py \
    --config configs/training_flux_depth.yaml \
    --model-name "/Users/gatilin/MyWork/models/ckpts/FLUX.1-schnell" \
    --data-path "data/vb_depth_pseudo_5k" \
    --batch-size 1 \
    --device mps \
    --epochs 3 \
    --lr 5e-5 \
    --log-every 1 \
    --resume "output/checkpoints_flux_depth/checkpoint-50" \
    --output-dir "output/checkpoints_flux_depth"
