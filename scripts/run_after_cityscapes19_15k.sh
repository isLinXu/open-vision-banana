#!/usr/bin/env bash
# After the CS-19 15k run finishes: official Cityscapes val, then FLUX RefCOCOg.
# Do not start this while another eval owns the GPU. Do not pkill the trainer.
set -euo pipefail
DEPLOY="${DEPLOY:-/comp_robot/home/xu.lin/workspaces/ovb-paper-eval-20261009}"
TRAIN_DIR="${TRAIN_DIR:-$DEPLOY/output/sana_1600m_cityscapes19_15k}"
LOG="$TRAIN_DIR/train.log"
PYTHON="${PYTHON:-/comp_robot/home/xu.lin/ovb_env/bin/python}"
export PYTHONPATH="${PYTHONPATH:-$DEPLOY/src}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPEN_VISION_BANANA_MODEL_PATH="${OPEN_VISION_BANANA_MODEL_PATH:-/comp_robot/home/xu.lin/workspaces/open-vision-banana/models/benchmark/sana_1600m_base_fp16}"

cd "$DEPLOY"
echo "[watch] waiting for Training complete in $LOG"
while ! grep -q "Training complete" "$LOG" 2>/dev/null; do
  if ! pgrep -f "scripts/train.py .*sana_1600m_cityscapes19_15k" >/dev/null; then
    echo "[watch] trainer exited before Training complete" >&2
    tail -40 "$LOG" >&2
    exit 1
  fi
  sleep 60
done

BEST=$(ls -d "$TRAIN_DIR"/checkpoint-*_best 2>/dev/null | sed 's/.*checkpoint-//' | sed 's/_best//' | sort -n | tail -1)
if [[ -z "${BEST:-}" ]]; then
  echo "[watch] no *_best checkpoint in $TRAIN_DIR" >&2
  exit 1
fi
ADAPTER="$TRAIN_DIR/checkpoint-${BEST}_best/lora_adapter"
echo "[watch] official Cityscapes val on $ADAPTER"
"$PYTHON" scripts/eval_cityscapes19.py \
  --cityscapes-root /comp_robot/home/xu.lin/data/paper_benchmarks/cityscapes \
  --lora-path "$ADAPTER" \
  --output-json "$DEPLOY/output/eval/cityscapes_val_cityscapes19_15k.json"

export OPEN_VISION_BANANA_MODEL_PATH=/comp_robot/home/xu.lin/workspaces/open-vision-banana/models/benchmark/flux1_dev
echo "[watch] FLUX RefCOCOg UMD val"
"$PYTHON" scripts/eval_refcocog.py \
  --model-name flux-dev \
  --quantize 4bit \
  --dtype bfloat16 \
  --num-inference-steps 20 \
  --refcocog-root /comp_robot/home/xu.lin/data/paper_benchmarks/refcocog/eval \
  --lora-path /comp_robot/home/xu.lin/ckpts/khazic_ovb/ovb-flux3_ctrl_seg_nominsnr_15k/ckpt/checkpoint-15000_final/lora_adapter \
  --output-json "$DEPLOY/output/eval/refcocog_umd_val_flux3_ctrl_seg_native.json"

echo "[watch] done"
