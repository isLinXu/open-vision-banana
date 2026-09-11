#!/bin/bash
# Monitor training and run inference when done
TRAIN_PID=537290
LOG=/tmp/train_3task_monitor.log
OUTPUT_DIR=output/sana_1600m_3task_full_clean/viz_results

echo "[$(date)] Starting training monitor for PID $TRAIN_PID" > $LOG

# Wait for training to complete
while kill -0 $TRAIN_PID 2>/dev/null; do
    # Log current progress every 10 minutes
    STEP=$(tail -1 /tmp/train_3task_full_clean.log 2>/dev/null | grep -oP 'step \K[0-9]+')
    echo "[$(date)] Training still running... step=$STEP" >> $LOG
    sleep 600
done

echo "[$(date)] Training process $TRAIN_PID has finished." >> $LOG

# Find best checkpoint
BEST_CKPT=$(ls -d output/sana_1600m_3task_full_clean/checkpoint-*_best/lora_adapter/lora.pt 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT=$(ls -d output/sana_1600m_3task_full_clean/checkpoint-*/lora_adapter/lora.pt 2>/dev/null | sort -t- -k2 -n | tail -1)
fi

echo "[$(date)] Best checkpoint: $BEST_CKPT" >> $LOG

# Get last val_loss from training log
LAST_VAL=$(grep "val_loss=" /tmp/train_3task_full_clean.log | tail -1)
echo "[$(date)] Last val info: $LAST_VAL" >> $LOG

# Run inference visualization
echo "[$(date)] Running inference visualization..." >> $LOG
cd /media/linxu/thinkHub/Thinking-face/open-vision-banana

/home/linxu/anaconda3/envs/llama/bin/python scripts/infer_3task_viz.py \
    --checkpoint "$BEST_CKPT" \
    --pair-manifest output/snvc_v5full_pairs/pairs_5task_v3_clean.jsonl \
    --output-dir "$OUTPUT_DIR" \
    --num-samples 5 \
    --num-inference-steps 20 \
    2>&1 | tee -a $LOG

echo "[$(date)] Inference complete." >> $LOG
echo "[$(date)] DONE." >> $LOG
