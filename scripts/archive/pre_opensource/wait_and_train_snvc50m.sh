#!/bin/bash
# Auto-start SNVC-50M tar mode training when index build completes
# Usage: nohup bash scripts/wait_and_train_snvc50m.sh > output/snvc50m_manifest/wait_and_train.log 2>&1 &

set -e

PROJECT_DIR="/media/linxu/thinkHub/Thinking-face/open-vision-banana"
INDEX_LOG="$PROJECT_DIR/output/snvc50m_manifest/build_tar_index_v3.log"
INDEX_FILE="$PROJECT_DIR/output/snvc50m_manifest/tar_index_v3.json"
TRAIN_CONFIG="$PROJECT_DIR/configs/sana_snvc50m_tar.yaml"
LLAMA_ENV="/home/linxu/anaconda3/envs/llama/bin/python"

cd "$PROJECT_DIR"

echo "$(date): Waiting for tar index build to complete..."

# Wait for index build process to finish
while ps aux | grep -q "[b]uild_tar_index_snvc50m_v3.py"; do
    # Show progress every 5 minutes
    tail -1 "$INDEX_LOG" 2>/dev/null || true
    sleep 300
done

echo "$(date): Index build process finished."

# Verify index file exists and is valid
if [ ! -f "$INDEX_FILE" ]; then
    echo "$(date): ERROR - Index file not found: $INDEX_FILE"
    exit 1
fi

# Check index file size (should be > 1 GB for full dataset)
INDEX_SIZE=$(stat -f%z "$INDEX_FILE" 2>/dev/null || stat -c%s "$INDEX_FILE" 2>/dev/null)
echo "$(date): Index file size: $((INDEX_SIZE / 1024 / 1024)) MB"

if [ "$INDEX_SIZE" -lt 100000000 ]; then
    echo "$(date): WARNING - Index file seems small (< 100 MB), might be incomplete"
fi

# Stop any existing training processes to free GPU
echo "$(date): Stopping any existing training processes..."
pkill -f "train_sana_512.py" 2>/dev/null || true
sleep 5

# Start training
echo "$(date): Starting SNVC-50M tar mode training..."
echo "$(date): Config: $TRAIN_CONFIG"
echo "$(date): Command: $LLAMA_ENV scripts/train_sana_512.py --config $TRAIN_CONFIG"

exec $LLAMA_ENV scripts/train_sana_512.py --config "$TRAIN_CONFIG"
