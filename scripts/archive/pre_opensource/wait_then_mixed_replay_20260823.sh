#!/bin/zsh
set -euo pipefail

# Wait until the ckpt-300 eval releases MPS, then start mixed replay.
# Bracket the pgrep patterns so this waiter is not counted as busy.

cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
OUT=output/sana_metric_mixed_replay_lr1e5_20260823
mkdir -p "$OUT/logs"
LOG="$OUT/logs/waiter.log"
BUSY='[e]valuate_paper_benchmarks.py|[t]rain_sana_512.py|[r]un_sana_kitti_ckpt300_eval'

echo "[waiter] started $(date '+%F %T'); waiting for MPS" | tee -a "$LOG"
while pgrep -f "$BUSY" >/dev/null; do
  echo "[waiter] $(date '+%H:%M:%S') eval still running" >> "$LOG"
  sleep 30
done
echo "[waiter] MPS idle at $(date '+%F %T'); launching mixed replay" | tee -a "$LOG"
exec ./scripts/run_sana_metric_mixed_replay_20260823.sh
