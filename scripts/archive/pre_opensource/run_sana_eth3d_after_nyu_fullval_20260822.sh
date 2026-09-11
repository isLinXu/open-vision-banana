#!/bin/zsh
set -euo pipefail

# Wait for the NYU full-val JSON, then run the ETH3D 2-scene subset.
# Exits without starting ETH3D if the NYU job finishes without a result.

cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722
OUT=output/sana_hypersim_metric_depth_lr1e5_20260822
NYU_JSON="$OUT/evals/nyu_delta1_step500_capped_fullval.json"
NYU_LOG="$OUT/logs/eval_nyu_500_capped_fullval.log"

echo "[chain] waiting for NYU full val → $NYU_JSON" | tee -a "$OUT/logs/runner.log"
while [[ ! -f "$NYU_JSON" ]]; do
  if ! pgrep -f 'eval_nyu_500_capped_fullval|nyu_delta1_step500_capped_fullval' >/dev/null; then
    if [[ ! -f "$NYU_JSON" ]]; then
      echo "[chain] NYU eval is not running and $NYU_JSON is missing" | tee -a "$OUT/logs/runner.log"
      exit 1
    fi
  fi
  sleep 30
done

echo "[chain] NYU full val present; starting ETH3D subset" | tee -a "$OUT/logs/runner.log"
exec ./scripts/run_sana_eth3d_subset_step500_20260822.sh
