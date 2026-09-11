#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Open-Vision-Banana Optimization Pipeline (P0-P3)
#
# This script orchestrates the full optimization pipeline. Run individual
# stages or the entire sequence. Must be run on the host Mac with MPS access.
#
# Usage:
#   # Run everything (P0 → P1 → P2)
#   bash scripts/run_optimization_pipeline.sh all
#
#   # Run only P0 (eval fix + re-evaluate)
#   bash scripts/run_optimization_pipeline.sh p0
#
#   # Run only P1 (grid search + depth finetune)
#   bash scripts/run_optimization_pipeline.sh p1
#
#   # Run only SANA resume training
#   bash scripts/run_optimization_pipeline.sh sana
#
#   # Run grid search only
#   bash scripts/run_optimization_pipeline.sh grid
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Configuration ──────────────────────────────────────────────────────────
# Best MAR checkpoint (3-task refine, step 3250)
MAR_CKPT="output/mar_vb_v5_full_clean_v2_3task_refine_384_20260811/ckpt_best.pt"
MAR_V1_CKPT="output/mar_vb_depth_v1/ckpt_best.pt"
MANIFEST="output/mar_vb_clean_pairs_v2/pairs.jsonl"
IMG_SIZE=384

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

stage="${1:-all}"

print_header() {
    echo -e "\n${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"
}

print_step() {
    echo -e "${GREEN}▶ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}  ℹ $1${NC}"
}

# ── P0: Fix colormap + Re-evaluate ────────────────────────────────────────
run_p0() {
    print_header "P0: Colormap Fix + Re-evaluation"

    # Step 1: Verify the fix is in place
    print_step "Step 1: Verify turbo colormap in DepthRGBMapper"
    python3 -c "
import sys; sys.path.insert(0, 'src')
from open_vision_banana.core.depth_mapper import get_available_colormaps
cms = get_available_colormaps()
assert 'turbo' in cms, f'turbo not found in {cms}'
print(f'  Available colormaps: {cms}')
print('  ✓ turbo colormap registered')
"
    print_info "v1 eval (mar_vb_eval_metrics.py) now uses turbo (was cube_edge)"
    print_info "v5 eval (mar_vb_eval_v5_clean.py) LUT was already correct (matches turbo)"

    # Step 2: Run v6 evaluation on best MAR checkpoint
    print_step "Step 2: Run v6 evaluation (turbo decoding, full paper metrics)"
    print_info "Checkpoint: ${MAR_CKPT}"
    print_info "This will take ~10-20 minutes on MPS"

    python3 scripts/mar_vb_eval_v6.py \
        --checkpoint "${MAR_CKPT}" \
        --manifest "${MANIFEST}" \
        --output output/mar_vb_v6_eval_p0 \
        --img-size ${IMG_SIZE} \
        --n-per-task 16 \
        --sample-iter 32 \
        --cfg 1.0

    print_info "Results saved to output/mar_vb_v6_eval_p0/metrics.json"

    # Step 3: Also re-evaluate v1 checkpoint with fixed colormap
    print_step "Step 3: Re-evaluate v1 depth checkpoint with turbo colormap"
    if [ -f "${MAR_V1_CKPT}" ]; then
        print_info "Checkpoint: ${MAR_V1_CKPT}"
        print_info "This was the checkpoint with AbsRel=5.52 (cube_edge decoding)"
        print_info "With turbo decoding, AbsRel should be much lower"

        python3 scripts/mar_vb_eval_metrics.py 2>&1 | tee output/mar_vb_depth_v1/v6_re_eval.log
        print_info "Re-evaluation complete - compare with original paper_metrics.json"
    else
        print_info "v1 checkpoint not found, skipping"
    fi

    # Step 4: Summary
    print_step "Step 4: Compare before/after"
    echo ""
    echo "  Original v5 eval results (rainbow LUT, correct):"
    python3 -c "
import json
for f in ['output/mar_vb_clean_eval_v2_cfg1_n8_refine/metrics.json',
          'output/mar_vb_clean_eval_v2_cfg2_n8_s32_refine/metrics.json']:
    try:
        d = json.load(open(f))
        dep = d.get('summary',{}).get('depth',{})
        print(f'  {f.split(\"/\")[-2]}: depth_mae={dep.get(\"depth_scalar_mae\",0):.4f} spearman={dep.get(\"depth_spearman\",0):.4f}')
    except: pass
" 2>/dev/null || true

    echo ""
    echo "  New v6 eval results (turbo DepthRGBMapper):"
    python3 -c "
import json
try:
    d = json.load(open('output/mar_vb_v6_eval_p0/metrics.json'))
    dep = d.get('summary',{}).get('depth',{})
    print(f'  v6_eval: AbsRel_raw={dep.get(\"depth_AbsRel_raw\",0):.4f} AbsRel_al={dep.get(\"depth_AbsRel_aligned\",0):.4f}')
    print(f'           delta1_al={dep.get(\"depth_delta1_aligned\",0):.4f} pearson={dep.get(\"depth_pearson\",0):.4f} spearman={dep.get(\"depth_spearman\",0):.4f}')
except Exception as e:
    print(f'  (results not yet available: {e})')
" 2>/dev/null || true
}

# ── P1: Grid Search + Depth Fine-tune ─────────────────────────────────────
run_grid() {
    print_header "P1: MAR Sampling Parameter Grid Search"
    print_info "Sweeping cfg × sample_iter to find best config per task"

    python3 scripts/mar_vb_grid_search.py \
        --checkpoint "${MAR_CKPT}" \
        --manifest "${MANIFEST}" \
        --output output/mar_vb_grid_search \
        --img-size ${IMG_SIZE} \
        --n-per-task 4 \
        --cfgs 0.5,1.0,1.5,2.0,3.0 \
        --iters 8,16,32,64 \
        --tasks depth,normal,structure

    print_info "Grid search results: output/mar_vb_grid_search/grid_search.json"
}

run_finetune() {
    print_header "P1: MAR Single-Task Depth Fine-Tuning"
    print_info "Transfer-learning from multitask checkpoint, depth-only"

    VB_OUT="output/mar_vb_depth_finetune" \
    VB_RESUME="${MAR_CKPT}" \
    VB_STEPS=5000 \
    VB_LR=3e-5 \
    VB_IMG_SIZE=${IMG_SIZE} \
    VB_BATCH=2 \
    VB_DROPOUT=0.1 \
    python3 scripts/mar_vb_depth_finetune.py

    print_info "Fine-tuned checkpoint: output/mar_vb_depth_finetune/ckpt_best.pt"

    # Evaluate the fine-tuned model
    print_step "Evaluating fine-tuned depth model"
    python3 scripts/mar_vb_eval_v6.py \
        --checkpoint output/mar_vb_depth_finetune/ckpt_best.pt \
        --manifest "${MANIFEST}" \
        --output output/mar_vb_depth_finetune_eval \
        --img-size ${IMG_SIZE} \
        --n-per-task 16 \
        --sample-iter 32 \
        --cfg 1.0 \
        --tasks depth
}

# ── P0: SANA Resume Training ──────────────────────────────────────────────
run_sana() {
    print_header "P0: SANA Resume Training (6000 → 20000 steps)"
    print_info "Two options:"
    print_info "  1. Full 11-task (same as original, broader coverage)"
    print_info "  2. 3-task focus (depth/normal/semantic, deeper per-task quality)"

    echo ""
    echo "  Starting 3-task focused training (recommended)..."
    echo ""

    bash scripts/sana_resume_train.sh --3task --steps 20000

    print_info "SANA training complete"
    print_info "Checkpoints in: output/sana_512_vb_3task/"
}

# ── Main dispatch ─────────────────────────────────────────────────────────
case "${stage}" in
    p0|P0)
        run_p0
        ;;
    grid|GRID)
        run_grid
        ;;
    finetune|FINETUNE)
        run_finetune
        ;;
    p1|P1)
        run_grid
        run_finetune
        ;;
    sana|SANA)
        run_sana
        ;;
    all|ALL)
        run_p0
        run_grid
        run_finetune
        run_sana
        ;;
    *)
        echo "Usage: $0 {p0|grid|finetune|p1|sana|all}"
        echo ""
        echo "Stages:"
        echo "  p0        - Fix colormap + re-evaluate MAR checkpoints"
        echo "  grid      - MAR sampling parameter grid search"
        echo "  finetune  - MAR single-task depth fine-tuning"
        echo "  p1        - Grid search + depth fine-tune"
        echo "  sana      - SANA resume training to 20000 steps"
        echo "  all       - Run everything (p0 → p1 → sana)"
        exit 1
        ;;
esac

echo ""
print_header "Pipeline Complete"
print_info "Review results in output/ subdirectories"
print_info "Key files to check:"
echo "    output/mar_vb_v6_eval_p0/metrics.json     - Re-evaluated MAR metrics"
echo "    output/mar_vb_grid_search/grid_search.json - Best cfg/iter per task"
echo "    output/mar_vb_depth_finetune_eval/metrics.json - Fine-tuned depth metrics"
echo "    output/sana_512_vb_3task/                  - SANA continued training"
