#!/usr/bin/env bash
# Vision Banana v7 — Automated experiment runner
# Usage: bash scripts/run_experiments.sh
#
# Runs three key experiments:
#   1. sana-600m + strength 0.3 + 3000 steps (DiT-based, lightweight)
#   2. ssd-1b   + strength 0.3 + 3000 steps (UNet-based, stronger baseline)
#   3. ssd-1b   + strength 1.0 + 3000 steps (pure noise, control: validates img2img necessity)
#
# All outputs go to output/vb_v7/<model>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=================================================="
echo "Vision Banana v7 — Experiment Runner"
echo "=================================================="
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "Device : $(python3 -c 'import torch; print(torch.device(\"mps\" if torch.backends.mps.is_available() else \"cpu\"))')"
echo "Data   : $(ls -d data/vb_* 2>/dev/null | wc -l) dataset dirs found"
echo "=================================================="
echo ""

mkdir -p "$PROJECT_ROOT/output/vb_v7"

# ─── Experiment 1: sana-600m (DiT, lightweight) ─────────────────────────────
run_exp1() {
    echo ""
    echo "▶ Experiment 1: sana-600m + img2img_strength=0.3 + 3000 steps"
    echo "  Purpose: Test DiT-based lightweight model with reduced texture leakage"
    echo "  Estimated time: ~90-120 min on MPS (Apple M1 Pro 32GB)"
    echo ""

    python3 "$SCRIPT_DIR/train_vb_v7.py" \
        --model sana-600m \
        --steps 3000 \
        --img2img-strength 0.3 \
        --colormaps cube_edge \
        --resolution 256 \
        --batch-size 1 \
        --grad-accum 8 \
        --lr 1e-4 \
        --vision-ratio 0.1 \
        --vision-tasks depth \
        --auto-select-data \
        --scheduler cosine \
        --ema-decay 0.9995 \
        --save-every 500 \
        --val-every 300 \
        --log-every 20 \
        --seed 42 \
        2>&1 | tee "$PROJECT_ROOT/output/vb_v7/exp1_sana_600m_strength03.log"

    echo ""
    echo "✓ Experiment 1 complete. Log: output/vb_v7/exp1_sana_600m_strength03.log"
    echo "  Output: output/vb_v7/sana-600m/training_log.json"
}

# ─── Experiment 2: ssd-1b + strength 0.3 (stronger UNet baseline) ────────────
run_exp2() {
    echo ""
    echo "▶ Experiment 2: ssd-1b + img2img_strength=0.3 + 3000 steps"
    echo "  Purpose: Strong UNet baseline with reduced texture leakage"
    echo "  Estimated time: ~120-180 min on MPS (Apple M1 Pro 32GB)"
    echo ""

    python3 "$SCRIPT_DIR/train_vb_v7.py" \
        --model ssd-1b \
        --steps 3000 \
        --img2img-strength 0.3 \
        --colormaps cube_edge \
        --resolution 384 \
        --batch-size 1 \
        --grad-accum 8 \
        --lr 5e-5 \
        --vision-ratio 0.1 \
        --vision-tasks depth \
        --auto-select-data \
        --scheduler cosine \
        --ema-decay 0.9995 \
        --save-every 500 \
        --val-every 300 \
        --log-every 20 \
        --seed 42 \
        2>&1 | tee "$PROJECT_ROOT/output/vb_v7/exp2_ssd_1b_strength03.log"

    echo ""
    echo "✓ Experiment 2 complete. Log: output/vb_v7/exp2_ssd_1b_strength03.log"
    echo "  Output: output/vb_v7/ssd-1b/training_log.json"
}

# ─── Experiment 3: ssd-1b + strength 1.0 (pure noise control) ────────────────
run_exp3() {
    echo ""
    echo "▶ Experiment 3: ssd-1b + img2img_strength=1.0 + 3000 steps (CONTROL)"
    echo "  Purpose: Validate that img2img conditioning is necessary."
    echo "  If strength=1.0 (pure noise) performs significantly worse than strength=0.3,"
    echo "  this confirms the img2img pipeline is essential for depth prediction."
    echo "  Estimated time: ~120-180 min on MPS (Apple M1 Pro 32GB)"
    echo ""

    python3 "$SCRIPT_DIR/train_vb_v7.py" \
        --model ssd-1b \
        --steps 3000 \
        --img2img-strength 1.0 \
        --colormaps cube_edge \
        --resolution 384 \
        --batch-size 1 \
        --grad-accum 8 \
        --lr 5e-5 \
        --vision-ratio 0.1 \
        --vision-tasks depth \
        --auto-select-data \
        --scheduler cosine \
        --ema-decay 0.9995 \
        --save-every 500 \
        --val-every 300 \
        --log-every 20 \
        --seed 42 \
        2>&1 | tee "$PROJECT_ROOT/output/vb_v7/exp3_ssd_1b_strength10.log"

    echo ""
    echo "✓ Experiment 3 complete. Log: output/vb_v7/exp3_ssd_1b_strength10.log"
    echo "  Output: output/vb_v7/ssd-1b/training_log.json (note: same model dir as exp2)"
    echo "  WARNING: exp3 will overwrite exp2's final checkpoint. Run separately or change output dir."
}

# ─── Experiment 4: ssd-1b + v6 best checkpoint warm-start (optional) ──────
run_exp4() {
    local v6_best="$PROJECT_ROOT/output/vb_v6/ssd-1b/vb_v6_ssd-1b_best_corr.pt"
    if [[ -f "$v6_best" ]]; then
        echo ""
        echo "▶ Experiment 4: ssd-1b warm-start from v6 best checkpoint + 3000 steps"
        echo "  Purpose: Continue learning from v6 best instead of from scratch"
        echo ""

        python3 "$SCRIPT_DIR/train_vb_v7.py" \
            --model ssd-1b \
            --steps 3000 \
            --img2img-strength 0.3 \
            --colormaps cube_edge \
            --resolution 384 \
            --batch-size 1 \
            --grad-accum 8 \
            --lr 5e-5 \
            --vision-ratio 0.1 \
            --vision-tasks depth \
            --auto-select-data \
            --scheduler cosine \
            --ema-decay 0.9995 \
            --save-every 500 \
            --val-every 300 \
            --log-every 20 \
            --seed 42 \
            --resume "$v6_best" \
            2>&1 | tee "$PROJECT_ROOT/output/vb_v7/exp4_ssd_1b_warmstart.log"

        echo ""
        echo "✓ Experiment 4 complete. Log: output/vb_v7/exp4_ssd_1b_warmstart.log"
    else
        echo ""
        echo "⚠ Experiment 4 skipped: v6 best checkpoint not found at $v6_best"
    fi
}

# ─── Main: parse args and run selected experiments ───────────────────────────
usage() {
    echo "Usage: bash scripts/run_experiments.sh [all|exp1|exp2|exp3|exp4]"
    echo ""
    echo "  all   — Run experiments 1, 2, 3 sequentially (default)"
    echo "  exp1  — sana-600m + strength 0.3"
    echo "  exp2  — ssd-1b   + strength 0.3"
    echo "  exp3  — ssd-1b   + strength 1.0 (control)"
    echo "  exp4  — ssd-1b   + warm-start from v6 best"
    exit 1
}

CMD="${1:-all}"

case "$CMD" in
    all)
        run_exp1
        run_exp2
        run_exp3
        ;;
    exp1)
        run_exp1
        ;;
    exp2)
        run_exp2
        ;;
    exp3)
        run_exp3
        ;;
    exp4)
        run_exp4
        ;;
    *)
        usage
        ;;
esac

echo ""
echo "=================================================="
echo "All requested experiments complete."
echo "=================================================="
echo "Results:"
ls -la "$PROJECT_ROOT/output/vb_v7/"*.log 2>/dev/null || true
echo ""
echo "To compare results:"
echo "  cat output/vb_v7/sana-600m/training_log.json | python3 -m json.tool | grep best_val_correlation"
echo "  cat output/vb_v7/ssd-1b/training_log.json   | python3 -m json.tool | grep best_val_correlation"
