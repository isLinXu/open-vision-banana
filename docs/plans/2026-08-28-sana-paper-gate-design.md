# SANA paper-method gate (2026-08-28)

## Goal

Reproduce the Vision-Banana **method** on SANA-600M after FLUX.1-schnell shared-LoRA failed the qualitative gate: one prompt-controlled generator, shared adapter, RGB task outputs, generation:vision ≈ 9:1. Not Table 1.

## Baseline

- `output/sana_hypersim_metric_depth_lr1e5_20260822/checkpoint-500`
- NYU Eigen+10m full val: δ1≈0.403 / AbsRel≈0.367 (capped)

## Data

- `data/sana_paper_gate_512/pairs.jsonl`: 1000 Hypersim metric_depth + 376 Cityscapes-19
- Generation: symlink to `data/flux_paper_gate_512/generation` (1968)

## Training

- Config: `configs/experiments/sana_paper_gate_from_hypersim500_20260828.yaml`
- Runner: `scripts/run_sana_paper_gate_20260828.sh`
- Resume weights-only, rank 16, lr 1e-5, vision_ratio 0.1, stop at 500
- Then NYU n=32, Cityscapes n=16, `viz_sana_paper_gate.py`

## Success (qualitative gate)

- Depth prediction in cube-edge palette resembling target (not RGB identity)
- Semantic prediction using Cityscapes flat colors (not green haze / checkerboard)
- Generation not pure noise

## Stage A follow-up (2026-08-28 evening)

Paper-gate mix (`vision_ratio=0.1`) from Hypersim-500 **failed**: NYU δ1≈0.018,
mIoU≈0.009; depth predictions landed on the wrong cube-edge face.

Next: **depth-only consolidation** before re-adding semantic/generation.

- Config: `configs/experiments/sana_depth_consolidate_from_hypersim500_20260828.yaml`
- Runner: `scripts/run_sana_depth_consolidate_20260828.sh`
- Resume Hypersim-500 weights-only → stop 300, `vision_ratio=1.0`, tasks=`[depth]`
- Gate to Stage B: enhanced viz on green face + NYU n=32 δ1 ≳ 0.28
