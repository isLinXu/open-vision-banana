# SANA depth-only indoor+outdoor replay (2026-08-30)

## Goal

Strengthen **depth-only** shared LoRA on SANA-600M until it clears a **dual gate**, before any semantic/generation mix.

Not Table 1. Method context: Vision-Banana shared LoRA + RGB task maps.

## Why this plan

- Hypersim-500 is the best quantitative base so far (NYU n=32 δ1≈0.357; full val historically ≈0.40).
- Blind continue to step 1000 previously **regressed** (δ1≈0.23).
- Stage B depth+semantic from 500 failed qualitatively and hurt depth (δ1≈0.253, mIoU≈0.022).
- NYU n=8 viz of Hypersim-500 shows **coarse blobs**, not sharp geometry — δ1≈0.36 is not “depth usable”.
- LoRA load fix (skip PEFT→Diffusers for SANA; manual rsLoRA merge) must stay in place for all eval/viz.

## Success criteria (dual gate)

1. **Quantitative:** NYU Depth Eigen crop + 10m cap, n=32, `delta_1.25` **≥ 0.55**.
2. **Qualitative:** NYU n=8 enhanced viz — majority of samples show readable wall/furniture structure vs GT (not pure color blobs).

**Process fail-safe:** if any gated eval has δ1 **< 0.33** (≈ baseline 0.357 − 0.03), **stop immediately** and keep the best prior checkpoint.

**Honest ceiling:** 0.55 from ~0.36 on SANA-600M may be unreachable. If best δ1 stays **< 0.45** after this recipe, stop and revisit data scale or backbone — do not auto-add semantic.

## Approach (chosen)

Resume **Hypersim-500** with **small LR**, **Hypersim-heavy + light KITTI replay**, and **frequent NYU gates**. Highest success odds under current MPS / SANA-600M constraints versus blind long training or restarting from scratch.

## Data (§2)

New manifest-only dataset (reuse pixels via absolute paths):

| Item | Value |
|------|--------|
| Dir | `data/sana_depth_indoor_outdoor_replay_512/` |
| Hypersim | 1000 from `data/hypersim_depth_512` |
| KITTI | 200 (seeded subsample of 298 from `data/kitti_metric_512`; exclude val drives / prior exclusions) |
| Total | 1200 (~5:1 indoor:outdoor) |
| Tasks | `metric_depth` only |

Do **not** reuse `metric_mixed_replay_512` as-is (200 Hypersim + 298 KITTI is outdoor-heavy).

Do **not** mix Cityscapes or generation in this phase.

Optional later (Phase 1b): expand raw Hypersim if a local source is available; do not block the first run on it.

`manifest.json` must record seed, KITTI indices, ratio, and metric RGB contract.

## Training (§3)

| Knob | Value |
|------|--------|
| Base | `output/sana_hypersim_metric_depth_lr1e5_20260822/checkpoint-500/lora.pt` |
| Resume | weights-only |
| Model | `sana-600m-512px` |
| LoRA | rank 16, alpha 32, rsLoRA |
| LR | `3e-6` |
| Steps (cap) | 500 |
| Batch | 1 × grad_acc 8 |
| `vision_ratio` | 1.0 |
| Tasks | `[depth]` |
| Flow | img2img, logit-normal t, min-SNR γ=5, cfg dropout 0.1 |
| EMA | on |
| Device | MPS, **single job only** |

### Cadence

- Save every **50** steps.
- Every **100** steps (and at stop): NYU n=32 → `evals/nyu_delta1_n32_step{N}.json`.
- Every **100** steps: NYU n=8 viz → `viz_nyu_n8_step{N}/`.
- Early stop on δ1 < 0.33; success stop if dual gate passes before 500.
- Write `gate_summary.json` with `best_step`, best δ1, dual-gate boolean.

### Output root

`output/sana_depth_replay_from_hypersim500_20260830/` (or dated equivalent)

- `checkpoint-*`, `logs/`, `evals/`, `viz_nyu_n8_step*/`, `gate_summary.json`

## Implementation checklist (§4)

1. **Build mix**
   - Script or one-shot: write `pairs.jsonl` + `manifest.json` under `data/sana_depth_indoor_outdoor_replay_512/`.
   - Seed fixed (e.g. 42); document KITTI sample IDs.
2. **Config**
   - `configs/experiments/sana_depth_replay_from_hypersim500_<date>.yaml` matching §3.
3. **Runner**
   - `scripts/run_sana_depth_replay_from_hypersim500_<date>.sh`:
     - refuse if MPS busy;
     - train in segments of 100 steps **or** one train with external gate loop;
     - after each gate: parse δ1, early-stop / continue;
     - sequential eval → viz → decide.
4. **Viz helper**
   - Reuse existing NYU n=8 viz path (or thin wrapper) with Eigen+10m panels + metrics sidecar.
5. **Baseline snapshot**
   - Before train: confirm LoRA load log shows SANA skip PEFT convert + manual merge `scale=8` + 58 layers;
   - record baseline δ1≈0.357 in `gate_summary` / runner log.
6. **Acceptance**
   - Dual gate pass → freeze that ckpt as depth base for a **later** semantic plan (new design, not this run).
   - Else → document best δ1 + viz links; no Stage-B-style mix.

## Out of scope

- Semantic / generation / paper-gate `vision_ratio=0.1` mixes.
- FLUX path.
- Changing LoRA rank/alpha in this phase.
- Parallel MPS jobs.

## References

- Baseline: `output/sana_hypersim_metric_depth_lr1e5_20260822/checkpoint-500`
- NYU qualitative: `…/viz_checkpoint-500_nyu_n8_20260830/`
- Failed Stage B: `output/sana_depth_sem_from_hypersim500_20260830/`
- Prior mixed replay (outdoor-heavy): `data/metric_mixed_replay_512` + `sana_metric_mixed_replay_lr1e5_20260823`
- LoRA load fix: `src/open_vision_banana/inference/generator.py` (`_prepare_lora_load` skips PEFT convert for `sana`)
