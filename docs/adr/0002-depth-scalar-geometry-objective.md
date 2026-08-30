# ADR-0002: Gate a differentiable scalar depth objective

## Status

Implemented and evaluated. Rejected for promotion in the current
SSD-1B/LoRA diffusion path.

## Context

The stage-4 geometry gate showed that decoded-RGB structure supervision and a
multi-scale T2I-Adapter did not produce stable train-set relative-depth
ordering. The remaining loss-semantics hypothesis was that RGB-local penalties
do not directly match the Pearson/Spearman evaluation contract.

The discriminator gate had to remain local on Apple MPS, preserve checkpoint
compatibility, consume no additional random numbers during training, and use
the same four-train/four-validation matched protocol as stage 4.

## Decision

Add an opt-in decoded-x0 scalar geometry objective:

1. Project each decoded RGB pixel onto every segment of the audited depth
   colormap using analytical segment projection.
2. Use a distance softmax to form a differentiable weighted curve position,
   avoiding the evaluator's hard nearest-segment `argmin` during training.
3. Supervise scalar position with Charbonnier reconstruction and
   horizontal/vertical scalar-gradient matching.
4. Add deterministic long-range ordinal comparisons at half- and
   quarter-image offsets. Ignore target pairs below a small separation margin
   and use a smooth logistic ranking penalty for the rest.

The feature is disabled by default. Overall, gradient, and ordinal weights are
stored in `SourceConditioningConfig`, but excluded from its architecture
signature because they do not change checkpoint tensor shapes.

## Alternatives considered

- Hard colormap inversion: evaluation-aligned but discontinuous at segment
  selection boundaries.
- Viridis luminance proxy: cheap but introduces a surrogate not used by the
  data contract or evaluator.
- O(N^2) all-pairs ranking: closest to global rank correlation but unnecessary
  and memory-heavy for the first gate.

## Verification

- Soft projection recovers known viridis positions and supplies finite,
  non-zero RGB gradients.
- Correctly ordered predictions score below reversed predictions.
- The loss backpropagates through a frozen VAE into LoRA parameters.
- One-step SSD-1B/MPS smoke completed at loss 1.100631, checkpointed, and
  reloaded in a fresh evaluation process.
- Focused tests: 26 passed. Full regression: 803 passed, 13 skipped. Ruff,
  mypy, and `git diff --check` passed.

## Matched outcome

All arms use four train and four validation images, 128 px, 60 optimizer
steps, four inference steps, rank/alpha 8/8, LR `5e-4`, seed 42, and no EMA.
Two scalar weights were tested because weight 0.2 changed the same-seed
one-step total loss by only about 0.022 relative to baseline.

### Raw metrics

| Arm | Split | RGB MAE | Position MAE | Pearson | Spearman |
|---|---|---:|---:|---:|---:|
| Baseline | train | 0.2182 | 0.3254 | -0.0076 | 0.0247 |
| Baseline | val | 0.2124 | 0.2741 | 0.0705 | -0.0411 |
| Scalar geometry, weight 0.2 | train | 0.2133 | 0.3118 | 0.0260 | 0.0533 |
| Scalar geometry, weight 0.2 | val | 0.2111 | 0.2845 | 0.0478 | -0.0656 |
| Scalar geometry, weight 1.0 | train | 0.2123 | 0.3065 | 0.0272 | 0.0185 |
| Scalar geometry, weight 1.0 | val | 0.2122 | 0.2884 | 0.0520 | -0.0548 |

### Configured calibration

| Arm | Split | RGB MAE | Position MAE | Pearson | Spearman |
|---|---|---:|---:|---:|---:|
| Baseline | train | 0.2267 | 0.3526 | 0.0108 | 0.0293 |
| Baseline | val | 0.1965 | 0.3005 | 0.0528 | -0.0415 |
| Scalar geometry, weight 0.2 | train | 0.2198 | 0.3353 | 0.0700 | 0.0751 |
| Scalar geometry, weight 0.2 | val | 0.1984 | 0.3045 | 0.0354 | -0.0627 |
| Scalar geometry, weight 1.0 | train | 0.2204 | 0.3342 | 0.0741 | 0.0681 |
| Scalar geometry, weight 1.0 | val | 0.2014 | 0.3075 | 0.0580 | -0.0259 |

## Consequences

The 0.2 arm moves both raw train correlations in the intended direction, but
the gains are small relative to per-split standard deviations around
0.20--0.35 and do not transfer to validation. Raising the weight to 1.0 does
not strengthen raw train ordering and still regresses raw validation
correlation. Therefore the loss-semantics hypothesis is insufficient in the
current diffusion/LoRA path.

- Keep the implementation as an opt-in diagnostic; do not enable it by
  default or combine it with the adapter.
- Stop scalar-loss weight tuning and do not scale this configuration.
- The next architecture gate should place a small discriminative scalar-depth
  head on frozen SSD/VAE features and first prove four-image overfit. This
  separates representation quality from iterative RGB diffusion decoding.
- If the discriminative head also cannot memorize, stop SSD-1B depth work and
  move to a pretrained monocular-depth backbone or metric-depth dataset.

## References

- `docs/plans/2026-07-27-viridis-scalar-geometry-gate-design.md`
- `docs/adr/0001-depth-geometry-gate-spatial-supervision.md`
- `docs/OPEN_VISION_BANANA_DEEP_ANALYSIS_v130.md`
