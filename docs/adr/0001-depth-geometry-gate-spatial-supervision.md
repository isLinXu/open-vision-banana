# ADR-0001: Compare decoded-x0 structure loss with multi-scale spatial conditioning

## Status

Accepted and evaluated for the stage-4 depth geometry gate. Neither variant
passed the promotion criterion.

## Context

SSD-1B's 8-channel source-conditioned path completes training, inference, and
checkpoint round trips. Across 32x100, 8x200, and high-learning-rate probes,
RGB and range errors improve while train/validation Pearson and Spearman depth
correlations remain near zero. The learned source-channel input weights are
non-zero, so the remaining question is whether stronger spatial supervision or
deeper spatial conditioning is required.

The gate must run locally on Apple MPS at 128 px, preserve legacy checkpoints,
avoid downloading additional weights, and produce an apples-to-apples result
from fixed train/validation samples and seeds.

## Decision

Implement two opt-in variants on the same 8-channel SSD/LoRA baseline:

1. **Structure loss**: decode the implied x0 through the frozen VAE and add a
   multi-scale Charbonnier reconstruction term plus horizontal/vertical
   gradient matching. This supplies a differentiable spatial objective without
   changing inference architecture.
2. **Spatial adapter**: use a randomly initialized Diffusers
   `T2IAdapter(full_adapter_xl)` with a 16x initial downscale and zero-initialized
   1x1 output projections. Inject its four source-image feature maps through
   `down_intrablock_additional_residuals` at every denoising step. This adds
   roughly 83M trainable parameters, far below a full ControlNet copy.

Both features remain disabled by default. Their weights and architecture
settings are stored in source-conditioning checkpoints, and inference refuses
an adapter-declared checkpoint if its adapter state is missing.

## Non-functional requirements

- **Performance:** a 128 px, batch-1 MPS optimizer step must remain practical
  for a 100-step gate; adapter features are computed once per inference image.
- **Compatibility:** existing conditioning version-1 checkpoints must load
  unchanged and retain pixel-identical behavior when both features are off.
- **Reliability:** structure loss must reject non-finite values; adapter feature
  shape mismatches must fail before a misleading training run.
- **Security/privacy:** all inputs, checkpoints, and metrics stay local; no new
  network service or untrusted model code is introduced.
- **Maintainability:** training and inference share the same adapter module and
  serialized contract; no pipeline monkey-patching or persistent hooks.

## Consequences

### Positive

- Separates “insufficient spatial loss” from “insufficient condition depth.”
- Keeps the existing source-conditioned path and legacy artifacts usable.
- Uses the native UNet residual interface, reducing custom integration code.

### Negative

- Decoded-x0 loss adds a differentiable VAE decode and therefore memory/time.
- The spatial adapter adds about 83M trainable parameters and checkpoint size.
- A successful tiny gate still does not prove held-out or metric-depth quality.

### Neutral

- Viridis projection remains evaluation/post-processing only; the new training
  loss operates on decoded RGB and image gradients.

## Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| VAE decode produces NaN/Inf | fail the step with a descriptive error |
| Random adapter initially destroys base output | zero-init all output projections |
| Adapter state is absent or incompatible | strict checkpoint load and shape checks |
| MAE falls but geometry stays flat | gate on train/val Pearson and Spearman |
| MPS memory pressure | 128 px, batch 1, no pixel loss duplication, adapter-only feature caching at inference |

## Alternatives considered

- **Full ControlNet:** stronger and proven, but duplicates a large fraction of
  SSD-1B and is too expensive for the first local discriminator gate.
- **More steps on the current path:** rejected because 8x200 and LR probes
  already improved colour without learning train-set ordering.
- **Plain pixel L1:** already showed unstable refinement and does not explicitly
  distinguish spatial gradients.
- **Replace diffusion with a discriminative depth head:** potentially simpler,
  but outside the current RGB-generative paradigm validation.

## Gate protocol

- Fixed model/data: SSD-1B, 128 px, four declared train samples, four declared
  validation samples, seed 42, rank/alpha 8/8, four inference timesteps.
- First gate: one optimizer step must produce finite gradients in the new
  module and checkpoint round-trip successfully.
- Short gate: compare identical step budgets. Advancement requires train
  Pearson and Spearman to improve materially over the matched baseline; RGB
  MAE alone is not a pass.

## Gate outcome

The matched comparison used SSD-1B, four declared train images, four declared
validation images, 128 px, rank/alpha 8/8, learning rate `5e-4`, 60 optimizer
steps, four inference timesteps, seed 42, and no EMA. The structure arm used a
Charbonnier weight of 5.0 and gradient weight of 1.0. The adapter arm used
scale 1.0 and downscale factor 16.

### Raw model output

| Arm | Split | RGB MAE | Relative-position MAE | Pearson | Spearman |
|---|---|---:|---:|---:|---:|
| Baseline | train | 0.2182 | 0.3254 | -0.0076 | 0.0247 |
| Baseline | val | 0.2124 | 0.2741 | 0.0705 | -0.0411 |
| Decoded-x0 structure | train | 0.2222 | 0.3215 | -0.0277 | -0.0276 |
| Decoded-x0 structure | val | 0.2171 | 0.2800 | 0.0816 | -0.0346 |
| Spatial adapter | train | 0.2158 | 0.3042 | 0.0004 | 0.0163 |
| Spatial adapter | val | 0.2199 | 0.3146 | -0.0040 | -0.1114 |

### Configured calibration

| Arm | Split | RGB MAE | Relative-position MAE | Pearson | Spearman |
|---|---|---:|---:|---:|---:|
| Baseline | train | 0.2267 | 0.3526 | 0.0108 | 0.0293 |
| Baseline | val | 0.1965 | 0.3005 | 0.0528 | -0.0415 |
| Decoded-x0 structure | train | 0.2239 | 0.3563 | -0.0232 | -0.0213 |
| Decoded-x0 structure | val | 0.1875 | 0.2924 | 0.1036 | -0.0231 |
| Spatial adapter | train | 0.2240 | 0.3423 | 0.0208 | -0.0022 |
| Spatial adapter | val | 0.2006 | 0.3038 | 0.0287 | -0.0600 |

The structure arm slightly improved validation correlation after calibration,
but train Pearson and Spearman both regressed and validation Spearman remained
negative. With only four samples and correlation standard deviations around
0.25--0.38, that isolated validation movement is not material evidence of
geometry learning. The adapter arm improved some train-set errors but made
both validation correlations worse, especially raw Spearman.

The adapter result is not explained by a disconnected or permanently
zero-initialized branch. After 60 steps, all four output-projection weight
tensors were fully non-zero, with norms 0.84, 1.66, 3.09, and 3.13. Its saved
checkpoint was also strictly loaded for fresh-process evaluation. The adapter
checkpoint occupied about 547 MB including optimizer state, versus 58 MB for
the other two arms, without a geometry benefit.

### Decision after the gate

- Do not scale either arm to a larger dataset or promote the adapter to a full
  ControlNet yet.
- Treat the result as evidence that adding RGB-local structure or spatial
  capacity alone does not align the objective with relative-depth ordering.
- The next discriminator gate should supervise a differentiable scalar
  viridis-curve position directly, using ordinal/ranking and scalar-gradient
  losses. Only after that objective learns train-set ordering should it be
  combined with the multi-scale adapter.
- Preserve both implementations as opt-in research paths; keep them disabled
  by default.

The scalar discriminator was subsequently executed in ADR-0002 and also
failed the ordering-memory gate, so it was not combined with the adapter.

## References

- `docs/OPEN_VISION_BANANA_DEEP_ANALYSIS_v110.md`
- `docs/plans/2026-07-26-ssd-depth-quality-stage2-design.md`
- `docs/adr/0002-depth-scalar-geometry-objective.md`
- Diffusers `T2IAdapter` and `UNet2DConditionModel.down_intrablock_additional_residuals`
