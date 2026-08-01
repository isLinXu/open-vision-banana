# Viridis scalar geometry gate design

## Context and success criterion

The stage-4 matched gate showed that decoded-RGB structure loss and an 82.8M
parameter multi-scale adapter do not make SSD-1B learn stable relative-depth
ordering. RGB error can improve while train and validation Pearson/Spearman
remain near zero. The next gate must therefore change loss semantics before
adding capacity.

Success is deliberately narrow: under the same four-train/four-validation,
128 px, 60-step, seed-42 protocol, train Pearson and Spearman must improve
materially over the matched baseline without relying on output calibration.
Validation movement is supporting evidence only because four samples have high
variance. A finite one-step MPS gradient and checkpoint round trip are required
before the formal run.

## Alternatives considered

1. Reuse the evaluator's hard nearest-segment viridis inverse. This is exact
   and differentiable within the selected segment, but `argmin` makes segment
   transitions discontinuous and can produce unstable gradients off-curve.
2. Use luminance as a scalar depth proxy. It is cheap and approximately
   monotonic for viridis, but it does not match the declared label contract or
   the evaluator and would introduce a new surrogate mismatch.
3. Softly project RGB onto every colormap segment and average positions with
   distance-based softmax weights. This is smooth, remains anchored to the
   registered viridis curve, and is inexpensive at 128 px because viridis has
   only a small number of segments.

Select option 3. Keep the implementation generic over registered colormaps so
the checkpoint's audited `target_colormap` remains the single source of truth.

## Components and data flow

`soft_colormap_position` accepts `[B,3,H,W]` RGB in `[0,1]`. For every
piecewise-linear colormap segment it computes the clamped analytical
projection, squared RGB distance, and curve position. A temperature-controlled
softmax over negative distance returns one scalar position per pixel. All
distance and softmax math runs in float32 for MPS stability.

`decoded_x0_scalar_geometry_loss` converts decoded prediction and RGB target
from `[-1,1]` to `[0,1]`, projects both to scalar positions, and combines:

- Charbonnier scalar-position reconstruction;
- horizontal/vertical scalar-gradient matching;
- deterministic long-range ordinal comparisons at half/quarter-image offsets.

Ordinal pairs whose target separation is below a small margin are ignored.
Remaining pairs use a smooth logistic ranking penalty, avoiding random pair
sampling so the variant consumes the same training RNG stream as the baseline.

The tuner decodes implied x0 once when any image-space loss is active. A new
overall weight plus gradient and ordinal component weights are serialized in
`SourceConditioningConfig`; they are training-only and therefore excluded from
the architecture compatibility signature. The feature is source-conditioned
depth only, requires an audited colormap, and remains disabled by default.

## Verification and gate protocol

- Unit tests cover on-curve projection accuracy, off-curve finite gradients,
  correct-vs-reversed ordinal behavior, validation errors, config round trips,
  and backpropagation through a frozen VAE.
- Run focused source-conditioning tests, Ruff, mypy, full pytest, and
  `git diff --check`.
- Run a one-step SSD-1B/MPS smoke and fresh-process inference load.
- Reuse the completed stage-4 baseline. Train one scalar-geometry arm for 60
  steps with all other hyperparameters fixed, then evaluate the same four
  train and four validation samples using raw and calibrated metrics.
- Do not combine the scalar loss with the adapter in this gate. Combination is
  allowed only if the scalar-only arm first demonstrates train-set ordering.

## Outcome

The implementation passed a one-step SSD-1B/MPS smoke (`loss=1.100631`),
checkpoint save, and fresh-process inference. The focused suite passed 26
tests; full regression passed 803 tests with 13 skipped. Ruff, mypy, and
`git diff --check` passed.

The formal 0.2-weight arm slightly improved raw train Pearson/Spearman from
`-0.0076/0.0247` to `0.0260/0.0533`, but validation regressed from
`0.0705/-0.0411` to `0.0478/-0.0656`. Because the same-seed one-step total loss
changed by only about 0.022 versus baseline, a single controlled weight probe
at 1.0 was run. It reached raw train `0.0272/0.0185` and validation
`0.0520/-0.0548`, so stronger weighting did not produce ordering memory.

The gate is rejected. The opt-in loss remains useful for diagnostics, but it
must not be combined with the adapter or scaled to more data. The next gate
should use a small discriminative scalar-depth head over frozen features to
separate representation quality from diffusion decoding and RGB generation.
