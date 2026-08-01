# ADR-0004: Match VAE-only and VAE+SSD depth-probe capacity

## Status

Accepted and evaluated. SSD feature contribution passed the stage-7 held-out
gate.

## Context

The stage-6 VAE+SSD head memorized four train images, but that result cannot
attribute the signal to SSD because the VAE latent was also present. A held-out
ablation must isolate representation information without changing decoder
capacity or optimization randomness.

## Decision

Use one six-input `ScalarDepthProbe` architecture for both arms. Apply a binary
mask after per-level projection: VAE-only activates only the VAE latent slot;
VAE+SSD activates all VAE, UNet down, mid, and final slots. Clone one initial
state and reuse one batch-index schedule. Train on 32 declared train images and
evaluate 16 disjoint official validation images for a fixed 300 steps.

Promotion requires validation `delta Pearson >= 0.05`, `delta Spearman >=
0.05`, and SSD validation MAE no more than 0.01 worse than VAE-only.

## Consequences

### Positive

- Equal parameter count, initialization, batches, optimizer, and budget.
- Directly measures information added by SSD features beyond VAE detail.
- Fixed endpoint avoids per-arm validation-step cherry-picking.

### Negative

- Only one train/validation subset and seed are tested in the first gate.
- The VAE-only arm retains inactive parameters, so nominal parameter count is
  equal while effective trainable connectivity is smaller by design.
- Sixteen validation images still produce uncertain generalization estimates.

### Neutral

- SSD features are computed for both arms but masked from the VAE-only head.
- This research path remains separate from generative RGB inference.

## Alternatives considered

- Separate smaller VAE head: rejected because decoder capacity differs.
- Zero raw SSD tensors: rejected because projection biases remain active.
- Train only at the best validation step: rejected for the primary gate because
  it permits different validation-selected budgets.

## Outcome

The formal offline MPS run used 32 declared train samples, 16 disjoint official
validation samples, 128 px, seed 42, batch 8, LR `1e-3`, and exactly 300 steps.
Both arms had 509,745 parameters, loaded the same state checksum
`c65788d8...43a95f2`, and consumed batch schedule checksum
`ebd2b251...affb073a`. The complete run took 36.3 seconds.

### Fixed endpoint

| Arm | Split | MAE | Pearson | Spearman |
|---|---|---:|---:|---:|
| VAE-only | train | 0.1256 | 0.8694 | 0.8569 |
| VAE-only | val | 0.2591 | 0.1425 | 0.1629 |
| VAE+SSD | train | 0.0670 | 0.9602 | 0.9535 |
| VAE+SSD | val | 0.2402 | 0.3663 | 0.3682 |

Validation deltas, computed as VAE+SSD minus VAE-only, were:

- Pearson: `+0.2238`
- Spearman: `+0.2053`
- MAE: `-0.0190` (improvement)
- RMSE: `-0.0145` (improvement)

Both correlation gains exceed the predeclared `+0.05` threshold by more than
4x, and the error guardrail also passes. The result is not dependent on a
validation-selected endpoint: VAE+SSD remained ahead at step 300 even though
its best observed validation correlation occurred near step 100.

### Decision after the gate

- Promote frozen SSD intermediate features to the next discriminative depth
  validation stage.
- Attribute a material portion of held-out geometry readout to SSD features,
  not only the VAE latent or head capacity.
- Keep the generative diffusion/LoRA/Adapter path stopped; this gate does not
  show that RGB diffusion can exploit the same information.
- The next gate should use multiple seeds and a larger split, with an internal
  development subset for checkpoint selection and official validation kept
  untouched until final evaluation.

## References

- `docs/plans/2026-07-27-frozen-feature-ablation-design.md`
- `docs/adr/0003-use-frozen-ssd-features-for-depth-probe.md`
