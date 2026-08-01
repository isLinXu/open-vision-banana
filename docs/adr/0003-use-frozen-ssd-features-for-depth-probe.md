# ADR-0003: Use frozen SSD/VAE features for a scalar-depth probe

## Status

Accepted and evaluated. The stage-6 four-image architecture gate passed.

## Context

The 8-channel LoRA diffusion baseline, decoded-RGB structure loss,
T2I-Adapter, and direct scalar ordinal objective all failed to learn stable
four-image relative-depth ordering. Continuing to tune the same diffusion
decoder cannot distinguish unavailable representation from an ineffective RGB
generation readout.

The next experiment must isolate representation readout, remain practical on
Apple MPS, and make no generalization claim.

## Decision

Freeze the locally cached SSD-1B pipeline and extract VAE latent, UNet down
block, mid block, and final features from one deterministic `t=0` forward pass
conditioned on the audited relative-depth prompt. Cache those features, release
the backbone, and train a lightweight FPN-style scalar-depth head.

The gate passes only when all three train metrics pass simultaneously:
Pearson at least 0.95, Spearman at least 0.95, and scalar MAE at most 0.05.

## Consequences

### Positive

- Separates frozen representation quality from stochastic diffusion decoding.
- Removes billion-parameter backpropagation from the overfit loop.
- Produces a small, inspectable checkpoint with a strict feature contract.

### Negative

- Forward hooks couple the probe to SDXL UNet block structure.
- Four-image memorization does not establish held-out depth quality.
- A sufficiently expressive head may memorize image-specific cues rather than
  semantic depth, so passing only authorizes a later validation gate.

### Neutral

- The probe is a research-only path and does not replace RGB task inference.
- Existing LoRA and spatial-adapter checkpoints are neither read nor changed.

## Alternatives considered

- VAE-only decoder: does not test SSD representations.
- UNet final output only: discards multi-scale intermediate information.
- Fine-tune the UNet with a depth head: fails to isolate frozen feature quality
  and restores the expensive optimization path this gate is meant to avoid.

## Outcome

The real offline SSD-1B/MPS run discovered this ordered feature contract:

| Feature | Channels | Spatial size for four 128 px images |
|---|---:|---:|
| VAE latent | 4 | 16x16 |
| UNet down 0 | 320 | 8x8 |
| UNet down 1 | 640 | 4x4 |
| UNet down 2 | 1280 | 4x4 |
| UNet mid | 1280 | 4x4 |
| UNet final | 4 | 16x16 |

The FPN head has 509,745 trainable parameters. With seed 42, LR `3e-3`, and
an upper bound of 2,000 full-batch steps, it passed at step 75 in 12.0 seconds
including model load, feature extraction, training, and artifact writing.

| Step | MAE | Pearson | Spearman |
|---:|---:|---:|---:|
| 1 | 0.1649 | 0.5239 | 0.5399 |
| 25 | 0.0740 | 0.9438 | 0.9398 |
| 50 | 0.0575 | 0.9619 | 0.9608 |
| 75 | 0.0330 | 0.9736 | 0.9740 |

Every individual image also passed all three thresholds:

| Sample | MAE | Pearson | Spearman |
|---|---:|---:|---:|
| `depth_hyp_real_00000.jpg` | 0.0357 | 0.9566 | 0.9619 |
| `depth_hyp_real_00001.jpg` | 0.0260 | 0.9716 | 0.9731 |
| `depth_hyp_real_00002.jpg` | 0.0284 | 0.9859 | 0.9803 |
| `depth_hyp_real_00003.jpg` | 0.0421 | 0.9804 | 0.9808 |

The checkpoint strictly reloads against the saved six-level feature contract.
The full repository regression passed 811 tests with 13 skipped; focused
tests, Ruff, mypy, and `git diff --check` passed.

### Decision after the gate

- Continue the SSD depth representation line; do not stop it.
- Treat the failed diffusion experiments as a readout/optimization failure,
  not evidence that SSD lacks all usable geometry.
- Do not claim generalization from four-image memorization.
- The next split-clean gate must compare VAE-only against VAE+SSD features on
  a larger train set and held-out validation. SSD features earn further work
  only if they improve validation ordering over the VAE-only control.

That matched comparison was subsequently executed in ADR-0004 and passed.

## References

- `docs/plans/2026-07-27-frozen-feature-depth-probe-design.md`
- `docs/adr/0002-depth-scalar-geometry-objective.md`
- `docs/adr/0004-match-vae-only-and-ssd-feature-ablation.md`
