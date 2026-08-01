# Frozen SSD/VAE scalar-depth probe design

## Requirements

### Functional

- Load the locally cached SSD-1B SDXL pipeline without modifying its weights.
- Encode four declared v3 train images through the frozen VAE and one fixed
  SSD UNet forward pass conditioned on the truthful relative-depth prompt.
- Capture deterministic multi-scale source features and cache them before
  training.
- Train only a small scalar-depth head and save its weights, feature contract,
  metrics, loss history, and rendered predictions.
- Stop early only when mean train Pearson and Spearman are both at least 0.95
  and scalar-position MAE is at most 0.05.

### Non-functional

- **Performance:** 128 px, four images, one-time MPS feature extraction; head
  training should finish in minutes rather than hours.
- **Memory:** cached features move to CPU before the base pipeline is released;
  only the head and four feature batches remain on MPS during optimization.
- **Reliability:** fixed manifest samples, prompt, timestep, and seed; fail on
  missing/shape-changing hook outputs or non-finite loss.
- **Compatibility:** no changes to source-conditioned LoRA checkpoints or
  production inference defaults. Probe checkpoints are a separate artifact.
- **Security/privacy:** offline local weights and data only; no Hub upload or
  external service.

## Alternatives

1. **VAE latent only.** Cheapest and easiest to decode, but a pass would only
   demonstrate that a convolutional autoencoder preserves image detail. It
   would not answer whether SSD features contain usable geometry.
2. **UNet final noise prediction only.** Uses SSD, but collapses the entire
   hierarchy into four output channels optimized for denoising rather than
   representation readout.
3. **VAE latent plus UNet down/mid features.** Uses fine VAE detail and the
   frozen UNet's multi-scale semantic hierarchy. A small FPN can read all
   scales without training or duplicating the backbone.

Select option 3. The feature set also includes the final UNet output as a cheap
diagnostic, but the head does not depend on a source-conditioned 8-channel
checkpoint or any previous LoRA.

## Architecture

```text
input RGB [-1,1]
       |
       +--> frozen VAE encoder --> latent z -------------------------+
       |                                                             |
       +--> z + fixed t=0 + relative-depth text                      |
                    |                                                 |
                    v                                                 |
             frozen SSD-1B UNet                                      |
          down-0, down-1, down-2, mid, final                         |
                    |                                                 |
                    +---------------- feature cache ------------------+
                                                                      v
                                                     1x1 projections to width C
                                                                      |
                                                     resize to latent resolution
                                                                      |
                                                          concat + 3x3 fusion
                                                                      |
                                                        three 2x upsample blocks
                                                                      |
                                                          sigmoid scalar depth
```

The extractor runs in inference mode with all VAE, text-encoder, and UNet
parameters frozen. Forward hooks are registered only for one call and removed
in `finally`. The VAE uses posterior mode, so extraction is deterministic.

`ScalarDepthProbe` receives the discovered channel contract, projects every
scale independently, upsamples to the VAE latent grid, fuses, and decodes to
the requested raster size. GroupNorm avoids batch-size dependence. Its loss is
scalar Smooth L1 plus scalar-gradient matching and a differentiable Pearson
term; Spearman remains an evaluation-only gate.

## Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Hook returns a tuple or unexpected object | extract the primary tensor and validate batch/spatial dimensions |
| Hook leaks across calls | register per call and always remove in `finally` |
| Base model accidentally trains | `requires_grad_(False)`, inference mode, and tests asserting no backbone gradients |
| Feature contract changes | serialize ordered names/channels and require exact match on load |
| Head trivially outputs a constant | gate on Pearson, Spearman, and MAE together |
| Four-image head memorizes but cannot generalize | this gate claims representation readout only; validation is explicitly out of scope |
| Gate still fails | stop SSD-1B depth work; do not tune diffusion loss or add ControlNet |

## Verification

- Synthetic extractor test with fake VAE/UNet proves feature ordering, hook
  cleanup, frozen parameters, and deterministic shapes.
- Head tests cover multiscale shape fusion, gradients, invalid contracts, and
  a tiny synthetic overfit.
- CLI helper tests cover manifest selection, threshold conjunction, and JSON
  serialization.
- Full repository regression, Ruff, mypy, and `git diff --check`.
- Real SSD-1B 128 px feature smoke before the full four-image gate.

## Outcome

The feature smoke and formal gate both completed offline on MPS. SSD-1B
produced six ordered levels with channels `4/320/640/1280/1280/4`. The
509,745-parameter head passed the four-image gate at step 75 of the 2,000-step
budget: scalar MAE `0.0330`, Pearson `0.9736`, and Spearman `0.9740`. All four
individual samples also passed MAE/Pearson/Spearman thresholds. Total runtime
was 12.0 seconds including base-model load and feature caching.

The gate therefore rejects the hypothesis that SSD/VAE frozen features contain
no readable geometry. It instead isolates the prior failures to diffusion-side
readout or optimization. The next experiment must remain discriminative and
split-clean, with a VAE-only control against VAE+SSD features; four-image
memorization alone does not authorize a generalization claim.
