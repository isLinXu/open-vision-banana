# ADR-0007: Promote the frozen-feature scalar-depth estimator

## Status

Accepted and production-validated.

## Context

The source-conditioned SSD diffusion path failed its eight-image raw geometry
gate after 4,800 steps, while the frozen SSD/VAE feature probe passed its
four-image gate and repeatedly improved held-out depth ordering over a matched
VAE-only control. The full 128 px, three-seed experiment reached official-val
MAE `0.1733`, Pearson `0.6527`, and Spearman `0.6345`.

The paused 1024 px diffusion run had reached only step 10 and had no checkpoint.
Its 512 px source data were being upsampled, and resuming it would consume far
more MPS time without resolving the failed diffusion RGB readout.

## Decision

Use frozen SSD/VAE features plus `ScalarDepthProbe` as the production depth
architecture. SSD and VAE weights remain frozen; only the lightweight scalar
head is trained.

Before a deployment refit, run matched VAE-only and VAE+SSD gates at 256 px and
then native 512 px. Continue to the next resolution only when the predeclared
held-out contribution gate passes. Endpoint selection uses the internal dev
split, and official validation is evaluated only after selection. When the
selected head is refit on a larger training split, preserve its selected sample
exposure (equivalent epochs), not its literal optimizer-step count.

Store each selected head as a versioned artifact containing:

- `model.safetensors` for tensors;
- `config.json` for the base model, feature contract, data split, resolution,
  selected step, metrics, and training settings;
- a SHA-256 checksum verified before loading.

Every saved weight must also produce raw scalar metrics and source/target/
prediction visualizations. Inference returns normalized relative depth and a
separate colormap rendering; the rendering is not used as the regression
target at inference time.

## Validation outcomes

Both resolution gates used 128 fit, 32 dev, and 32 official-validation samples
with seeds 42, 43, and 44. The feature contribution gate passed at both
resolutions:

| Resolution | VAE+SSD MAE | Pearson | Spearman | Pearson delta | Spearman delta |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.1810 | 0.5830 | 0.5665 | +0.1607 | +0.1593 |
| 512 | 0.1824 | 0.5786 | 0.5782 | +0.1641 | +0.1678 |

The first 256 px full-data refit did not preserve the dev-selected sample
exposure and was under-trained. It is superseded by the exposure-scaled refit.
The corrected run trained on all 2,181 current train samples and evaluated the
115 official-validation samples once. Across three seeds it reached MAE
`0.1509`, Pearson `0.7005`, and Spearman `0.6786`.

The native 512 px production refit then used the same 2,181/115 split and
exposure-scaled schedule. Its three-seed official-validation aggregate is MAE
`0.12960 +/- 0.00202`, RMSE `0.18642 +/- 0.00219`, Pearson
`0.77146 +/- 0.00579`, and Spearman `0.75653 +/- 0.00517`:

| Seed | Refit steps | MAE | RMSE | Pearson | Spearman |
|---:|---:|---:|---:|---:|---:|
| 42 | 4,686 | 0.13177 | 0.18823 | 0.76460 | 0.74936 |
| 43 | 5,112 | 0.13012 | 0.18769 | 0.77103 | 0.75886 |
| 44 | 5,112 | **0.12690** | **0.18335** | **0.77875** | **0.76137** |

Seed 44 is the deployment candidate because it is best on all four official
validation metrics. The production artifacts and aggregate report are stored
under
`output/ssd1b_depth_probe_production_refit_512_full_exposure_scaled_20260728/`.
All three `model.safetensors` files passed checksum verification.

Fresh-process inference reloaded every saved artifact, re-extracted frozen
SSD-1B features, and generated 12 source/target/prediction triptychs plus a
contact sheet per seed. On that fixed 12-image official-validation subset,
Pearson was `0.83140`, `0.84264`, and `0.83663` for seeds 42, 43, and 44.
These subset metrics are a reload and visualization check, not replacements
for the 115-image official metrics. The independent outputs are stored under
the production directory's `independent_inference/` subdirectory.

The current train count is one lower than the historical 128 px Stage-9 run
because `depth_hyp_real_00285` was subsequently identified as an all-black
source image and moved to the dataset quarantine. The current 2,181/115 split
contains every remaining depth pair and has no overlap.

## Consequences

### Positive

- Uses the only SSD depth path with repeated held-out evidence.
- Removes billion-parameter backpropagation and stochastic denoising.
- Gives inference a strict, inspectable feature and preprocessing contract.
- Safe tensor loading and checksums reduce checkpoint integrity risk.

### Negative

- The estimator remains coupled to SSD-1B/SDXL intermediate block structure.
- 256/512 feature extraction and CPU caching are still computationally costly.
- Relative-depth quality is not metric-depth calibration.
- This architecture does not generalize automatically to semantic or normal
  tasks.

### Neutral

- Frozen feature caches use half precision on CPU and are transferred to MPS
  per batch to control unified-memory pressure.
- Historical `.pt` probe checkpoints remain readable through
  `torch.load(weights_only=True)` but all new model artifacts use safetensors.

## Alternatives considered

**Resume the 1024 px source-conditioned diffusion run**

Rejected because it has no checkpoint, upsamples 512 px data, and does not
address the failed generative readout gate.

**Immediately adopt ControlNet-XL**

Deferred. A pretrained depth-conditioned ControlNet solves a different input
contract, and the deterministic feature path already has stronger evidence.

**Fine-tune additional SSD convolution blocks**

Retained only as a small controlled generative research gate. It is not
authorized for a large run until raw eight-image geometry thresholds pass.

## Failure modes and mitigations

- Feature contract drift: reject the checkpoint before inference.
- Artifact corruption: verify SHA-256 before loading tensors.
- Resolution memory pressure: keep half-precision features on CPU and stream
  minibatches to MPS.
- Endpoint leakage: select on dev, then evaluate official validation once.
- Misleading RGB quality: report raw scalar MAE/Pearson/Spearman and keep the
  colormap visualization secondary.

## References

- `docs/adr/0003-use-frozen-ssd-features-for-depth-probe.md`
- `docs/adr/0004-match-vae-only-and-ssd-feature-ablation.md`
- `docs/adr/0005-multiseed-dev-selected-frozen-depth-evaluation.md`
- `docs/plans/2026-07-26-ssd-depth-quality-stage2-design.md`
