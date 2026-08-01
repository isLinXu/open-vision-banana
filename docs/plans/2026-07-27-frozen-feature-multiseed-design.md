# Stage 8 Frozen-Feature Multi-Seed Evaluation Design

## Goal

Determine whether SSD intermediate features improve held-out relative-depth
generalization beyond a matched VAE-only control across multiple random seeds.

## Protocol

The declared `train.depth` manifest is selected in manifest order and split
deterministically into `train_fit` and `dev`. The declared `val.depth` names are
kept as an untouched `official_val` split. Each seed uses the same feature cache,
head initialization, replacement-sampling schedule, optimizer, loss, and step
budget for both arms. Checkpoints are selected independently by a joint dev
score, then evaluated exactly once on official validation at the selected
endpoint. Official validation metrics never affect selection.

The VAE-only arm uses the first (VAE latent) slot of the six-input head; the
VAE+SSD arm uses all slots. Both retain the same nominal parameter count and
state checksum. The experiment reports per-seed and aggregate mean/std metrics,
selection steps, split checksums, and VAE+SSD minus VAE-only deltas.

## Promotion gate

Promote the frozen SSD path only when aggregate official-val Pearson and
Spearman deltas each meet the configured threshold, at least two of three seeds
improve both correlations, and the aggregate MAE regression remains within the
guardrail. The development and official-val correlation deltas must have the
same sign; otherwise the result is inconclusive and the generative diffusion
path remains stopped.

## Non-functional requirements

- No data overlap between fit, dev, and official validation.
- Split and schedule checksums make reruns auditable.
- Strict JSON output rejects NaN/Infinity.
- Features remain frozen and are cached once per selected image.
- The CLI remains usable for small smoke runs by reducing sample counts,
  resolution, steps, and seed count.
