# ADR-0005: Multi-seed frozen-depth evaluation with dev-selected endpoints

## Status

Accepted; implementation pending.

## Context

Stage 7 showed a held-out SSD-feature gain for one seed and one fixed endpoint,
but checkpoint choice and seed variance were not measured. Reusing official
validation during step selection would overstate generalization.

## Decision

Split declared training names deterministically into `train_fit` and `dev`, keep
manifest-declared `official_val` untouched, and run matched VAE-only/VAE+SSD
arms for each requested seed. Select one endpoint per arm and seed using only a
joint dev score (Pearson and Spearman, with MAE as a tie-breaker). Evaluate the
selected checkpoint once on official validation. Aggregate mean/std metrics and
per-seed deltas, and require both correlation gains across the configured seed
majority plus an MAE guardrail for promotion.

## Consequences

Positive: reduces endpoint and seed cherry-picking, preserves the Stage 7
capacity control, and produces an auditable estimate of SSD feature value.

Negative: caching and training cost increase roughly with the selected image
count and seed count; dev selection can still be noisy for very small subsets.

## Alternatives rejected

- Selecting on official validation: leaks the endpoint criterion.
- Different heads for the two arms: confounds representation and capacity.
- One seed only: cannot expose unstable gains.
- Resuming diffusion/LoRA/ControlNet: outside the failed generative path gate.
