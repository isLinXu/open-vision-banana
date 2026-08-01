# ADR-0006: Audited multi-task LoRA runner before full training

## Status

Accepted; implementation in progress.

## Context

The frozen SSD feature probe now has full-dataset relative-depth evidence, but
the complete multi-task generation model has not passed task-level gates. The
repository contains multiple task directories with different target semantics,
and the legacy multi-task script depends on a manifest format absent from the
current v3 dataset.

## Decision

Add a manifest-driven audit and runner around the existing SDXL/SSD LoRA path.
Start with depth, normal, semantic, edge/structure, and generation. Require
train/val disjointness and complete declarations before model loading. Preserve
task-specific prompts and sampling ratios in checkpoint metadata. Keep
instance/referring/keypoint/object-detection/multiview/depth-edge/specular
removal audit-only until their target and metric contracts are explicit.

## Consequences

Positive: prevents silent task omission, makes smoke runs reproducible, and
keeps the validated frozen-depth result separate from unvalidated generative
claims.

Negative: initial scope does not cover every directory in v3; multi-task LoRA
quality may still be poor and requires separate per-task validation.

## Rejected alternatives

- Reusing `train_v5_multitask.py` with an improvised JSONL conversion.
- Running full data before a finite-loss/checkpoint smoke.
- Claiming production quality from a single aggregate loss.
