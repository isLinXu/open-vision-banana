# ADR-0001: Use One Shared Adapter for the Reproduction Baseline

- Status: Accepted
- Date: 2026-07-25

## Context

The paper's central claim is that a prompt-controlled image generator can share one representation across vision tasks. Task heads, separate adapters, or MoE routing could improve individual metrics but would test a different architecture.

## Decision

The reproduction track freezes the base generator and trains one shared LoRA/rsLoRA adapter for semantic, instance, referring/reasoning, depth, normal, and generation-preservation samples. Task selection happens through prompts and RGB output conventions. Model-specific target modules come from the registry when config uses `target_modules: null`.

## Consequences

- Cross-task transfer and interference are measured directly.
- One checkpoint serves all paper tasks.
- Capacity changes are limited to rank/target-module ablations in the optimization track.
- Task-specific adapters, heads, or routing require a separate ADR and cannot replace baseline results.
