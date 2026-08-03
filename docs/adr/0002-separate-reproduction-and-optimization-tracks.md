# ADR-0002: Separate Reproduction and Optimization Tracks

- Status: Accepted
- Date: 2026-07-25

## Context

The closed paper system does not disclose its base weights, data, exact mix ratio, LoRA rank, optimizer, or schedule. Combining guessed reproduction choices with project optimizations would make results impossible to interpret.

## Decision

Maintain two canonical configs. `paper_reproduction_baseline.yaml` changes only the minimum required to exercise the paper method on an open base. `paper_optimized_multitask.yaml` contains explicitly labeled project extensions. Optimization experiments change one factor group at a time and are compared against immutable baseline manifests.

## Consequences

- Project results cannot be presented as numerical reproduction of Nano Banana Pro.
- Paper-disclosed, inferred, and extended choices remain distinguishable.
- Any promoted optimization must pass five-task, generation-preservation, stability, and efficiency gates.
