# Multi-Task LoRA Runner Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement an audited five-task SDXL/SSD LoRA smoke runner and decide
whether full multi-task training is safe to launch.

**Architecture:** Audit `train_val_split.json` first, then load only declared
paired datasets and preserve task prompts. Reuse frozen-backbone LoRA training,
write deterministic metadata/checkpoints, and run task-level finite-loss and
validation sanity gates before any full run.

**Tech Stack:** Python, PyTorch, diffusers, existing dataset/tuner utilities,
pytest.

---

### Task 1: Data audit helpers

Add a script that reports task counts, train/val overlap, missing paired files,
and a stable manifest checksum. Reject unsupported or incomplete requested tasks.

### Task 2: Test contracts

Add tests for supported-task validation, disjointness, checksum determinism,
ratio normalization, and strict checkpoint metadata.

### Task 3: Runner

Implement a lightweight runner that invokes the existing LoRA tuner only after
the audit passes, with task ratios, seed, resolution, max steps, and output
metadata exposed as CLI arguments.

Follow-up implementation adds manifest-backed held-out datasets for all five
tasks, named per-task validation losses in checkpoint metadata, normalized
`--task-ratios`, and deterministic `--task-schedule` coverage for smoke gates.
Explicit v3 training defaults to all declared training samples; smoke caps are
still available, and generation validation is filtered before capping.

The first full-data run is now active at
`output/multitask_lora_v3_full_256/`: 19,118 train samples, 320 held-out
validation samples, 256px, rank 128, no-replacement traversal, and a 3,000
optimizer-step ceiling. Quality promotion remains pending its checkpoints.

### Task 4: Verification

Run audit-only, synthetic/CPU smoke, and a small offline SSD-1B five-task smoke.
Run focused and full regression tests.

### Task 5: Decision

Update the deep analysis with actual task coverage and smoke results. Launch a
full multi-task run only if the smoke passes finite-loss/checkpoint/task gates.
