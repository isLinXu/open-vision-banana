# Frozen-Feature Multi-Seed Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reproducible Stage 8 CLI comparing matched VAE-only and VAE+SSD
heads with train/dev/official-val splits and multiple seeds.

**Architecture:** Cache frozen SSD/VAE features once. For each seed, clone one
head state and schedule, train both masks, select checkpoints from dev only,
then score official validation at those endpoints. Emit strict JSON with
checksums, histories, per-seed metrics, aggregates, and the promotion gate.

**Tech Stack:** Python, PyTorch, pytest, existing `ScalarDepthProbe` and
`VisionTaskDataset` utilities.

---

### Task 1: Specify deterministic split and selection contracts

**Files:** `scripts/compare_frozen_depth_features.py`, `tests/test_depth_probe.py`

Add helpers for fit/dev/official-val selection, split checksums, dev joint-score
checkpoint selection, and mean/std aggregation. Test disjointness, deterministic
checksums, and that changing official-val metrics cannot change a selected step.

### Task 2: Implement multi-seed matched training

Extend the comparison CLI with `--dev-samples`, `--seeds`, and dev-only endpoint
selection. Preserve matched initialization, schedules, parameter counts, and
feature masks for each seed. Evaluate official validation only after selection.

### Task 3: Verify locally

Run focused depth-probe tests, the full test suite, and a tiny CPU/synthetic
smoke where model feature extraction is replaced by test fixtures if needed.

### Task 4: Run Stage 8 and document the result

Run the offline MPS experiment with at least three seeds and a larger subset,
record result artifacts under `output/`, and update the deep analysis and Stage 2
plan with per-seed/aggregate metrics and the promotion decision.
