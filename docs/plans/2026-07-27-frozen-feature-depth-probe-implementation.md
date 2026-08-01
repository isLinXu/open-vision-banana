# Frozen Feature Depth Probe Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and run a frozen SSD/VAE multi-scale feature probe that must overfit four relative-depth images.

**Architecture:** A read-only extractor caches VAE latent and fixed-timestep SSD UNet hierarchy tensors. A small FPN-style convolutional head projects, fuses, and upsamples the cached tensors to one scalar relative-depth channel.

**Tech Stack:** Python 3.11, PyTorch/MPS, Diffusers SDXL/SSD-1B, pytest, Ruff, mypy.

---

### Task 1: Specify feature and head behavior with tests

**Files:**
- Create: `tests/test_depth_probe.py`
- Create: `src/open_vision_banana/training/depth_probe.py`

**Step 1: Write failing extractor tests**

Use fake VAE and UNet blocks to require an ordered feature contract containing
VAE latent, each down-block output, mid output, and final output. Assert hooks
are removed and all backbone parameters remain gradient-free.

**Step 2: Run tests to verify failure**

Run: `pytest -q tests/test_depth_probe.py`

Expected: collection fails because `depth_probe` does not exist.

**Step 3: Write failing head and metric tests**

Require exact output size, finite non-zero gradients, rejection of channel
mismatches, joint gate thresholds, and a synthetic two-image overfit.

### Task 2: Implement the frozen feature extractor

**Files:**
- Create: `src/open_vision_banana/training/depth_probe.py`

**Step 1: Implement `FrozenSDXLDepthFeatureExtractor`**

Freeze VAE/UNet, encode posterior mode, build SDXL text/time conditioning,
register per-call hooks, run at fixed timestep, validate tensors, and remove
hooks in `finally`.

**Step 2: Run extractor tests**

Run: `pytest -q tests/test_depth_probe.py -k extractor`

Expected: extractor tests pass.

### Task 3: Implement the scalar probe and gate helpers

**Files:**
- Modify: `src/open_vision_banana/training/depth_probe.py`

**Step 1: Implement `ScalarDepthProbe`**

Use per-scale 1x1 projections, GroupNorm/SiLU, latent-grid interpolation,
fusion convolutions, three resize-convolution blocks, and sigmoid output.

**Step 2: Implement loss and metrics**

Add Smooth L1, scalar gradient, differentiable Pearson loss, exact evaluation
metrics, and a threshold conjunction helper.

**Step 3: Run all module tests**

Run: `pytest -q tests/test_depth_probe.py`

Expected: all tests pass.

### Task 4: Implement the four-image overfit CLI

**Files:**
- Create: `scripts/overfit_frozen_depth_probe.py`
- Modify: `tests/test_depth_probe.py`

**Step 1: Add manifest and result tests**

Test deterministic selection of the first four declared train images and
serialization of finite metrics and feature contracts.

**Step 2: Implement CLI**

Load SSD-1B offline, audit viridis, cache one sample at a time to CPU, release
the backbone, train the head on MPS, evaluate every fixed interval, stop on the
joint gate, and save `depth_probe.pt`, `result.json`, and prediction PNGs.

**Step 3: Run CLI unit tests**

Run: `pytest -q tests/test_depth_probe.py`

Expected: all tests pass without loading real model weights.

### Task 5: Verify software and real feature extraction

**Files:**
- Verify only

**Step 1: Run static and focused checks**

Run: `ruff check src/open_vision_banana/training/depth_probe.py scripts/overfit_frozen_depth_probe.py tests/test_depth_probe.py`

Run: `mypy src/open_vision_banana/training/depth_probe.py`

Run: `pytest -q tests/test_depth_probe.py tests/test_source_conditioning.py`

**Step 2: Run full regression**

Run: `pytest -q`

Expected: no failures.

**Step 3: Run one-image real SSD smoke**

Run the CLI with `--num-samples 1 --steps 1 --resolution 128` and verify a
finite loss, discovered feature contract, checkpoint, and result JSON.

### Task 6: Execute and document the gate

**Files:**
- Modify: `docs/adr/0003-use-frozen-ssd-features-for-depth-probe.md`
- Modify: `docs/OPEN_VISION_BANANA_DEEP_ANALYSIS_v110.md`
- Modify: `docs/plans/2026-07-26-ssd-depth-quality-stage2-design.md`

**Step 1: Run four-image overfit**

Use 128 px, seed 42, full batch four, at most 2000 steps, and evaluate every 25
steps. Do not use validation data.

**Step 2: Apply the gate**

Pass only if mean train Pearson/Spearman are each at least 0.95 and MAE is at
most 0.05. Otherwise mark the SSD-1B depth mainline stopped.

**Step 3: Record exact outcome**

Write metrics, runtime, parameter count, feature shapes, and the next decision
to the ADR and reports. Do not stage or commit unrelated dirty-worktree files.

