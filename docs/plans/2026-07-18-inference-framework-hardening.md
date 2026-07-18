# Inference Framework Hardening Implementation Plan

> **For Codex:** Implement the tasks in this plan incrementally and verify each task before moving on.

**Goal:** Improve the inference framework's input contracts, tiled decoding behavior, and observability without changing the paper-aligned RGB task representation.

**Architecture:** Keep `VisionBanana` as the public facade and `BaseGenerator` as the diffusion engine. Add centralized input validation at the facade boundary, make tiled inference preserve task metadata and output shape, and add structured per-call diagnostics through an optional callback/logger payload. Avoid model downloads and keep tests generator-free.

**Tech Stack:** Python, PyTorch, pytest, existing `VisionBanana` decoders.

---

### Task 1: Define inference input contracts

**Files:**
- Modify: `src/open_vision_banana/inference/pipeline.py`
- Test: `tests/test_integration.py`

Validate image rank, channel count, finite values, value range, batch/task constraints, and `strength`/tile arguments before touching the generator. Use clear `ValueError` messages.

### Task 2: Harden tiled inference contracts

**Files:**
- Modify: `src/open_vision_banana/inference/pipeline.py`
- Test: `tests/test_integration.py`

Reject invalid overlap/tile sizes, preserve the caller's `height`/`width` arguments, and ensure stitched `rgb_output` matches the original image raster. Add synthetic tests for continuous and discrete tile paths.

### Task 3: Add inference diagnostics

**Files:**
- Modify: `src/open_vision_banana/inference/pipeline.py`
- Test: `tests/test_integration.py`

Expose optional `metadata` in prediction results with task, batch size, input/output shape, seed, and inference settings. Keep existing result keys backward compatible.

### Task 4: Run targeted verification

Run the inference integration tests, paper benchmark tests, Ruff, Mypy, compile checks, and `git diff --check`. Report environment-only failures separately.
