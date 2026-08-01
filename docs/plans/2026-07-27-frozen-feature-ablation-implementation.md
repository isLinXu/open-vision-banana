# Frozen Feature Ablation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run a capacity- and randomness-matched held-out comparison of VAE-only and VAE+SSD scalar-depth probes.

**Architecture:** Extend the existing six-level head with a post-projection binary feature mask. A new split-clean CLI extracts one shared feature cache, clones identical head initialization, reuses one batch schedule, and reports fixed-endpoint validation deltas.

**Tech Stack:** Python 3.11, PyTorch/MPS, Diffusers SSD-1B, pytest, Ruff, mypy.

---

### Task 1: Test feature masking and contribution gate

**Files:**
- Modify: `tests/test_depth_probe.py`
- Modify: `src/open_vision_banana/training/depth_probe.py`

**Steps:**
1. Add a failing test that masked projected levels cannot change output when
   their raw tensors change.
2. Add failing validation tests for mask length/value errors.
3. Add failing tests requiring both correlation deltas and the MAE guardrail.
4. Run `pytest -q tests/test_depth_probe.py` and confirm failure.
5. Implement the minimal mask and gate helpers; rerun until green.

### Task 2: Implement matched experiment helpers

**Files:**
- Create: `scripts/compare_frozen_depth_features.py`
- Modify: `tests/test_depth_probe.py`

**Steps:**
1. Test disjoint manifest selection for 32 train and 16 validation names.
2. Test deterministic batch schedules and state-dict checksums.
3. Implement shared feature caching, arm training, fixed endpoint evaluation,
   strict checkpoint writing, and result serialization.
4. Assert both arm parameter counts, initialization checksums, and batch
   schedule checksums are identical in the report.

### Task 3: Verify software and real smoke

**Files:**
- Verify only

**Steps:**
1. Run `pytest -q tests/test_depth_probe.py tests/test_source_conditioning.py`.
2. Run Ruff, mypy, and `git diff --check`.
3. Run the complete test suite.
4. Run a 4-train/2-val, 1-step real SSD smoke and verify both checkpoints and
   finite held-out metrics.

### Task 4: Execute the formal matched gate

**Files:**
- Output: `output/ssd1b_frozen_feature_ablation_32x16_300_128_seed42/`

**Steps:**
1. Extract shared features for 32 declared train and 16 official val images.
2. Train VAE-only and VAE+SSD for exactly 300 matched steps, batch 8, LR
   `1e-3`, seed 42, evaluating every 25 steps.
3. Compare fixed endpoint validation Pearson, Spearman, and MAE.
4. Promote SSD features only if both correlation deltas are at least 0.05 and
   MAE does not regress by more than 0.01.

### Task 5: Record the decision

**Files:**
- Modify: `docs/adr/0004-match-vae-only-and-ssd-feature-ablation.md`
- Modify: `docs/plans/2026-07-26-ssd-depth-quality-stage2-design.md`
- Modify: `docs/OPEN_VISION_BANANA_DEEP_ANALYSIS_v110.md`

**Steps:**
1. Record feature contract, checksums, endpoint histories, deltas, runtime, and
   exact split sizes.
2. State whether SSD intermediate features earn a larger validation run.
3. Preserve all unrelated dirty-worktree changes and do not stage or commit.

