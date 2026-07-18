# Independent Report Hardening Implementation Plan

> **For Codex:** Implement the tasks in this plan incrementally and verify each task before moving on.

**Goal:** Address the independent report's highest-risk reliability findings while preserving the existing SD3 and training changes.

**Architecture:** Keep the current lazy model registry and pipeline boundaries. Harden checkpoint deserialization, make MPS memory detection platform-aware, and expose the broader LoRA defaults already used by model-specific registry entries. Add focused tests around registry capability metadata, nested ONNX export resolution, and safe checkpoint loading.

**Tech Stack:** Python, PyTorch, pytest, ONNX export helpers.

---

### Task 1: Harden checkpoint model loading

**Files:**
- Modify: `src/open_vision_banana/training/trainer_interface.py`
- Test: `tests/test_phase2.py`

Use `torch.load(..., weights_only=True)` for flat `model.pt` state dictionaries. Preserve the existing PEFT adapter path and keep RNG restoration separate because RNG payloads contain Python/NumPy state objects.

### Task 2: Make MPS memory detection platform-aware

**Files:**
- Modify: `src/open_vision_banana/training/optimization.py`
- Test: add focused unit coverage in `tests/test_training_utils.py`

Resolve total memory from CUDA properties, then `psutil`, then `os.sysconf`; retain a documented fallback only when no platform API is available.

### Task 3: Expand generic LoRA defaults and registry tests

**Files:**
- Modify: `src/open_vision_banana/config.py`
- Modify: `tests/test_data_config.py`
- Modify: `tests/test_model_registry.py`

Use attention output and FFN targets in the generic LoRA configuration, while preserving explicit per-family registry targets. Add capability-matrix assertions for every registered config.

### Task 4: Verify nested ONNX export resolution

**Files:**
- Test: `tests/test_onnx_export.py`

Add a nested `VisionBanana.generator._t2i_pipe.transformer` mock test. No implementation change is needed because the current working tree already contains the report's proposed resolver.

### Task 5: Run focused regression checks

Run the changed test modules, `git diff --check`, and import/compile checks. Record any environment-only failures separately from code failures.
