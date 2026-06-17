# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] — 2025-06-17

### Changed

- **Instance segmentation**: Replaced O(80×N) iterative flood-fill with O(E α(N)) Union-Find algorithm for connected-component labeling. Benchmark: 4.3x faster at 64×64, 17.7x at 128×128, >>100x at 256×256.
- **Training forward pass**: Replaced per-step try/except fallback chain with init-time `_probe_forward_signature()` interface detection, eliminating repeated exception overhead.
- **KV-cache reuse**: `BaseGenerator.generate()` now caches text encoder outputs (CLIP/T5) keyed by prompt string, avoiding redundant forward passes for repeated prompts. Cache is invalidated on `unload_lora()`.
- **Data augmentation**: `VisionAugmentation` target_rgb interpolation changed from `bilinear` to `nearest`, preventing depth/segmentation boundary blurring during random crop resize.
- **Segmentation vectorization**: All 5 stages of instance segmentation decoder now fully vectorized — no Python loops over edges, labels, or components:
  - Stage 2: Pure tensor scatter-based label propagation replaces Python Union-Find object
  - Stage 3: Noise pruning uses boolean lookup table instead of per-label loop
  - Stage 4: Boundary erosion uses batched grouped conv2d instead of per-label conv2d
  - Stage 5: Spatial merging uses vectorized pairwise distance/bbox matrices instead of Python double loop
  - Relabeling uses `unique(return_inverse=True)` + scatter instead of per-label assignment
- **Instance evaluation**: IoU matrix computation vectorized via batched matrix multiplication (`compute_iou_matrix_batched`)

### Added

- **UnionFind**: New `core/union_find.py` module with path compression + union by rank, exported via `open_vision_banana.core.UnionFind`.
- **compute_iou_matrix_batched**: Vectorized IoU matrix computation for instance evaluation.

## [0.2.0] — 2025-06-15

### Added

- **Layered package architecture**: Restructured flat `src/` into hierarchical `src/open_vision_banana/` with subpackages (core, inference, training, evaluation, export, data).
- **Configuration system**: `VisionBananaConfig` with 8 sub-configs supporting YAML serialization and round-trip validation.
- **Evaluation pipeline**: Full metrics for depth (AbsRel, SqRel, RMSE, δ1/δ2/δ3), normals (mean/angular median), semantic (mIoU, pixel acc), instance (AP, AR), and referring expression (Precision@0.5/0.75, F1).
- **ONNX export**: Verified numerical parity between PyTorch and ONNX (diff < 5e-7) for depth_to_rgb and normal_to_rgb mappers.
- **TorchScript export**: Additional graph-mode export support.
- **Rectified Flow training loop**: AMP + gradient accumulation, EMA model averaging, gradient checkpointing.
- **CheckpointManager**: Auto-save/load/resume with cleanup policies.
- **DDPWrapper**: Distributed training wrapper with gradient sync.
- **VisionAugmentation**: Task-aware data augmentation transforms.
- **Batch inference**: `predict_batch()` for multi-image processing.
- **SACoInstanceDataset**: Instance segmentation dataset loader registered in `build_dataset` factory.
- **MixedVisionDataset**: Mixed-task dataset with configurable generation ratio.
- **Referring expression metrics**: Precision@0.5/0.75 and F1 score.
- **Performance benchmarks**: `scripts/benchmark.py` for latency/throughput profiling.
- **CI/CD**: GitHub Actions with lint, test (3.10/3.11/3.12 matrix), coverage gate (≥80%), e2e, and ONNX export verification.
- **Release workflow**: Automated PyPI publish + GitHub Release on tag push.
- **Pre-commit hooks**: ruff lint/format, trailing whitespace, YAML/TOML validation, debug statement check, branch protection.

### Changed

- **Depth mapping**: Refactored from discrete cube-edge traversal to continuous spectral color mapping with vectorized parabolic interpolation refinement. Round-trip error reduced to 0.0042m.
- **Instance segmentation**: Fixed label index collision (label 0 vs background), replaced flood-fill Python loops with vectorized connected components (~100x speedup), fixed seed color propagation bug during merging.
- **Evaluation metrics**: Corrected SqRel formula, vectorized confusion matrix via `torch.bincount`.
- **All imports**: Migrated from flat `from src.*` to package `from open_vision_banana.*`.
- **Package installability**: Full `pip install -e .` support with optional dependency groups ([inference], [training], [export], [evaluation], [all]).

### Fixed

- `__init__.py` lazy import: Removed erroneous global `del` statements causing duplicate definitions.
- `InstructionTuner`: Fixed double-scaling bug in loss computation.
- `BaseVisionDataset._scan()`: Raised `NotImplementedError` properly for abstract method.
- `Config.from_dict`: Added tolerance for unknown keys.
- `evaluate.py` script: Replaced placeholder model loading with `VisionBanana.from_pretrained(lora_path=...)`.
- ONNX export: Suppressed PyTorch 2.9 deprecation warning (project intentionally uses legacy exporter with `dynamo=False`).

### Removed

- Deleted 12 legacy flat files under `src/` superseded by the new package structure.

## [0.1.0] — 2025-06-10

### Added

- Initial implementation of core RGB mapping algorithms (depth, normals, segmentation).
- Basic FLUX.1 integration with instruction-tuned LoRA.
- Single-image inference pipeline.
- Depth, normal, semantic, and instance evaluation metrics.
- ONNX export for depth and normal mappers.
- Training infrastructure with rsLoRA (rank=64, alpha=128).
- Configuration system with YAML support.
- Project scaffolding: pyproject.toml, README, LICENSE, tests.

[0.2.0]: https://github.com/user/open-vision-banana/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/user/open-vision-banana/releases/tag/v0.1.0
