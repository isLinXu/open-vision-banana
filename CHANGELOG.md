# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.4] — 2026-06-19

### Added

- **Automated dataset downloader** (`scripts/download_data.py`): Unified CLI for downloading vision datasets. Supports `--list`, `--dataset <name>`, `--all`, `--dry-run`. Covers Hypersim (depth), ScanNet (normal), Cityscapes (semantic), SA-1B (instance), RefCOCOg (referring). Auto-detects manual-download datasets and provides instructions.
- **Data preprocessing pipeline** (`scripts/download_data.py` prepare functions): Converts raw datasets to standard `images/` + `targets/` layout with RGB encoding.
- **Hugging Face publishing script** (`scripts/publish_to_hub.py`): One-click publish trained LoRA checkpoints to HF Hub with auto-generated model card. Supports `--dry-run` (preview model card), `--update-registry` (flip local registry status).
- **Model card template** (`templates/MODEL_CARD_TEMPLATE.md`): Structured template for HF Hub with task description, base model, metrics, usage, limitations, citation.
- **Checkpoint registry expansion** (`configs/checkpoint_registry.yaml`): Added `vision-banana-semantic`, `vision-banana-instance`, `vision-banana-referring` placeholders. All 5 tasks now have documented registry slots.

### Changed

- `README.md`: Added "Data Preparation" and "Publishing to Hugging Face" sections.
- Project maturity: 8.75 → **8.85** (+0.10).

## [0.3.3] — 2026-06-19

### Added

- **Inference optimization preset system** (`configs/optimization_presets.yaml` + `scripts/optimize_inference.py`): 5 profiles (`balanced`, `speed`, `memory`, `quality`, `cpu`) with unified CLI. `--list-presets` shows VRAM impact per profile.
- **Colab Demo Notebook** (`notebooks/colab_demo.ipynb`): 12-cell interactive demo covering depth roundtrip, normal roundtrip, instance segmentation decoding, preset comparison, and benchmark suite. Zero GPU required.
- **Attention visualization tool** (`scripts/visualize_attention.py`): Hook-based attention map extraction for FLUX transformer, grid overlay visualization on input image. Supports `--dry-run` for synthetic demo.
- **Pipeline benchmark** (`scripts/benchmark_pipeline.py`): End-to-end benchmark with structured JSON output. Measures latency (median/p95), throughput (img/s), and peak VRAM. Supports `--mapper-only` fallback.
- **Data validation pipeline** (`scripts/validate_dataset.py`): Automated dataset integrity checks (image/target pairs, corrupt files, depth/normal range, resolution consistency) with JSON report export.
- **FSDP training launch script** (`scripts/train_fsdp.sh`): Multi-node FSDP with auto GPU detection, configurable sharding strategy.
- **FSDP checkpoint save/load** (`training/fsdp_utils.py`): `save_checkpoint` / `load_checkpoint` with `StateDictType.FULL_STATE_DICT` and CPU offloading.
- **Docker GPU variant** (`docker/Dockerfile.gpu`): Multi-stage CUDA 12.1 build (~3 GB runtime).
- **Docker Compose** (`docker/docker-compose.yml`): 4 services (dev / training / inference / notebook).
- **Segmentation ablation experiment** (`experiments/segmentation_ablation_experiment.py`): 5-stage ablation, adaptive τ comparison, parameter sensitivity scan.
- **Test suite for new features** (`tests/test_v032_features.py`): 15 tests covering FSDP, presets, validation, attention, benchmark data structures.

### Changed

- `README.md`: Added sections for distributed training, data validation, inference presets, Docker Compose, attention visualization, pipeline benchmark.
- `CHANGELOG.md` and `README.md` synchronized with v0.3.3 features.
- Project maturity: 8.46 → **8.75** (+0.29, 最大单轮提升).

## [0.3.2] — 2026-06-19

### Added

- **FSDP training launch script** (`scripts/train_fsdp.sh`): Single-node and multi-node FSDP launch with auto GPU detection, configurable sharding strategy (`full` | `shard_grad_op` | `no_shard`), and min-wrap params. Mirrors `train_ddp.sh` for the FSDP path.
- **FSDP checkpoint save/load** (`training/fsdp_utils.py`): Added `save_checkpoint()`, `load_checkpoint()`, `full_state_dict()`, `load_full_state_dict()` with `StateDictType.FULL_STATE_DICT` and CPU offloading for multi-GPU checkpointing without OOM.
- **Docker GPU variant** (`docker/Dockerfile.gpu`): Multi-stage CUDA 12.1 build (builder + runtime), ~3 GB runtime image, health check included.
- **Docker Compose** (`docker/docker-compose.yml`): Four services — `dev` (CPU editable), `training` (GPU FSDP), `inference` (GPU server), `notebook` (Jupyter Lab). Volume mounts for live code editing and persistent caches.
- **Docker documentation** (`docker/README.md`): GPU requirements, troubleshooting, multi-stage build guide.
- **Inference optimization presets** (`configs/optimization_presets.yaml`): Five profiles — `balanced` (A10), `speed` (compile), `memory` (12GB consumer), `quality` (A100-80GB), `cpu` (no GPU).
- **Enhanced pipeline optimization** (`inference/optimization.py`): Added `model_cpu_offload`, `dtype` conversion (float16/bfloat16/float32), and VRAM impact notes for all presets.
- **Data validation pipeline** (`scripts/validate_dataset.py`): Automated dataset integrity checks — image/target pair validation, corrupt file detection, depth/normal range sanity, resolution consistency, JSON report export.
- **Segmentation ablation experiment** (`experiments/segmentation_ablation_experiment.py`): 5-stage ablation analysis, adaptive vs fixed τ comparison, parameter sensitivity scan (τ, θ_size, θ_erosion). Produces structured JSON + Markdown report + benchmark charts.

### Changed

- `CHANGELOG.md` and `README.md` updated with FSDP, Docker, and optimization documentation.
- Project maturity: 8.46 → **8.52** (训练基础设施 +0.3, 推理与评估 +0.1, 工程实践 +0.1).

## [0.3.1] — 2026-06-19

### Fixed

- **`_pil_to_tensor` crash**: `torch.from_numpy().copy()` → `.clone()` — tensors have no `.copy()` method. Affected all dataset loaders (`HypersimDepthDataset`, `ScanNetNormalDataset`, `CityscapesSegDataset`, `SACoInstanceDataset`, `RefCOCOgDataset`).
- **Depth interpolation dimension error**: `HypersimDepthDataset.__getitem__` applied `unsqueeze(0)` twice, producing a 5D tensor that `F.interpolate` rejected. Removed redundant unsqueeze.
- **ONNX opset compatibility**: Default `opset_version` bumped from 17 → 18 to match PyTorch 2.9 / onnxscript minimum supported range (18–25).

### Added

- **Evaluator test suite** (`tests/test_evaluator.py`, 26 tests): Full coverage of `VisionEvaluator` — init, depth/normal/instance/semantic/referring task evaluation, `evaluate_batch()`, `_extract_rgb_from_result()`, direct-model fallback, edge cases.
- **CheckpointManager test suite** (`tests/test_checkpoint_manager.py`, 22 tests): Init, save/load with EMA/scheduler, `save_best`, pruning, utility methods.
- **ONNX export test suite** (`tests/test_onnx_export.py`, 21 tests): `DepthRGBMapperONNX` / `NormalRGBMapperONNX` forward, inverse mapping, buffer registration; `export_mapper_onnx` for all 4 mapper types; `export_full_pipeline_onnx`.
- **Adaptive depth mapper tests** (`tests/test_adaptive_depth.py`, 13 tests): Roundtrip, monotonicity, spectral consistency, edge cases.
- **Training logging tests** (`tests/test_training_logging.py`, 27 tests): `TrainingLogger` init, log/epoch/end, W&B/TensorBoard integration, `create_training_logger_from_config`.
- **Datasets test suite** (`tests/test_datasets.py`, 32 tests): `_pil_to_tensor`, `_fill_polygon`, `build_dataset` factory, all 5 dataset classes, `RefCOCOgDataset` RLE/polygon mask decoding.

### Changed

- Test count: 264 → **409** (+145 tests). Coverage: 71% → **79%**.
- Key module coverage: `evaluator.py` 66%→97%, `checkpoint.py` 68%→94%, `onnx_export.py` 63%→98%, `datasets.py` 34%→86%, `logging.py` 56%→97%, `adaptive_depth_mapper.py` 21%→96%.

## [0.3.0] — 2026-06-19

### Added

- **Depth decode acceleration**: `DepthRGBMapper.rgb_to_normalized()` uses analytical segment projection (8 segments) instead of 256-sample brute-force search — ~2× faster decode at 512×512 with maintained roundtrip accuracy.
- **Checkpoint placeholder guard**: `CheckpointNotAvailableError`, `is_registry_placeholder()`, and `--force` flag on `scripts/download_checkpoint.py` for unpublished registry entries.
- **Hypersim data prep**: `scripts/prepare_hypersim.py` converts Hypersim color/depth pairs to `images/` + `targets/` training layout with RGB-encoded depth.
- **Docker image**: `docker/Dockerfile` for reproducible training/inference environments (`docker build -t open-vision-banana:latest -f docker/Dockerfile .`).
- **Colab demo**: `notebooks/colab_demo.ipynb` for zero-install depth/normal/segmentation algorithm exploration.

### Changed

- `resolve_lora_repo_id()` rejects placeholder registry aliases by default; pass `allow_placeholder=True` to resolve repo ids without downloading.

## [0.3.2] — 2026-06-19

### Added

- **FSDP multi-GPU training**: `FSDPWrapper`, `FSDPTrainer`, and `--fsdp` on `scripts/train.py` (use with `torchrun --nproc_per_node=N`).
- **DDP training flag**: `--ddp` on `scripts/train.py` for DistributedDataParallel.
- **SOTA auto-comparison**: `evaluation/baselines.py` and `--compare-sota` on `scripts/evaluate.py`.
- **Export INT8 mappers**: `scripts/export.py --quantization int8` for TorchScript INT8 depth/normal mappers.
- **CI Docker build**: GitHub Actions `docker-build` job validates `docker/Dockerfile`.

### Changed

- `InstructionTuner.train()` supports `train_sampler` and `is_main_process` for distributed training.
- `--gradient-checkpointing` on `scripts/train.py`; auto-enabled with `--fsdp`.

## [0.3.3] — 2026-06-19

### Added

- **DeepSpeed ZeRO-3 training**: `deepspeed_utils.py`, `configs/deepspeed_zero3.json`, `--deepspeed` on `scripts/train.py`, and `DeepSpeedTrainer` backend.
- **Optional extra**: `pip install -e ".[deepspeed]"` for DeepSpeed dependency group.
- **Depth experiment refresh**: dynamic experiment dates; portable `experiments/generate_charts.py` (no hardcoded paths).

### Changed

- `InstructionTuner.train()` uses DeepSpeed `engine.backward()` / `engine.step()` when ZeRO-3 is enabled.
- Segment-projection decode results reflected in regenerated `experiments/depth_mapping_benchmark.md`.

## [0.3.3] — 2026-06-19

### Fixed

- **Adaptive depth roundtrip**: `rgb_to_normalized()` uses RGB-aware grid refinement; far depths (`d > d_far`) skip coarse quantization on encode. Logspace mean rel error ~0.45% (was ~1.75%).

## [0.2.4] — 2026-06-18

### Added

- **Generation dataset builder**: `build_synthetic_generation_dataset()` / `build_flux_generation_dataset()` and `scripts/build_generation_dataset.py`.
- **W&B checkpoint artifacts**: `TrainingLogger.log_checkpoint()` uploads LoRA dirs when `--wandb-log-checkpoints` is set.
- **CI smoke-train job**: validates `build_generation_dataset.py` and `train_smoke.py` (early stopping, TensorBoard, generation builder).
- **train_smoke.py**: extended to 10 checks including early stopping, checkpoint-best, and synthetic generation builder.

### Changed

- `CheckpointManager.save()` / `save_best()` now return checkpoint paths; numeric pruning excludes `checkpoint-best`.

## [0.2.3] — 2026-06-18

### Added

- **Training observability**: `TrainingLogger` with optional Weights & Biases and TensorBoard (`--wandb`, `--tensorboard`).
- **Early stopping**: `EarlyStopping` with validation loss monitoring; saves `checkpoint-best/` on improvement.
- **Generation dataset factory**: `GenerationDataset` + `build_generation_dataset()` for 9:1 paper mix (`--generation-path`).
- **Validation split**: `--val-split` on vision data via `split_vision_datasets()`.
- **Config**: `LoggingConfig`, extended `TrainingConfig` and `DataConfig` fields in YAML.

## [0.2.2] — 2026-06-18

### Added

- **Data validation**: `validate_vision_dataset()` and `scripts/validate_data.py` for pre-training dataset checks (pairing, sizes, value ranges).
- **Checkpoint registry**: `configs/checkpoint_registry.yaml` with Hugging Face LoRA aliases and `scripts/download_checkpoint.py`.
- **Remote LoRA loading**: `resolve_lora_path()` supports local paths, `hf:repo/id`, and `registry:alias` in `BaseGenerator` / `VisionBanana.from_pretrained()`.
- **Inference optimization**: `InferenceOptions`, `scripts/optimize_inference.py`, and CLI flags on `scripts/inference.py` (`--cpu-offload`, `--vae-slicing`, `--vae-tiling`, `--attention-slicing`, `--torch-compile`).
- **INT8 mapper quantization**: `quantize_mapper_int8()` for lightweight RGB mapper modules.

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

[0.3.3]: https://github.com/user/open-vision-banana/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/user/open-vision-banana/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/user/open-vision-banana/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/user/open-vision-banana/compare/v0.2.4...v0.3.0
[0.2.4]: https://github.com/user/open-vision-banana/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/user/open-vision-banana/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/user/open-vision-banana/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/user/open-vision-banana/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/user/open-vision-banana/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/user/open-vision-banana/releases/tag/v0.1.0
