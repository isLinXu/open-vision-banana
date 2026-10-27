# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **RES expression validation**: accept the existing canonical prompt when its shared builder normalizes a leading article, so valid Rea2Seg/LLM-Seg expressions such as "A red car" are not rejected during preparation or loading.

- **nuScenes depth benchmark**: `evaluate_paper_benchmarks.py --nuscenes-root` failed before loading any model with `TypeError: build_dataset() got multiple values for argument 'task'`, because the depth task of `nuScenesDataset` was passed as an adapter keyword while `build_dataset`'s first argument is also named `task`. The benchmark now uses a `nuscenes_depth` adapter key (`NuScenesDepthDataset`), as NYU uses `nyu_depth`.

- **Expression-conditioned RES**: keep one prompt per expression, resize binary masks with nearest-neighbor interpolation, and share the versioned white/black decoding contract across training, inference and evaluation. Preserve disconnected targets and provide an explicit legacy color decoder. Strict RefCOCOg evaluation rejects incomplete annotations and declares its image range so official and portable loaders preserve the same input pixels.

- **Inference prompts for LoRAs from the manifest trainers**: `scripts/train.py` (like `train_sana_512.py` / `train_flux.py`) trains depth and normal with "Generate a relative depth map. ..." / "Generate an RGB surface-normal map. ...", while `VisionBanana` and `scripts/inference.py` prompted with PromptBuilder's "Generate a metric depth visualization of this image.", which such a LoRA never saw. Unified-trainer checkpoints now record in `metadata.json` (`task_prompts`) the prompt of each task trained with a single prompt, `VisionBanana.from_pretrained` uses that record for a local LoRA, and `scripts/inference.py --prompt` sets the prompt for checkpoints without the record.
- **SANA-frame loss terms under rectified flow (FLUX)**: the depth-edge and normal-angular aux losses rebuild the target endpoint as `sample - sigma * v`, and Min-SNR weighting takes a sigma, both in SANA's frame. `TrainingLosses` passed them rectified-flow `(t, v)` unchanged: the aux losses rebuilt the source image and penalised a correct prediction (both are enabled in `flux_3task_full_clean.yaml`), and Min-SNR was mirrored along the path (on by default). `TrainingLosses` now converts with its `family` (`FlowScheduler.to_sana_frame`: `sigma = 1 - t`, `v -> -v`). `scripts/train_flux.py` interpolates the other way round (`t` weights the source), so its Min-SNR was already right and its aux losses now only negate `v`; its `mix_rectified_flow` docstring is corrected.
- **FLUX 13-task config quantization**: `flux_13task_multitask_30k.yaml` now sets `quantize: "4bit"` like the other flux configs, so `scripts/train.py` loads it in 4-bit instead of unquantized (`scripts/train_flux.py` always used NF4 and is unaffected).
- **Multi-GPU training with the unified trainer**: since the unified-trainer rewrite, every multi-GPU launcher failed at argument parsing (`--ddp` / `--fsdp` / `--deepspeed` were unknown) and the trainer had no distributed code. Under torchrun, `scripts/train.py` now trains data-parallel: per-rank data shards, LoRA gradients averaged across ranks, logging, checkpoints and validation on rank 0. `--ddp` is accepted; `--fsdp` / `--deepspeed` fail with a message pointing to the DDP launchers. `launch_training_a100_80g.sh`, `launch_training_a100_80g_ddp.sh` and `launch_h20_ddp.sh` share `scripts/_gpus.sh` for the GPU count, which honours `--gpus-per-node` and `CUDA_VISIBLE_DEVICES` instead of always using every GPU `nvidia-smi` lists.
- **SANA VAE encode**: `vae_encode` now reads `EncoderOutput.latent` from SANA's `AutoencoderDC`; it previously fell back to a non-existent `.sample`, so SANA training through `scripts/train.py` crashed on the first step with `AttributeError`. Latent extraction is now a single `extract_vae_latents` helper shared by `vae_encode`, `vae_encode_sana` and `scripts/train_sana_512.py`.
- **Hardcoded path cleanup**: Replaced all `/media/linxu/` and `/Users/gatilin/` paths with portable HF Hub IDs and relative paths across 30+ config files, 8 scripts, and 1 source module (`model_registry.py`).
- **DeepSpeedTrainer export**: Added missing import in `training/__init__.py` (was in `__all__` but not imported).
- **VisionGenerator reference**: Fixed non-existent class import in `evaluate_depth_standard.py` → `BaseGenerator`.
- **Serving version sync**: Updated hardcoded version in `serving/fastapi_app.py` from 0.4.2 to 0.5.0.
- **docs/quick-start.md**: Fixed 3 broken script references and version badge (v0.4.2 → v0.5.0).
- **Colab notebook**: Replaced placeholder GitHub URL (`github.com/user/...`) with actual repo URL.
- **Experiment configs**: Updated all archived script references to include `archive/pre_opensource/` prefix.

### Added

- **Main depth/reasoning extension**: prepare and record 20,000 additional Hypersim frames and the pinned Rea2Seg/LLM-Seg sources, then retain 13,267 depth frames with maximum native encoding error ≤5 cm and 16,116 referring rows whose images retain their targets under every allowed 512px crop. Provides an E1 six-task continuation configuration, source/quality ledgers, resumable downloads, pinned filtering/merging and cross-task held-out checks. The shared merger preserves `metric_depth` paths. See [the experiment and release record](docs/main_depth_reasoning.md); no new training result is claimed.

- **Supervised RES reasoning sources**: integrate PR #26's offline Rea2Seg adapter and a deterministic LLM-Seg40K COCO train subset through a shared native binary-pair exporter. Reuse existing prompt/mask contracts, manifest loading, merging and portable RES bundles; add source/exclusion audits, optional explicit cleaning, pinned supplement downloads and controlled R0/R1/R2 configurations.

- **Qwen-Image-2.1 LoRA training and inference**: `qwen-image-2.1` (`QwenImage21Pipeline`, diffusers>=0.41 with transformers>=5.17) trains with the source image given the way the pipeline's editing mode gives it: as Qwen3-VL visual tokens through the pipeline's own prompt encoder, and as reference latents placed before the target tokens (`edit_forward.qwen21_edit_forward`, sigma in and `noise - x0` out, converted to the repo frame). `vae_encode` / `vae_decode` handle its RGBA VAE with per-channel `latents_mean` / `latents_std`. Training takes batch size 1 (the transformer reads one token layout per batch) and a resolution that is a multiple of 32, checked before the model loads; CFG dropout blanks the prompt and keeps the source image. At inference, `output_resolution` keeps the input's area up to the model's native 1024 instead of always resizing the source to 1024 px.

- **Mixed RES source audit**: reconcile C3VG/SeqTR `instances_nogoogle.json` with official RefCOCO-family expressions, shared COCO images and released native masks. Publish the equivalent source and visual audit as a supplement without duplicating training data or changing benchmark splits.

- **Portable RES data pipeline**: official RefCOCO/RefCOCO+/RefCOCOg conversion, provenance and leakage audits, checksum-verified Hub bundles, public `RESManifestDataset`, native-resolution evaluation and a supervised SANA training configuration. See [RES data and evaluation](docs/res_data.md) for the pinned public release, split accounting and compatibility guidance. No trained checkpoint or benchmark accuracy is claimed.

### Changed

- **README beautification**: Added badges, table of contents, centered header, improved visual hierarchy. Reduced from 438 to ~360 lines.
- **scripts/README.md rewrite**: Updated to accurately reflect all 54 active scripts (was referencing 14 non-existent scripts).
- **Script archival**: Moved 6 unrelated/duplicate scripts to `archive/pre_opensource/` (Crablet desktop scripts, macOS dev scripts).

## [0.5.0] — 2026-09-10

### Changed (Pre-Open-Source Release)

- **README rewrite**: Streamlined from 892 to ~340 lines. Rewrote project introduction with clear feature highlights. Added SANA-600M/1600M training and inference examples. Simplified dataset sections (removed verbose training dataset lists). Consolidated distributed training commands to reference `docs/quick-start.md`. Updated all code examples for consistency.
- **Script archival**: Moved 241 temporary/experimental scripts to `scripts/archive/pre_opensource/`. Remaining 57 core scripts cover: training (`train.py`, `train_sana_512.py`), inference (`inference.py`, `infer_sana_512.py`), evaluation (`evaluate.py`, `eval_checkpoints.py`), data preparation (`prepare_hypersim.py`, `download_data.py`), and launch scripts.
- **LoRA registry update**: `sana-depth-relative-v1` status changed from `local` to `available` (v1.0.0). Metrics documented: Pearson R=0.790, Spearman R=0.778, MAE=0.147.
- **`.gitignore` update**: Added `.tools/` exclusion for local development tools.
- **Version bump**: `__version__` updated to `0.5.0`.

### Added

- **`data/tar_dataset.py`**: TAR archive dataset reader for SNVC-50M compressed data.
- **SANA configs**: `sana_1600m_3task.yaml`, `sana_1600m_5task.yaml`, `sana_snvc50m_3task.yaml`, `sana_snvc_linux_5task.yaml`.

## [0.4.2] — 2026-06-27

### Fixed (P1)

- **torch.ao.quantization deprecation** (`inference/optimization.py`): `quantize_mapper_int8()` migrated from `torch.ao.quantization.quantize_dynamic` (removed in PyTorch ≥ 2.10) to `torchao.quantization.quantize_` with `Int8DynamicActivationInt8WeightConfig`. Automatic fallback to legacy API on platforms without torchao/Triton (e.g. MPS). Pure-tensor ONNX mappers (no `nn.Linear`/`nn.Conv2d`) now short-circuit to a no-op instead of producing an empty quantized model.
- **CHANGELOG timestamp typos** (`CHANGELOG.md`): v0.1.0, v0.2.0, and v0.2.1 entries had year "2025" instead of "2026".
- **DeepSpeedTrainer parameter redundancy** (`training/trainer_interface.py`): Removed `max_grad_norm_clip` and `gradient_accumulation_steps_native` — these duplicated `max_grad_norm` and `gradient_accumulation_steps` respectively, causing confusion about which parameter controlled clipping. The single `max_grad_norm` now serves both the DeepSpeed config and NativeTrainer's `clip_grad_norm_`.
- **AccelerateTrainer.optimizer_step() contract violation** (`training/trainer_interface.py`): Was a no-op `pass` with a docstring telling users not to call it — violated `TrainerInterface` contract. Now properly delegates to `self._optimizer.step()` and raises `RuntimeError` with a clear message if `setup_optimizer()` was not called first.
- **_probe_forward_signature exception-based control flow** (`training/tuner.py`): Replaced 3 sequential `try/except TypeError` model calls with `inspect.signature` static analysis. Falls back to a single trial call only when the signature has `**kwargs` but no named params (e.g. PEFT wrapper with `(*args, **kwargs)`). Eliminates unnecessary model forward calls during init.

### Changed (P2)

- **VisionTaskDataset._load_data optimization** (`training/tuner.py`): Replaced `sorted(iterdir())` + per-file extension probing with `os.scandir` + a target-file stem index for O(1) lookup. Avoids O(N×K) extension checks on large datasets (tens of thousands of samples).
- **README Project Structure sync** (`README.md`): Updated directory tree to include all 43 source files (was missing 15 files: `adaptive_depth_mapper.py`, `split.py`, `validation.py`, `generation.py`, `checkpoint.py` in both `inference/` and `training/`, `optimization.py`, `trainer_interface.py`, `early_stopping.py`, `logging.py`, `lr_finder.py`, `fsdp_utils.py`, `deepspeed_utils.py`, `baselines.py`). Test list updated from 12 to 35 entries. Added `docker/`, `experiments/`, `notebooks/`, `templates/` directories.
- **README Citation fix** (`README.md`): Replaced placeholder `author={Google Research}` with actual paper authors: Shengbang Bang, Hezhen Hu, Zheng Qian, Chengyu Wang, Yi Tai, Yann LeCun, Tong Wang.

### Tests

- All 467 tests pass. Ruff lint + format clean on modified files.

### Changed

- **`_extract_instances` vectorized** (`core/segmentation.py`): Replaced per-label Python loop (K GPU→CPU round-trips for `img[mask].mean()`) with scatter_add + bincount aggregation. Mean colors for all components computed in a single pass, reducing K sync points to 1.
- **`_compute_adaptive_tau` vectorized** (`core/segmentation.py`): Replaced manual gradient differencing with `F.conv2d` + Sobel kernels. Eliminates 3 temporary tensor allocations (`gx`, `gy`, `gradient_magnitude` reallocations) and produces cleaner gradients.
- **GPU sync elimination** (`core/segmentation.py`): 4 `.item()` call sites wrapped with `int()` to make GPU→CPU synchronization explicit and prevent implicit float-coercion syncs.
- **`_tensor_to_pil` GPU fix** (`inference/generator.py`): `t.min()` on GPU caused an implicit synchronization before `.cpu()`. Reordered to `.cpu()` first, then min/max check — avoids blocking the GPU pipeline.
- **Config DRY refactor** (`config.py`): Extracted repeated `from_dict` boilerplate (8 classes × 4 lines each) into shared `_filtered_from_dict()` helper. ~50 lines of duplicated `inspect.signature` logic removed.
- **Import cleanup** (`inference/pipeline.py`): `collections.abc.Callable` replaces `typing.Callable` (PEP 585). `InferenceOptions` imported at module level instead of string-annotated. `contextlib.suppress` replaces bare try/except.
- **Import ordering** (`training/__init__.py`): Alphabetical ordering for deepspeed/fsdp imports.
- **Lint compliance**: All 46 source + test files pass `ruff check` and `ruff format`. Fixed B904 (raise from exc), B905 (zip strict), B007 (unused loop var), F841 (unused vars). N817 added to ignore list (FSDP acronym is PyTorch convention).

## [0.4.0] — 2026-06-23

### Added

- **Core algorithm performance benchmark** (`experiments/core_algorithm_benchmark.py`): Latency / throughput / memory baselines for depth, normal, and segmentation across resolutions 64–2048. Verifies the paper's Union-Find ~100× speedup claim (137× at 256²).
- **Segmentation optimization regression tests** (`tests/test_segmentation_v040.py`): Covers the S4 tiny-component pre-filter and S5 vectorized merging.

### Changed

- **Single-source version**: `pyproject.toml` now reads `version` dynamically from `open_vision_banana.__version__` (was hardcoded and drifting between `0.2.0` / `0.3.3`). `__version__` bumped to `0.4.0`.
- **SegmentationDecoder S5 (`_spatial_merging`) vectorized**: Replaced the Python greedy merge-pair loop (O(P) scalar `.item()` calls, P up to K²) with a tensor Union-Find over the merge-adjacency graph. Computes the same connected-component partition with no Python-level loops.

### Performance

- **SegmentationDecoder S4 (`_boundary_artifact_elimination`) pre-filter**: Components with < 9 pixels (which can never survive a 3×3 erosion) are dropped before the multi-hot conv. This removes a pathological blowup on noisy / low-resolution inputs — random 64×64 decode dropped from **~6.8 s → ~16 ms (≈430×)**, 256×256 from ~400 ms → ~120 ms. Exact (not approximate): such components were always removed by erosion anyway.

### Fixed

- **Fragile performance tests** (`tests/test_modules.py::TestPerformance`): Replaced absolute wall-clock thresholds (which failed under parallel CI load) with warmup + median-of-runs timing, generous regression-only bounds, and explicit correctness assertions.

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

## [0.1.0] — 2026-06-10

### Added

- Initial implementation of core RGB mapping algorithms (depth, normals, segmentation).
- Basic FLUX.1 integration with instruction-tuned LoRA.
- Single-image inference pipeline.
- Depth, normal, semantic, and instance evaluation metrics.
- ONNX export for depth and normal mappers.
- Training infrastructure with rsLoRA (rank=64, alpha=128).
- Configuration system with YAML support.
- Project scaffolding: pyproject.toml, README, LICENSE, tests.

[0.5.0]: https://github.com/user/open-vision-banana/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/user/open-vision-banana/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/user/open-vision-banana/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/user/open-vision-banana/compare/v0.3.4...v0.4.0
[0.3.4]: https://github.com/user/open-vision-banana/compare/v0.3.3...v0.3.4
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
