# Scripts Directory

Active scripts for Open-Vision-Banana training, inference, evaluation, and utilities.

## Core Scripts

| Script | Description |
|--------|-------------|
| `train.py` | Main training entry point (multi-backend support) |
| `train_smoke.py` | CPU smoke test (1-2 epochs, synthetic data) |
| `train_lora_mps.py` | MPS-specific LoRA training |
| `inference.py` | Single-image inference |
| `evaluate.py` | Model evaluation on benchmark datasets |
| `benchmark.py` | Core algorithm benchmark (no GPU required) |
| `benchmark_pipeline.py` | Full pipeline benchmark |
| `export.py` | Model export (checkpoint → deployable format) |
| `export_onnx.py` | ONNX export |
| `demo.py` | Interactive demo |
| `download_model.py` | Download base diffusion models from HuggingFace Hub (23 families, resume support) |

## Data Pipeline

| Script | Description |
|--------|-------------|
| `download_flux_model.py` | Download FLUX.1 model weights |
| `download_checkpoint.py` | Download pretrained checkpoints |
| `download_data.py` | Download training datasets |
| `download_datasets.py` | Download evaluation datasets |
| `build_generation_dataset.py` | Build synthetic generation dataset |
| `generate_synthetic_data.py` | Generate synthetic training data |
| `distill_depth_pseudo_labels.py` | Distill depth pseudo-labels |
| `prepare_depth_dataset.py` | Prepare depth estimation dataset |
| `prepare_hypersim.py` | Prepare Hypersim dataset |
| `evaluate_paper_benchmarks.py` | Preflight and evaluate decoded paper benchmarks |
| `validate_data.py` | Validate dataset integrity |
| `validate_dataset.py` | Comprehensive dataset validation |

## Utilities

| Script | Description |
|--------|-------------|
| `monitor_training_mps.py` | Monitor MPS training (memory, temperature) |
| `optimize_inference.py` | Inference optimization (quantization, compilation) |
| `fix_segmentation.py` | Fix segmentation decoder issues |
| `publish_to_hub.py` | Publish model to HuggingFace Hub |
| `visualize_attention.py` | Attention map visualization |
| `verify_training_mps.py` | Verify MPS training correctness |
| `validate_multi_backbone_training.py` | Validate multi-backbone training paths |

## Shell Scripts

| Script | Description |
|--------|-------------|
| `launch_training_mps.sh` | Launch training on Apple Silicon MPS |
| `train_ddp.sh` | Multi-GPU DDP training launcher |
| `train_ddp_example.sh` | DDP example with sample config |
| `train_fsdp.sh` | FSDP training launcher |
| `run_experiments.sh` | Run experiment matrix |
| `submit_batches.sh` | Submit batch jobs |
| `build-release.sh` | Build release package |
| `generate-icons.sh` | Generate app icons |

## Archive

Historical versions are in `archive/`:
- `archive/v1_to_v3/` — v1-v3 training/eval/inference scripts (deprecated)
- `archive/v4_to_v6/` — v4-v6 training scripts (deprecated)

The latest training script is `train.py` (v7+), which supports all model backends via `model_registry`.
