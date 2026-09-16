# Scripts Directory

Active scripts for Open-Vision-Banana training, inference, evaluation, and utilities.

## Training

| Script | Description |
|--------|-------------|
| `train.py` | Main training entry point (multi-backend: FLUX, SANA, SDXL, etc.) |
| `train_sana_512.py` | SANA-600M/1600M 512px LoRA training |
| `train_smoke.py` | CPU smoke test (1-2 epochs, synthetic data) |
| `train_and_visualize.py` | Training + inline visualization |
| `merge_sana_lora.py` | Merge SANA LoRA adapter into base model |

## Inference

| Script | Description |
|--------|-------------|
| `inference.py` | Single-image / batch inference for all 10 vision tasks |
| `infer_sana_512.py` | SANA-512 img2img inference with flow matching |
| `infer_checkpoint_viz.py` | Checkpoint inference + visualization |
| `demo.py` | Core algorithm demo (no GPU required) |

## Evaluation

| Script | Description |
|--------|-------------|
| `evaluate.py` | Basic evaluation on benchmark datasets |
| `evaluate_model.py` | Full evaluation (DDP, resume, SOTA comparison) |
| `evaluate_depth_standard.py` | Standard depth evaluation with colormap |
| `evaluate_generation_preservation.py` | Generation preservation metrics (FID, CLIP) |
| `evaluate_paper_benchmarks.py` | Paper benchmark preflight + evaluation |
| `eval_checkpoints.py` | Batch evaluation across checkpoint series |
| `qualitative_eval.py` | Qualitative visualization grids (depth/normal/seg) |
| `qualitative_eval_robust.py` | Robust qualitative eval with retry mechanism |
| `run_paper_judges.py` | Run Gemini/CLIP paper judges |
| `benchmark.py` | Core algorithm benchmark (no GPU required) |
| `benchmark_pipeline.py` | Full pipeline benchmark |

## Data Preparation

| Script | Description |
|--------|-------------|
| `prepare_hypersim.py` | Prepare Hypersim depth + normal dataset |
| `prepare_depth_dataset.py` | Prepare depth estimation dataset |
| `prepare_blendedmvs.py` | Prepare BlendedMVS dataset |
| `prepare_cityscapes_19class.py` | Prepare Cityscapes 19-class segmentation |
| `prepare_kitti_metric.py` | Prepare KITTI metric depth |
| `prepare_synthetic_depth.py` | Prepare synthetic depth data |
| `prepare_tartanair.py` | Prepare TartanAir dataset |
| `build_generation_dataset.py` | Build synthetic generation dataset |
| `build_gzip_index.py` | Pre-build gzip seek index for SNVC-50M tar stream |
| `build_gzip_index_snvc50m.py` | Split tar.gz stream gzip index builder |
| `build_task_tar_index.py` | Split full tar index into per-task indexes |
| `build_volume_map.py` | Build file→volume mapping for SNVC-50M |
| `build_volume_map_fast.py` | Fast volume mapping via ISIZE estimation |
| `download_data.py` | Download training datasets |
| `download_all_datasets.py` | Download all datasets at once |
| `validate_dataset.py` | Comprehensive dataset validation |

## Model Download & Export

| Script | Description |
|--------|-------------|
| `download_model.py` | Download base diffusion models (23 families) |
| `download_all_models.py` | Download all model weights |
| `download_checkpoint.py` | Download pretrained LoRA checkpoints |
| `convert_sana_1600m_to_diffusers.py` | Convert SANA-1600M .pth to diffusers format |
| `export.py` | RGB mapper ONNX export |
| `export_onnx.py` | ONNX export (all mappers) |
| `publish_to_hub.py` | Publish model to HuggingFace Hub |

## Utilities

| Script | Description |
|--------|-------------|
| `optimize_inference.py` | Inference optimization (quantization, offloading) |
| `visualize_attention.py` | Attention map visualization |
| `validate_training_config.py` | Validate training config files |

## Shell Launch Scripts

| Script | Description |
|--------|-------------|
| `launch_training_mps.sh` | Apple Silicon MPS single-GPU |
| `launch_h20_single.sh` | H20 single-GPU |
| `launch_h20_ddp.sh` | H20 multi-GPU DDP |
| `launch_h20_deepspeed.sh` | H20 DeepSpeed ZeRO |
| `launch_h20_fsdp.sh` | H20 FSDP |
| `launch_training_a100_80g.sh` | A100-80G single-GPU |
| `launch_training_a100_80g_ddp.sh` | A100-80G multi-GPU DDP |
| `launch_training_a100_80g_deepspeed.sh` | A100-80G DeepSpeed |
| `launch_training_a100_80g_fsdp.sh` | A100-80G FSDP |
| `train_a100_40g_deepspeed.sh` | A100-40G DeepSpeed (with offload) |
| `train_ddp.sh` | Generic multi-GPU DDP launcher |
| `train_fsdp.sh` | Generic FSDP launcher |
| `unified_launch.sh` | Unified launcher (auto-detects hardware) |

## Archive

Historical and archived scripts are in `archive/`:
- `archive/pre_opensource/` — Pre-open-source experimental scripts (245 files)
