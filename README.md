# Open-Vision-Banana

**Image Generators are Generalist Vision Learners** — Open-source implementation of Vision Banana.

Vision Banana demonstrates that state-of-the-art visual understanding can emerge from image generation pretraining. By instruction-tuning FLUX.1 [schnell] on a mixture of its original training data alongside a small amount of vision task data, we obtain a model that achieves SOTA zero-shot transfer on segmentation, depth estimation, and surface normal estimation — all through the lens of image generation.

## Core Idea

> **Vision task outputs are parameterized as RGB images.** The model generates colored visualizations that can be decoded back to structured vision outputs. This eliminates the need for task-specific architectures — only the prompt changes.

## Supported Tasks

| Task | Type | Prompt Style |
|------|------|-------------|
| Semantic Segmentation | 2D | Color mapping: `{"cat": "red", "background": "yellow"}` |
| Instance Segmentation | 2D | "Each dog is colored differently" |
| Referring Expression Segmentation | 2D | "The man in pink t shirt is rendered solid white" |
| Metric Depth Estimation | 3D | Power transform + cube-edge colormap |
| Surface Normal Estimation | 3D | Direct linear mapping (x,y,z) → (R,G,B) |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Vision Banana                       │
│                                                          │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────┐  │
│  │Prompt Builder│──▶│  FLUX.1     │──▶│  RGB Output  │  │
│  │             │   │  [schnell]   │   │              │  │
│  └─────────────┘   └─────────────┘   └──────────────┘  │
│         │                │               │              │
│         ▼                ▼               ▼              │
│    Depth Decoder   Normal Decoder   Segmentation Decoder│
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │          Instruction Tuning (rsLoRA)             │   │
│  │  ┌─────────────────────────────────────────┐     │   │
│  │  │ LoRA rank=64, alpha=128                │     │   │
│  │  │ rsLoRA scaling: α/√r                   │     │   │
│  │  │ Data mix: 9:1 vision-to-language        │     │   │
│  │  │ Params trained: 0.5%-1% of total        │     │   │
│  │  │ Generation preserved: 90%+              │     │   │
│  │  └─────────────────────────────────────────┘     │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Evaluation Pipeline                  │   │
│  │  ┌────────────────────────────────────────────┐  │   │
│  │  │ Depth: AbsRel, SqRel, RMSE, δ<1.25       │  │   │
│  │  │ Normal: Mean Angular Error, Cosine Sim     │  │   │
│  │  │ Semantic: mIoU, Pixel Accuracy             │  │   │
│  │  │ Instance: AP@0.5, AP@0.75, mAP@[0.5:0.95]│  │   │
│  │  └────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Core only (no GPU dependencies)
pip install -e .

# With inference support (requires GPU)
pip install -e ".[inference]"

# With training support
pip install -e ".[training]"

# Full installation
pip install -e ".[all]"
```

## Quick Start

### Core Algorithms (no GPU required)

```python
from open_vision_banana.core import DepthRGBMapper, NormalRGBMapper
from open_vision_banana.core import SegmentationDecoder
import torch

# Depth → RGB → Depth (roundtrip)
mapper = DepthRGBMapper()
depth = torch.rand(1, 1, 512, 512) * 50.0
rgb = mapper.depth_to_rgb(depth)
recovered = mapper.rgb_to_depth(rgb)

# Normal → RGB → Normal
mapper = NormalRGBMapper()
normal = torch.randn(1, 3, 512, 512)
normal = normal / normal.norm(dim=1, keepdim=True).clamp(min=1e-8)
rgb = mapper.normal_to_rgb(normal)

# Instance segmentation decoding
decoder = SegmentationDecoder()
instances = decoder.decode_instance(rgb)
```

### Inference (requires GPU + 24GB VRAM)

```python
from open_vision_banana import VisionBanana

# Load model (local path, hf:repo/id, or registry:alias)
model = VisionBanana.from_pretrained(lora_path="registry:vision-banana")

# 16GB GPU: enable CPU offload
model = VisionBanana.from_pretrained(
    lora_path="path/to/checkpoint",
    enable_sequential_cpu_offload=True,
)

# Depth estimation
result = model.predict(image=input_image, task="depth")
```

CLI with memory optimizations:

```bash
# List registered checkpoints
python scripts/download_checkpoint.py --list

# Download LoRA from Hugging Face (when published)
python scripts/download_checkpoint.py --alias vision-banana --output-dir checkpoints/

# Inference on 16GB GPU
python scripts/inference.py --image input.jpg --task depth \
  --lora-path registry:vision-banana --cpu-offload --vae-slicing
```

### Data Validation (before training)

Validate custom datasets in ``images/`` + ``targets/`` layout:

```bash
python scripts/validate_data.py --data-path my_tiny_data/
python scripts/validate_data.py --data-path my_tiny_data/ --strict --min-pairs 10
```

Prepare Hypersim depth data for instruction tuning:

```bash
python scripts/prepare_hypersim.py \
  --source /path/to/hypersim \
  --output data/hypersim_depth/ \
  --resolution 512 \
  --validate
```

```python
from open_vision_banana.data import validate_vision_dataset

report = validate_vision_dataset("my_tiny_data/")
print(report.summary())
assert report.ok
```

### Inference Optimization

```bash
# Show available presets and VRAM impact
python scripts/optimize_inference.py --list-presets

# Use a preset profile (balanced | speed | memory | quality | cpu)
python scripts/optimize_inference.py --profile balanced --run --image data/sample.jpg --task depth

# Benchmark mapper throughput with preset
python scripts/optimize_inference.py --profile speed --benchmark-mappers --resolution 512
```

| Preset | Target VRAM | Speed | Quality | Key Options |
|--------|-------------|-------|---------|-------------|
| `balanced` | ~16 GB (A10/RTX 3090) | ★★★ | ★★★★ | float16 + model_cpu_offload + VAE slicing |
| `speed` | ~20 GB (A100/H100) | ★★★★★ | ★★★ | float16 + torch.compile + VAE tiling |
| `memory` | ~8 GB (RTX 3060) | ★★ | ★★★ | float16 + sequential offload + INT8 mappers |
| `quality` | ~40 GB (A100-80GB) | ★★★ | ★★★★★ | bfloat16 + max-autotune compile |
| `cpu` | RAM only | ★ | ★★★ | float32 + sequential offload |

### Attention Visualization

```bash
# Dry-run (synthetic attention, no model download)
python scripts/visualize_attention.py --image input.jpg --prompt "depth estimation" --output output/attn/ --dry-run

# Real attention extraction (requires FLUX weights)
python scripts/visualize_attention.py --image input.jpg --prompt "segmentation of a dog" --output output/attn/
```

### Pipeline Benchmark

```bash
# Full pipeline benchmark (requires GPU + model)
python scripts/benchmark_pipeline.py --profile balanced --tasks depth,normal

# Mapper-only benchmark (no GPU required)
python scripts/benchmark_pipeline.py --mapper-only --resolution 512 --num-samples 50

# Export JSON results
python scripts/benchmark_pipeline.py --mapper-only --output benchmark_results.json
```

### Data Preparation

```bash
# List all supported datasets with download instructions
python scripts/download_data.py --list

# Download and prepare a specific dataset
python scripts/download_data.py --dataset hypersim-depth --output data/hypersim/

# Prepare from existing raw download (for datasets requiring login)
python scripts/prepare_hypersim.py \
  --source /path/to/hypersim_raw \
  --output data/hypersim_depth/ \
  --resolution 512 --validate

# Validate prepared dataset before training
python scripts/validate_dataset.py --data-path data/hypersim_depth/ --task depth
```

### Publishing to Hugging Face

```bash
# Preview the model card before publishing
python scripts/publish_to_hub.py \
  --checkpoint output/checkpoints/best_model.pt \
  --repo-id open-vision-banana/vision-banana-depth-v1 \
  --task depth \
  --description "Depth estimation LoRA trained on Hypersim" \
  --dry-run

# Publish and update local registry
python scripts/publish_to_hub.py \
  --checkpoint output/checkpoints/best_model.pt \
  --repo-id open-vision-banana/vision-banana-depth-v1 \
  --task depth \
  --update-registry
```

### Training

```bash
# Basic training with validation + early stopping
python scripts/train.py \
  --data-path my_tiny_data/ \
  --val-split 0.1 \
  --early-stopping-patience 5 \
  --epochs 10

# 9:1 generation mix + W&B logging + checkpoint artifacts
python scripts/train.py \
  --data-path vision_data/ \
  --generation-path generation_data/ \
  --vision-ratio 0.1 \
  --wandb --wandb-log-checkpoints

# TensorBoard
python scripts/train.py --data-path my_tiny_data/ --tensorboard
tensorboard --logdir output/runs

# Multi-GPU FSDP (8×A100 example)
torchrun --nproc_per_node=8 scripts/train.py \
  --data-path vision_data/ \
  --fsdp --gradient-checkpointing \
  --batch-size 2 --epochs 10

# Multi-GPU DeepSpeed ZeRO-3
deepspeed --num_gpus=8 scripts/train.py \
  --deepspeed --deepspeed-config configs/deepspeed_zero3.json \
  --data-path vision_data/ --gradient-checkpointing \
  --batch-size 2 --epochs 10
```

Generation dataset layout for 9:1 mix:

```bash
# Build synthetic generation data (no GPU)
python scripts/build_generation_dataset.py --output-dir gen_data/ --num-samples 200

# Or from custom prompts
python scripts/build_generation_dataset.py \
  --output-dir gen_data/ --prompts-file prompts.txt --mode synthetic

# FLUX mode (GPU + weights)
python scripts/build_generation_dataset.py \
  --output-dir gen_data/ --prompts-file prompts.txt --mode flux
```

```
generation_data/
  images/          ← RGB images
  prompts.json     ← {"photo.jpg": "a cat on a windowsill", ...}
```

```python
from open_vision_banana.training import InstructionTuner, VisionTaskDataset
from open_vision_banana.data import build_dataset, MixedVisionDataset, build_generation_dataset

# Create datasets
depth_ds = build_dataset("depth", root_dir="/path/to/hypersim")
gen_ds = build_generation_dataset("/path/to/generation_data")
mixed_ds = MixedVisionDataset(
    vision_datasets=[depth_ds],
    generation_dataset=gen_ds,
    vision_ratio=0.1,
)

# Train
tuner = InstructionTuner(model, lora_rank=64, lora_alpha=128)
tuner.train(mixed_ds, val_dataset=val_ds, early_stopping_patience=5)
```

### Configuration

```python
from open_vision_banana import VisionBananaConfig

# Load from YAML
config = VisionBananaConfig.from_yaml("configs/training_config.yaml")

# Modify and save
config.lora.rank = 32
config.optimizer.lr = 1e-4
config.to_yaml("my_config.yaml")
```

### Export

```python
from open_vision_banana.export import export_mapper_onnx

# Export depth mapper to ONNX
export_mapper_onnx("depth_to_rgb", "depth_mapper.onnx")

# Export normal mapper to ONNX
export_mapper_onnx("normal_to_rgb", "normal_mapper.onnx")
```

### Evaluation

```python
from open_vision_banana.evaluation import evaluate_depth, evaluate_normals

# Depth metrics
metrics = evaluate_depth(pred_depth, gt_depth)
# → {"AbsRel": 0.12, "SqRel": 0.45, "RMSE": 3.2, ...}

# Normal metrics
metrics = evaluate_normals(pred_normals, gt_normals)
# → {"Mean Angular Error (°)": 15.3, "Cosine Similarity": 0.96, ...}
```

CLI with SOTA comparison:

```bash
python scripts/evaluate.py --task depth --data-path /data/hypersim --compare-sota
python scripts/evaluate.py --task semantic --data-path /data/cityscapes --output-json metrics.json
```

## Project Structure

```
open-vision-banana/
├── src/open_vision_banana/
│   ├── __init__.py              # Lazy imports, version
│   ├── config.py                # Hierarchical config (8 sub-configs)
│   ├── core/                    # Core algorithms (no external deps)
│   │   ├── depth_mapper.py      # Power transform + cube-edge colormap
│   │   ├── normal_mapper.py     # Linear normal ↔ RGB mapping
│   │   ├── segmentation.py      # 5-stage instance segmentation decoder
│   │   ├── union_find.py       # Union-Find (path compression + union by rank)
│   │   └── rgb_mapper.py       # Unified encode/decode interface
│   ├── inference/               # Model loading and generation
│   │   ├── generator.py         # FLUX.1 + LoRA wrapper
│   │   ├── pipeline.py          # VisionBanana unified model
│   │   ├── prompt.py            # Task-specific prompt builder
│   │   └── decoder.py           # Visualization decoder
│   ├── training/                # LoRA instruction tuning
│   │   ├── tuner.py             # InstructionTuner + VisionTaskDataset
│   │   ├── utils.py             # EMA, DDP, augmentation, gradient checkpointing
│   │   └── checkpoint.py        # CheckpointManager
│   ├── evaluation/              # Metrics for all vision tasks
│   │   ├── depth.py             # AbsRel, SqRel, RMSE, δ thresholds
│   │   ├── normal.py            # Angular error, cosine similarity
│   │   ├── semantic.py          # mIoU, pixel accuracy
│   │   ├── instance.py          # AP@0.5, AP@0.75, mAP
│   │   ├── referring.py         # IoU, Precision@threshold
│   │   └── evaluator.py         # Unified VisionEvaluator
│   ├── export/                  # Model export
│   │   └── onnx_export.py       # ONNX + TorchScript export
│   └── data/                    # Dataset loaders
│       ├── datasets.py           # Hypersim, ScanNet, Cityscapes, RefCOCOg
│       └── mixed.py             # 9:1 mixed dataset
├── tests/
│   ├── test_modules.py          # Core algorithm tests
│   ├── test_training_utils.py   # Training utility tests
│   ├── test_evaluation.py       # Evaluation metric tests
│   ├── test_e2e.py              # ONNX export verification
│   └── test_data_config.py      # Data + config tests
├── configs/
│   ├── training_config.yaml     # Default training configuration
│   └── checkpoint_registry.yaml # Hugging Face LoRA checkpoint aliases
├── scripts/
│   ├── demo.py                  # Algorithm demo (no GPU needed)
│   ├── validate_data.py         # Dataset format validation
│   ├── build_generation_dataset.py  # Synthetic / FLUX generation data builder
│   ├── download_checkpoint.py   # Hugging Face LoRA download
│   ├── optimize_inference.py    # Inference optimization helper
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── inference.py             # Inference script
├── docs/                        # Documentation
├── paper/                       # Reference paper
└── pyproject.toml               # pip-installable package config
```

## Key Algorithms

### 1. Depth → RGB Mapping (Power Transform + Spectral Colormap)

Maps unbounded depth [0, ∞) to bounded RGB [0, 1]³ via power transform:

```
f(d, λ, c) = 1 - (1 - d/(λc))^(λ+1)
```

Then maps via a **cube-edge traversal colormap** (black → blue → cyan → green → yellow → white) with piecewise-linear interpolation. Parameters: λ=-3, c=10/3.

### 2. Normal → RGB Mapping (Direct Linear)

Camera-space normals (right-handed: +x right, +y up, +z toward camera):

```
R = (1-x)/2,  G = (1+y)/2,  B = (1+z)/2
```

### 3. Instance Segmentation Decoding (5-Stage Clustering)

From Appendix A of the paper:

1. **Background Initialization**: τ=14 color tolerance
2. **Color Similarity Grouping**: Union-Find with 16-connectivity (8 + 8 knight's move, O(E α(N)), ~100x faster than flood-fill)
3. **Noise Pruning**: θ_size = 2×10⁻⁴
4. **Boundary Artifact Elimination**: 3×3 erosion, θ_erosion=0.1
5. **Spatially Constrained Merging**: γ=5.0

### 4. Instruction Tuning (rsLoRA)

- LoRA rank=64, alpha=128 with rsLoRA scaling (α/√r)
- 9:1 visual-to-language data mixing ratio
- Only 0.5%-1% of total parameters trained
- Preserves 90%+ generation capability

## Benchmarks (from paper)

| Task | Benchmark | Metric | Vision Banana | Previous SOTA |
|------|-----------|--------|---------------|---------------|
| Semantic Seg | Cityscapes | mIoU | 69.9 | 65.2 (SAM 3) |
| Instance Seg | SA-Co/Gold | cgF1 | 47.5 | 24.6 (OWLv2) |
| Referring Seg | RefCOCOg | cIoU | 73.8 | 67.7 (LocHeads) |
| Referring Seg | ReasonSeg | gIoU | 79.3 | 77.0 (SAM3+GPT) |
| Depth | 6-dataset avg | δ1 | 0.882 | 0.823 (UniK3D) |
| Normal | Indoor avg | mean ↓ | 15.05 | 15.72 (Lotus-2) |

### Distributed Training (DDP / FSDP)

```bash
# DDP — single node, 4 GPUs
bash scripts/train_ddp.sh --data-path vision_data/ --epochs 5

# FSDP — single node, 8 GPUs (recommended for FLUX.1 12B)
bash scripts/train_fsdp.sh --nodes 1 --gpus-per-node 8 \
  --data-path vision_data/ --epochs 5 --gradient-checkpointing

# Multi-node FSDP (2 nodes × 8 GPUs)
# Node 0 (master):
bash scripts/train_fsdp.sh --nodes 2 --node-rank 0 \
  --master-addr 10.0.0.1 --data-path /shared/vision_data/
# Node 1:
bash scripts/train_fsdp.sh --nodes 2 --node-rank 1 \
  --master-addr 10.0.0.1 --data-path /shared/vision_data/
```

### Data Validation (Pre-training)

```bash
# Validate dataset integrity before training
python scripts/validate_dataset.py --data-path vision_data/ --task all --output validation_report.json

# Checks: image/target pair integrity, corrupt files, depth/normal range sanity, resolution consistency
```

### Inference Optimization Presets

```bash
# Use a preset profile (balanced | speed | memory | quality | cpu)
python scripts/optimize_inference.py --preset configs/optimization_presets.yaml --profile balanced

# Profiles:
#   balanced  — A10 / RTX 3090, model CPU offload + float16
#   speed     — torch.compile + float16, fastest
#   memory    — sequential CPU offload + VAE tiling, fits 12GB
#   quality   — bfloat16 + max-autotune compile, A100-80GB
#   cpu       — no GPU required, universally compatible
```

## Docker

Build and run a reproducible environment:

```bash
# CPU image (core algorithms, inference)
docker build -t open-vision-banana:latest -f docker/Dockerfile .

# GPU image (training + CUDA inference, multi-stage build)
docker build -t open-vision-banana:gpu -f docker/Dockerfile.gpu .

# Docker Compose — development, training, or Jupyter
docker compose -f docker/docker-compose.yml up dev      # editable CPU dev
docker compose -f docker/docker-compose.yml up training # GPU FSDP training
docker compose -f docker/docker-compose.yml up notebook # Jupyter on :8888
```

See `docker/README.md` for GPU requirements, troubleshooting, and multi-stage build details.

# Core algorithms (CPU)
docker run --rm open-vision-banana:latest \
  python -c "from open_vision_banana.core import DepthRGBMapper; print('ok')"

# GPU inference (requires nvidia-container-toolkit)
docker run --gpus all -v $(pwd)/data:/data open-vision-banana:latest \
  python scripts/inference.py --image /data/input.jpg --task depth
```

Interactive Colab demo: [`notebooks/colab_demo.ipynb`](notebooks/colab_demo.ipynb)

## Citation

```bibtex
@article{visionbanana2025,
  title={Image Generators are Generalist Vision Learners},
  author={Google Research},
  year={2025}
}
```

## License

This project is for research purposes. The base FLUX.1 model follows its own license terms.
