# Open-Vision-Banana

**Image Generators are Generalist Vision Learners** — Open-source implementation of Vision Banana.

Vision Banana demonstrates that state-of-the-art visual understanding can emerge from image generation pretraining. By instruction-tuning diffusion models (FLUX.1, ZImage, Qwen-Image, SANA, and more) on a mixture of their original training data alongside a small amount of vision task data, we obtain models that achieve SOTA zero-shot transfer on segmentation, depth estimation, and surface normal estimation — all through the lens of image generation.

## Core Idea

> **Vision task outputs are parameterized as RGB images.** The model generates colored visualizations that can be decoded back to structured vision outputs. This eliminates the need for task-specific architectures — only the prompt changes.

## Supported Tasks

| Task | Type | Prompt Style | Status |
|------|------|-------------|--------|
| Semantic Segmentation | 2D | Color mapping: `{"cat": "red", "background": "yellow"}` | ✅ |
| Instance Segmentation | 2D | "Each dog is colored differently" | ✅ |
| Referring Expression Segmentation | 2D | "The man in pink t shirt is rendered solid white" | ✅ |
| Metric Depth Estimation | 3D | Power transform + 6 colormaps (cube-edge/plasma/inferno/viridis/grayscale/rainbow) | ✅ |
| Surface Normal Estimation | 3D | Direct linear mapping (x,y,z) → (R,G,B) | ✅ |
| Edge Detection | 2D | "Edges are white, background is black" | ✅ Speculative |
| Optical Flow | 2D | HSV encoding: direction=hue, magnitude=saturation | ✅ Speculative |
| 2D Keypoint Detection | 2D | Colored dots per keypoint type | ✅ Speculative |
| Object Detection | 2D | Colored bounding box per class | ✅ Speculative |
| Depth Edge Detection | 3D | Depth discontinuity → white edges | ✅ Speculative |

> **Speculative tasks** are based on the paper's mention of "depth estimation, surface normal estimation, segmentation, **etc.**" and the [rainbow colormap mentioned on the project website](https://vision-banana.github.io).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Vision Banana                       │
│                                                          │
│  ┌─────────────┐   ┌─────────────────────────┐   ┌──────┐│
│  │Prompt Builder│──▶│  Diffusion Model        │──▶│ RGB  ││
│  │             │   │  (FLUX.1 / ZImage /       │   │Output││
│  └─────────────┘   │   Qwen-Image / SANA ...) │   └──────┘│
│         │          └─────────────────────────┘      │    │
│         ▼                │                         ▼    │
│    Depth Decoder    Backbone (Transformer/UNet) Segmentation Decoder│
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │          Instruction Tuning (rsLoRA)             │   │
│  │  ┌─────────────────────────────────────────┐     │   │
│  │  │ LoRA rank=32~64, alpha=64~128          │     │   │
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

> 📖 **GPU Training Guide**: See [`docs/quick-start.md`](docs/quick-start.md) for the complete H20 / A100 / RTX training quick-start with copy-paste commands.

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

### Inference (requires GPU + 16-24GB VRAM)

```python
from open_vision_banana import VisionBanana

# Load model (local path, hf:repo/id, or registry:alias)
# Supported models: flux-schnell, z-image, qwen-image, sana-600m, lumina2, chroma, etc.
model = VisionBanana.from_pretrained(lora_path="registry:vision-banana")

# Or specify a different base model (default is FLUX.1-schnell)
model = VisionBanana.from_pretrained(
    lora_path="path/to/checkpoint",
    model_name="z-image",  # or "qwen-image", "sana-600m", etc.
)

# 16GB GPU: enable CPU offload
model = VisionBanana.from_pretrained(
    lora_path="path/to/checkpoint",
    enable_sequential_cpu_offload=True,
)

# Depth estimation
result = model.predict(image=input_image, task="depth")
```

**Supported Base Models** (via model registry):

| Model | Family | Size | License | VRAM |
|-------|--------|------|---------|------|
| FLUX.1-schnell | flux | 12B | Apache-2.0 | ~24GB |
| Z-Image-Turbo | z-image | 6B | Apache-2.0 | ~24GB |
| Qwen-Image | qwen-image | ~6B | Apache-2.0 | ~24GB |
| SANA-600M | sana | 600M | Apache-2.0 | ~12GB |
| Lumina2 | lumina2 | ~3B | Apache-2.0 | ~16GB |
| Chroma | chroma | ~3B | MIT | ~16GB |

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

> **Production Tip**: Use `scripts/unified_launch.sh` for all training modes — it auto-detects your GPU type and recommends the best strategy. See [`docs/quick-start.md`](docs/quick-start.md) for the complete GPU training guide (H20 / A100 / RTX).

```bash
# ─── Recommended: Universal launcher (auto-detects GPU & mode) ───
bash scripts/unified_launch.sh --config configs/unified_training_config.yaml

# Override mode manually
bash scripts/unified_launch.sh --mode ddp     --config configs/training_config.yaml
bash scripts/unified_launch.sh --mode fsdp    --config configs/training_config.yaml --gradient-checkpointing
bash scripts/unified_launch.sh --mode deepspeed --config configs/training_config.yaml

# Dry-run preview (no execution)
bash scripts/unified_launch.sh --config configs/unified_training_config.yaml --dry-run

# Environment variable overrides
OVB_BATCH_SIZE=4 OVB_LR=1e-4 bash scripts/unified_launch.sh --config configs/training_config.yaml
```

**Legacy direct training** (for custom scripts or debugging):

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

**Hardware-specific quick launch** (see `docs/quick-start.md` for full details):

| GPU | VRAM | Config | Launcher |
|-----|------|--------|----------|
| H20 | 96GB | `configs/training_h20_single.yaml` | `bash scripts/launch_h20_single.sh` |
| H20 ×8 | 96GB×8 | `configs/training_h20_ddp_8x.yaml` | `bash scripts/launch_h20_ddp.sh` |
| A100-80GB | 80GB | `configs/training_a100_80g.yaml` | `bash scripts/launch_training_a100_80g.sh` |
| A100-40GB | 40GB | `configs/training_a100_40g_deepspeed.yaml` | `bash scripts/train_a100_40g_deepspeed.sh` |
| RTX 4090 | 24GB | `configs/unified_training_config.yaml` | `bash scripts/unified_launch.sh --mode deepspeed` |
| MPS (Apple) | Unified | `configs/training_mps.yaml` | `bash scripts/unified_launch.sh --mode mps` |

**Multi-model training** (train on ZImage, Qwen-Image, SANA, etc.):

```bash
# Train on Z-Image-Turbo instead of FLUX.1
sed -i 's|black-forest-labs/FLUX.1-schnell|z-image|g' configs/unified_training_config.yaml
bash scripts/unified_launch.sh --config configs/unified_training_config.yaml
```

All registered models: `flux-schnell`, `z-image`, `qwen-image`, `sana-600m`, `sana-1600m`, `lumina2`, `chroma`, `sdxl`, `sd3`, `pixart`, `cogview4`, `flux2-dev`, ...

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
│   ├── model_registry.py        # Multi-model registry (18+ families: FLUX, ZImage, Qwen, SANA, Lumina2...)
│   ├── core/                    # Core algorithms (no external deps)
│   │   ├── depth_mapper.py      # Power transform + 6 colormaps (cube-edge, plasma, inferno, viridis, grayscale, rainbow)
│   │   ├── adaptive_depth_mapper.py  # Adaptive quantization mapper (roundtrip error ~0.34%)
│   │   ├── normal_mapper.py     # Linear normal ↔ RGB mapping
│   │   ├── segmentation/        # 5-stage instance segmentation decoder
│   │   │   ├── decoder.py       # Adjacent + seed-based Union-Find
│   │   │   └── utils.py         # Segmentation utilities
│   │   ├── edge_mapper.py       # Edge detection: white edges on black background
│   │   ├── flow_mapper.py       # Optical flow: HSV → RGB encoding/decoding
│   │   ├── keypoint_mapper.py   # 2D keypoint detection: colored dots (speculative)
│   │   ├── object_detection_mapper.py  # Object detection: colored bbox outlines (speculative)
│   │   ├── depth_edge_mapper.py # Depth edge detection: Sobel gradient → white edges (speculative)
│   │   ├── union_find.py       # Union-Find (path compression + union by rank)
│   │   └── rgb_mapper.py       # Unified encode/decode interface for 10 tasks
│   ├── inference/               # Model loading and generation
│   │   ├── checkpoint.py        # LoRA checkpoint resolution (local / HF / registry)
│   │   ├── generator.py         # Multi-model + LoRA wrapper (18+ families)
│   │   ├── pipeline.py          # VisionBanana unified model (10 tasks)
│   │   ├── prompt.py            # Task-specific prompt builder (10 tasks)
│   │   ├── decoder.py           # Visualization decoder (10 tasks)
│   │   └── optimization.py      # Inference optimization presets + INT8 quantization
│   ├── training/                # LoRA instruction tuning
│   │   ├── tuner.py             # InstructionTuner + VisionTaskDataset (multi-model aware)
│   │   ├── trainer_interface.py # TrainerInterface ABC + 5 backends (Native/DDP/FSDP/DeepSpeed/Accelerate)
│   │   ├── checkpoint.py        # CheckpointManager
│   │   ├── early_stopping.py    # EarlyStopping with best-checkpoint saving
│   │   ├── logging.py           # TrainingLogger (W&B / TensorBoard)
│   │   ├── lr_finder.py         # Learning rate finder
│   │   ├── utils.py             # EMA, VisionAugmentation, gradient checkpointing
│   │   ├── fsdp_utils.py        # FSDP wrapper (save/load with CPU offload)
│   │   ├── deepspeed_utils.py   # DeepSpeed ZeRO-3 initialization
│   │   ├── latent_cache.py      # Latent caching for faster training
│   │   └── optimization.py      # Training optimization utilities
│   ├── evaluation/              # Metrics for all vision tasks
│   │   ├── depth.py             # AbsRel, SqRel, RMSE, δ thresholds
│   │   ├── normal.py            # Angular error, cosine similarity
│   │   ├── semantic.py          # mIoU, pixel accuracy
│   │   ├── instance.py          # AP@0.5, AP@0.75, mAP
│   │   ├── referring.py         # IoU, Precision@threshold
│   │   ├── edge.py              # Edge detection metrics (precision, recall, F1)
│   │   ├── flow.py              # Optical flow metrics (EPE, AE, F1)
│   │   ├── baselines.py         # SOTA benchmark comparison data
│   │   └── evaluator.py         # Unified VisionEvaluator
│   ├── export/                  # Model export
│   │   └── onnx_export.py       # ONNX + TorchScript export
│   └── data/                    # Dataset loaders
│       ├── datasets.py           # Hypersim, ScanNet, Cityscapes, SA-Co, RefCOCOg
│       ├── mixed.py             # 9:1 mixed dataset with multi-colormap support
│       ├── split.py             # Train/validation split utility
│       ├── validation.py        # Dataset integrity validation
│       ├── generation.py        # GenerationDataset (synthetic / FLUX modes)
│       └── generation_builder.py # Generation dataset builder
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

- LoRA rank=32~64, alpha=64~128 with rsLoRA scaling (α/√r) — model-family aware defaults
- 9:1 visual-to-language data mixing ratio
- Only 0.5%-1% of total parameters trained
- Preserves 90%+ generation capability
- Supports 18+ model families via unified registry

## Benchmarks (from paper)

| Task | Benchmark | Metric | Vision Banana | Previous SOTA |
|------|-----------|--------|---------------|---------------|
| Semantic Seg | Cityscapes | mIoU | 69.9 | 65.2 (SAM 3) |
| Instance Seg | SA-Co/Gold | cgF1 | 47.5 | 24.6 (OWLv2) |
| Referring Seg | RefCOCOg | cIoU | 73.8 | 67.7 (LocHeads) |
| Referring Seg | ReasonSeg | gIoU | 79.3 | 77.0 (SAM3+GPT) |
| Depth | 6-dataset avg | δ1 | 0.882 | 0.823 (UniK3D) |
| Normal | Indoor avg | mean ↓ | 15.05 | 15.72 (Lotus-2) |

## Datasets

> **Important:** The paper explicitly states that **no training data from evaluation benchmarks is included** in the instruction-tuning mixture. All evaluation benchmarks listed below must be **strictly held out** and used only for validation.

### Evaluation Benchmarks (Hold-out Only)

These datasets are used to reproduce the paper's quantitative results in Table 1. They must **not** be mixed into training data.

| Task | Benchmark | Metric | Paper Result | Size | License | Status |
|------|-----------|--------|-------------|------|---------|--------|
| **Semantic Seg** | [Cityscapes val](https://www.cityscapes-dataset.com/) | mIoU ↑ | 0.699 | 5K images | Non-commercial (registration required) | ✅ Available |
| **Instance Seg** | [SA-Co/Gold](https://github.com/facebookresearch/sam3/tree/main/scripts/eval/gold) | pmF1 ↑ | 0.540 | 500-query subset | MetaCLIP/SA-1B terms | ✅ Annotations public |
| **Referring Seg** | [RefCOCOg UMD val](https://github.com/lichengunc/refer) | cIoU ↑ | 0.738 | 26K images | COCO (CC BY 4.0) | ✅ Available |
| **Reasoning Seg** | [ReasonSeg val](https://github.com/dvlab-reasoning/ReasonSeg) | gIoU ↑ | 0.793 | 1,218 samples | Academic | ✅ Available |
| **Depth** | [NYU v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) | δ1 / AbsRel | 0.948 / 0.081 | 1,449 RGB-D | Academic | ✅ Available |
| **Depth** | [KITTI](http://www.cvlibs.net/datasets/kitti/) | δ1 / AbsRel | 0.915 / 0.107 | 23K frames | Academic (CC BY-NC-SA 3.0) | ✅ Registration required |
| **Depth** | [DIODE](https://diode-dataset.org/) | δ1 / AbsRel | 0.917 / 0.108 | 41K frames | Academic | ✅ Available |
| **Depth** | [ETH3D](https://www.eth3d.net/) | δ1 / AbsRel | 0.935 / 0.103 | 454 frames | Academic | ✅ Available |
| **Depth** | [iBims-1](https://github.com/atlas1992/customized_iBims1) | δ1 / AbsRel | 0.934 / 0.078 | 100 images | Academic | ✅ Available |
| **Depth** | [nuScenes](https://www.nuscenes.org/) | δ1 / AbsRel | 0.643 / 0.219 | 40K keyframes | Academic (registration) | ✅ Available |
| **Normal** | [NYU v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) | mean/median angle ↓ | 17.78° / 8.88° | 1,449 RGB-D | Academic | ✅ Available |
| **Normal** | [ScanNet](http://www.scan-net.org/) | mean/median angle ↓ | 13.82° / 11.56° | 2.5M views | Academic (registration) | ✅ Available |
| **Normal** | [DIODE](https://diode-dataset.org/) | mean/median angle ↓ | — | 41K frames | Academic | ✅ Available |
| **Normal** | [Virtual KITTI](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) | mean/median angle ↓ | 29.06° / 10.70° | 21K frames | Academic | ✅ Available |
| **Generation** | [GenAI-Bench](https://github.com/TIGER-AI-Lab/GenAI-Bench) | Win Rate ↑ | 53.5% | 1,600 prompts | Apache 2.0 | ✅ `pip install genaibench` |
| **Generation** | [ImgEdit](https://github.com/PKU-YuanGroup/ImgEdit) | Win Rate ↑ | 47.8% | 811 test cases | Academic | ✅ Available |

### Training Datasets (Public Alternatives)

The paper uses **in-house synthetic annotations** and **rendering engine outputs** that are not publicly available. The following public datasets can be used as **functional replacements** for instruction-tuning.

#### Core Training Set

> **Hypersim 1.9TB 太大？** 论文同款合成数据，但完整下载需要约 1.9TB。以下提供**轻量级替代方案**（17-50GB），功能等价。

**方案 A：轻量级（推荐，~50 GB 总训练数据）**

| Dataset | Task | Size | Why | Download |
|---------|------|------|-----|----------|
| **Virtual KITTI 2** | Depth + Normal + Flow + Instance | ~50 GB | 合成完美标注，室外场景，覆盖 4 个任务 | [naverlabs.com](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) |
| **BlendedMVS** | Depth | ~17 GB | 多视图立体，深度质量高 | [GitHub](https://github.com/YoYo000/MVSNet) |
| **COCO 2017** | Instance + Keypoint + Detection | ~25 GB | 标准多任务基础数据集 | [cocodataset.org](https://cocodataset.org/) |
| **ADE20K** | Semantic | ~1 GB | 语义分割标准训练集 | [sceneparsing.csail.mit.edu](http://sceneparsing.csail.mit.edu/) |
| **RefCOCO / RefCOCO+ / RefCOCOg** | Referring Expression | ~100 MB | 指代表达标准 | [github.com/lichengunc/refer](https://github.com/lichengunc/refer) |
| **ReasonSeg** | Reasoning Segmentation | ~50 MB | 推理指令训练 | [github.com/dvlab-reasoning/ReasonSeg](https://github.com/dvlab-reasoning/ReasonSeg) |
| **FlyingThings3D** | Optical Flow | ~40 GB | 光流标准训练数据 | [lmb.informatik.uni-freiburg.de](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) |
| **BSDS500** | Edge Detection | ~100 MB | 经典边缘检测数据集 | [berkeley.edu](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) |

**方案 B：论文同款（~70 GB）**

| Dataset | Task | Size | Why | Download |
|---------|------|------|-----|----------|
| **Hypersim（子集，前 8 场景）** | Depth + Normal + Semantic + Instance | ~33 GB | 论文同款合成数据，只取前 8 场景 | [GitHub](https://github.com/apple/ml-hypersim) |
| **COCO 2017** | Instance + Keypoint + Detection | ~25 GB | 标准多任务基础 | [cocodataset.org](https://cocodataset.org/) |
| **ADE20K** | Semantic | ~1 GB | 语义分割标准 | [sceneparsing.csail.mit.edu](http://sceneparsing.csail.mit.edu/) |
| **RefCOCO / RefCOCO+ / RefCOCOg** | Referring Expression | ~100 MB | 指代表达标准 | [github.com/lichengunc/refer](https://github.com/lichengunc/refer) |
| **ReasonSeg** | Reasoning Segmentation | ~50 MB | 推理指令训练 | [github.com/dvlab-reasoning/ReasonSeg](https://github.com/dvlab-reasoning/ReasonSeg) |
| **FlyingThings3D** | Optical Flow | ~40 GB | 光流标准训练 | [lmb.informatik.uni-freiburg.de](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) |
| **BSDS500** | Edge Detection | ~100 MB | 经典边缘检测 | [berkeley.edu](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) |

> **建议**：先使用方案 A（轻量级）快速验证 pipeline，再按需扩展到方案 B 或完整 Hypersim。论文中 vision data 仅占训练混合的 ~10%，所以数据量不需要很大。

#### Extended Training Set

| Dataset | Task | Size | Notes |
|---------|------|------|-------|
| Mapillary Vistas | Semantic | ~25 GB | Street scenes; non-commercial license |
| COCO-Stuff | Semantic | Part of COCO | Stuff annotations for semantic segmentation |
| PACO-LVIS | Part-level Semantic | ~5 GB | Part-level segmentation annotations |
| PASCAL-Part | Part-level Semantic | ~1 GB | Part-level segmentation |
| PartImageNet | Part-level Semantic | ~2 GB | Part-level segmentation |
| Taskonomy | Depth + Normal | ~400 GB | Large-scale synthetic indoor (Stanford) |
| TartanAir | Depth + Flow | Large | Synthetic indoor/outdoor drone views |
| MPI Sintel | Flow | ~5 GB | Movie scenes; high-quality flow |
| KITTI 2015 | Flow | Small | Real-world sparse flow |
| COCO Keypoints | Keypoint | In COCO 2017 | 17 keypoint human pose annotations |
| MPII Human Pose | Keypoint | ~2 GB | Classic human pose dataset |
| Objects365 | Detection | ~50 GB | 365-class detection; large scale |
| OpenImages | Detection | ~30 GB | 600-class detection |

### Data Preparation Quick Start

```bash
# List all supported datasets with download instructions
python scripts/download_data.py --list

# Download and prepare a specific dataset
python scripts/download_data.py --dataset hypersim-depth --output data/hypersim/

# Prepare Hypersim for training (the paper's equivalent synthetic data)
python scripts/prepare_hypersim.py \
  --source /path/to/hypersim \
  --output data/hypersim_depth/ \
  --resolution 512 \
  --validate

# Validate prepared dataset before training
python scripts/validate_dataset.py --data-path data/hypersim_depth/ --task depth
```

### Training Data Mixing Strategy

The paper uses a **9:1 generation-to-vision data ratio** (≈10% vision task data). We recommend:

```python
from open_vision_banana.data import MixedVisionDataset, build_generation_dataset
from open_vision_banana.data import build_dataset

# Vision datasets (≈10% of training)
vision_datasets = [
    build_dataset("depth", root_dir="data/hypersim"),
    build_dataset("normal", root_dir="data/hypersim"),
    build_dataset("semantic", root_dir="data/ade20k"),
    build_dataset("instance", root_dir="data/coco"),
    build_dataset("referring", root_dir="data/refcocog"),
    build_dataset("flow", root_dir="data/flyingthings"),
]

# Generation data (≈90% of training)
gen_ds = build_generation_dataset("data/generation/")

# Mixed with multi-colormap augmentation for depth
mixed_ds = MixedVisionDataset(
    vision_datasets=vision_datasets,
    generation_dataset=gen_ds,
    vision_ratio=0.1,
    colormaps=["cube_edge", "plasma", "inferno", "viridis", "grayscale", "rainbow"],
)
```

For the full dataset research report (including all download links, license details, and preprocessing pipelines), see [`DATASET_RESEARCH_REPORT.md`](DATASET_RESEARCH_REPORT.md).

### Distributed Training (DDP / FSDP / DeepSpeed)

```bash
# ─── H20 (96GB) ───
# Single GPU
bash scripts/launch_h20_single.sh --data-path vision_data/ --epochs 5

# 8× H20 DDP
bash scripts/launch_h20_ddp.sh --data-path /shared/data

# 8× H20 FSDP (max throughput, 1024² resolution)
bash scripts/launch_h20_fsdp.sh --data-path /shared/data --gradient-checkpointing

# Multi-node 2×8 H20
# Node 0 (master):
bash scripts/launch_h20_ddp.sh \
    --nodes 2 --node-rank 0 \
    --master-addr 10.0.0.1 \
    --data-path /shared/data
# Node 1 (worker):
bash scripts/launch_h20_ddp.sh \
    --nodes 2 --node-rank 1 \
    --master-addr 10.0.0.1 \
    --data-path /shared/data

# ─── A100-80GB ───
# Unified auto-launcher
bash scripts/launch_training_a100_80g.sh --data-path /shared/data

# 8× A100 DDP
bash scripts/launch_training_a100_80g_ddp.sh \
    --config configs/training_a100_80g.yaml

# 8× A100 FSDP + gradient-checkpointing
bash scripts/launch_training_a100_80g_fsdp.sh --gradient-checkpointing

# ─── A100-40GB (must use DeepSpeed ZeRO-3) ───
# 8× A100 production
bash scripts/train_a100_40g_deepspeed.sh \
    --mode multi \
    --data-path /shared/data \
    --wandb \
    --wandb-run-name "a100_40g_depth_v1"

# Single-card debug (auto CPU offload)
bash scripts/train_a100_40g_deepspeed.sh \
    --mode single \
    --data-path /data/small
```

**Legacy launch scripts** (still available):

```bash
# DDP — single node, 4 GPUs
bash scripts/train_ddp.sh --data-path vision_data/ --epochs 5

# FSDP — single node, 8 GPUs
bash scripts/train_fsdp.sh --nodes 1 --gpus-per-node 8 \
  --data-path vision_data/ --epochs 5 --gradient-checkpointing

# Multi-node FSDP (2 nodes × 8 GPUs)
bash scripts/train_fsdp.sh --nodes 2 --node-rank 0 \
  --master-addr 10.0.0.1 --data-path /shared/vision_data/
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
# List available presets and VRAM impact
python scripts/optimize_inference.py --list-presets

# Use a preset profile (balanced | speed | memory | quality | cpu)
python scripts/optimize_inference.py --profile balanced --run --image data/sample.jpg --task depth

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

```bash
# Core algorithms (CPU)
docker run --rm open-vision-banana:latest \
  python -c "from open_vision_banana.core import DepthRGBMapper; print('ok')"

# GPU inference (requires nvidia-container-toolkit)
docker run --gpus all -v $(pwd)/data:/data open-vision-banana:latest \
  python scripts/inference.py --image /data/input.jpg --task depth
```

Interactive Colab demo: [`notebooks/colab_demo.ipynb`](notebooks/colab_demo.ipynb)

## Citation

If you use this codebase or build upon the Vision Banana paradigm, please cite the original paper:

```bibtex
@article{visionbanana2025,
  title={Image Generators are Generalist Vision Learners},
  author={Shengbang Bang and Hezhen Hu and Zheng Qian and Chengyu Wang and Yi Tai and Yann LeCun and Tong Wang},
  year={2025},
  note={Google Research}
}
```

## License

This project is for research purposes. The base FLUX.1 model follows its own license terms.
