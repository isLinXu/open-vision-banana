<div align="center">

# 🍌 Open-Vision-Banana

**Image Generators are Generalist Vision Learners**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-1000+-green.svg)](tests/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.20329-b31b1b.svg)](https://arxiv.org/abs/2604.20329)

*Open-source implementation of the Vision Banana paradigm — achieving SOTA zero-shot transfer on segmentation, depth, and normal estimation through image generation.*

</div>

---

## Table of Contents

- [Overview](#overview)
- [Supported Tasks](#supported-tasks)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Key Algorithms](#key-algorithms)
- [Benchmarks](#benchmarks-from-paper)
- [Evaluation & Export](#evaluation--export)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

---

## Overview

Vision Banana demonstrates that **state-of-the-art visual understanding can emerge from image generation pretraining**. By instruction-tuning diffusion models on a mixture of their original training data alongside vision task data, we obtain models that achieve strong zero-shot transfer on segmentation, depth estimation, and surface normal estimation — all through the lens of image generation.

> **Key insight**: All vision tasks are parameterized as RGB images, enabling seamless integration with generative models while preserving 90%+ generation capability.

**Highlights:**
- 🎨 **Unified RGB paradigm** — All 10 vision tasks encoded as RGB, decoded back to structured outputs
- 🔧 **Multi-model support** — FLUX.1, SANA, Z-Image, Qwen-Image, Lumina2, Chroma via unified registry
- 📊 **10 vision tasks** — Segmentation, depth, normal, edge, flow, keypoint, detection, depth-edge
- ⚡ **Efficient tuning** — rsLoRA trains only 0.5%-1% of parameters
- 🚀 **Complete pipeline** — Training, inference, evaluation, export, and serving

---

## Supported Tasks

| Task | Encoding | Status |
|------|----------|:------:|
| Semantic Segmentation | Color mapping: `{"cat": "red", "background": "yellow"}` | ✅ |
| Instance Segmentation | "Each dog is colored differently" | ✅ |
| Referring Expression Segmentation | "The man in pink t shirt is rendered solid white" | ✅ |
| Metric Depth Estimation | Power transform + 7 colormaps (cube-edge, plasma, inferno, viridis, grayscale, rainbow, turbo) | ✅ |
| Surface Normal Estimation | Direct linear mapping (x,y,z) → (R,G,B) | ✅ |
| Edge Detection | "Edges are white, background is black" | ✅ |
| Optical Flow | HSV encoding: direction=hue, magnitude=saturation | ✅ |
| 2D Keypoint Detection | Colored dots per keypoint type | ✅ |
| Object Detection | Colored bounding box per class | ✅ |
| Depth Edge Detection | Depth discontinuity → white edges | ✅ |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Vision Banana                             │
│                                                                  │
│   ┌──────────────┐    ┌─────────────────────────┐    ┌────────┐ │
│   │ Prompt Builder│───▶│   Diffusion Model       │───▶│  RGB   │ │
│   │              │    │   (FLUX / SANA / ZImage / │    │ Output │ │
│   └──────────────┘    │    Qwen-Image / ...)     │    └────────┘ │
│          │             └─────────────────────────┘        │      │
│          ▼                       │                        ▼      │
│    Depth Decoder           Backbone               Segmentation   │
│                              (Transformer/UNet)      Decoder    │
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │              Instruction Tuning (rsLoRA)                 │  │
│   │   LoRA rank=16~64, alpha=32~128, rsLoRA scaling         │  │
│   │   Data mix: 9:1 vision-to-language                      │  │
│   │   Params trained: 0.5%-1% of total                      │  │
│   └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
# Core only (no GPU dependencies)
pip install -e .

# With inference support (requires GPU)
pip install -e ".[inference]"

# With training support
pip install -e ".[training]"

# Full installation (all features)
pip install -e ".[all]"
```

---

## Quick Start

### Core Algorithms (no GPU required)

```python
from open_vision_banana.core import DepthRGBMapper, NormalRGBMapper, SegmentationDecoder
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
model = VisionBanana.from_pretrained(lora_path="registry:vision-banana")

# Or specify a different base model
model = VisionBanana.from_pretrained(
    lora_path="path/to/checkpoint",
    model_name="sana-1600m",  # or "z-image", "qwen-image", etc.
)

# 16GB GPU: enable CPU offload
model = VisionBanana.from_pretrained(
    lora_path="path/to/checkpoint",
    enable_sequential_cpu_offload=True,
)

# Depth estimation
result = model.predict(image=input_image, task="depth")
```

### Supported Base Models

| Model | Family | Size | License | VRAM |
|-------|--------|------|---------|------|
| FLUX.1-schnell | flux | 12B | Apache-2.0 | ~24GB |
| Z-Image-Turbo | z-image | 6B | Apache-2.0 | ~24GB |
| Qwen-Image | qwen-image | ~6B | Apache-2.0 | ~24GB |
| SANA-600M | sana | 600M | Apache-2.0 | ~12GB |
| SANA-1600M | sana | 1600M | Apache-2.0 | ~16GB |
| Lumina2 | lumina2 | ~3B | Apache-2.0 | ~16GB |
| Chroma | chroma | ~3B | MIT | ~16GB |

### CLI Inference

```bash
# Download checkpoint
python scripts/download_checkpoint.py --alias vision-banana --output-dir checkpoints/

# Inference on 16GB GPU
python scripts/inference.py --image input.jpg --task depth \
  --lora-path registry:vision-banana --cpu-offload --vae-slicing

# SANA multi-task inference
python scripts/infer_sana_512.py \
  --image input.jpg --task multi \
  --lora-path path/to/multitask_lora --steps 20
```

---

## Training

> 📖 **GPU Training Guide**: See [`docs/quick-start.md`](docs/quick-start.md) for H20 / A100 / RTX quick-start with copy-paste commands.

### Basic Training

```bash
# Universal launcher (auto-detects GPU & mode)
bash scripts/unified_launch.sh --config configs/unified_training_config.yaml

# Override mode manually
bash scripts/unified_launch.sh --mode ddp       --config configs/training_config.yaml
bash scripts/unified_launch.sh --mode fsdp      --config configs/training_config.yaml
bash scripts/unified_launch.sh --mode deepspeed --config configs/training_config.yaml
```

### SANA Training

```bash
# SANA-1600M multi-task training
PYTHONPATH=src python scripts/train_sana_512.py \
  --config configs/sana_1600m_3task.yaml

# SANA with SNVC-50M dataset
PYTHONPATH=src python scripts/train_sana_512.py \
  --config configs/sana_snvc50m_3task.yaml
```

### Hardware-specific Launch

| GPU | VRAM | Launcher |
|-----|------|----------|
| H20 | 96GB | `bash scripts/launch_h20_single.sh` |
| H20 ×8 | 96GB×8 | `bash scripts/launch_h20_ddp.sh` |
| A100-80GB | 80GB | `bash scripts/launch_training_a100_80g.sh` |
| A100-40GB | 40GB | `bash scripts/train_a100_40g_deepspeed.sh` |
| RTX 4090 | 24GB | `bash scripts/unified_launch.sh --mode deepspeed` |
| MPS (Apple) | Unified | `bash scripts/unified_launch.sh --mode mps` |

### Data Preparation

```bash
# List all supported datasets
python scripts/download_data.py --list

# Download and prepare a dataset
python scripts/download_data.py --dataset hypersim-depth --output data/hypersim/

# Validate before training
python scripts/validate_dataset.py --data-path data/hypersim_depth/ --task depth
```

### Inference Optimization

| Preset | Target VRAM | Speed | Quality | Key Options |
|--------|-------------|:-----:|:-------:|-------------|
| `balanced` | ~16 GB | ★★★ | ★★★★ | float16 + model_cpu_offload + VAE slicing |
| `speed` | ~20 GB | ★★★★★ | ★★★ | float16 + torch.compile + VAE tiling |
| `memory` | ~8 GB | ★★ | ★★★ | float16 + sequential offload + INT8 mappers |
| `quality` | ~40 GB | ★★★ | ★★★★★ | bfloat16 + max-autotune compile |
| `cpu` | RAM only | ★ | ★★★ | float32 + sequential offload |

---

## Key Algorithms

### 1. Depth → RGB Mapping

Maps unbounded depth [0, ∞) to bounded RGB [0, 1]³ via **power transform**:

```
f(d, λ, c) = 1 - (1 - d/(λc))^(λ+1)
```

Then maps via the **first-order 3D Hilbert path** (black → red → yellow → green → cyan → blue → magenta → white) with piecewise-linear interpolation.

**Parameters**: λ=-3, c=10/3

### 2. Normal → RGB Mapping

Camera-space normals (right-handed: +x right, +y up, +z toward camera):

```
R = (1-x)/2,  G = (1+y)/2,  B = (1+z)/2
```

### 3. Instance Segmentation Decoding

5-stage clustering from Appendix A of the paper:

1. **Background Initialization** — τ=14 color tolerance
2. **Color Similarity Grouping** — Union-Find with 16-connectivity (O(E α(N)), ~100× faster than flood-fill)
3. **Noise Pruning** — θ_size = 2×10⁻⁴
4. **Boundary Artifact Elimination** — 3×3 erosion, θ_erosion=0.1
5. **Spatially Constrained Merging** — γ=5.0

### 4. Instruction Tuning (rsLoRA)

- LoRA rank=16~64, alpha=32~128 with rsLoRA scaling (α/√r)
- 9:1 visual-to-language data mixing ratio
- Only 0.5%-1% of total parameters trained
- Preserves 90%+ generation capability
- Supports 18+ model families via unified registry

---

## Benchmarks (from paper)

| Task | Benchmark | Metric | Vision Banana | Previous SOTA |
|------|-----------|:------:|:-------------:|:-------------:|
| Semantic Seg | Cityscapes | mIoU | **69.9** | 65.2 (SAM 3) |
| Instance Seg | SA-Co/Gold | cgF1 | **47.5** | 24.6 (OWLv2) |
| Referring Seg | RefCOCOg | cIoU | **73.8** | 67.7 (LocHeads) |
| Referring Seg | ReasonSeg | gIoU | **79.3** | 77.0 (SAM3+GPT) |
| Depth | 6-dataset avg | δ1↑ | **0.882** | 0.823 (UniK3D) |
| Normal | Indoor avg | mean°↓ | **15.05** | 15.72 (Lotus-2) |

---

## Evaluation & Export

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

```bash
# CLI with SOTA comparison
python scripts/evaluate.py --task depth --data-path /data/hypersim --compare-sota
python scripts/evaluate.py --task semantic --data-path /data/cityscapes --output-json metrics.json
```

### Export

```python
from open_vision_banana.export import export_mapper_onnx

# Export depth/normal mappers to ONNX
export_mapper_onnx("depth_to_rgb", "depth_mapper.onnx")
export_mapper_onnx("normal_to_rgb", "normal_mapper.onnx")
```

---

## Docker

```bash
# CPU image (core algorithms, inference)
docker build -t open-vision-banana:latest -f docker/Dockerfile .

# GPU image (training + CUDA inference)
docker build -t open-vision-banana:gpu -f docker/Dockerfile.gpu .

# Docker Compose
docker compose -f docker/docker-compose.yml up dev      # editable CPU dev
docker compose -f docker/docker-compose.yml up training # GPU FSDP training
docker compose -f docker/docker-compose.yml up notebook # Jupyter on :8888
```

Interactive Colab demo: [`notebooks/colab_demo.ipynb`](notebooks/colab_demo.ipynb)

---

## Project Structure

```
open-vision-banana/
├── src/open_vision_banana/
│   ├── core/                    # Core algorithms (no external deps)
│   │   ├── depth_mapper.py      # Power transform + 7 colormaps
│   │   ├── adaptive_depth_mapper.py
│   │   ├── normal_mapper.py     # Linear normal ↔ RGB mapping
│   │   ├── segmentation/        # 5-stage instance segmentation decoder
│   │   ├── edge_mapper.py       # Edge detection
│   │   ├── flow_mapper.py       # Optical flow HSV encoding
│   │   ├── keypoint_mapper.py   # 2D keypoint detection
│   │   ├── object_detection_mapper.py
│   │   ├── depth_edge_mapper.py
│   │   ├── union_find.py        # Union-Find (path compression)
│   │   └── rgb_mapper.py        # Unified encode/decode interface
│   ├── inference/               # Model loading and generation
│   │   ├── generator.py         # Multi-model + LoRA wrapper
│   │   ├── pipeline.py          # VisionBanana unified model
│   │   ├── prompt.py            # Task-specific prompt builder
│   │   ├── decoder.py           # Visualization decoder
│   │   └── optimization.py      # Inference optimization presets
│   ├── training/                # LoRA instruction tuning
│   │   ├── tuner.py             # InstructionTuner + VisionTaskDataset
│   │   ├── trainer_interface.py # 5 backends (Native/DDP/FSDP/DeepSpeed/Accelerate)
│   │   └── ...
│   ├── evaluation/              # Metrics for all vision tasks
│   ├── export/                  # ONNX + TorchScript export
│   └── data/                    # Dataset loaders
├── configs/                     # Training & experiment configs
├── scripts/                     # Training, inference, evaluation scripts
├── tests/                       # 1000+ tests across 79 files
├── docker/                      # CPU/GPU Docker images
└── notebooks/                   # Colab demo
```

---

## Citation

If you use this codebase or build upon the Vision Banana paradigm, please cite the original paper:

```bibtex
@article{visionbanana2026,
  title={Image Generators are Generalist Vision Learners},
  author={Gabeur, Valentin and others},
  year={2026},
  eprint={2604.20329},
  archivePrefix={arXiv}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The base models (FLUX.1, SANA, etc.) follow their own respective license terms.
