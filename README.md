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
| Metric Depth Estimation | 3D | Power transform + spectral colormap |
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

# Load model
model = VisionBanana.from_pretrained(lora_path="path/to/checkpoint")

# Depth estimation
result = model.predict(image=input_image, task="depth")

# Semantic segmentation
result = model.predict(
    image=input_image,
    task="semantic",
    class_color_map={"cat": (255, 255, 0), "background": (128, 0, 128)},
)

# Normal estimation
result = model.predict(image=input_image, task="normal")

# Batched inference
results = model.predict_batch(images=batch, task="depth")
```

### Training

```python
from open_vision_banana.training import InstructionTuner, VisionTaskDataset
from open_vision_banana.data import build_dataset, MixedVisionDataset

# Create datasets
depth_ds = build_dataset("depth", root_dir="/path/to/hypersim")
normal_ds = build_dataset("normal", root_dir="/path/to/scannet")
mixed_ds = MixedVisionDataset(vision_datasets=[depth_ds, normal_ds])

# Train
tuner = InstructionTuner(model, lora_rank=64, lora_alpha=128)
tuner.train(mixed_ds, epochs=10, batch_size=8)
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

## Project Structure

```
open-vision-banana/
├── src/open_vision_banana/
│   ├── __init__.py              # Lazy imports, version
│   ├── config.py                # Hierarchical config (8 sub-configs)
│   ├── core/                    # Core algorithms (no external deps)
│   │   ├── depth_mapper.py      # Power transform + spectral colormap
│   │   ├── normal_mapper.py     # Linear normal ↔ RGB mapping
│   │   ├── segmentation.py      # 5-stage instance segmentation decoder
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
│   └── training_config.yaml     # Default training configuration
├── scripts/
│   ├── demo.py                  # Algorithm demo (no GPU needed)
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

Then applies a continuous spectral colormap with vectorized parabolic interpolation for sub-segment precision. Parameters: λ=-3, c=10/3.

### 2. Normal → RGB Mapping (Direct Linear)

Camera-space normals (right-handed: +x right, +y up, +z toward camera):

```
R = (1-x)/2,  G = (1+y)/2,  B = (1+z)/2
```

### 3. Instance Segmentation Decoding (5-Stage Clustering)

From Appendix A of the paper:

1. **Background Initialization**: τ=14 color tolerance
2. **Color Similarity Grouping**: Vectorized connected components (~100x faster than Python flood-fill)
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
