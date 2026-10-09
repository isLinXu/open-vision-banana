# Open-Vision-Banana — Agent Operations Guide

> This document is structured for AI agents to quickly understand and operate the project.
> It covers training, inference, evaluation workflows, and the relationship to the original paper.

---

## 1. Project Overview & Paper Relationship

**Paper**: "Image Generators are Generalist Vision Learners" (arXiv: 2604.20329)
**Authors**: Valentin Gabeur et al.
**PDF**: `paper/Image Generators are Generalist Vision Learners.pdf`

### Core Thesis
The paper demonstrates that **instruction-tuning a pre-trained image generator** on a mixture of its original training data + vision task data (parameterized as RGB images) yields a model that, per the paper, achieves SOTA zero-shot transfer on segmentation, depth, and normal estimation while retaining its generation capability (human win rates 53.5% GenAI-Bench / 47.8% ImgEdit vs NBP; the "90%+" figure is a project shorthand, not a number from the paper).

### Disclaimer: Unofficial Implementation
**This is an unofficial, community-driven open-source reimplementation.** It is NOT affiliated with, endorsed by, or verified by the original paper authors. The paper's official code (if any) is separate from this repository.

### This Repository's Relationship to the Paper
- **Unofficial open-source reimplementation** of the Vision Banana paradigm described in the paper
- **Implements** the core RGB parameterization algorithms for vision tasks (depth, normal, segmentation, etc.)
- **Extends** the paper's approach to multiple backbone families beyond FLUX.1 (SANA, Z-Image, Qwen-Image, etc.) — *most are registered in code but not all have been end-to-end verified*
- **Provides** training, inference, and evaluation pipelines — *the unified training framework was recently built and is undergoing verification*
- **Does NOT claim** to reproduce the paper's exact benchmark numbers; no released checkpoints matching paper SOTA yet

### Implementation Status
| Paper Concept | Status | Location |
|---|---|---|
| RGB parameterization of vision tasks | ✅ Implemented | `src/open_vision_banana/core/` |
| Depth power transform + Hilbert colormap | ✅ Implemented | `src/open_vision_banana/core/depth_mapper.py` |
| Normal linear mapping | ✅ Implemented | `src/open_vision_banana/core/normal_mapper.py` |
| 5-stage instance segmentation decoding | ✅ Implemented | `src/open_vision_banana/core/segmentation/` |
| Union-Find color grouping | ✅ Implemented | `src/open_vision_banana/core/union_find.py` |
| rsLoRA instruction tuning | ✅ Implemented | `src/open_vision_banana/training/` |
| Flow matching (linear_flow + rectified-flow) | ✅ Implemented | `src/open_vision_banana/training/flow_matching.py` |
| SANA-1600M multi-task training | ✅ Verified (legacy + unified) | `scripts/train_sana_512.py`, `scripts/train.py` |
| FLUX.1-dev (12B) quantized training | ✅ Verified (legacy + unified) | `scripts/train_flux.py`, `scripts/train.py` |
| 9:1 vision-to-language data mix | ⚠️ Code exists, not fully verified | `src/open_vision_banana/training/trainer.py` |
| Paper benchmark reproduction | ❌ Not yet achieved | — |
| Released checkpoints matching SOTA | ❌ Not available | — |

### What Is NOT Done (Honest Boundaries)
- **No paper-matching benchmark results**: The paper reports SOTA on Cityscapes (mIoU 69.9), SA-Co/Gold (cgF1 47.5), etc. This repo has NOT reproduced these numbers.
- **Not all 10 vision tasks are equally verified**: Depth, normal, and semantic segmentation have the most testing. Tasks like optical flow, keypoint detection, and object detection have mapper code but limited end-to-end training verification.
- **Not all registered models are verified**: FLUX and SANA have been trained. Z-Image, Qwen-Image, Lumina2, Chroma are registered in `model_registry.py` but have NOT been trained end-to-end. Instruction editors (`qwen-image-edit`, `qwen-image-edit-2509`, `longcat-image-edit`, `flux2-klein`, `joyimage-edit`, `joyimage-edit-plus`, `instruct-pix2pix`, `sdxl-instruct-pix2pix`, `hunyuan-image3-instruct`, `hunyuan-image3-instruct-distil`, and `flux-kontext`) have a training forward. The diffusers editors have been executed on the official transformer and UNet classes built at tiny width. HunyuanImage-3.0-Instruct builds its token batch with `prepare_model_inputs` and calls the published forward; a full fine-tune of that 80B checkpoint has not been run. Qwen and Joy still receive the text embeddings the trainer already encoded; `encode_text` does not insert their vision-language image tokens. `hunyuan-image` remains the HunyuanImage 2.1 text-to-image pipeline.
- **Export/serving modules**: Code exists in `src/open_vision_banana/export/` and `src/open_vision_banana/serving/` but integration with the new unified training framework has not been verified.
- **Generation preservation**: The paper claims 90%+ generation capability is preserved. This has NOT been measured in this repo.

---

## 2. Environment Setup

```bash
# Conda environment
conda activate llama  # Python 3.10+, torch 2.7.1+cu128

# Install project
cd /media/linxu/thinkHub/Thinking-face/open-vision-banana
pip install -e ".[all]"

# Required env vars
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_API_KEY=<your-key>  # Only for W&B logging
```

### Key Paths
| Item | Path |
|---|---|
| Project root | `/media/linxu/thinkHub/Thinking-face/open-vision-banana` |
| Source code | `src/open_vision_banana/` |
| Training configs | `configs/` |
| Training output | `output/<run_name>/` |
| Scripts | `scripts/` |
| Tests | `tests/` |
| Paper PDF | `paper/Image Generators are Generalist Vision Learners.pdf` |

---

## 3. Training

### 3.1 Unified Training Framework (Recommended)

The project provides a **unified multi-model training framework** via `scripts/train.py`.
This replaces the legacy model-specific scripts (`train_sana_512.py`, `train_flux.py`).

#### Entry Point
```bash
PYTHONPATH=src python scripts/train.py --config <config.yaml> [--wandb] [--steps N] [--lr LR]
```

#### Architecture
```
scripts/train.py
  → TrainingConfig (config.py)     # YAML + CLI overrides
  → Trainer (trainer.py)           # Core training loop
    → ModelBackend (model_backend.py)  # Model abstraction
    → TrainingLosses (losses.py)       # Loss computation
    → TrainingCallbacks (callbacks.py) # Logging + checkpointing
    → FlowScheduler (flow_matching.py) # Flow matching
```

#### Module Responsibilities
| Module | File | Responsibility |
|---|---|---|
| `ModelBackend` | `training/model_backend.py` | Wraps model_registry: quantization, LoRA, latent packing, forward pass |
| `FlowScheduler` | `training/flow_matching.py` | SANA: linear_flow + flow_shift; FLUX/SD3: rectified-flow |
| `TrainingConfig` | `training/config.py` | Dataclass config from YAML, CLI override support |
| `TrainingLosses` | `training/losses.py` | Flow MSE + Min-SNR weighting + depth/normal aux losses |
| `TrainingCallbacks` | `training/callbacks.py` | W&B sync, checkpoint save, best-model tracking |
| `Trainer` | `training/trainer.py` | Gradient accumulation, EMA, validation loop |

#### SANA Training Example
```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/sana_1600m_3task_full_clean.yaml \
  --wandb
```

Key config fields for SANA:
- `model_name: "sana-1600m"` → uses SanaTransformer backbone + DC-AE VAE
- `flow_type: "linear_flow"` + `flow_shift: 3.0`
- `resolution: 512`, `batch_size: 4`, `gradient_accumulation: 4` (YAML field name; CLI: `--grad-accum`)
- Tasks: depth + normal + semantic (3-task mixture)

#### FLUX Training Example
```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/flux_3task_full_clean.yaml \
  --wandb
```

Key config fields for FLUX:
- `model_name: "flux-dev"` → 12B transformer with 4-bit NF4 quantization
- `quantize: "4bit"` → fits in 24GB VRAM
- `flow_type: "rectified_flow"` (no flow_shift)
- Latent packing: 2×2 patches → `[B, (H/2)*(W/2), C*4]`
- Guidance: FLUX uses `guidance` kwarg in forward pass

#### CLI Override Examples
```bash
# Override steps, lr, output dir
PYTHONPATH=src python scripts/train.py \
  --config configs/sana_1600m_3task_full_clean.yaml \
  --steps 10000 --lr 5e-5 --output-dir output/custom_run

# Override batch size and resolution
PYTHONPATH=src python scripts/train.py \
  --config configs/flux_3task_full_clean.yaml \
  --batch-size 2 --resolution 256
```

### 3.2 Legacy Training Scripts

Preserved for backward compatibility. NOT recommended for new experiments.

```bash
# SANA legacy
PYTHONPATH=src python scripts/train_sana_512.py \
  --config configs/sana_1600m_3task.yaml

# FLUX legacy
PYTHONPATH=src python scripts/train_flux.py \
  --config configs/flux_3task_full_clean.yaml --wandb
```

### 3.3 Hardware-Specific Launch

| GPU | VRAM | Command |
|---|---|---|
| H20 | 96GB | `bash scripts/launch_h20_single.sh` |
| H20 ×8 | 96GB×8 | `bash scripts/launch_h20_ddp.sh` |
| A100-80GB | 80GB | `bash scripts/launch_training_a100_80g.sh` |
| A100-40GB | 40GB | `bash scripts/train_a100_40g_deepspeed.sh` |
| RTX 4090 | 24GB | `PYTHONPATH=src python scripts/train.py --config <yaml>` |
| MPS (Apple) | Unified | `PYTHONPATH=src python scripts/train.py --config <yaml>` |

### 3.4 Training Output Structure

Expected layout (created by `TrainingCallbacks.on_checkpoint`):

```
output/<run_name>/
├── train.log                    # Training log
├── checkpoint-step-{N}/         # Periodic checkpoints
│   ├── lora.pt                  # LoRA weights + optimizer state + metadata
│   ├── lora_adapter/            # PEFT-format adapter (save_pretrained)
│   └── metadata.json            # step, val_loss, lora_rank/alpha, model_name
├── checkpoint-step-{N}_best/    # Best model (by validation loss)
│   ├── lora.pt
│   ├── lora_adapter/
│   └── metadata.json
└── wandb/                       # W&B run directory (if --wandb)
```

### 3.5 Available Config Files

| Config | Model | Tasks | Notes |
|---|---|---|---|
| `sana_1600m_3task_full_clean.yaml` | SANA-1600M | depth+normal+seg | Clean 3-task |
| `sana_1600m_5task.yaml` | SANA-1600M | depth+normal+seg | SNVC-50M dataset (named "5task" historically, 3 tasks active) |
| `flux_3task_full_clean.yaml` | FLUX.1-dev (12B) | depth+normal+seg | Quantized 4-bit |
| `sana_snvc50m_3task.yaml` | SANA-1600M | 3 tasks | SNVC-50M dataset |
| `training_mps.yaml` | SANA | depth | MPS (Apple Silicon) |
| `training_mps_multitask.yaml` | SANA | multi | MPS multi-task |

---

## 4. Inference

### 4.1 Python API

```python
from open_vision_banana.inference.pipeline import VisionBanana

# Load model
model = VisionBanana.from_pretrained(
    model_name="black-forest-labs/FLUX.1-schnell",  # or "sana-1600m"
    lora_path="path/to/lora/checkpoint",
    device="cuda",
)

# Single task inference
result = model.predict(image=input_tensor, task="depth")
# result keys: "rgb_output", "depth", "visualization", etc.

result = model.predict(image=input_tensor, task="normal")
result = model.predict(image=input_tensor, task="semantic", class_color_map=color_map)
result = model.predict(image=input_tensor, task="referring", expression="the red car")
```

### 4.2 CLI Inference

```bash
# Single image
PYTHONPATH=src python scripts/inference.py \
  --image input.jpg --task depth \
  --lora-path path/to/lora --output-dir output/

# Batch inference
PYTHONPATH=src python scripts/inference.py \
  --image-dir /path/to/images --task normal \
  --lora-path path/to/lora

# With optimizations for 16GB GPU
PYTHONPATH=src python scripts/inference.py \
  --image input.jpg --task depth \
  --lora-path registry:vision-banana \
  --cpu-offload --vae-slicing

# SANA multi-task visualization on a manifest's val split (input / target / prediction)
PYTHONPATH=src python scripts/infer_sana_512.py \
  --data-dir /path/to/dataset --pair-manifest /path/to/pairs.jsonl \
  --checkpoint path/to/checkpoint/lora.pt \
  --model-name sana-1600m --lora-rank 32 --lora-alpha 64 \
  --num-inference-steps 20 --output-dir output/infer_sana
```

### 4.3 Supported Tasks for Inference

| Task | CLI `--task` | Required Args | Output Keys | E2E Verified |
|---|---|---|---|---|
| Depth | `depth` | — | `rgb_output`, `depth` | ✅ |
| Normal | `normal` | — | `rgb_output`, `normal` | ✅ |
| Semantic Seg | `semantic` | `--class-map` (optional) | `rgb_output`, `mask` | ✅ |
| Instance Seg | `instance` | `--object-name` | `rgb_output`, `instances` | ⚠️ Code exists, limited testing |
| Referring Seg | `referring` | `--expression` (required) | `rgb_output`, `mask` | ⚠️ Code exists, limited testing |
| Edge | `edge` | — | `rgb_output`, `edge_mask` | ⚠️ Mapper code, not training-verified |
| Optical Flow | `flow` | — | `rgb_output`, `flow` | ⚠️ Mapper code, not training-verified |
| Keypoint | `keypoint` | — | `rgb_output`, `keypoints` | ⚠️ Mapper code, not training-verified |
| Object Detection | `object_detection` | — | `rgb_output`, `bboxes` | ⚠️ Mapper code, not training-verified |
| Depth Edge | `depth_edge` | — | `rgb_output`, `edge_mask` | ⚠️ Mapper code, not training-verified |

### 4.4 Inference Optimization Presets

| Preset | VRAM | Speed | Quality | Key Options |
|---|---|---|---|---|
| `balanced` | ~16 GB | ★★★ | ★★★★ | float16 + model_cpu_offload + VAE slicing |
| `speed` | ~20 GB | ★★★★★ | ★★★ | float16 + torch.compile + VAE tiling |
| `memory` | ~8 GB | ★★ | ★★★ | float16 + sequential offload + INT8 mappers |
| `quality` | ~40 GB | ★★★ | ★★★★★ | bfloat16 + max-autotune compile |
| `cpu` | RAM only | ★ | ★★★ | float32 + sequential offload |

---

## 5. Evaluation

### 5.1 CLI Evaluation

```bash
# Evaluate depth on a dataset
PYTHONPATH=src python scripts/evaluate.py \
  --task depth --data-path /path/to/eval/data \
  --lora-path output/lora_checkpoint \
  --output-json metrics.json

# Compare against paper SOTA
PYTHONPATH=src python scripts/evaluate.py \
  --task depth --data-path /path/to/data \
  --compare-sota

# Semantic segmentation with class count
PYTHONPATH=src python scripts/evaluate.py \
  --task semantic --num-classes 19 \
  --data-path /path/to/cityscapes

# Distributed evaluation (multi-GPU)
PYTHONPATH=src torchrun --nproc_per_node=4 scripts/evaluate.py \
  --task depth --data-path /path/to/data --ddp

# Quick smoke test (limit batches)
PYTHONPATH=src python scripts/evaluate.py \
  --task normal --data-path /path/to/data \
  --max-batches 5
```

### 5.2 Python Evaluation API

```python
from open_vision_banana.evaluation.evaluator import VisionEvaluator
from open_vision_banana.inference.pipeline import VisionBanana

# Load model
model = VisionBanana.from_pretrained(lora_path="path/to/lora")

# Create evaluator
evaluator = VisionEvaluator(model, device="cuda")

# Evaluate on a dataloader
metrics = evaluator.evaluate(
    dataloader=eval_dataloader,
    task="depth",           # or "normal", "semantic", "instance", etc.
    num_classes=19,         # for semantic segmentation
    max_batches=None,       # None = full dataset
)

# Metrics by task:
# depth:     AbsRel, SqRel, RMSE, logRMSE, δ1, δ2, δ3
# normal:    Mean Angular Error (°), Cosine Similarity, RMSE
# semantic:  mIoU, pixel accuracy
# instance:  cgF1, cIoU
# referring: cIoU, gIoU
# edge:      F1, precision, recall, ODS
```

### 5.3 Evaluation Task → Dataset Mapping

| `--task` | Dataset Key | Expected Data Format |
|---|---|---|
| `depth` | `hypersim-depth` | RGB + depth maps |
| `normal` | `normal` | RGB + normal maps |
| `semantic` | `cityscapes` | RGB + segmentation masks |
| `instance` | `sa-co` | RGB + instance annotations |
| `referring` | `refcocog` | RGB + expression + mask |
| `edge` | `coco-stuff` | RGB + edge masks |
| `flow` | `sintel` | RGB + flow fields |
| `keypoint` | `coco-detection` | RGB + keypoint annotations |
| `object_detection` | `coco-detection` | RGB + bbox annotations |
| `depth_edge` | `depth` | RGB + depth maps |
| `generation_preservation` | `coco` | RGB + generation metrics |

---

## 6. Key Source Code Architecture

```
src/open_vision_banana/
├── core/                        # Paper algorithms (no external deps)
│   ├── depth_mapper.py          # Power transform + 7 colormaps
│   ├── normal_mapper.py         # Linear (x,y,z) → (R,G,B)
│   ├── segmentation/            # 5-stage instance decoder
│   ├── union_find.py            # Union-Find with path compression
│   ├── rgb_mapper.py            # Unified encode/decode interface
│   └── ...                      # edge, flow, keypoint, object_detection mappers
│
├── inference/                   # Model loading + generation
│   ├── pipeline.py              # VisionBanana (main model class)
│   ├── generator.py             # BaseGenerator (multi-model + LoRA)
│   ├── prompt.py                # PromptBuilder (task-specific prompts)
│   ├── decoder.py               # VisualizationDecoder
│   └── optimization.py          # InferenceOptions presets
│
├── training/                    # Unified training framework
│   ├── model_backend.py         # ModelBackend: quantization, LoRA, packing, forward
│   ├── flow_matching.py         # FlowScheduler: linear_flow + rectified_flow
│   ├── config.py                # TrainingConfig dataclass + YAML parsing
│   ├── losses.py                # TrainingLosses: flow MSE + Min-SNR + aux
│   ├── callbacks.py             # TrainingCallbacks: W&B + checkpoint
│   ├── trainer.py               # Trainer: main training loop
│   ├── tuner.py                 # Legacy InstructionTuner
│   ├── trainer_interface.py     # 5 distributed backends
│   ├── sana_depth_aux.py        # Depth edge endpoint loss
│   ├── normal_aux_loss.py       # Normal angular loss
│   └── sana_flow.py             # Min-SNR weight computation
│
├── evaluation/                  # Metrics for all 10 tasks
│   ├── evaluator.py             # VisionEvaluator (unified runner)
│   ├── depth.py, normal.py, semantic.py, instance.py, ...
│   └── baselines.py             # SOTA comparison tables
│
├── model_registry.py            # Central registry: load_pipeline, apply_lora, etc.
├── config.py                    # VisionBananaConfig
└── task_prompts.py              # Task prompt templates
```

---

## 7. Model Registry

The `model_registry.py` is the central abstraction for multi-model support.

### Supported Models
| Alias | HF Path | Family | Size | Training Verified |
|---|---|---|---|---|
| `flux-dev` | `black-forest-labs/FLUX.1-dev` | flux | 12B | ✅ (3-task, 4-bit quant) |
| `flux-schnell` | `black-forest-labs/FLUX.1-schnell` | flux | 12B | ⚠️ Registered, not trained |
| `sana-1600m` | `Efficient-Large-Model/Sana_1600M_512px_diffusers` | sana | 1.6B | ✅ (3-task, 5-task) |
| `sana-600m` | `Efficient-Large-Model/SANA_600M_1024px_diffusers` | sana | 600M | ⚠️ Registered, not trained |
| `z-image` | `Tongyi-MAI/Z-Image` | z-image | 6B | ⚠️ Registered, not trained |
| `qwen-image` | `Qwen/Qwen-Image` | qwen-image | ~6B | ⚠️ Registered, not trained |
| `lumina2` | `Alpha-VLLM/Lumina-Image-2.0` | lumina2 | ~3B | ⚠️ Registered, not trained |
| `chroma` | `lodestones/Chroma1-HD` | chroma | ~3B | ⚠️ Registered, not trained |

### Key Registry Functions
```python
from open_vision_banana.model_registry import (
    load_pipeline,      # Load diffusers pipeline
    apply_lora,         # Apply LoRA adapters
    encode_text,        # Text encoding
    vae_encode,         # Image → latent
    vae_decode,         # Latent → image
    backbone_forward,   # Model-specific forward pass
    get_model_config,   # Get ModelConfig for a model
)
```

---

## 8. Data Preparation

```bash
# List all supported datasets
python scripts/download_data.py --list

# Download and prepare
python scripts/download_data.py --dataset hypersim-depth --output data/hypersim/

# Validate before training
python scripts/validate_dataset.py --data-path data/hypersim_depth/ --task depth

# Build manifest datasets (for training)
python scripts/build_v3_dataset.py          # Build v3 multi-task dataset
python scripts/build_snvc50m_manifest.py    # Build SNVC-50M manifest
```

---

## 9. Testing

```bash
# Run all tests (tests/ is a flat directory — no subdirectories)
pytest tests/

# Run tests by name pattern
pytest tests/ -k "core"           # Core algorithm tests
pytest tests/ -k "training"       # Training infrastructure tests
pytest tests/ -k "inference"      # Inference tests
pytest tests/ -k "evaluation"     # Evaluation tests
pytest tests/ -k "model_registry" # Model registry tests

# Run with coverage
pytest tests/ --cov=src/open_vision_banana --cov-report=term-missing
```

---

## 10. Git Conventions

- **Commit format**: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, etc.)
- **Branch**: `main`
- **Lint**: `ruff check --fix .` then `ruff format .`
- **Pre-commit**: `.pre-commit-config.yaml` is configured

---

## 11. Quick Reference Commands

```bash
# === TRAINING ===
# SANA 3-task
PYTHONPATH=src python scripts/train.py --config configs/sana_1600m_3task_full_clean.yaml --wandb

# FLUX 3-task
PYTHONPATH=src python scripts/train.py --config configs/flux_3task_full_clean.yaml --wandb

# === INFERENCE ===
# Single image depth
PYTHONPATH=src python scripts/inference.py --image input.jpg --task depth --lora-path output/lora

# === EVALUATION ===
# Depth evaluation with SOTA comparison
PYTHONPATH=src python scripts/evaluate.py --task depth --data-path /data/eval --compare-sota

# === TESTING ===
pytest tests/ -v

# === LINT ===
ruff check --fix src/ tests/
ruff format src/ tests/
```

