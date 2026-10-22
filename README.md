<div align="center">

# 🍌 Open-Vision-Banana

**Image Generators are Generalist Vision Learners**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-1000+-green.svg)](tests/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.20329-b31b1b.svg)](https://arxiv.org/abs/2604.20329)

*Unofficial open-source implementation of the Vision Banana paradigm (RGB-parameterized vision tasks via image generation). The paper reports SOTA zero-shot transfer; this repo has not yet reproduced those results — see [Status](#benchmarks-from-paper).*

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
- [Checkpoints](#checkpoints)
- [Paper Alignment Statement](#paper-alignment-statement)
- [Benchmarks](#benchmarks-from-paper)
- [Evaluation & Export](#evaluation--export)
- [Project Structure](#project-structure)
- [Roadmap & TODO](#roadmap--todo)
- [Citation](#citation)
- [License](#license)

---

## Overview

Vision Banana demonstrates that **state-of-the-art visual understanding can emerge from image generation pretraining**. By instruction-tuning diffusion models on a mixture of their original training data alongside vision task data, we obtain models that achieve strong zero-shot transfer on segmentation, depth estimation, and surface normal estimation — all through the lens of image generation.

> **Key insight**: All vision tasks are parameterized as RGB images, enabling seamless integration with generative models while (per the paper) retaining the base model's generation capability. Generation retention has not been measured in this repo.

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

### Instruction-editing models

These checkpoints take a source image plus an instruction. Each one has a training forward that follows that model's official conditioning. The diffusers classes have been executed at tiny width. A full fine-tune of those Hub checkpoints has not been run, except the in-progress FLUX.1 Kontext depth run.

| Name | Pipeline | Notes |
|------|----------|--------|
| `flux-kontext` | `FluxKontextPipeline` | In-context tokens. Non-commercial. |
| `qwen-image-edit`, `qwen-image-edit-2509` | Qwen edit / edit-plus | Packed 2×2 latents. Text encoding does not insert the VL image tokens. |
| `longcat-image-edit` | `LongCatImageEditPipeline` | Packed tokens, modality 2 for the source. |
| `flux2-klein` | `Flux2KleinPipeline` | Flattened reference tokens. |
| `joyimage-edit`, `joyimage-edit-plus` | Joy image slots | Text encoding does not insert VL image tokens. |
| `instruct-pix2pix`, `sdxl-instruct-pix2pix` | InstructPix2Pix | Source channels concatenated. `image_guidance_scale` stays 1.5. |
| `hunyuan-image3-instruct`, `hunyuan-image3-instruct-distil` | `generate_image` | 80B MoE. The checkpoint directory name cannot contain `.`. Training calls `prepare_model_inputs`. |

`hunyuan-image` stays the HunyuanImage 2.1 text-to-image pipeline.

### CLI Inference

```bash
# Download checkpoint
python scripts/download_checkpoint.py --alias vision-banana --output-dir checkpoints/

# Inference on 16GB GPU
python scripts/inference.py --image input.jpg --task depth \
  --lora-path registry:vision-banana --cpu-offload --vae-slicing

# SANA multi-task visualization on a manifest's val split (input / target / prediction)
python scripts/infer_sana_512.py \
  --data-dir /path/to/dataset --pair-manifest /path/to/pairs.jsonl \
  --checkpoint path/to/checkpoint/lora.pt \
  --model-name sana-1600m --lora-rank 32 --lora-alpha 64 \
  --num-inference-steps 20 --output-dir output/infer_sana
```

---

## Training

> 📖 **GPU Training Guide**: See [`docs/quick-start.md`](docs/quick-start.md) for H20 / A100 / RTX quick-start with copy-paste commands.

For supervised referring expression segmentation, see the [RES data and evaluation guide](docs/res_data.md): validated RefCOCO-family data, a versioned binary mask contract, a pinned Hugging Face release, and native-resolution evaluation.

### Unified Training Framework

The project provides a **unified multi-model training framework** via `scripts/train.py`,
supporting all registered models (SANA, FLUX, SD3, etc.) and multi-task training (depth, normal, segmentation, etc.).

```bash
# SANA-1600M multi-task training
PYTHONPATH=src python scripts/train.py \
  --config configs/sana_1600m_3task_full_clean.yaml --wandb

# FLUX.1-dev multi-task training
PYTHONPATH=src python scripts/train.py \
  --config configs/flux_3task_full_clean.yaml --wandb

# Override any config parameter via CLI
PYTHONPATH=src python scripts/train.py \
  --config configs/sana_1600m_3task_full_clean.yaml \
  --steps 10000 --lr 5e-5 --output-dir output/custom_run
```

The framework consists of modular components:

| Module | File | Description |
|--------|------|-------------|
| `ModelBackend` | `training/model_backend.py` | Model abstraction (quantization, LoRA, packing, forward) |
| `FlowScheduler` | `training/flow_matching.py` | Unified flow matching (SANA linear_flow + rectified-flow) |
| `TrainingConfig` | `training/config.py` | YAML config + CLI overrides via dataclass |
| `TrainingLosses` | `training/losses.py` | Flow MSE + Min-SNR + task weighting + aux losses |
| `TrainingCallbacks` | `training/callbacks.py` | W&B logging + checkpoint + validation tracking |
| `Trainer` | `training/trainer.py` | Core training loop assembling all modules |

### Legacy Training Scripts

Model-specific scripts are preserved for reference and backward compatibility:

```bash
# SANA training (legacy)
PYTHONPATH=src python scripts/train_sana_512.py \
  --config configs/sana_1600m_3task.yaml

# FLUX training (legacy)
PYTHONPATH=src python scripts/train_flux.py \
  --config configs/flux_3task_full_clean.yaml --wandb
```

### Hardware-specific Launch

| GPU | VRAM | Launcher |
|-----|------|----------|
| H20 | 96GB | `bash scripts/launch_h20_single.sh` |
| H20 ×8 | 96GB×8 | `bash scripts/launch_h20_ddp.sh` |
| A100-80GB | 80GB | `bash scripts/launch_training_a100_80g.sh` |
| A100-40GB | 40GB | `bash scripts/train_a100_40g_deepspeed.sh` |
| RTX 4090 | 24GB | `PYTHONPATH=src python scripts/train.py --config <yaml>` |
| MPS (Apple) | Unified | `PYTHONPATH=src python scripts/train.py --config <yaml>` |

### Data Preparation

One-command fetch of every public source in the paper-aligned recipe (registry:
`configs/datasets/sources.tsv`; details and the download → prepare → train path in
[`docs/DATASETS.md`](docs/DATASETS.md)):

```bash
bash scripts/download_datasets.sh --list              # registered sources, grouped train / eval / replay
bash scripts/download_datasets.sh --group train       # Hypersim, TartanAir, ADE20K, COCO+LVIS, RefCOCO(+), ...
bash scripts/download_datasets.sh --only omniedit     # generation replay shards
```

Legacy per-dataset helpers:

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
- Aims to preserve generation capability (not yet measured here; the paper reports human win rates of 53.5% / 47.8% vs the base model)
- Supports 18+ model families via unified registry

---

## Checkpoints

Published LoRAs are at [`khazic/ovb-*`](https://huggingface.co/khazic?search=ovb-). The repo id is `khazic/ovb-<run>`.

| Name pattern | Base | What to load |
|--------------|------|----------------|
| `ovb-fix_*`, `ovb-best3_*`, `ovb-ctrl_*`, `ovb-main_*` | SANA-1600M | `checkpoint-*_final` (prefer `ema_adapter` when that directory exists) |
| `ovb-flux*`, `ovb-flux2*`, `ovb-flux3*`, `ovb-flux4*` | FLUX.1-dev | same layout; `metadata.json` records `model_name: flux-dev` |

Task-matched runs:

- Depth: [`ovb-fix_ctrl_depth`](https://huggingface.co/khazic/ovb-fix_ctrl_depth), [`ovb-flux3_ctrl_depth_nominsnr_15k`](https://huggingface.co/khazic/ovb-flux3_ctrl_depth_nominsnr_15k)
- Normals: [`ovb-fix_ctrl_normal`](https://huggingface.co/khazic/ovb-fix_ctrl_normal), [`ovb-flux3_ctrl_normal_nominsnr_15k`](https://huggingface.co/khazic/ovb-flux3_ctrl_normal_nominsnr_15k)
- Semantic / referring: [`ovb-fix_ctrl_seg`](https://huggingface.co/khazic/ovb-fix_ctrl_seg), [`ovb-fix_seg_plus_gcg`](https://huggingface.co/khazic/ovb-fix_seg_plus_gcg), [`ovb-flux3_ctrl_seg_nominsnr_15k`](https://huggingface.co/khazic/ovb-flux3_ctrl_seg_nominsnr_15k)

The `fix_*` depth LoRAs were trained as closer-brighter grayscale. Decode them with `--depth-colormap grayscale --invert-depth`. A separate local SANA-1600M Hypersim run (cube-edge metric metres, not a Hub `ovb-*` repo) is listed separately.

## Paper Alignment Statement

> What "aligned with the paper" means here, and what it cannot mean. Audited on 2026-10-22 against arXiv:2604.20329v3 (Sections 2–3, Appendix A–D). Details and line references: [`docs/ABLATION_BASELINE.md`](docs/ABLATION_BASELINE.md).

**Three categories, kept separate on purpose:**

1. **Stated by the paper and matched here (method-level, ~96%).** Every formula, constant, prompt string and evaluation protocol the paper writes down has a one-to-one implementation:
   depth power transform (λ = −3, c = 10/3) and the cube-edge colormap with its analytic inverse (`core/depth_mapper.py`); normal mapping R=(1−x)/2, G=(1+y)/2, B=(1+z)/2 (`core/normal_mapper.py`); the four semantic prompt styles of Fig. 2, the instance prompts of Fig. 3 and the referring prompts of Fig. 4, verbatim (`inference/prompt.py`, `referring_contract.py`); the five-stage instance decoder of Appendix A with τ = 14, θ_size = 2×10⁻⁴, θ_erosion = 0.1, γ = 5, 16-connectivity (`core/segmentation/`); the SA-Co cgF1 = 100·pmF1·IL_MCC with the Gemini 3.1 Flash-Lite presence prompt and the single-turn Gemini 2.5 Pro ReasonSeg rewrite (`evaluation/instance.py`, `evaluation/gemini.py`); metric depth evaluated without scale alignment or intrinsics; the five tasks, low-ratio mixing with generation data, and the zero-shot rule that no evaluation benchmark enters training (path- and image-id-level audits in `scripts/build_qwen_baseline_manifest.py`). The only remaining items the paper shows but does not benchmark are covered as optional training data (Fig. 4 (a)/(b) multi-expression rows). Two readings the paper leaves open — Eq. (3) seed comparison and the Eq. (6) area basis — are exposed as `seed_merge_rule` / `spatial_merge_area_basis` with the paper-literal reading as default.

2. **Not stated by the paper; chosen here and recorded.** The paper gives no architecture, optimizer, learning rate, schedule, batch size, warmup, weight decay, gradient clipping, loss form, loss weights, mixing ratio, seed, resolution, inference steps or CFG, and no benchmark depth caps/crops. The values in `configs/experiments/*_paper_aligned.yaml` (AdamW 1e-4, rsLoRA r32/α64, 512 px, `vision_ratio` 0.5, NYU 10 m Eigen / KITTI 80 m Garg, …) are this repository's choices, written to `effective_config.json` per run and marked "not a paper constant" in the config comments. They can be reproduced, but they cannot be "aligned".

3. **Closed, not reproducible by any open implementation.** The Nano Banana Pro base model, the in-house 2D annotations and internal 3D renders, and the human pairwise generation study (GenAI-Bench 53.5 % / ImgEdit 47.8 %). This repository substitutes Qwen-Image-2.1 / Qwen-Image-Edit, public data (SNVC, Hypersim, TartanAir, BlendedMVS, COCO, LVIS, Rea2Seg) and an FID/LPIPS proxy that can only show the absence of severe generation degradation. **The paper's benchmark numbers are therefore not a target of this repository and are reported in the next section for reference only.**

## Benchmarks (from paper)

> These are the **paper's** numbers (Nano Banana Pro base, undisclosed in-house data). This repository has **not** reproduced them. Measured scores, gates, and blocked benches are in [`docs/EXPERIMENTAL_RESULTS_2026-10-10.md`](docs/EXPERIMENTAL_RESULTS_2026-10-10.md).

| Task | Benchmark | Metric | Vision Banana | Previous SOTA |
|------|-----------|:------:|:-------------:|:-------------:|
| Semantic Seg | Cityscapes | mIoU | **69.9** | 65.2 (SAM 3) |
| Instance Seg | SA-Co/Gold | cgF1 | **47.5** | 24.6 (OWLv2) |
| Referring Seg | RefCOCOg | cIoU | **73.8** | 73.4 (SAM 3 + Gemini 2.5 Pro) |
| Referring Seg | ReasonSeg | gIoU | **79.3** | 77.0 (SAM 3 + Gemini 2.5 Pro) |
| Depth | 6-dataset avg | δ1↑ | **0.882** | 0.823 (UniK3D) |
| Depth | 4-dataset avg (NYU/ETH3D/DIODE/KITTI) | δ1↑ | **0.929** | 0.918 (Depth Anything 3) |
| Normal | 4-dataset avg | mean°↓ | **18.928** | 19.642 (Lotus-2) |

### Paper capability scorecard (this repository)

The paper evaluates three axes. This table is the current measured-or-blocked status, not Table 1. Paper numbers stay in the left column only.

| Axis | Benchmark | Paper | This repo | Status |
|------|-----------|------:|-----------|--------|
| 2D | RefCOCOg UMD val cIoU | 0.738 | SANA `ovb-fix_seg_plus_gcg` 0.143 (n=4896) | measured |
| 2D | ReasonSeg val gIoU | 0.793 | — | adapter ready; dataset not on the eval machine. Paper score uses Gemini 2.5 Pro query rewrite |
| 2D | Cityscapes val mIoU | 0.699 | — | blocked: official `gtFine` is not available. Do not use `cityscapes8` |
| 2D | SA-Co/Gold cgF1 | 47.5 | — | adapter ready (query JSONL + pmF1/IL_MCC/cgF1). Dataset not present. Paper score uses Gemini 3.1 Flash-Lite NP filter |
| 3D depth | NYU δ1 / AbsRel | 0.948 / 0.081 | SANA gray 0.066 / 1.990; SANA metric 0.124 / 1.251; FLUX gray 0.155 / 1.472 | measured, gate fail |
| 3D depth | iBims-1 | 0.934 / 0.078 | SANA gray 0.101; SANA metric 0.179; FLUX gray 0.179 | measured, gate fail |
| 3D depth | ETH3D | 0.935 / 0.103 | SANA gray 0.186; SANA metric 0.292; FLUX gray 0.163 | measured, gate fail |
| 3D depth | DIODE-Indoor | 0.917 / 0.108 | SANA gray 0.083; SANA metric 0.184; FLUX gray 0.140 | measured, gate fail |
| 3D depth | KITTI | 0.915 / 0.107 | SANA gray 0.264; SANA metric 0.051; FLUX gray 0.095 | measured, gate fail |
| 3D depth | nuScenes | 0.643 / 0.219 | — | adapter now returns metric `gt_depth` (80 m cap, no paper crop). LiDAR projections not on the eval machine |
| 3D normal | DIODE-Indoor mean° | 13.818 | SANA 24.96 (gates pass); FLUX 91.09 | measured |
| 3D normal | NYU / ScanNet / VKITTI | 17.778 / 15.052 / 29.063 | — | official normal maps not in the eval tree |
| Generation | GenAI-Bench T2I win | 53.5% | — | paper metric is human pairwise. Prompt adapter ready; no human judge run |
| Generation | ImgEdit win | 47.8% | — | paper metric is human pairwise. Instruction-pair adapter ready; no human judge run |

`--benchmark 2d|3d|generation|all` now groups those jobs. ReasonSeg, SA-Co/Gold, nuScenes, GenAI-Bench, and ImgEdit roots are `--reasonseg-root`, `--saco-root`, `--nuscenes-root`, `--genai-bench-root`, `--imgedit-root`.

### Measurements in this repository

These scores use the official splits, native ground-truth resolution, and the pixel-weighted metrics in `scripts/evaluate_paper_benchmarks.py`. They are not Table 1. `delta_1.25_aligned` is a per-image median-scale diagnostic. The depth gate requires metric δ1 at least 0.10 above the best non-oracle trivial baseline and AbsRel no worse than that baseline. Every depth row below failed that gate.

SANA relative depth, [`ovb-fix_ctrl_depth`](https://huggingface.co/khazic/ovb-fix_ctrl_depth) EMA, 20 steps, grayscale with bright=near:

| Dataset | n | δ1 | AbsRel | aligned δ1 |
|---------|--:|---:|-------:|-----------:|
| NYU Eigen + 10 m | 654 | 0.066 | 1.990 | 0.427 |
| DIODE indoor | 325 | 0.083 | 550.1 | 0.322 |
| KITTI Garg + 80 m | 1000 | 0.264 | 0.436 | 0.471 |
| ETH3D public DSLR + 80 m | 454 | 0.186 | 1.765 | 0.418 |
| iBims-1 | 100 | 0.101 | 229.5 | 0.345 |

SANA metric depth, local Hypersim cube-edge EMA (`sana_1600m_hypersim_metric_depth`, step 3000), 20 steps, no luminance invert:

| Dataset | n | δ1 | AbsRel | aligned δ1 |
|---------|--:|---:|-------:|-----------:|
| NYU Eigen + 10 m | 654 | 0.124 | 1.251 | 0.323 |
| DIODE indoor | 325 | 0.184 | 18.62 | 0.494 |
| KITTI Garg + 80 m | 1000 | 0.051 | 0.773 | 0.241 |
| ETH3D public DSLR + 80 m | 454 | 0.292 | 0.543 | 0.486 |
| iBims-1 | 100 | 0.179 | 23.02 | 0.393 |

SANA normals, [`ovb-fix_ctrl_normal`](https://huggingface.co/khazic/ovb-fix_ctrl_normal) EMA, DIODE indoor n=325: mean angle 24.96°, median 17.01°. Both normal gates passed. This is not the paper's 13.818°. NYU, ScanNet and VKITTI normal maps are not in the eval tree.

SANA referring, [`ovb-fix_seg_plus_gcg`](https://huggingface.co/khazic/ovb-fix_seg_plus_gcg) step-30000 adapter (no EMA), RefCOCOg UMD val n=4896: cIoU 0.143, gIoU 0.158, Precision@0.5 0.089. The paper's cIoU is 0.738. Cityscapes semantic mIoU is not measured: official `gtFine` is not available here.

FLUX.1-dev specialist `ovb-flux3_ctrl_depth_nominsnr_15k`, 20 steps, 4-bit, closer-brighter grayscale:

| Dataset | n | δ1 | AbsRel | aligned δ1 |
|---------|--:|---:|-------:|-----------:|
| NYU Eigen + 10 m | 654 | 0.155 | 1.472 | 0.202 |
| DIODE indoor | 325 | 0.140 | 2433 | 0.219 |
| KITTI Garg + 80 m | 1000 | 0.095 | 0.778 | 0.178 |
| ETH3D public DSLR + 80 m | 454 | 0.163 | 1.398 | 0.206 |
| iBims-1 | 100 | 0.179 | 2096 | 0.206 |

FLUX.1-dev specialist `ovb-flux3_ctrl_normal_nominsnr_15k`, DIODE indoor n=325: mean angle 91.09°, median 91.51°. That is chance-level orientation, not the paper's 13.818°.

FLUX.1 Kontext Hypersim metric EMA finished 3000/3000 (`flow_loss=0.1526`). NYU official_test (654, cube-edge, 28 steps) is still running. `ovb-flux3_ctrl_seg_nominsnr_15k` has no Cityscapes `gtFine` here, so semantic mIoU is not measured.

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

Paper-protocol eval scores a decoded task against the official split at native ground-truth resolution. Relative grayscale depth LoRAs need the invert flags. Metric cube-edge checkpoints omit them.

```bash
PYTHONPATH=src python scripts/evaluate_paper_benchmarks.py \
  --benchmark depth --model-name sana-1600m \
  --lora-path checkpoints/ovb-fix_ctrl_depth/ema_adapter \
  --dtype float32 --native-resolution --num-inference-steps 20 \
  --depth-colormap grayscale --invert-depth \
  --nyu-root data/nyu/official_test \
  --output-json output/eval/nyu_fix_ctrl_depth.json

PYTHONPATH=src python scripts/check_depth_gate.py \
  --model-json output/eval/nyu_fix_ctrl_depth.json --benchmark depth/nyu \
  --baseline-json output/eval/nyu_trivial.json

# FLUX.1-dev on a 24 GB GPU (4-bit transformer + 8-bit T5)
PYTHONPATH=src python scripts/evaluate_paper_benchmarks.py \
  --benchmark depth --model-name flux-dev --quantize 4bit \
  --lora-path checkpoints/ovb-flux3_ctrl_depth/lora_adapter \
  --dtype bfloat16 --native-resolution --num-inference-steps 20 \
  --depth-colormap grayscale --invert-depth \
  --nyu-root data/nyu/official_test \
  --output-json output/eval/nyu_flux3_ctrl_depth.json
```

Capability groups: `--benchmark 2d` (Cityscapes, RefCOCOg, ReasonSeg, SA-Co/Gold), `--benchmark 3d` (NYU, iBims, ETH3D, DIODE-Indoor, KITTI, nuScenes, plus normals), `--benchmark generation` (GenAI-Bench / ImgEdit proxies). Single-task flags still work. Normals use `--benchmark normal`. Referring uses `--refcocog-root` and `--reasonseg-root`. Instance uses `--saco-root`. Semantic Cityscapes uses `--cityscapes-root`.

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
│   ├── training/                # Unified training framework
│   │   ├── model_backend.py     # Multi-model abstraction (SANA/FLUX/SD3)
│   │   ├── flow_matching.py     # Unified flow matching scheduler
│   │   ├── config.py            # YAML config + CLI overrides
│   │   ├── losses.py            # Flow MSE + Min-SNR + aux losses
│   │   ├── callbacks.py         # W&B logging + checkpoint
│   │   ├── trainer.py           # Core training loop
│   │   ├── edit_forward.py      # Instruction-editor training layouts
│   │   ├── tuner.py             # Legacy InstructionTuner
│   │   ├── trainer_interface.py # 5 backends (Native/DDP/FSDP/DeepSpeed/Accelerate)
│   │   └── ...
│   ├── evaluation/              # Metrics, trivial baselines, depth/normal gates
│   ├── export/                  # ONNX + TorchScript export
│   └── data/                    # Dataset loaders
├── configs/                     # Training & experiment configs
├── scripts/                     # Training, inference, evaluation scripts
├── tests/                       # 1000+ tests across 79 files
├── docker/                      # CPU/GPU Docker images
└── notebooks/                   # Colab demo
```

---

## Roadmap & TODO

> This is an **unofficial, community-driven reimplementation**. We are transparent about what is done and what is not.

### High Priority
- [x] Unified trainer used for the published SANA LoRAs and the FLUX.1-dev LoRAs under `khazic/ovb-*`
- [x] Paper-protocol adapters for 2D (Cityscapes, RefCOCOg, ReasonSeg, SA-Co/Gold), 3D (NYU, iBims, ETH3D, DIODE indoor, KITTI, nuScenes, normals) and generation prompt/edit sets
- [ ] Paper Table 1 reproduction (current measurements are far from those numbers; Cityscapes `gtFine`, ReasonSeg, SA-Co/Gold, nuScenes, GenAI-Bench and ImgEdit data are not all present)
- [ ] Public checkpoint whose paper-protocol scores match the paper
- [ ] 9:1 vision-to-language data mix verification

### Medium Priority
- [ ] Multi-model training verification (Z-Image, Qwen-Image, Lumina2, Chroma)
- [ ] Additional vision task training (edge, flow, keypoint, object_detection, depth_edge)
- [ ] Generation preservation measurement (FID/CLIP on COCO)
- [ ] Export/serving integration with unified training framework

### Low Priority
- [ ] DDP/FSDP/DeepSpeed verification with unified `Trainer`
- [ ] EMA verification in actual training runs
- [ ] Legacy script deprecation (`train_sana_512.py`, `train_flux.py`)
- [ ] Comprehensive test coverage for new unified training modules
- [ ] Dataset preparation pipeline documentation

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
