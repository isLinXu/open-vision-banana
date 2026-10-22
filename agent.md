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
| Low-ratio mixing with generation/editing replay (`vision_ratio` + `generation` task) | ✅ Implemented; ratio value is a repo choice (paper gives none) | `src/open_vision_banana/training/trainer.py`, `scripts/build_generation_replay_manifest.py` |
| Zero-shot protocol (path + COCO image-id audits) | ✅ Implemented | `scripts/build_qwen_baseline_manifest.py` |
| Instance training data (COCO / LVIS, Fig. 3 prompts + backgrounds) | ✅ Implemented | `scripts/build_instance_manifest.py` |
| Referring contract (Fig. 4 (c)/(d) verbatim, train = eval prompt) | ✅ Implemented | `src/open_vision_banana/referring_contract.py` |
| Paper benchmark reproduction | ❌ Not yet achieved | — |
| Released checkpoints matching SOTA | ❌ Not available | — |

### What Is NOT Done (Honest Boundaries)
- **No paper-matching benchmark results**: The paper reports SOTA on Cityscapes (mIoU 69.9), SA-Co/Gold (cgF1 47.5), etc. This repo has NOT reproduced these numbers.
- **Not all 10 vision tasks are equally verified**: Depth, normal, and semantic segmentation have the most testing. Tasks like optical flow, keypoint detection, and object detection have mapper code but limited end-to-end training verification.
- **Not all registered models are verified**: FLUX and SANA have been trained. Z-Image, Qwen-Image, Lumina2, Chroma are registered in `model_registry.py` but have NOT been trained end-to-end. Instruction editors (`qwen-image-edit`, `qwen-image-edit-2509`, `longcat-image-edit`, `flux2-klein`, `joyimage-edit`, `joyimage-edit-plus`, `instruct-pix2pix`, `sdxl-instruct-pix2pix`, `hunyuan-image3-instruct`, `hunyuan-image3-instruct-distil`, and `flux-kontext`) have a training forward. The diffusers editors have been executed on the official transformer and UNet classes built at tiny width. HunyuanImage-3.0-Instruct builds its token batch with `prepare_model_inputs` and calls the published forward; a full fine-tune of that 80B checkpoint has not been run. Qwen and Joy still receive the text embeddings the trainer already encoded; `encode_text` does not insert their vision-language image tokens. `hunyuan-image` remains the HunyuanImage 2.1 text-to-image pipeline.
- **Export/serving modules**: Code exists in `src/open_vision_banana/export/` and `src/open_vision_banana/serving/` but integration with the new unified training framework has not been verified.
- **Generation preservation**: The paper reports human pairwise win rates (53.5% GenAI-Bench / 47.8% ImgEdit). This repo has an FID/LPIPS proxy (`scripts/evaluate_generation_preservation.py`) that can only rule out severe degradation; it is not the paper protocol.
- **Paper alignment, in one sentence**: everything the paper *states* (formulas, constants, prompt strings, Appendix A algorithm, evaluation protocols, data rules) is implemented one-to-one (~96% method-level, see README "Paper Alignment Statement"); everything the paper *does not state* (architecture, optimizer, lr, schedule, batch, ratio, seed, resolution, inference steps) is a recorded repo choice; the base model, in-house data and human study are closed and are not reproduced.

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
| Project root | the repository checkout (examples below use `/media/linxu/thinkHub/Thinking-face/open-vision-banana`; substitute your own) |
| Source code | `src/open_vision_banana/` |
| Dataset registry | `configs/datasets/sources.tsv` (one line per source; `scripts/download_datasets.sh` reads it) |
| Raw downloads / prepared data | `data/raw/` (default `DATA_ROOT`) / `data/prep/`, `data/merged/`, `data/qwen_baseline/` |
| Data pipeline guide | `docs/DATASETS.md` |
| Training configs | `configs/` |
| Training output | `output/<run_name>/` |
| Scripts | `scripts/` |
| Tests | `tests/` |
| Paper PDF | `paper/Image Generators are Generalist Vision Learners.pdf` |

---

## 3. Data Pipeline (download → prepare → merge → audit)

The paper's five tasks plus generation replay come from public sources listed in
`configs/datasets/sources.tsv` (`group=train|eval|replay`; `eval` rows are paper
benchmarks and are never pulled by `--group train`). Full table and per-source
commands: `docs/DATASETS.md`.

```bash
# 1. fetch (ModelScope / HF / URL; login-walled sets print instructions)
bash scripts/download_datasets.sh --list
HYPERSIM_SCENES="ai_001_001 ai_001_002" OMNIEDIT_SHARDS=2 bash scripts/download_datasets.sh --group train
bash scripts/download_datasets.sh --group eval

# 2. prepare, one script per source (all write manifest-relative pairs.jsonl)
PYTHONPATH=src python scripts/prepare_hypersim.py --source data/raw/hypersim --output data/prep/hypersim \
    --tasks depth,normal,semantic,instance            # ray distance -> planar z; normal_cam; NYU40; instances
PYTHONPATH=src python scripts/prepare_tartanair.py --source data/raw/tartanair --output data/prep/tartanair
PYTHONPATH=src python scripts/build_openvocab_semantic_manifest.py --images data/raw/ade20k/ADEChallengeData2016/images/training \
    --labels data/raw/ade20k/ADEChallengeData2016/annotations/training --classes configs/ade20k_150_classes.json \
    --source-name ade20k --label-offset 1 --ignore-index -1 --out-dir data/prep/ade20k
PYTHONPATH=src python scripts/build_instance_manifest.py --annotations data/raw/coco2017/annotations/instances_train2017.json \
    --images-dir data/raw/coco2017/train2017 --targets-dir data/prep/inst/targets --output data/prep/inst/instance.jsonl \
    --data-root data/prep/inst --exclude-image-ids "data/raw/refer/refcocog/refs(umd).p"
PYTHONPATH=src python scripts/build_res_manifest.py --annotations-root data/raw/refer --image-root data/raw/coco2017/train2014 --output-dir data/prep/res
PYTHONPATH=src python scripts/build_generation_replay_manifest.py --kind edit --preset omniedit --records data/raw/omniedit/data \
    --images-dir data/prep/replay/images --data-root data/prep/replay --output data/prep/replay/omniedit.jsonl

# 3. merge (rebases paths, gives every task a val slice) and 4. audit (zero-shot + metric depth)
PYTHONPATH=src python scripts/merge_multitask_manifest.py --inputs data/prep/*/pairs.jsonl data/prep/inst/instance.jsonl \
    data/prep/replay/omniedit.jsonl --out data/merged/pairs.jsonl --val-percent 2
PYTHONPATH=src python scripts/build_qwen_baseline_manifest.py --base-manifest data/merged/pairs.jsonl \
    --output data/qwen_baseline/pairs.jsonl --data-root data/merged --min-cityscapes-share 0 \
    --exclude-image-ids "data/raw/refer/refcocog/refs(umd).p"
```

Rules the pipeline enforces (do not bypass them in ad-hoc scripts):
- **Zero-shot**: no evaluated benchmark enters training — path patterns (`ZERO_SHOT_BENCHMARK_PATTERNS`) *and* COCO image ids (RefCOCOg UMD val/test live inside COCO train2014). Cityscapes **train** in the mixture breaks the Table 2 zero-shot claim.
- **Metric depth only**: targets must be cube-edge metric PNGs (`is_valid_metric_cube_edge_target`); Hypersim `depth_meters` is ray distance and is converted to planar z by default.
- **Contracts**: semantic rows carry their exact `mapping`; instance targets are black-background Fig. 3 renders; referring is the Fig. 4 (c)/(d) white/black prompt; generation replay for instruction editors must be edit pairs (the Trainer rejects identity rows).
- **Tests that guard the chain**: `tests/test_pipeline_smoke_all_tasks.py` (scripts → merge → audit → `ManifestVisionDataset` batches for all six tasks) and `tests/test_trainer_system_all_tasks.py` (the real `Trainer` end-to-end with a CPU fake backend). Any new source must pass both.

## 4. Training

### 4.1 Unified Training Framework (Recommended)

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

#### Paper-aligned recipe (recommended)
```bash
PYTHONPATH=src python scripts/train.py --config configs/experiments/qwen_image_21_paper_aligned.yaml \
  --pair-manifest data/qwen_baseline/pairs.jsonl --data-dir data/qwen_baseline --wandb
```
`qwen_image_21_paper_aligned.yaml` / `qwen_image_edit_paper_aligned.yaml`: six tasks
(`depth, normal, semantic, instance, referring, generation`), `vision_ratio: 0.5`
(the paper only says the vision share is "very low"), multi-colormap depth
augmentation, four Fig. 2 semantic prompt styles with 50 % colour remapping,
aspect-preserving crops, rsLoRA r32/α64, AdamW 1e-4 (optimizer block is a repo
choice, recorded in `effective_config.json`). Qwen-Image-2.1 needs `batch_size: 1`
(raise `gradient_accumulation`) and a resolution that is a multiple of 32.

Continuing from an earlier LoRA on **new data**: set `checkpoint.resume` to the
checkpoint and `checkpoint.resume_weights_only: true` (fresh optimizer / schedule /
step; the LoRA rank and target modules must match). Plain `resume` restores the
full state for an interrupted run.

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

### 4.2 Legacy Training Scripts

Preserved for backward compatibility. NOT recommended for new experiments.

```bash
# SANA legacy
PYTHONPATH=src python scripts/train_sana_512.py \
  --config configs/sana_1600m_3task.yaml

# FLUX legacy
PYTHONPATH=src python scripts/train_flux.py \
  --config configs/flux_3task_full_clean.yaml --wandb
```

### 4.3 Hardware-Specific Launch

| GPU | VRAM | Command |
|---|---|---|
| H20 | 96GB | `bash scripts/launch_h20_single.sh` |
| H20 ×8 | 96GB×8 | `bash scripts/launch_h20_ddp.sh` |
| A100-80GB | 80GB | `bash scripts/launch_training_a100_80g.sh` |
| A100-40GB | 40GB | `bash scripts/train_a100_40g_deepspeed.sh` |
| RTX 4090 | 24GB | `PYTHONPATH=src python scripts/train.py --config <yaml>` |
| MPS (Apple) | Unified | `PYTHONPATH=src python scripts/train.py --config <yaml>` |

### 4.4 Training Output Structure

Expected layout (created by `TrainingCallbacks.on_checkpoint`):

```
output/<run_name>/
├── train.log                    # Training log
├── checkpoint-{N}/              # Periodic checkpoints
│   ├── lora.pt                  # LoRA weights + optimizer state + metadata
│   ├── lora_adapter/            # PEFT-format adapter (save_pretrained)
│   └── metadata.json            # step, val_loss, lora_rank/alpha, model_name
├── checkpoint-step-{N}_best/    # Best model (by validation loss)
│   ├── lora.pt
│   ├── lora_adapter/
│   └── metadata.json
└── wandb/                       # W&B run directory (if --wandb)
```

### 4.5 Available Config Files

| Config | Model | Tasks | Notes |
|---|---|---|---|
| `experiments/qwen_image_21_paper_aligned.yaml` | Qwen-Image-2.1 | depth+normal+semantic+instance+referring+generation | **Recommended**; paper-aligned recipe, see above |
| `experiments/qwen_image_edit_paper_aligned.yaml` | Qwen-Image-Edit-2509 | same six | Editing backbone; replay must be edit pairs |
| `experiments/qwen_image_21_shared_baseline.yaml` / `_shared_smoke.yaml` | Qwen-Image-2.1 | depth+normal+semantic+referring | Vision-only ablation baseline / smoke |
| `sana_1600m_3task_full_clean.yaml` | SANA-1600M | depth+normal+seg | Clean 3-task (legacy T2I backbone) |
| `sana_1600m_5task.yaml` | SANA-1600M | depth+normal+seg | SNVC-50M dataset (named "5task" historically, 3 tasks active) |
| `flux_3task_full_clean.yaml` | FLUX.1-dev (12B) | depth+normal+seg | Quantized 4-bit |
| `sana_snvc50m_3task.yaml` | SANA-1600M | 3 tasks | SNVC-50M dataset |
| `training_mps.yaml` | SANA | depth | MPS (Apple Silicon) |
| `training_mps_multitask.yaml` | SANA | multi | MPS multi-task |

---

## 5. Inference

### 5.1 Python API

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

### 5.2 CLI Inference

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

### 5.3 Supported Tasks for Inference

| Task | CLI `--task` | Required Args | Output Keys | E2E Verified |
|---|---|---|---|---|
| Depth | `depth` | — | `rgb_output`, `depth` | ✅ |
| Normal | `normal` | — | `rgb_output`, `normal` | ✅ |
| Semantic Seg | `semantic` | `--class-map` (optional) | `rgb_output`, `mask` | ✅ |
| Instance Seg | `instance` | `--object-name` | `rgb_output`, `instances` | ✅ Decoder + training data pipeline tested (Appendix A round trips); no released checkpoint |
| Referring Seg | `referring` | `--expression` (required) | `rgb_output`, `mask` | ✅ Fig. 4 contract shared by data/inference/eval; RefCOCOg cIoU 0.16–0.23 measured on early LoRAs |
| Edge | `edge` | — | `rgb_output`, `edge_mask` | ⚠️ Mapper code, not training-verified |
| Optical Flow | `flow` | — | `rgb_output`, `flow` | ⚠️ Mapper code, not training-verified |
| Keypoint | `keypoint` | — | `rgb_output`, `keypoints` | ⚠️ Mapper code, not training-verified |
| Object Detection | `object_detection` | — | `rgb_output`, `bboxes` | ⚠️ Mapper code, not training-verified |
| Depth Edge | `depth_edge` | — | `rgb_output`, `edge_mask` | ⚠️ Mapper code, not training-verified |

### 5.4 Inference Optimization Presets

| Preset | VRAM | Speed | Quality | Key Options |
|---|---|---|---|---|
| `balanced` | ~16 GB | ★★★ | ★★★★ | float16 + model_cpu_offload + VAE slicing |
| `speed` | ~20 GB | ★★★★★ | ★★★ | float16 + torch.compile + VAE tiling |
| `memory` | ~8 GB | ★★ | ★★★ | float16 + sequential offload + INT8 mappers |
| `quality` | ~40 GB | ★★★ | ★★★★★ | bfloat16 + max-autotune compile |
| `cpu` | RAM only | ★ | ★★★ | float32 + sequential offload |

---

## 6. Evaluation

### 6.1 CLI Evaluation

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

### 6.2 Python Evaluation API

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

### 6.3 Evaluation Task → Dataset Mapping

Paper benchmarks (`scripts/evaluate_paper_benchmarks.py`; `--instance-decoding key=value`
overrides the Appendix A decoder for ablations). Downloaded layouts via
`bash scripts/download_datasets.sh --group eval`; login-walled sets print instructions.

| Paper table | Benchmark | Dataset class | Registry row / source |
|---|---|---|---|
| T2 | Cityscapes val (19 cls, mIoU) | `CityscapesSegDataset` | `cityscapes` (manual) + `setup_cityscapes_official.py` |
| T3 | SA-Co/Gold (cgF1, Gemini 3.1 Flash-Lite presence) | `SACoInstanceDataset` | `saco-gold` (HF gated, manual) |
| T4 | RefCOCOg UMD val (cIoU) | `RefCOCOgDataset(protocol="umd")` | `refcocog` (parquet → REFER) |
| T5 | ReasonSeg val (gIoU, Gemini 2.5 Pro rewrite) | `ReasonSegDataset` | `reasonseg` |
| T6 | NYU / iBims-1 / ETH3D / DIODE-Indoor / KITTI / nuScenes depth | `NYUDepthDataset`, `IBimsDepthDataset`, `ETH3DDataset`, `DIODEDataset`, `KITTIDataset`, `nuScenesDataset` | `nyu-depth-v2-*`, `ibims1` (manual), `eth3d-dslr-*`, `diode-val`, `kitti-depth` (manual), `nuscenes` (manual) |
| T7 | NYUv2 / DIODE-indoor / ScanNet / VKitti normals | `NYUNormalDataset`, `DIODENormalDataset`, `ScanNetNormalDataset`, `VirtualKITTINormalDataset` | `nyu-normals` (manual), `diode-val-normals`, `scannet-normals` (manual), `vkitti2` (manual; normals derived from depth + intrinsics by the dataset) |
| App. D | GenAI-Bench / ImgEdit-Bench (paper: human win rate; repo: FID/LPIPS proxy only) | `GenAIBenchDataset`, `ImgEditDataset` | `genai-bench`, `imgedit-bench` |

Legacy `scripts/evaluate.py --task` keys:

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

## 7. Key Source Code Architecture

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
├── data/                        # Dataset adapters + pipeline helpers
│   ├── hypersim.py              # normal_cam / NYU40 / instance channels -> paper RGB targets
│   ├── depth_geometry.py        # Hypersim ray distance -> planar z (60° HFOV focal rule)
│   ├── instance_palette.py      # ≥60-apart instance colours shared by COCO/LVIS + Hypersim builders
│   ├── openvocab_semantic.py    # Fig. 2 prompt styles, sampler, parse_prompt_mapping
│   └── res_contract.py / res_dataset.py  # binary RES rows
├── referring_contract.py        # Fig. 4 (c)/(d) template shared by data, inference, evaluation
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

## 8. Model Registry

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

## 9. Data Preparation (legacy helpers)

The current pipeline is Section 3; this section keeps the RES-specific guide and the older per-dataset helpers.

### RES manifests and evaluation

Use the dedicated workflow in [docs/res_data.md](docs/res_data.md) for RefCOCO-family referring segmentation. The `binary_white_black_v1` contract requires a per-expression prompt, native binary target, source protocol and split provenance. Train with `configs/sana_res_binary_v1.yaml` and its test-disjoint `training.jsonl`; use the complete official manifests for evaluation. Do not turn Flickr30K/VG boxes into segmentation ground truth. Existing colored-background checkpoints must explicitly select `legacy_color_distance_v1`.

The published dataset is supervised training data, not a zero-shot reproduction. Keep claims limited to verified data/interface behavior until actual training and held-out evaluation are completed. The release ID and full build/download/train/evaluation commands are in the linked guide.

The mixed-seg `instances_nogoogle.json` supplement is an equivalent serialization, not additional supervision. Use `scripts/audit_res_mixed.py` to verify it against official sources and the canonical release; train with the existing `training.jsonl` rather than concatenating duplicate or held-out records. See the same guide for its pinned source archive and visual review.

```bash
# List all supported datasets
python scripts/download_data.py --list

# Download and prepare
python scripts/download_data.py --dataset hypersim-depth --output data/hypersim/

# Validate before training
python scripts/validate_dataset.py --data-path data/hypersim_depth/ --task depth

# Legacy manifest builders (SNVC-50M is no longer the recommended source; see Section 3)
python scripts/build_v3_dataset.py          # Build v3 multi-task dataset
python scripts/build_snvc50m_manifest.py    # Build SNVC-50M manifest
```

---

## 10. Testing

The independent `RES Contract` workflow runs on Python 3.10–3.12 for paths used by RES (or by manual dispatch), without model downloads or private local data. It checks RES lint/format/types, source builds, training conditioning, inference and native evaluation; `.coveragerc-res` measures the new RES protocol/data modules only. Use `.github/workflows/res.yml` to reproduce these checks locally. RES work should preserve other tasks' loading and metric behavior and leave general CI, performance drivers and unrelated tests unchanged; existing whole-repository failures are reported separately.

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

## 11. Git Conventions

- **Commit format**: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, etc.)
- **Branch**: `main`
- **Lint**: `ruff check --fix .` then `ruff format .`
- **Pre-commit**: `.pre-commit-config.yaml` is configured

---

## 12. Quick Reference Commands

```bash
# === DATA (Section 3) ===
bash scripts/download_datasets.sh --list
HYPERSIM_SCENES="ai_001_001 ai_001_002" bash scripts/download_datasets.sh --group train
PYTHONPATH=src python scripts/merge_multitask_manifest.py --inputs data/prep/*/pairs.jsonl --out data/merged/pairs.jsonl
PYTHONPATH=src python scripts/build_qwen_baseline_manifest.py --base-manifest data/merged/pairs.jsonl \
    --output data/qwen_baseline/pairs.jsonl --data-root data/merged --exclude-image-ids "data/raw/refer/refcocog/refs(umd).p"

# === TRAINING ===
# Paper-aligned six-task recipe (recommended)
PYTHONPATH=src python scripts/train.py --config configs/experiments/qwen_image_21_paper_aligned.yaml \
    --pair-manifest data/qwen_baseline/pairs.jsonl --data-dir data/qwen_baseline --wandb
# Resume on new data from an earlier LoRA (fresh optimizer/schedule)
PYTHONPATH=src python scripts/train.py --config <yaml> --resume output/<run>/checkpoint-<N> --resume-weights-only

# SANA 3-task (legacy T2I backbone)
PYTHONPATH=src python scripts/train.py --config configs/sana_1600m_3task_full_clean.yaml --wandb

# FLUX 3-task
PYTHONPATH=src python scripts/train.py --config configs/flux_3task_full_clean.yaml --wandb

# === INFERENCE ===
# Single image depth
PYTHONPATH=src python scripts/inference.py --image input.jpg --task depth --lora-path output/lora

# === EVALUATION ===
# Paper benchmarks (Tables 2-7); --instance-decoding ablates the Appendix A decoder readings
PYTHONPATH=src python scripts/evaluate_paper_benchmarks.py --model-name qwen-image-2.1 --lora-path output/<run>/checkpoint-<N>_best \
    --nyu-root data/raw/nyu_depth_v2 --refcocog-root data/raw/refer/refcocog --reasonseg-root data/raw/reasonseg
# Depth evaluation with SOTA comparison (legacy entry point)
PYTHONPATH=src python scripts/evaluate.py --task depth --data-path /data/eval --compare-sota

# === TESTING ===
pytest tests/ -v

# === LINT ===
ruff check --fix src/ tests/
ruff format src/ tests/
```

