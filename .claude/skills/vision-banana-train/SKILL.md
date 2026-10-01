---
name: vision-banana-train
description: One-stop guided Open-Vision-Banana workflow — environment setup, model selection, data preparation, training config, background training, loss monitoring, evaluation, and inference. TRIGGER when the user wants to train a vision model (depth/normal/segmentation) with Open-Vision-Banana, run a full training pipeline, evaluate vision tasks, or asks to "训练视觉模型" / "用 vision banana 做训练" / "train a vision LoRA". SKIP for general questions about the codebase or non-training tasks.
---

# Open-Vision-Banana One-Stop Vision Training Workflow

Guide the user through a complete Open-Vision-Banana run: **environment setup → model selection → data preparation → training config → background training → loss monitoring → evaluation → inference**.

The core of this skill is **interactive guidance**: at every key decision point use `AskUserQuestion` and never make irreversible assumptions on the user's behalf.

The working directory is the repo root (`/media/linxu/thinkHub/Thinking-face/open-vision-banana`). All commands must prefix `PYTHONPATH=src` to ensure correct imports.

> **Conda environment**: All commands run under `conda activate llama` (Python 3.10+, torch 2.7.1+cu128). When launching background jobs, use the full conda python path: `/home/linxu/anaconda3/envs/llama/bin/python`.

> **HuggingFace mirror**: Always set `export HF_ENDPOINT=https://hf-mirror.com` for model downloads in mainland China.

> **Artifacts directory**: Training outputs go to `output/<run_name>/`. Generated configs, logs, and evaluation results should be organized under `output/`. Do NOT scatter files in the repo root.

> **Language**: communicate with the user in whatever language they request, or otherwise follow the active language setting/preference.

---

## Progress board (show only when progress changes)

This workflow has many stages, so the user must always be able to see **where we are**: which steps are done, which is in progress, and which remain.

**Rule:** Render the **progress board** only when the board state changes, or when the user explicitly asks for status/progress. A board state changes when any step marker changes (`[ ]` / `[~]` / `[x]` / `[-]`). Do **not** repeat the board on routine status updates, repeated confirmations, log polling, or consecutive messages within the same stage if the markers are unchanged.

Use these markers:
- `[x]` done
- `[~]` in progress (the step you are working on right now)
- `[ ]` not started
- `[-]` skipped (e.g. evaluation when the user chose "training only")

Render it as a compact checklist inside a fenced code block. **Always wrap the entire progress board in triple backticks** (use `text` as the info string, or no info string) so Markdown preserves every line break and marker exactly. **The board's language must follow the active conversation language / the user's request.**

```
进度看板
[x] 0. 确认整体流程（范围 / 执行方式）
[x] 1. 环境准备
[~] 2. 模型选择与下载
[ ] 3. 数据准备
[ ] 4. 训练配置
[ ] 5. 启动训练
[ ] 6. 效果评估
[ ] 7. 推理验证
```

Guidelines:
- Mark steps `[-]` instead of dropping them when they don't apply to the chosen flow (e.g. "training only" skips 6 & 7).
- After Stage 0, adapt the board to the chosen branch (mark skipped steps `[-]`) and keep that shape for the rest of the run.
- Do **not** render the board as plain paragraph text or as Markdown task-list checkboxes (`- [x] ...`); both can change the intended layout or marker semantics.
- Keep it terse — one line per step. When the board is shown, put it at the **top** of your message, then continue with the actual content / question below it.
- Update markers as soon as a step's status changes; show the updated board once at that transition, and never show a stale board.

---

## Stage 0: Confirm the overall flow

Before doing anything, use a **single `AskUserQuestion` call that asks all three of the most important branches together** (one question object each, in the same call) so the user can settle the whole shape of the run in one step:

1. **Flow scope**:
   - Training only (data prep + config + training)
   - Full flow (also includes evaluation + inference validation)

2. **Model backbone** — which generative model to use as the backbone:
   - **SANA-1600M** (recommended, 1.6B params, fits 24GB VRAM) → `sana-1600m`
   - **FLUX.1-dev** (12B params, requires 4-bit quantization for 24GB) → `flux-dev`
   - **Other registered models** (z-image, qwen-image, lumina2, chroma) → registered but NOT training-verified

3. **Vision tasks** — which tasks to train on:
   - **3-task mixture** (depth + normal + segmentation) — recommended default
   - **Single task** (depth only, normal only, or seg only)
   - **Custom task mixture** (user-specified combination)

Record the user's choices and branch the later stages accordingly.

---

## Stage 1: Environment setup

Verify the environment is ready:

1. **Check conda environment**:
   ```bash
   conda activate llama
   python --version  # Should be 3.10+
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```

2. **Check project installation**:
   ```bash
   cd /media/linxu/thinkHub/Thinking-face/open-vision-banana
   pip show open-vision-banana  # Check if installed
   # If not installed:
   pip install -e ".[all]"
   ```

3. **Set required environment variables**:
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   export WANDB_API_KEY=<your-key>  # Only for W&B logging
   ```

4. **Check GPU availability**:
   ```bash
   nvidia-smi
   ```
   Read each GPU's **VRAM usage** and **utilization**. Treat a GPU as **free** when its VRAM usage is near-empty (e.g. ≲ 1 GB) and utilization is low.

   - **No free GPU:** tell the user which GPUs are busy and use `AskUserQuestion` to let them choose: wait, share, or specify a particular index.
   - **Multiple free GPUs:** use `AskUserQuestion` to ask single-card vs multi-card.
   - **Record the chosen GPU index** and reuse it via `CUDA_VISIBLE_DEVICES=<idx>` for all subsequent commands.

---

## Stage 2: Model selection & download

Use `AskUserQuestion` to confirm the model choice (defaulting to the recommendation from Stage 0).

1. **Check if model is already available locally**:
   ```bash
   # Check HF cache
   ls ~/.cache/huggingface/hub/ | grep -i "<model_keyword>"
   # Or check custom path
   ls /media/linxu/thinkHub/ckpts/<model_path>/
   ```

2. **Verify completeness, not just existence.** A cached directory can exist while being incomplete. Verify:
   - Total size is in the expected GB range
   - All `*.safetensors` shards are present
   - No `*.incomplete` files

3. **If not present locally**, confirm the **download source**:
   - **HuggingFace Hub** (via mirror)
   - **ModelScope** (often faster in mainland China)

   Then start a download task in the background:
   ```bash
   # HuggingFace (via mirror)
   HF_ENDPOINT=https://hf-mirror.com huggingface-cli download <model_id> --local-dir <path>

   # ModelScope
   modelscope download --model <model_id> --local_dir <path>
   ```

   **Model → HF Path mapping** (from `model_registry.py`):
   | Alias | HF Path |
   |---|---|
   | `sana-1600m` | `Efficient-Large-Model/Sana_1600M_512px_diffusers` |
   | `flux-dev` | `black-forest-labs/FLUX.1-dev` |
   | `sana-600m` | `Efficient-Large-Model/SANA_600M_1024px_diffusers` |
   | `z-image` | `Tongyi-MAI/Z-Image` |
   | `qwen-image` | `Qwen/Qwen-Image` |
   | `lumina2` | `Alpha-VLLM/Lumina-Image-2.0` |
   | `chroma` | `lodestones/Chroma1-HD` |

   Keep advancing the config work while it downloads, and report download progress periodically (interval ≤ 100 seconds).

---

## Stage 3: Data preparation

The project uses **RGB parameterization** for vision tasks — all tasks are encoded as RGB images:
- **Depth**: power transform + Hilbert/spectral colormap → RGB
- **Normal**: linear (x,y,z) → (R,G,B) mapping
- **Segmentation**: color palette encoding

Use `AskUserQuestion` to ask about the data source:

### Option A: Use existing training data
Check if training data is already prepared:
```bash
# Check for v3 multi-task dataset
ls data/v3_multitask/ 2>/dev/null
# Check for SNVC-50M
ls /media/linxu/datasets/SN-VC-50M/ 2>/dev/null
# Check for pair manifests
ls data/pair_manifests/ 2>/dev/null
```

### Option B: Build a new dataset
```bash
# Build v3 multi-task dataset
PYTHONPATH=src python scripts/build_v3_dataset.py

# Build SNVC-50M manifest
PYTHONPATH=src python scripts/build_snvc50m_manifest.py

# Validate dataset before training
PYTHONPATH=src python scripts/validate_dataset.py --data-path <path> --task <task>
```

### Option C: Use custom data
Ask for the data file path(s) and task type. The data should follow the pair manifest format:
- Each entry: `{image_path, target_path, task}` or tar-based streaming format

---

## Stage 4: Training configuration

### 4.1 Select a base config

Based on the model and tasks chosen in Stage 0, select the appropriate config:

| Config | Model | Tasks | Notes |
|---|---|---|---|
| `sana_1600m_3task_full_clean.yaml` | SANA-1600M | depth+normal+seg | Clean 3-task (recommended for SANA) |
| `flux_3task_full_clean.yaml` | FLUX.1-dev (12B) | depth+normal+seg | Quantized 4-bit |
| `sana_1600m_5task.yaml` | SANA-1600M | depth+normal+seg | SNVC-50M dataset |
| `sana_snvc50m_3task.yaml` | SANA-1600M | 3 tasks | SNVC-50M dataset |
| `training_mps.yaml` | SANA | depth | MPS (Apple Silicon) |

### 4.2 Print key parameters for confirmation

Show the user a table of key config parameters:

| Parameter | Value | Source |
|---|---|---|
| `model_name` | `sana-1600m` or `flux-dev` | Stage 0 |
| `flow_type` | `linear_flow` (SANA) / `rectified_flow` (FLUX) | Config |
| `resolution` | 512 | Config |
| `batch_size` | 4 | Config |
| `gradient_accumulation` | 32 | Config (YAML field; CLI: `--grad-accum`) |
| `lora_rank` | 32 | Config |
| `lora_alpha` | 64 | Config |
| `rslora` | true | Config |
| `lr` | 3e-5 | Config |
| `steps` | 5000 | Config |
| `tasks` | depth, normal, seg | Config |
| `data_dir` | `<path>` | Stage 3 |
| `output_dir` | `output/<run_name>` | Config |

### 4.3 Ask for adjustments

Use `AskUserQuestion` to ask whether any parameters need adjusting:
- Learning rate, steps, batch size
- LoRA rank/alpha
- Resolution
- Output directory name

**CLI override syntax** (no need to edit yaml):
```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/<base>.yaml \
  --steps 10000 --lr 5e-5 --output-dir output/custom_run \
  --batch-size 2 --resolution 256
```

---

## Stage 5: Launch training

### 5.1 Generate run directory
```bash
TS=$(date +%Y%m%d_%H%M)
RUN_ID="<model>_<tasks>_${TS}"
RUN_DIR="output/${RUN_ID}"
mkdir -p "${RUN_DIR}"
```

### 5.2 Start training in background

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=<idx> nohup /home/linxu/anaconda3/envs/llama/bin/python scripts/train.py \
  --config configs/<selected>.yaml \
  --wandb > "${RUN_DIR}/train.log" 2>&1 &

# Multi-GPU (DDP)
CUDA_VISIBLE_DEVICES=<idx0,idx1> nohup /home/linxu/anaconda3/envs/llama/bin/python -m torch.distributed.run \
  --nproc_per_node=2 scripts/train.py \
  --config configs/<selected>.yaml \
  --wandb > "${RUN_DIR}/train.log" 2>&1 &
```

### 5.3 Monitor training

**Periodically check status** (interval ≤ 100 seconds):

```bash
# Check process is alive
pgrep -af "scripts/train.py"

# Tail training log
tail -20 "${RUN_DIR}/train.log"

# Extract latest loss (logged every logging_steps)
grep -E "step=|loss=" "${RUN_DIR}/train.log" | tail -5

# Check GPU usage
nvidia-smi

# Check W&B (if --wandb)
# Visit https://wandb.ai/<entity>/<project>
```

**Progress extraction** — from tqdm progress-bar lines:
```bash
# Current step / total steps
tr '\r' '\n' < "${RUN_DIR}/train.log" | grep -oE "[0-9]+/[0-9]+" | tail -1
```

**Each status report must include BOTH the current run state AND the current loss.**

### 5.4 Checkpoint structure

Training outputs (created by `TrainingCallbacks.on_checkpoint`):
```
output/<run_name>/
├── train.log
├── checkpoint-step-{N}/
│   ├── lora.pt                  # LoRA weights + optimizer state + metadata
│   ├── lora_adapter/            # PEFT-format adapter (save_pretrained)
│   └── metadata.json            # step, val_loss, lora_rank/alpha, model_name
├── checkpoint-step-{N}_best/    # Best model (by validation loss)
│   ├── lora.pt
│   ├── lora_adapter/
│   └── metadata.json
└── wandb/                       # W&B run directory (if --wandb)
```

### 5.5 Resume from checkpoint

If training is interrupted:
```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/<selected>.yaml \
  --resume output/<run_name>/checkpoint-step-{N}/lora.pt
```

---

## Stage 6: Evaluation (optional, if user chose full flow)

After training completes, evaluate the best checkpoint.

### 6.1 Qualitative evaluation (visualization)

```bash
# SANA multi-task visualization
PYTHONPATH=src python scripts/infer_sana_512.py \
  --data-dir <dataset_dir> --pair-manifest <pairs.jsonl> \
  --checkpoint output/<run_name>/checkpoint-step-{N}_best/lora.pt \
  --model-name sana-1600m --lora-rank 32 --lora-alpha 64 \
  --output-dir output/eval_<run_name>/
```

### 6.2 Quantitative evaluation (metrics)

```bash
# Depth evaluation
PYTHONPATH=src python scripts/evaluate.py \
  --task depth --data-path <eval_data> \
  --lora-path output/<run_name>/checkpoint-step-{N}_best/lora.pt \
  --output-json output/eval_<run_name>/depth_metrics.json

# Normal evaluation
PYTHONPATH=src python scripts/evaluate.py \
  --task normal --data-path <eval_data> \
  --lora-path output/<run_name>/checkpoint-step-{N}_best/lora.pt

# Semantic segmentation
PYTHONPATH=src python scripts/evaluate.py \
  --task semantic --num-classes 19 \
  --data-path <cityscapes_path> \
  --lora-path output/<run_name>/checkpoint-step-{N}_best/lora.pt

# Compare against paper SOTA
PYTHONPATH=src python scripts/evaluate.py \
  --task depth --data-path <eval_data> --compare-sota

# Quick smoke test (limit batches)
PYTHONPATH=src python scripts/evaluate.py \
  --task normal --data-path <eval_data> --max-batches 5

# Distributed evaluation (multi-GPU)
PYTHONPATH=src torchrun --nproc_per_node=<n> scripts/evaluate.py \
  --task depth --data-path <eval_data> --ddp
```

**Metrics by task**:
- depth: AbsRel, SqRel, RMSE, logRMSE, δ1, δ2, δ3
- normal: Mean Angular Error (°), Cosine Similarity, RMSE
- semantic: mIoU, pixel accuracy
- instance: cgF1, cIoU
- referring: cIoU, gIoU
- edge: F1, precision, recall, ODS

### 6.3 Evaluation dataset mapping

| `--task` | Dataset Key | Expected Format |
|---|---|---|
| `depth` | `hypersim-depth` | RGB + depth maps |
| `normal` | `normal` | RGB + normal maps |
| `semantic` | `cityscapes` | RGB + segmentation masks |
| `instance` | `sa-co` | RGB + instance annotations |
| `referring` | `refcocog` | RGB + expression + mask |

---

## Stage 7: Inference validation (optional)

Load the trained checkpoint for inference on custom images.

### 7.1 Single image inference

```bash
# Depth estimation
PYTHONPATH=src python scripts/inference.py \
  --image <input.jpg> --task depth \
  --lora-path output/<run_name>/checkpoint-step-{N}_best/lora.pt \
  --output-dir output/infer_<run_name>/

# Normal estimation
PYTHONPATH=src python scripts/inference.py \
  --image <input.jpg> --task normal \
  --lora-path output/<run_name>/checkpoint-step-{N}_best/lora.pt

# Semantic segmentation
PYTHONPATH=src python scripts/inference.py \
  --image <input.jpg> --task semantic \
  --lora-path output/<run_name>/checkpoint-step-{N}_best/lora.pt
```

### 7.2 Batch inference

```bash
PYTHONPATH=src python scripts/inference.py \
  --image-dir <path_to_images> --task depth \
  --lora-path output/<run_name>/checkpoint-step-{N}_best/lora.pt
```

### 7.3 Python API

```python
from open_vision_banana.inference.pipeline import VisionBanana

model = VisionBanana.from_pretrained(
    model_name="sana-1600m",  # or "flux-dev"
    lora_path="output/<run_name>/checkpoint-step-{N}_best/lora.pt",
    device="cuda",
)

result = model.predict(image=input_tensor, task="depth")
# result keys: "rgb_output", "depth", "visualization", etc.
```

### 7.4 Inference optimization presets

| Preset | VRAM | Key Options |
|---|---|---|
| `balanced` | ~16 GB | float16 + model_cpu_offload + VAE slicing |
| `speed` | ~20 GB | float16 + torch.compile + VAE tiling |
| `memory` | ~8 GB | float16 + sequential offload + INT8 mappers |
| `quality` | ~40 GB | bfloat16 + max-autotune compile |

---

## Final summary (end of the whole workflow)

When all chosen stages are finished, give a concise closing summary and then **stop**. The summary may include ONLY:

- the produced artifacts (checkpoint directory, best model path, evaluation results);
- the training result (total steps / final loss / best val_loss / runtime);
- the evaluation result (metrics by task, comparison with paper SOTA if available);
- the inference output paths (visualization images);
- the ready-to-run commands for future inference.

**Do NOT propose or offer any follow-up work** — no suggestions about further training, hyperparameter tuning, deployment, or anything similar. Simply describe what was produced and stop.

---

## General requirements

- Ask first with `AskUserQuestion` at every irreversible or ambiguous decision; do not assume.
- **Confirm before every config change or training launch** — print key parameters as a table AND call `AskUserQuestion` to get explicit confirmation.
- Run **long-running** commands (model download, training) **in the background + report periodically** (interval ≤ 100 seconds). Evaluation and inference are usually quick and do NOT need background.
- **Detect GPU status once before running and pin the free device(s)** (see Stage 1). Reuse the same index for all GPU commands via `CUDA_VISIBLE_DEVICES`.
- **All commands must prefix `PYTHONPATH=src`** to ensure correct module imports.
- **Use conda python path for background jobs**: `/home/linxu/anaconda3/envs/llama/bin/python` (not just `python`).
- **Keep `model_name` consistent across train / inference / evaluation.** A mismatch produces errors or silent failures.
- Never destroy existing checkpoints or data files; always use new directories for new runs.
- **Honest boundaries**: This is an unofficial reimplementation. Paper-matching SOTA has NOT been achieved. Not all models/tasks are equally verified.

---

## Quick reference commands

```bash
# === ENVIRONMENT ===
conda activate llama
export HF_ENDPOINT=https://hf-mirror.com

# === TRAINING (SANA 3-task) ===
PYTHONPATH=src python scripts/train.py --config configs/sana_1600m_3task_full_clean.yaml --wandb

# === TRAINING (FLUX 3-task) ===
PYTHONPATH=src python scripts/train.py --config configs/flux_3task_full_clean.yaml --wandb

# === BACKGROUND TRAINING ===
CUDA_VISIBLE_DEVICES=0 nohup /home/linxu/anaconda3/envs/llama/bin/python scripts/train.py \
  --config configs/sana_1600m_3task_full_clean.yaml --wandb > output/train.log 2>&1 &

# === RESUME TRAINING ===
PYTHONPATH=src python scripts/train.py --config configs/<yaml> --resume output/<run>/checkpoint-step-N/lora.pt

# === INFERENCE ===
PYTHONPATH=src python scripts/inference.py --image input.jpg --task depth --lora-path output/<run>/checkpoint-step-N_best/lora.pt

# === EVALUATION ===
PYTHONPATH=src python scripts/evaluate.py --task depth --data-path <data> --compare-sota

# === TESTING ===
pytest tests/ -k "core"
pytest tests/ -k "training"

# === MONITORING ===
pgrep -af "scripts/train.py"
tail -20 output/<run>/train.log
nvidia-smi
```
