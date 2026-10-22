---
name: vision-banana-train
description: One-stop guided Open-Vision-Banana workflow — environment setup, model selection, public-data download and preparation (six paper tasks), paper-aligned training config, background training, loss monitoring, paper-benchmark evaluation, and inference. TRIGGER when the user wants to train a vision model (depth/normal/semantic/instance/referring) with Open-Vision-Banana, run the full data → training pipeline, evaluate against the paper's benchmarks, or asks to "训练视觉模型" / "用 vision banana 做训练" / "train a vision LoRA". SKIP for general questions about the codebase or non-training tasks.
---

# Open-Vision-Banana One-Stop Vision Training Workflow

Guide the user through a complete run: **environment → backbone → data (download → prepare → merge → audit) → training config → background training → monitoring → evaluation → inference**.

The core of this skill is **interactive guidance**: at every key decision point use `AskUserQuestion` and never make irreversible assumptions on the user's behalf.

The working directory is the repository root. All Python commands must prefix `PYTHONPATH=src`.

> **Environment**: the repo's Python environment (conda `llama` on the lab machine; any Python ≥ 3.10 env with `pip install -e ".[all]"` elsewhere). For background jobs use the environment's **absolute python path** (`which python` after activation), not bare `python`.

> **Mirrors**: `export HF_ENDPOINT=https://hf-mirror.com` for Hugging Face; ModelScope sources need `modelscope login` once for `OpenDataLab/*` datasets (Hypersim, ADE20K full).

> **Artifacts**: training outputs go to `output/<run_name>/`; raw data to `data/raw/` (`DATA_ROOT`), prepared manifests to `data/prep/`, `data/merged/`, `data/qwen_baseline/`. Do NOT scatter files in the repo root.

> **Honesty**: unofficial reimplementation. Everything the paper *states* is implemented one-to-one (README "Paper Alignment Statement"); the paper's base model, data and human study are closed, so **paper benchmark numbers are not a target** and must never be presented as expected results.

> **Language**: communicate in whatever language the user requests, otherwise follow the active language setting.

---

## Progress board (show only when progress changes)

Render the **progress board** only when a step marker changes (`[ ]` / `[~]` / `[x]` / `[-]`) or when the user asks for status. Never repeat it on routine log polling.

Markers: `[x]` done · `[~]` in progress · `[ ]` not started · `[-]` skipped.

Always wrap the board in a fenced code block so line breaks survive; its language follows the conversation.

```
进度看板
[x] 0. 确认整体流程（范围 / 底座 / 任务）
[x] 1. 环境准备
[~] 2. 底座模型选择与下载
[ ] 3. 数据：下载 → 预处理 → 合并 → 零样本审计
[ ] 4. 训练配置
[ ] 5. 启动训练
[ ] 6. 论文基准评估
[ ] 7. 推理验证
```

- Mark steps `[-]` instead of dropping them when they do not apply (e.g. "training only" skips 6 & 7).
- Put the board at the **top** of the message when shown, then the actual content.

---

## Stage 0: Confirm the overall flow

One `AskUserQuestion` call with three questions:

1. **Flow scope** — training only (data + config + training) / full flow (+ evaluation + inference).

2. **Backbone**:
   - **Qwen-Image-2.1** (`qwen-image-2.1`, recommended) — natively image-conditioned (source image as Qwen3-VL visual tokens + reference latents), the closest open analogue of the paper's editor base. `batch_size` must be 1 (use `gradient_accumulation`), resolution a multiple of 32.
   - **Qwen-Image-Edit-2509** (`qwen-image-edit-2509`) — editing backbone; generation replay must be **edit pairs**.
   - **SANA-1600M** (`sana-1600m`) — text-to-image backbone, 24 GB friendly; the original ablation baseline (semantic/depth were weak in that regime, see `docs/ABLATION_BASELINE.md`).
   - **FLUX.1-dev** (`flux-dev`) — 12B, needs 4-bit quantization.
   - Other registered models/editors exist but are not training-verified.

3. **Tasks**:
   - **Paper six-task mixture** (recommended): `depth, normal, semantic, instance, referring, generation` — the `generation` rows are the paper's generation-preservation mixing.
   - Vision-only subset (e.g. depth + normal + semantic) — note that the paper warns pure vision fine-tuning erodes generation.
   - Single task.

Record the choices; they drive Stages 2–4.

---

## Stage 1: Environment setup

```bash
python --version                                # ≥ 3.10
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pip show open-vision-banana || pip install -e ".[all]"
python -c "import peft, accelerate, diffusers, wandb"           # training deps (missing -> ~35 test failures, not a baseline)
python -c "import h5py, pyarrow, pycocotools, ijson, modelscope" # data pipeline deps
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_API_KEY=<key>                      # only with --wandb
nvidia-smi
```

GPU rule: a GPU is **free** when VRAM use is ≲ 1 GB and utilisation is low. No free GPU → `AskUserQuestion` (wait / share / pick index). Multiple free → ask single vs multi-card. Pin the choice with `CUDA_VISIBLE_DEVICES=<idx>` for every GPU command.

---

## Stage 2: Backbone selection & download

Confirm with `AskUserQuestion` (default = Stage 0 choice). Check the local cache first and **verify completeness** (expected size, all `*.safetensors`, no `*.incomplete`).

```bash
HF_ENDPOINT=https://hf-mirror.com hf download <model_id> --local-dir <path>     # HF
modelscope download --model <model_id> --local_dir <path>                       # ModelScope
```

| Alias | Hub id |
|---|---|
| `qwen-image-2.1` | `Qwen/Qwen-Image-2.1` |
| `qwen-image-edit-2509` | `Qwen/Qwen-Image-Edit-2509` |
| `sana-1600m` | `Efficient-Large-Model/Sana_1600M_512px_diffusers` |
| `flux-dev` | `black-forest-labs/FLUX.1-dev` |

Downloads run in the background; keep working on Stage 3 meanwhile and report progress every ≤ 100 s.

---

## Stage 3: Data — download → prepare → merge → audit

All sources are public and registered in `configs/datasets/sources.tsv` (`group=train|eval|replay`); the full table, sizes and the exact per-source prepare command live in **`docs/DATASETS.md`**. Use `AskUserQuestion` to choose between:

- **A. Existing prepared manifest** — the user has a `pairs.jsonl`; validate it (Stage 3.4) and go to Stage 4.
- **B. Fresh public-data build** (default) — the steps below.
- **C. Custom data** — rows `{"task", "input", "target", "prompt", "split"}` with manifest-relative paths; targets must follow the RGB contracts (`core/`); run Stage 3.4.

### 3.1 Download
```bash
bash scripts/download_datasets.sh --list
# first run: small footprint (~50 GB); scale with the env knobs later
HYPERSIM_SCENES="ai_001_001 ai_001_002 ai_001_003 ai_001_004" TARTANAIR_CHUNKS=1 OMNIEDIT_SHARDS=2 \
  bash scripts/download_datasets.sh --group train
bash scripts/download_datasets.sh --group eval      # paper benchmarks; login-walled ones print instructions
```
Ask before pulling > 100 GB. Hypersim scenes are 2–8 GB each; COCO images 20 GB; OmniEdit 5 GB/shard.

### 3.2 Prepare (one script per source; each writes a manifest-relative `pairs.jsonl`)
```bash
PYTHONPATH=src python scripts/prepare_hypersim.py --source data/raw/hypersim --output data/prep/hypersim --tasks depth,normal,semantic,instance
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
```
Non-negotiables: Hypersim `depth_meters` is **ray distance** (the script converts to planar z by default); generation replay for Qwen editors must be edit pairs (`--kind edit`; the Trainer rejects identity rows); `--exclude-image-ids refs(umd).p` keeps RefCOCOg val images out of COCO-derived training rows.

### 3.3 Merge + zero-shot audit
```bash
PYTHONPATH=src python scripts/merge_multitask_manifest.py \
  --inputs data/prep/hypersim/pairs.jsonl data/prep/tartanair/pairs.jsonl data/prep/ade20k/pairs.jsonl \
           data/prep/inst/instance.jsonl data/prep/res/pairs.jsonl data/prep/replay/omniedit.jsonl \
  --out data/merged/pairs.jsonl --val-percent 2
PYTHONPATH=src python scripts/build_qwen_baseline_manifest.py --base-manifest data/merged/pairs.jsonl \
  --output data/qwen_baseline/pairs.jsonl --data-root data/merged \
  --exclude-image-ids "data/raw/refer/refcocog/refs(umd).p"
```
Then run the machine gate and paste its table to the user:
```bash
PYTHONPATH=src python scripts/check_experiment_gates.py data --merged-audit data/merged/pairs.audit.json \
  --final-manifest data/qwen_baseline/pairs.jsonl --hypersim-manifest data/prep/hypersim/manifest.json --model-name qwen-image-2.1
```
Exit code 1 = do not launch. Also read out: `rejected_benchmark_leak_rows` / `rejected_excluded_image_id_rows` (what the zero-shot rule removed), `rejected_non_metric_depth_rows` (what failed the cube-edge check), and `protocol` / `zero_shot_claim_allowed`.

### 3.4 Validate before training
```bash
PYTHONPATH=src python -c "
from open_vision_banana.data.validation import validate_manifest_contract
print(validate_manifest_contract('data/qwen_baseline/pairs.jsonl', model_name='qwen-image-2.1', data_root='data/qwen_baseline').summary())"
```
It flags: no attention LoRA targets, text-length truncation of semantic legends, relative-depth prompts on metric targets, grayscale/non-cube-edge depth, multi-prompt tasks that break `uniform_task_prompts`. The same audit is **enforced inside `Trainer`** at launch (errors abort; `training.skip_preflight: true` bypasses and records the errors in `effective_config.json`) — running it here just gives you the report before the backbone download.

---

## Stage 4: Training configuration

### 4.1 Base config

| Config | Backbone | Tasks | Notes |
|---|---|---|---|
| `configs/experiments/qwen_image_21_paper_aligned.yaml` | Qwen-Image-2.1 | six | **Recommended**; paper-aligned recipe |
| `configs/experiments/qwen_image_edit_paper_aligned.yaml` | Qwen-Image-Edit-2509 | six | editing backbone |
| `configs/experiments/qwen_image_21_shared_baseline.yaml` | Qwen-Image-2.1 | depth+normal+semantic+referring | vision-only ablation |
| `configs/sana_1600m_3task_full_clean.yaml` | SANA-1600M | depth+normal+seg | legacy T2I |
| `configs/flux_3task_full_clean.yaml` | FLUX.1-dev | depth+normal+seg | 4-bit |

### 4.2 Print key parameters for confirmation

| Parameter | paper_aligned value | Source |
|---|---|---|
| `model.name` | `qwen-image-2.1` | Stage 0 |
| `data.tasks` | depth, normal, semantic, instance, referring, generation | Stage 0 |
| `data.vision_ratio` | 0.5 (paper: "very low", no number → repo choice) | config |
| `data.depth_colormaps` | cube_edge, plasma, inferno, viridis, grayscale | paper §3.2 |
| `data.semantic_prompt_styles` / `semantic_color_remap_prob` | json_named, json_tuple, natural, hex / 0.5 | paper Fig. 2 |
| `data.aspect_preserving_crop` | true | repo choice |
| `model.resolution` | 512 (multiple of 32) | repo choice |
| `training.batch_size` × `gradient_accumulation` | 1 × 32 | Qwen constraint |
| `lora.rank` / `alpha` / `rs_lora` | 32 / 64 / true | repo choice |
| `training.lr` / `optimizer` | 1e-4 / AdamW wd 0.01 β(0.9, 0.999) | repo choice |
| `training.steps` | 30000 | repo choice |
| `checkpoint.output_dir` | `output/<run_name>` | — |

State explicitly which values are paper constants and which are repository choices.

### 4.3 Ask for adjustments

`AskUserQuestion` for lr / steps / accumulation / rank / resolution / run name. CLI overrides (no yaml edits):
```bash
PYTHONPATH=src python scripts/train.py --config configs/experiments/qwen_image_21_paper_aligned.yaml \
  --pair-manifest data/qwen_baseline/pairs.jsonl --data-dir data/qwen_baseline \
  --steps 10000 --lr 5e-5 --output-dir output/<run_name> --print-effective-config
```
Always run `--print-effective-config` once and show the result before launching.

---

## Stage 5: Launch training

```bash
TS=$(date +%Y%m%d_%H%M); RUN_ID="<model>_<tasks>_${TS}"; RUN_DIR="output/${RUN_ID}"; mkdir -p "${RUN_DIR}"
PY=$(which python)

# single GPU
CUDA_VISIBLE_DEVICES=<idx> PYTHONPATH=src nohup "$PY" scripts/train.py \
  --config configs/experiments/qwen_image_21_paper_aligned.yaml \
  --pair-manifest data/qwen_baseline/pairs.jsonl --data-dir data/qwen_baseline \
  --output-dir "${RUN_DIR}" --wandb > "${RUN_DIR}/train.log" 2>&1 &

# multi GPU (LoRA gradients are all-reduced; no FSDP/DeepSpeed in the unified trainer)
CUDA_VISIBLE_DEVICES=<i,j> PYTHONPATH=src nohup "$PY" -m torch.distributed.run --nproc_per_node=2 scripts/train.py \
  --config ... --output-dir "${RUN_DIR}" > "${RUN_DIR}/train.log" 2>&1 &
```

**Monitor** every ≤ 100 s; each report must include run state **and** current loss:
```bash
pgrep -af "scripts/train.py"
tail -20 "${RUN_DIR}/train.log"
grep -E "step=|loss=" "${RUN_DIR}/train.log" | tail -5
tr '\r' '\n' < "${RUN_DIR}/train.log" | grep -oE "[0-9]+/[0-9]+" | tail -1
nvidia-smi
```
Once the first checkpoint exists: `PYTHONPATH=src python scripts/check_experiment_gates.py run --run-dir "${RUN_DIR}" --vision-ratio <value>` (tasks, sampling probabilities, preflight, optimizer block, checkpoints, finite val losses). After evaluation: `... eval --eval-json <json> --require AbsRel,cIoU`.

**Checkpoint layout** (`TrainingCallbacks.on_checkpoint`):
```
output/<run_name>/
├── effective_config.json
├── train.log
├── checkpoint-{N}/          lora.pt (LoRA + optimizer + scheduler + EMA + step), lora_adapter/, metadata.json
├── checkpoint-{N}_best/     best validation (EMA weights when EMA is on)
└── checkpoint-{N}_final/
```

**Resume**:
```bash
# interrupted run: full state
PYTHONPATH=src python scripts/train.py --config <yaml> --resume output/<run>/checkpoint-<N>
# new data / new schedule from an earlier LoRA: weights only (rank + target modules must match)
PYTHONPATH=src python scripts/train.py --config <yaml> --resume output/<run>/checkpoint-<N> --resume-weights-only
```
`steps` is absolute: a full-state resume from step 5000 with `steps: 30000` runs 25 000 more.

---

## Stage 6: Evaluation (full flow only)

Use the **paper protocols** via `scripts/evaluate_paper_benchmarks.py` on the `_best` checkpoint. Only evaluate benchmarks whose data the user actually has (Stage 3.1 `--group eval`; login-walled sets print instructions).

```bash
PYTHONPATH=src python scripts/evaluate_paper_benchmarks.py \
  --model-name qwen-image-2.1 --lora-path output/<run>/checkpoint-<N>_best \
  --nyu-root data/raw/nyu_depth_v2 --diode-root data/raw/diode \
  --refcocog-root data/raw/refer/refcocog --reasonseg-root data/raw/reasonseg \
  --output-json output/eval_<run>/paper_benchmarks.json
# Appendix A decoder ablation (two readings the paper leaves open)
#   --instance-decoding seed_merge_rule=pixel_and_seed,spatial_merge_area_basis=pixel
```

| Paper table | Benchmark | Metric | Needs |
|---|---|---|---|
| T2 | Cityscapes val | mIoU (19 classes, one fixed prompt) | manual download + `setup_cityscapes_official.py` |
| T3 | SA-Co/Gold | cgF1 (Gemini 3.1 Flash-Lite presence filter) | gated HF download + `GEMINI_API_KEY` |
| T4 / T5 | RefCOCOg UMD val / ReasonSeg val | cIoU / gIoU (ReasonSeg: Gemini 2.5 Pro single-turn rewrite) | `refcocog`, `reasonseg` rows |
| T6 | NYU, iBims-1, ETH3D, DIODE, KITTI, nuScenes | AbsRel, δ1 — metric, **no scale alignment, no intrinsics** | NYU/DIODE/ETH3D direct; others manual |
| T7 | NYUv2, DIODE, ScanNet, VKitti normals | mean / median angular error | DIODE direct; others manual |
| App. D | GenAI-Bench / ImgEdit-Bench | paper: human win rate; repo: FID/LPIPS **proxy only** | `scripts/evaluate_generation_preservation.py` |

Report `AbsRel_aligned` / `delta_1.25_aligned` only as diagnostics — they are not the paper protocol. Qualitative grids: `scripts/infer_sana_512.py` (SANA) or `scripts/inference.py --image-dir`.

---

## Stage 7: Inference validation (optional)

```bash
PYTHONPATH=src python scripts/inference.py --image <input.jpg> --task depth \
  --model-name qwen-image-2.1 --lora-path output/<run>/checkpoint-<N>_best --output-dir output/infer_<run>/
PYTHONPATH=src python scripts/inference.py --image <input.jpg> --task referring --expression "the red car" --lora-path ...
PYTHONPATH=src python scripts/inference.py --image <input.jpg> --task instance --object-name dog --lora-path ...
```
Python API: `VisionBanana.from_pretrained(model_name=..., lora_path=...).predict(image=tensor, task="depth")`.
Keep `model_name` identical across train / eval / inference; the checkpoint's recorded task prompts are reused automatically (referring always rebuilds from the current expression).

---

## Final summary (end of the whole workflow)

Give a concise closing summary and **stop**. Include only: produced artifacts (run dir, `_best` path, `effective_config.json`, eval JSON), training result (steps / final loss / best val loss / runtime), evaluation results **labelled with the protocol used** and without comparing to paper numbers as targets, inference output paths, and ready-to-run commands. **Do NOT propose follow-up work.**

---

## General requirements

- Ask first with `AskUserQuestion` at every irreversible or ambiguous decision.
- Confirm before every config change or launch: print the parameter table **and** ask.
- Long-running commands (downloads > 1 GB, training) run in the background with periodic reports (≤ 100 s). Evaluation and inference usually do not.
- Pin the GPU once (`CUDA_VISIBLE_DEVICES`) and reuse it.
- Prefix `PYTHONPATH=src`; use the env's absolute python for background jobs.
- Never bypass the zero-shot audit or the metric-depth filter with ad-hoc manifests; never put `group=eval` sources into training.
- Never destroy existing checkpoints or data; new runs get new directories.
- Honest boundaries: unofficial reimplementation; paper-matching numbers are not a goal; generation preservation is only a proxy here.

---

## Quick reference

```bash
# DATA
bash scripts/download_datasets.sh --list
HYPERSIM_SCENES="ai_001_001 ai_001_002" bash scripts/download_datasets.sh --group train
PYTHONPATH=src python scripts/prepare_hypersim.py --source data/raw/hypersim --output data/prep/hypersim --tasks depth,normal,semantic,instance
PYTHONPATH=src python scripts/merge_multitask_manifest.py --inputs data/prep/*/pairs.jsonl --out data/merged/pairs.jsonl
PYTHONPATH=src python scripts/build_qwen_baseline_manifest.py --base-manifest data/merged/pairs.jsonl --output data/qwen_baseline/pairs.jsonl --data-root data/merged --exclude-image-ids "data/raw/refer/refcocog/refs(umd).p"

# TRAIN
PYTHONPATH=src python scripts/train.py --config configs/experiments/qwen_image_21_paper_aligned.yaml --pair-manifest data/qwen_baseline/pairs.jsonl --data-dir data/qwen_baseline --wandb
PYTHONPATH=src python scripts/train.py --config <yaml> --resume output/<run>/checkpoint-<N> --resume-weights-only

# EVAL / INFER
PYTHONPATH=src python scripts/evaluate_paper_benchmarks.py --model-name qwen-image-2.1 --lora-path output/<run>/checkpoint-<N>_best --nyu-root data/raw/nyu_depth_v2 --refcocog-root data/raw/refer/refcocog
PYTHONPATH=src python scripts/inference.py --image input.jpg --task depth --model-name qwen-image-2.1 --lora-path output/<run>/checkpoint-<N>_best

# TESTS that guard the pipeline
PYTHONPATH=src:. pytest tests/test_pipeline_smoke_all_tasks.py tests/test_trainer_system_all_tasks.py tests/test_download_datasets_script.py -q
```
