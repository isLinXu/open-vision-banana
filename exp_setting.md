# Experiment Launch Settings (`exp_setting.md`)

Single source of truth for launching the **first real runs on the public-data pipeline**
(download → prepare → merge → audit → train → infer → evaluate). Every value below is
either a paper constant (marked **P**) or a repository choice (marked **R**); nothing here
claims to reproduce Nano Banana Pro or the paper's numbers.

Status of the chain before these experiments:

| segment | verified on real backbone (GPU) | verified on fixtures / fake backend |
|:--|:--|:--|
| Qwen-Image-2.1 LoRA training (`scripts/train.py`) | yes — T7 in `docs/ABLATION_BASELINE.md` (30k steps, SNVC-era data) | yes — `tests/test_trainer_system_all_tasks.py` |
| Public-data six-task pipeline (registry → manifests) | **no, this is what E0 establishes** | yes — `tests/test_pipeline_smoke_all_tasks.py` |
| Inference (`VisionBanana`, `QwenImage21Pipeline` edit layout) | yes — T7 depth / semantic | — |
| Paper-protocol evaluation (`evaluate_paper_benchmarks.py`) | yes — Cityscapes mIoU, NYU AbsRel/δ1 | yes — contract tests |
| Instance / referring / generation evaluation with Qwen | **no** (referring measured on SANA only) | yes |
| Gemini-dependent protocols (SA-Co cgF1, ReasonSeg rewrite) | **no** — need `GEMINI_API_KEY` | unit tests with stubs |

---

## 0. Hardware & environment

| item | setting |
|:--|:--|
| GPU | 1× 80 GB (H100 / H20 / A100-80G) for Qwen-Image-2.1 at 512 px, bf16, batch 1. 2× for E1 if available (DDP; LoRA grads all-reduced) |
| Disk | ≥ 300 GB free: raw ≈ 60 GB (E0) → ≈ 150 GB (E1), prepared targets ≈ 2× raw, checkpoints ≈ 400 MB each |
| Python env | repo env, `pip install -e ".[all]"`, plus `h5py pyarrow pycocotools ijson modelscope` |
| Mirrors / logins | `export HF_ENDPOINT=https://hf-mirror.com`; `modelscope login` (OpenDataLab Hypersim/ADE20K); `WANDB_API_KEY` optional; `GEMINI_API_KEY` only for E3 |
| Backbone | `Qwen/Qwen-Image-2.1` (bf16, ~40 GB). Verify: all `*.safetensors`, no `*.incomplete` |
| Paths | `DATA_ROOT=data/raw`, prepared `data/prep/*`, merged `data/merged`, final `data/qwen_baseline`, runs `output/<run>` |

Pin one GPU: `export CUDA_VISIBLE_DEVICES=<idx>`; use the env's absolute python for `nohup`.

---

## E-1 — Close the evidence gaps on existing checkpoints (evaluation only, ~1 GPU-day)

Review items 3–5 (EXPERIMENT_REVIEW_2026-10-22): the old table has training-complete runs
without conclusion-complete evaluations. No training here; everything evaluates checkpoints that
already exist on HF.

| job | checkpoint | what is missing | command |
|:--|:--|:--|:--|
| Qwen 30k full eval | `khazic/ovb-qwen_image_21_shared_uniform/checkpoint-30000_final` | only 5k mid-run numbers exist | `evaluate_paper_benchmarks.py --model-name qwen-image-2.1 --lora-path <ckpt> --cityscapes-root ... --nyu-root ... --refcocog-root ... --reasonseg-root ... --diode-root ...` |
| DIODE normals for T1/T2 | `khazic/ovb-sana_1600m_shared_{uniform,sqrt}` | the third decision metric of the T5 rule | same with `--model-name sana-1600m --diode-root data/raw/diode` (needs `diode-val-normals`) |
| Post-fix SANA control | retrain `sana_1600m_shared_baseline.yaml` (attention LoRA + 300 tokens) with the **same** manifest as T1 | old 0.009–0.017 mIoU predates the fix | rebuild the T1 manifest with `build_qwen_baseline_manifest.py --protocol in_domain` (it contains Cityscapes train), then `train.py --config configs/experiments/sana_1600m_shared_baseline.yaml`, then the same eval |

Decision rule stays the pre-registered one: `sqrt` replaces `uniform` only if ≥ 2 of {Cityscapes mIoU,
NYU δ1, DIODE mean°} improve and none regresses. Until the DIODE column is filled, `sqrt` is a
**candidate**; status in `ABLATION_BASELINE.md` is written as `训练 done / 结论 pending`. Old SNVC
depth numbers stay labelled "relative-depth supervision"; they are not metric-depth evidence.
The SANA shared runs trained on Cityscapes train + Rea2Seg → their Cityscapes / ReasonSeg numbers are
in-domain, and a rebuilt manifest must use `--protocol in_domain` explicitly (strict is the default).

## E0 — Pipeline smoke on real data (½ day)

Goal: prove download → prepare → audit → 500 training steps → inference → one metric
works on the real backbone with the public data. **No quality target.**

### E0.1 Data (small)
```bash
HYPERSIM_SCENES="ai_001_001 ai_001_002" TARTANAIR_CHUNKS=1 OMNIEDIT_SHARDS=1 \
  bash scripts/download_datasets.sh --group train        # ≈ 30 GB (COCO 20 GB dominates)
bash scripts/download_datasets.sh --only refcocog,reasonseg,nyu-depth-v2-splits,nyu-depth-v2-labeled
```
Expected: `done: N fetched, 0 failed`. Hypersim zips must pass `unzip -t`.

Prepare exactly as `docs/DATASETS.md` lists (Hypersim with `--tasks depth,normal,semantic,instance`),
then merge with `--val-percent 2` and audit with `--exclude-image-ids "data/raw/refer/refcocog/refs(umd).p"`.

**Gates** — run `python scripts/check_experiment_gates.py data --merged-audit data/merged/pairs.audit.json --final-manifest data/qwen_baseline/pairs.jsonl --hypersim-manifest data/prep/hypersim/manifest.json --model-name qwen-image-2.1`; exit code 1 = abort. The table is what it checks:

| file | field | required |
|:--|:--|:--|
| `data/merged/pairs.audit.json` | `per_task_dropped` | `{}` |
| | `per_task_split` | every task has `train` and `val` ≥ 1 |
| `data/qwen_baseline/pairs.audit.json` | `rejected_non_metric_depth_rows` | 0 |
| | `rejected_benchmark_leak_rows`, `rejected_excluded_image_id_rows` | reported, and the only RefCOCOg rows removed are its val/test |
| `data/prep/hypersim/manifest.json` | `metric_depth.convention` / `tasks.normal_backfacing_frames` | `planar_z_depth` / 0 |
| `python -c "...validate_manifest_contract(...)"` (see SKILL Stage 3.4; the Trainer re-runs it at launch and aborts on errors) | summary | no `ERROR` lines |

### E0.2 Train 500 steps
```bash
RUN=output/e0_qwen21_six_smoke_$(date +%Y%m%d_%H%M)
PYTHONPATH=src python scripts/train.py --config configs/experiments/qwen_image_21_paper_aligned.yaml \
  --pair-manifest data/qwen_baseline/pairs.jsonl --data-dir data/qwen_baseline \
  --steps 500 --warmup-steps 50 --save-every 250 --eval-every 250 --output-dir "$RUN" --print-effective-config
```
Then `python scripts/check_experiment_gates.py run --run-dir "$RUN" --vision-ratio 0.5` (tasks = six,
`vision_ratio` 0.5, sampling probabilities generation 0.5 / others 0.1, preflight not skipped and error-free,
optimizer block recorded, `checkpoint-250`, `_best` and `500_final` present, finite val losses).

### E0.3 Inference + one metric per family
```bash
PYTHONPATH=src python scripts/inference.py --model-name qwen-image-2.1 --lora-path "$RUN/checkpoint-500_final" \
  --image-dir data/qwen_baseline/<a few val inputs> --task depth --output-dir "$RUN/infer_depth"
PYTHONPATH=src python scripts/inference.py --model-name qwen-image-2.1 --lora-path "$RUN/checkpoint-500_final" \
  --image <val image> --task referring --expression "<expression from the row>" --output-dir "$RUN/infer_ref"
PYTHONPATH=src python scripts/evaluate_paper_benchmarks.py --model-name qwen-image-2.1 --lora-path "$RUN/checkpoint-500_final" \
  --nyu-root data/raw/nyu_depth_v2 --refcocog-root data/raw/refer/refcocog --max-samples 50 \
  --output-json "$RUN/eval_smoke.json"
```
Pass criterion: outputs decode (depth map not constant, referring mask non-empty on ≥ 1 sample) and
`python scripts/check_experiment_gates.py eval --eval-json "$RUN/eval_smoke.json" --require AbsRel,cIoU`
passes (present and finite). Values themselves are irrelevant at 500 steps.

---

## E1 — Paper-aligned six-task run (main experiment, ~2–3 days on 1 GPU)

### E1.1 Data (full first tier)
```bash
HYPERSIM_SCENES="$(python - <<'EOF'
# 60 scenes ≈ 200 GB raw; spread across the ai_0xx families for layout diversity
print(" ".join(f"ai_{a:03d}_{b:03d}" for a in range(1, 11) for b in range(1, 7)))
EOF
)" TARTANAIR_CHUNKS=6 OMNIEDIT_SHARDS=8 bash scripts/download_datasets.sh --group train
```
Also `ade20k` and `coco-stuff` (already in `--group train`), `lvis-v1`. Prepare with `semantic_variants 2`
for Hypersim and `--variants 2` for ADE20K/COCO-Stuff; instance with `--min-instances 1`.

Target row counts after audit (R, chosen so the vision half is balanced; adjust with `merge --cap`):

| task | rows (train) | sources |
|:--|--:|:--|
| depth | ~15 k | Hypersim 60 scenes (~6 k frames) + TartanAir 6 chunks |
| normal | ~6 k | Hypersim |
| semantic | ~40 k | ADE20K 20 k × 2 variants + Hypersim 6 k × 2 + COCO-Stuff capped |
| instance | ~40 k | COCO 2017 + LVIS (`iscrowd`/non-exhaustive filtered) + Hypersim |
| referring | ~120 k expressions | RefCOCO + RefCOCO+ train (RefCOCOg val/test images excluded everywhere) |
| generation | ~40 k edit pairs | OmniEdit 8 shards |

Use `merge_multitask_manifest.py --cap 40000` so no task dominates its 0.1 share.

### E1.2 Config (= `configs/experiments/qwen_image_21_paper_aligned.yaml`)

| group | key | value | P/R | note |
|:--|:--|:--|:-:|:--|
| model | `name` / `resolution` / `dtype` | qwen-image-2.1 / 512 / bf16 | R | 512 multiple of 32; paper does not state resolution |
| data | `tasks` | depth, normal, semantic, instance, referring, generation | P | the paper's five + generation mixing |
| data | `vision_ratio` | 0.5 | R | paper: "very low" share, no number; E2 ablates |
| data | `task_sampling` | uniform within the vision share | R | |
| data | `depth_colormaps` | cube_edge, plasma, inferno, viridis, grayscale | P | §3.2, Eq. 1 with λ=−3, c=10/3 |
| data | `semantic_prompt_styles` / `semantic_color_remap_prob` | 4 styles / 0.5 | P / R | Fig. 2 styles; remap probability is ours |
| data | `aspect_preserving_crop` | true | R | |
| lora | `rank` / `alpha` / `rs_lora` / targets | 32 / 64 / true / attention + MLP | R | paper: full model, closed |
| training | `batch_size` × `gradient_accumulation` | 1 × 32 | R | Qwen 2.1 forces batch 1 |
| training | `lr` / `warmup` / schedule | 1e-4 / 500 / cosine | R | |
| training | `optimizer` | AdamW, wd 0.01, β (0.9, 0.999), eps 1e-8 | R | recorded in `effective_config.json` |
| training | `steps` | 30 000 | R | T7 reached usable semantics by 5 k |
| training | `ema` | on, decay 0.999 | R | `_best` saves EMA weights |
| training | `cfg_dropout_rate` | 0.1 | R | keeps unconditional ability for CFG at inference |
| checkpoint | `save_every` / `eval_every` | 2 500 / 2 500 | R | |

Launch:
```bash
RUN=output/e1_qwen21_paper_aligned_$(date +%Y%m%d_%H%M)
CUDA_VISIBLE_DEVICES=<idx> PYTHONPATH=src nohup "$(which python)" scripts/train.py \
  --config configs/experiments/qwen_image_21_paper_aligned.yaml \
  --pair-manifest data/qwen_baseline/pairs.jsonl --data-dir data/qwen_baseline \
  --output-dir "$RUN" --wandb > "$RUN/train.log" 2>&1 &
```
Monitor every ≤ 100 s: `tail`, `grep -E "step=|loss="`, `nvidia-smi`. Expect ~3.5 s/step at bf16 on H100
(≈ 30 h for 30 k). If interrupted: `--resume "$RUN/checkpoint-<N>"` (full state).

### E1.3 Evaluation (at 5 k, 15 k, 30 k; always the `_best` of that window)
```bash
PYTHONPATH=src python scripts/evaluate_paper_benchmarks.py --model-name qwen-image-2.1 --lora-path "$RUN/checkpoint-<N>_best" \
  --nyu-root data/raw/nyu_depth_v2 --diode-root data/raw/diode --eth3d-root data/raw/eth3d \
  --refcocog-root data/raw/refer/refcocog --reasonseg-root data/raw/reasonseg \
  --cityscapes-root data/raw/cityscapes \
  --output-json "$RUN/eval_<N>.json" --output-csv "$RUN/eval_<N>.csv"
PYTHONPATH=src python scripts/evaluate_generation_preservation.py --model-name qwen-image-2.1 \
  --lora-path "$RUN/checkpoint-<N>_best" --dataset-root data/raw/genai_bench --output-json "$RUN/gen_<N>.json"
```
Record (protocol-faithful numbers only; `_aligned` keys are diagnostics):

| table | metric | reference points we *do* have |
|:--|:--|:--|
| T2 Cityscapes val | mIoU (19 cls, one prompt) | SANA shared 0.009–0.017; SANA semantic-only 0.266; Qwen 2.1 shared 5 k **0.235** |
| T6 NYU | AbsRel / δ1, metric, no alignment | constant-3 m baseline δ1 0.352; SANA shared 0.065 |
| T6 DIODE-Indoor, ETH3D | AbsRel / δ1 | first measurements |
| T7 DIODE-indoor normals | mean / median angle | first measurement |
| T4 RefCOCOg UMD val | cIoU | SANA 0.16–0.23 |
| T5 ReasonSeg val | gIoU (needs Gemini rewrite; without key: raw-query diagnostic) | SANA 0.19–0.21 |
| App. D | FID / LPIPS proxy vs base model on GenAI-Bench prompts | SANA degraded 105 → 388 |

**Success criteria for E1 (internal, not paper numbers):** Cityscapes mIoU ≥ 0.30 at 30 k; NYU δ1 clearly
above the constant-depth baseline (> 0.45); RefCOCOg cIoU ≥ 0.35; generation FID within 1.3× of the base model.
Missing any one of these is a *finding*, not a failure — write it into `docs/ABLATION_BASELINE.md`.

---

## E2 — `vision_ratio` ablation (after E1; 2 × 10 k steps)

Same data and config as E1; only `data.vision_ratio ∈ {0.2, 0.8}` (E1 = 0.5 is the third point), 10 k
steps each, evaluated with the E1.3 command at 10 k. Question: does a lower vision share preserve
generation (App. D proxy) without collapsing the vision metrics — the paper's qualitative claim.
Launch with `--vision-ratio 0.2 --steps 10000 --output-dir output/e2_vr02_...` (CLI override, no yaml copy).

## E3 — Appendix A decoder readings (evaluation only, no training)

The paper leaves two readings open; evaluate E1's `_best` four ways on instance data
(SA-Co Gold if the gated download + `GEMINI_API_KEY` are available, otherwise the held-out COCO val instances):
```bash
for seed in pixel_only pixel_and_seed; do for area in bbox pixel; do
PYTHONPATH=src python scripts/evaluate_paper_benchmarks.py --model-name qwen-image-2.1 --lora-path "$RUN/checkpoint-30000_best" \
  --saco-root data/raw/saco_gold --instance-decoding seed_merge_rule=$seed,spatial_merge_area_basis=$area \
  --output-json "$RUN/eval_decoder_${seed}_${area}.json"
done; done
```
Default (paper-aligned) = `pixel_only` + `bbox`.

## E4 — Continue E1 on re-cleaned data (when new data arrives)

```bash
PYTHONPATH=src python scripts/train.py --config configs/experiments/qwen_image_21_paper_aligned.yaml \
  --pair-manifest data/qwen_baseline_v2/pairs.jsonl --data-dir data/qwen_baseline_v2 \
  --resume "$RUN/checkpoint-30000_best" --resume-weights-only --steps 15000 --output-dir output/e4_...
```
Weights only: fresh AdamW / warmup / step counter; LoRA rank and targets must match the source checkpoint.
`best_val_loss` resets automatically because the manifest fingerprint changes.

---

## Bookkeeping (every run)

1. Keep `effective_config.json`, both audit JSONs, `data/prep/*/manifest.json` and the eval JSON/CSV **inside the run directory** (copy the audits in).
2. Add one row to the results table in `docs/ABLATION_BASELINE.md`: run id, data tier, steps, the T2/T4/T6/T7/App. D numbers with their protocol, and a one-line finding.
3. Never report `_aligned` depth, FID/LPIPS, or raw-query ReasonSeg as paper-protocol numbers.
4. If a run changes the data distribution, lr or schedule, start from `--resume-weights-only`, not a full-state resume.

## Fixed since the 2026-10-22 review (so E0+ are not affected)

- Mixed-batch generation replay now starts from noise per sample (train and validation); the old
  batch-wide switch gave T2I replay rows a zero velocity target on T2I backbones.
- Instance `background_color` rides from manifest row → batch → `decode_instance`; non-black Fig. 3
  backgrounds are decoded correctly in evaluation. `num_instances` = visible instances.
- `build_qwen_baseline_manifest.py` defaults to `--protocol strict` (Cityscapes rejected, COCO rows need
  `--exclude-image-ids`, no upsampling); the audit records `zero_shot_claim_allowed`, `manifest_sha256`
  and per-source evaluation-id overlap.
- `Trainer` runs `validate_manifest_contract` at launch and aborts on contract errors
  (`training.skip_preflight` records them instead).

## Known gaps to keep in mind

- VKITTI normals (T7) are derived from VKITTI2 depth + `intrinsic.txt` by `VirtualKITTINormalDataset` (no normal GT ships); the derivation is exact on analytic planes but inherits the renderer's depth discretisation (1 cm) at grazing angles — report it as such.
- SA-Co cgF1 and ReasonSeg's rewrite need a Gemini key; without it the scripts fall back to diagnostics and say so.
- iBims-1, ScanNet, nuScenes, KITTI, Cityscapes are login/request-walled; the registry prints the instructions.
- Generation preservation here is a proxy; the paper's metric is a human pairwise study.
