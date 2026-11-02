# LingBot 50k full-pool experiments / 全量数据实验方案

Reviewed against main `7a41d27b75f0cd275d46833bc72ebb0590781638` (2026-11-02).
This is an experiment implementation and execution plan, **not new training results**.
Background: [#88](https://github.com/isLinXu/open-vision-banana/issues/88),
[#91](https://github.com/isLinXu/open-vision-banana/issues/91),
[#93](https://github.com/isLinXu/open-vision-banana/issues/93).

## Research question and historical evidence

Does adding the full LingBot increment improve **unaligned metric depth** at a fixed
optimizer budget and initialization? Keep capacity, LR, decoder, inference resolution
and calibration fixed before attributing improvements to the new training mixture.

Published #88 NYU results (654 samples; not newly rerun here):

| Run | HF revision | AbsRel | RMSE (m) | δ1 raw | δ1 median-aligned |
| --- | --- | ---: | ---: | ---: | ---: |
| C1 | `be6d643b8b8f58aa641c318a4ac02a1ff95b69cf` | .584609 | 2.290240 | .348319 | .845190 |
| T2 resumed | `ce70f055b47d07e72bdfb1344e5ff05c6d0f7188` | 1.178240 | 3.549415 | .179114 | .757900 |

Repositories: [C1](https://huggingface.co/khazic/ovb-i88_c1_depth_cube100_20261030_1602),
[T2](https://huggingface.co/khazic/ovb-i88_t2_depth_replay_resume_20261030).
The aligned .845 is not metric δ1. Current alignment is per-image median scale,
not a general affine fit. Do not use aligned scores to claim closure of the raw
metric gap to another paper. Historical T2 resumed at 6k with optimizer state;
that repository does not represent an additional independent 10k continuation.

#91 now has 10k full-set results; #93's earlier missing-endpoint checklist is stale.
Three newer full-finetuning LR probes have 2k full NYU results:

| LR | Fixed HF revision | NYU AbsRel / raw δ1 |
| --- | --- | --- |
| 5e-6 | `3457e5f202116871c3f56c845db2f13a514da209` | 1.40124 / .12438 |
| 1e-5 | `9d978f46c8638bfc6522677d1586dcb7fdcf97c5` | 2.24092 / .02936 |
| 2e-5 | `17f09c5a988186127d60bb17f6d7afb203cace0d` | 2.68207 / .01064 |

The repos are respectively `khazic/ovb-ft_mf_lr5e-6_t1_cube_20261102_0045`,
`khazic/ovb-ft_mf_lr1e-5_t1_cube_20261102_0051`,
`khazic/ovb-ft_mf_lr2e-5_t1_cube_20261102_0138`, file `eval_2000/eval_all.json`.
A configured 10k budget does not establish completed 10k evaluation. These three
points do not establish that full tuning fails independently of LR. Keep that
investigation separate from the data comparison.

## Data identity and interpretation

[LingBot release](https://huggingface.co/datasets/Melo111/open-vision-banana-depth/tree/78946c7b0aff52b87fe2ad15143e93625a43b139/lingbot-depth-v1):
50,000 training pairs, 49 RobbySim RRT scenes, revision
`78946c7b0aff52b87fe2ad15143e93625a43b139`, fingerprint
`e3a39e02e2b4383d341d59884ed2dc90ef8e5bacdad466744486b2db4c55863c`.
CC BY-NC-SA 4.0, following the upstream card body despite conflicting YAML;
other components retain their own licenses. See the release's provenance and audit reports.

Original 1280×960 RGB JPEG, native cube-edge PNG and float32 metre NPY are retained.
Source uint16 millimetres are divided by 1000 once; no per-image normalization or
test-GT scale fitting. Default prompt: `Generate a metric depth visualization of this image.`
Use metric_power (λ=-3, c=10/3). The native codec MAE 2.563 mm and max error
4.938 cm describe encoding loss, not source accuracy or model error.

**Code review correction:** the current trainer reads PNG targets; T2 decodes the
resized/cropped cube RGB and recolours it. It does **not** use the NPY as a direct
metric loss or as the source for colour augmentation. Retain this behavior in
both arms. The NPY remains useful for audits and later offline diagnostics.
Changing this target-generation path requires a separate paired experiment.

| Scale statistic | LingBot native pixels | NYU scoring regions |
| --- | ---: | ---: |
| Mean | 2.098 m | 2.743 m |
| <1 m | 17.88% | 3.04% |
| ≥5 m | 3.14% | 6.55% |
| >10 m | .015% | excluded by evaluation mask |

Nine-bin base-2 JSD to NYU is .265, versus .365 for the sampled Hypersim increment
and .439 for weighted TartanAir samples. Original Hypersim 8,924 rows were not
equally sampled, so these comparisons do not describe every old training row.
The new source is still too near; synthetic appearance, 90° horizontal FOV,
tilted/upward views and crop geometry differ. Its archive-prefix sampling is not
uniform over the complete LingBot corpus. NYU aggregate statistics informed its
design: label the experiment benchmark-informed and add an external benchmark
that was not used for selection. Closer depth marginals are a hypothesis, not proof of gain.

The actual historical effective configs, rather than planned row counts, recorded:

| Task | Existing train rows | With LingBot |
| --- | ---: | ---: |
| depth | 60,852 | 110,852 |
| generation | 16,492 | 16,492 |
| normal | 8,680 | 8,680 |
| semantic | 39,133 | 39,133 |
| instance | 38,596 | 38,596 |
| referring | 49,048 | 49,048 |
| All six | 212,801 | 262,801 |

Old depth comprises TartanAir 38,661, original Hypersim 8,924 and additional
Hypersim 13,267. Freeze the actual base/replay versions and manifest SHA; pinning
the supplement alone is insufficient. Rea2Seg/LLM-Seg provide reasoning training;
ReasonSeg is evaluation-only. Numbers are train rows, not distinct images.
If a shared cleanup changes them, declare the new expected counts and rerun the
old-data controls. Never selectively clean only an experimental arm.

## Implementability audit

| Requirement | Implementation / boundary |
| --- | --- |
| Pinned `.tar.xz` bundle | Existing `download_res_dataset.py`, tar auto-detection, checksum verification |
| PNG/NPY paths, C1 and five palettes | Existing loader; NPY audit-only as explained above |
| Six full configurations | `configs/experiments/lingbot_{d0,d1,p0,p1,m0,m1}.yaml` |
| Strict paired manifests | New `prepare_lingbot_experiments.py`; no missing-file skipping/caps/new splits |
| Frozen identity and global batch launch | New `run_lingbot_experiment.py`; default validates/prints, `--execute` trains |
| Source/scene/palette consumption | Per-rank `exposure-rank*-*.jsonl`, including dropped-gradient batches |
| Partial restore safety | Loader now rejects `BUILD_INCOMPLETE` for depth-only as well as referring |
| Internal overlap | Decoded RGB and COCO IDs, plus new-vs-base decoded RGB duplicate rejection |
| External benchmark decontamination | Separate required audit; not established by internal overlap checks |
| Standard full benchmarks | Existing paper evaluator, preflight/smoke/shard merger |
| Custom development-set metric inference | Follow-up implementation; built-in validation is flow loss only |
| Distance bins, off-palette diagnostics, paired/cluster bootstrap | Follow-up analysis; not automatically produced by this PR |
| Per-source sampling weights | Not supported by current sampler; no invented YAML key is added |
| Full GPU training readiness | Requires resource check, actual weight loading and short training smoke |

The generic merger silently skips missing files; the new preparation entry point
fails instead and preserves both manifest row order and splits. Asset size checks
are always applied; `--verify-asset-hashes` also rereads all native assets. Restoring
the pinned bundle already verifies transport hashes. External pHash/CLIP candidates
need visual/source confirmation: a similarity hit alone is not proof of leakage.
The release's NYU screen is partial evidence, not exhaustive decontamination.

## Matrix and budget

| Arms | Tasks / probabilities | Palettes | Data difference |
| --- | --- | --- | --- |
| D0 / D1 | depth .75, replay .25 | cube_edge | old full pool / old + 50k |
| P0 / P1 | depth .75, replay .25 | five uniformly | old full pool / old + 50k |
| M0 / M1 | replay .10, five vision tasks .18 each | cube_edge | old six-task / new six-task |

Run D first, then P, then M, seed42. Repeat **both D arms** with seeds43/44 even
for a zero/negative result. Six initial runs cost 60k optimizer steps; D replication
adds 40k. M0 is a new six-task cube control, not historical five-palette T0; #91's
no-normal recipe gives depth .225 and is not a substitute for M0's .18.

Use one E2B online parent for all arms:
`khazic/ovb-e2b_cap10k_s42_20261026_1016` revision
`036bc7285371e24b9e52ab1b0ad9aea9ea337f27`, `ckpt/checkpoint-10000_final`.
Pin and verify the corresponding base model separately. Resume **weights only**,
fresh optimizer/scheduler/EMA, rather than each arm's best historical checkpoint.
Do not add the parent adapter again on a base that already contains it.

Common settings: Qwen-Image-2.1, BF16, 512, rsLoRA r32/α64 and C1 target modules;
AdamW LR3e-5, betas(.9,.999), eps1e-8, weight decay.01; cosine10k, warmup500;
global batch32 = 8×1×4, gradient checkpointing, clipping1/drop10, EMA.9999/warmup500.
Keep img2img flow, sigma0, logit-normal, shift1, CFG dropout.05, min_snr=false,
existing spatial augmentation and zero auxiliary losses. Save/validate every2k,
log every50. Config parsing is not a substitute for a GPU smoke test.

At 10k, D/P expect 240k depth and 80k replay draws. LingBot is 45.10% of the new
depth pool: about108,252 LingBot and131,748 old-depth draws. Old exposure therefore
falls at fixed budget. The primary contrast measures the **new mixture's practical
effect**, not scale distribution alone. “Full data” means all eligible rows are
available, not that replacement sampling visits every image. No cap10k restriction.

Exposure files are cumulative **within each rank/invocation**. Use the last record,
not the sum of snapshots. Counts include discarded-gradient batches; unique counts
cannot be summed across ranks as global uniqueness. Resumes get separate files,
not restored lifetime counters. `complete` denotes the consumed-data loop ending,
not proof that final checkpoint save or evaluation succeeded. No RNG is consumed
by this audit. Existing untagged datasets do not emit these files.

If 10k is inconclusive, pre-register paired20k runs with the same schedule from the
same parent; do not only extend D1. Source-weight sweeps (25/45.1/60%) and
scale-matched comparisons require a separately tested sampler and old-depth audit;
do not fit source weights on NYU test or duplicate manifest rows to imitate weights.

## Reproducible preparation and launch

Commands below are run from this checkout; use new output directories. `$BASE_SHA`,
`$PARENT_SHA` and `$BASE_REV` are independently verified immutable identities, not
placeholders to bypass. The launcher checks the parent `lora.pt` hash and adapter
architecture/online metadata. It records the base revision and hashes model_index;
it does not claim to rehash every base weight shard.

```bash
export PYTHONPATH="$PWD/src"
python scripts/download_res_dataset.py \
  --repo-id Melo111/open-vision-banana-depth \
  --revision 78946c7b0aff52b87fe2ad15143e93625a43b139 \
  --subfolder lingbot-depth-v1 --output-dir data/prep/lingbot_depth_v1

python scripts/prepare_lingbot_experiments.py \
  --baseline data/qwen_main_depth_reasoning/pairs.jsonl \
  --baseline-sha256 "$BASE_SHA" --lingbot-dir data/prep/lingbot_depth_v1 \
  --output-dir data/lingbot_experiments --verify-asset-hashes

python scripts/run_lingbot_experiment.py --arm D1 \
  --data-root data/lingbot_experiments --parent "$PARENT" \
  --parent-sha256 "$PARENT_SHA" --base-model-dir "$BASE_MODEL" \
  --base-model-revision "$BASE_REV" --world-size 8 --seed 42 \
  --output-dir output/lingbot_d1_s42
# After resource/weight/data audits and GPU smoke, repeat with --execute.
```

Preparation rejects existing output directories and keeps incomplete markers on
write failure. It requires frozen train/val rows for all six tasks, and defaults
to verifying60,852 depth/16,492 replay train rows. A reviewed shared cleanup may use
`--expected-depth-train` / `--expected-replay-train`; log the change. It does not
manufacture a validation split from the release's50k rows. Native assets consume
about287GB plus51GB compressed; allow at least400GB working space.

The launcher preserves global batch32 for1/2/4/8 processes. Check idle owned GPUs
before using CUDA_VISIBLE_DEVICES; this PR does not reserve cards or upgrade the
shared environment. Interrupted-run recovery uses the existing trainer's full-state
resume, **not** this fresh-continuation wrapper; preserve separate exposure attempts.

## Intermediate and final evaluation

1. Freeze source/manifest/split/weights and run data preflight. Exact overlap failures
   stop the build; review/repair the shared baseline, never silently drop records.
2. CPU loader/codec checks, then short GPU training/backward/save/reload smoke in an
   independent directory. Source metadata should not change pixels, prompts or RNG.
3. Record step0. At2k/4k/6k/8k retain fixed development samples, loss, gradient drops
   and exposure statistics. Built-in validation currently samples at most40/task and
   measures flow loss. Implement the custom depth development evaluator before
   claiming intermediate metric-depth curves. Existing external sampled-test curves
   may be reproduced for historical diagnostics only, not for picking hyperparameters.
4. Primary comparison uses fixed10k **online** checkpoint, not lowest-loss `_best` or
   independently selected endpoints. EMA is a separate table.
5. Full NYU654, ETH3D454, DIODE-Indoor325; RefCOCOg4896, ReasonSeg200, Cityscapes500
   and fixed200-prompt generation proxy monitor capability retention. Validate actual
   readiness; missing datasets/normal/instance results are missing, never zero.

Keep BF16,50 inference steps, guidance4, strength.85,512 canvas,batch1, seed42 plus
original sample index, native GT, default depth prompt/cube_edge/metric_power.
NYU uses rows45:471, cols41:601 and .001<GT≤10m; preserve each other benchmark's
own crop/cap. Batch/shard changes must preserve original-index seeds. Generation
must retain true_cfg_scale semantics. ReasonSeg original-query scores are diagnostic;
FID/LPIPS proxies are not human preference or a “capability retention percentage”.

```bash
python scripts/check_eval_checkpoint.py --checkpoint "$CHECKPOINT"
python scripts/evaluate_paper_benchmarks.py \
  --benchmark depth --nyu-root "$NYU_ROOT" --split test \
  --model-name qwen-image-2.1 --lora-path "$CHECKPOINT" \
  --device cuda --dtype bfloat16 --num-inference-steps 50 \
  --guidance-scale 4 --strength .85 --seed 42 --resolution 512 \
  --batch-size 1 --native-resolution --depth-colormap cube_edge \
  --depth-curve metric_power --output-json "$OUT/eval_nyu.json"
```

First run `--preflight-only`, then an independent `--max-samples 1` smoke. The full
command has no sample cap. For ETH3D/DIODE use their roots and val split. Parallelize
with existing `--num-shards/--shard-index` and `merge_eval_shards.py`; never average
shard RMSE. GT/valid-pixel coverage, skipped/failed samples, weight identity and
protocol must match before accepting results.

Primary metrics: raw AbsRel, RMSE,δ1. Median-aligned metrics only diagnose structure.
Distance bins [0,1),[1,2),[2,3),[3,5),[5,10],>10m; median(pred)/median(GT), log-scale
bias, NaN/Inf, near-endpoint/off-palette outputs and boundary errors are planned
diagnostics, not fields this PR pretends already exist. First measure GT→RGB→decode
and the actual resize/crop path. Do not silently discard extreme predictions or
add a universal10m cap. Prediction-dependent masks need coverage reporting; any
metric-protocol fix requires both controls to be rerun and historical results separated.

No test-GT calibration. A single global scale fitted on independent calibration
images is a separate raw/calibrated experiment. Add a pre-registered external
KITTI/nuScenes evaluation only after verifying its exact split, data and protocol.
Keep all50k in training; if an additional RobbySimVal development pool is prepared,
audit source/scene overlap and format separately rather than calling the release a
train/val dataset. Pooled NYU statistics already informed data design; do not reuse
NYU test to optimize mixture ratios or choose checkpoints.

## Decision and reporting

Proposed pre-registered practical threshold: mean NYU raw AbsRel reduction≥5%,
direction consistent in at least2/3 training seeds, paired/scene-cluster bootstrap95%
CI of the difference excluding0; mean RMSE deterioration≤2%,δ1 loss≤1 percentage
point. These are design thresholds, not measured noise floors. Never treat pixels
as independent replicates. Report individual seeds and uncertainty; implement the
paired accumulator/bootstrap analysis before claiming statistical significance.

If ETH3D/DIODE AbsRel worsens>5%, qualify the result as NYU/indoor specialization.
If only aligned scores improve, metric scale remains unresolved. If raw improves
and aligned stays flat, scale-prior improvement is plausible but source causality
is not proven. M1 must additionally report any>2pt segmentation/referring loss.
Negative results and failed gates remain in the report.

Save effective config, code/base/adapter/data hashes, exposure attempts, checkpoint
metadata, full JSON/CSV, masks/coverage and predictions. Keep HF historical results,
server reruns, smoke, sampled and full results separate. Historical ~14s/sample
suggests ~5.6 GPU-hours for three depth datasets/model, or ~25.4 for five prepared
benchmarks (6,529 samples), excluding model load/Cityscapes/generation. This is a
planning estimate, not certified current throughput or a training-time estimate.
