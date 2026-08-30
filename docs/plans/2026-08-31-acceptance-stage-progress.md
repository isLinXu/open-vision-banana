# Open-Vision-Banana Acceptance Stage Progress

Date: 2026-08-31

This document records the execution evidence for the acceptance order. It is
not a model-quality claim for the paper numbers.

## Data And Preflight

The locally available, benchmark-shaped data is:

| Benchmark | Split | Samples | Fingerprint | Status |
| --- | --- | ---: | --- | --- |
| NYU depth | val/test export | 654 | `fa5664dace3f625fe58e23d2fdbc30207089d6514b87d54ab4257fbd6b5edb17` | available |
| Cityscapes semantic | val | 50 | `42f49c9ea07dde199dcf2d54ba191dcac2d7e3dfa41d31060096079f9eff37c4` | available |
| NYU normal | val | 0 | n/a | blocked: raw H5 contains only `rgb` and `depth` |

The other requested depth/normal roots (iBims, ETH3D, DIODE, KITTI,
ScanNet, VKITTI) are not present locally. The all-benchmark preflight therefore
remains intentionally failing instead of silently reducing the benchmark.

Preflight artifacts:

- `output/acceptance_stage/nyu_depth_preflight.json`
- `output/acceptance_stage/cityscapes_semantic_preflight.json`
- `output/acceptance_stage/normal_preflight.json`

## Specialist Evidence

### Depth

The existing Hypersim metric-depth specialist is
`output/sana_hypersim_metric_depth_lr1e5_20260822/checkpoint-500/lora_adapter`.
The current evaluator was run on NYU val n=32 with 20 SANA steps, float32 MPS,
and hard-fail non-finite handling:

- `output/acceptance_stage/depth_specialist_nyu_n32_strict.json`
- 32/32 samples, `skipped_samples=0`
- AbsRel `0.4367`, delta-1.25 `0.3578`, aligned delta-1.25 `0.5432`

An earlier complete n=654 report exists at
`output/sana_hypersim_metric_depth_lr1e5_20260822/evals/nyu_delta1_step500_capped_fullval.json`
with delta-1.25 `0.4030` and zero skipped samples. Its fingerprint predates the
relative-path manifest contract, so it is retained as legacy evidence and is
not relabeled as a current-fingerprint acceptance result.

The planned indoor/outdoor replay was also executed through its first gate
from the same baseline, using 1,200 metric-depth training pairs (1,000
Hypersim + 200 KITTI). Step 100 produced
`output/sana_depth_replay_from_hypersim500_20260830/evals/nyu_delta1_n32_step100.json`:
delta-1.25 `0.2859`, `skipped_samples=0`. This is below the configured early-stop
threshold `0.33`, so the runner stopped and retained the original Hypersim-500
checkpoint as the best depth baseline. The replay is a negative result, not a
candidate for shared-LoRA promotion.

### Semantic

The Cityscapes-19 specialist checkpoint is
`output/sana_cityscapes19_specialist_lr1e5_20260822/checkpoint-500/lora_adapter`.
The run used all 376 training pairs and was intentionally stopped at step 500;
the 50-image current full-val evaluation is:

- `output/acceptance_stage/semantic_specialist_cityscapes_fullval_strict.json`
- mIoU `0.0300`, pixel accuracy `0.0849`, valid pixels `11,864,004`
- no samples were skipped and the JSON contains no non-finite values

This is above the historical near-zero result but far below a usable semantic
specialist. Shared-LoRA training must not start from this checkpoint.

### Normal

No official normal target data is present. The NYU raw files expose only
`rgb`/`depth`; deriving normals from depth would not be an official normal
benchmark and is not accepted for this gate. No normal specialist is promoted.

## Hard-Fail And API Checks

An explicit MPS fp16 compatibility run on Cityscapes produced a non-finite
generator output at evaluation batch 16 and terminated immediately. It did not
write partial metrics. The default evaluator remains float32 on MPS, and the
new `--allow-mps-reduced-dtype` flag is opt-in only; it never enables NaN/Inf
skipping. The pre-existing depth replay runner now passes this flag explicitly
for its declared fp16 compatibility gates; its gate still fails hard if a
non-finite output appears.

The SANA batch API and depth benchmark mask broadcasting were fixed and covered
by regression tests. Full test suite at this stage: `1031 passed, 15 skipped,
20 warnings`.

## Promotion Decision

Depth has a viable specialist baseline but needs a current-fingerprint full-val
rerun when compute is available. Semantic is not yet usable, and normal is
data-blocked. Consequently the shared LoRA stage and generation replay from a
new shared checkpoint are **not started**. The next valid transition is:

1. Acquire and audit official normal targets plus missing benchmark roots.
2. Rework semantic specialist capacity/target calibration and rerun its full
   19-class gate.
3. Re-run current-contract depth full val, then train shared LoRA only after all
   specialist gates pass.
