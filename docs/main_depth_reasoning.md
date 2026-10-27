# Main six-task extension: metric depth and reasoning references

Continue the **E1 six-task configuration with `vision_ratio=0.5`**, adding the
following training-only supplements. This is E4 data expansion, with fresh
optimizer/schedule state; it is not a new benchmark result.

| Addition | Before quality filtering | Retained training rows | Supervision |
| --- | ---: | ---: | --- |
| Hypersim filtered-v1 | 20,000 | **13,267** | Native RGB and planar metric depth, scene-disjoint from recorded E1 sources |
| Rea2Seg reasoning-v1 | 15,275 | **11,812** | Relations, reasoning and spatial references; the entire ReasonSeg category remains excluded |
| LLM-Seg COCO reasoning-v1 | 5,000 | **4,304** | Longer reasoning expressions, original RLE-derived masks |
| **Total increment** | **40,275** | **29,383** | **Depth 13,267 + referring 16,116** |

The initial 20k depth preparation is recorded, but only the quality-filtered
subset is published for training. A user-approved quality-first decision removes
high-error depth frames and images that can lose their referring target in a crop;
the release deliberately does not replace rejected frames to force the count to 20k.

Use the two published **supplement** manifests, not the complete R2 training
manifest: R2 would add the entire single-task RefCOCO-family base again.
Normal, semantic, instance and generation replay remain in the main mixture.
The [release record](main_depth_reasoning_release.json) records the actual data,
source pins, fingerprints and validation results. No GPU training or accuracy
improvement is claimed by this PR.

## Why extend E1

The four supplied evaluations use matching dataset fingerprints and inference
settings: 50 steps, CFG 4, seed 42, 512px inference and native-resolution scoring.
Their reported results are:

| Evaluation | Cityscapes mIoU ↑ | RefCOCOg cIoU ↑ | ReasonSeg gIoU ↑* | NYU δ1 ↑ | DIODE δ1 ↑ | ETH3D δ1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E2, ratio 0.2, eval_10000 | .2484 | .3765 | .3004 | .1196 | .0828 | .1749 |
| E2, ratio 0.8, eval_10000 | .2957 | .3842 | .2896 | .0521 | .0597 | .1418 |
| E1, ratio 0.5, eval_10000 | .3307 | .3371 | .2659 | .0425 | .0654 | .0397 |
| E1, ratio 0.5, eval_15000 | .2938 | .4095 | .3013 | .1640 | .1410 | .0912 |

Sources: [E2 0.2](https://huggingface.co/khazic/ovb-e2_vr02_s42_20261026_1708/tree/16dd068e0ce8708a52c22bcb30a285e418907046/eval_10000),
[E2 0.8](https://huggingface.co/khazic/ovb-e2_vr08_s42_20261026_1837/tree/c2d05776c7be4fc3ae90ce5527a6c0905b2e49f1/eval_10000),
[E1 10k](https://huggingface.co/khazic/ovb-e1_qwen21_paper_aligned_20261025_0250/tree/f32ca2ebfb1bd81f9401efc3b4e89c2ba6b0dca7/eval_10000),
[E1 15k](https://huggingface.co/khazic/ovb-e1_qwen21_paper_aligned_20261025_0250/tree/f32ca2ebfb1bd81f9401efc3b4e89c2ba6b0dca7/eval_15000).
*ReasonSeg remains diagnostic under its recorded query protocol.*

E1's later evaluation is the strongest candidate here for referring and
NYU/DIODE depth, but not for Cityscapes or ETH3D. The E2 schedules end at 10k
whereas E1 schedules 30k; these are not isolated ratio comparisons at equal
learning-rate histories. Normal, instance and generation-preservation results
are also missing. Keep the balanced ratio and expand the weak data pools;
do not declare a universal winner or attribute the change solely to ratio.

The evaluation metadata points to `_best` checkpoints, while the pinned Hub
releases expose different final online artifacts. The trainer's
`--resume-weights-only` loads the **online `lora_state_dict`**, even from a best
directory; it does not select the validation EMA adapter. Before continuing,
identify and hash the actual adapter to load and re-evaluate it. Do not label
the published E1 30k online adapter as the evaluated 15k best/EMA adapter.

## Hypersim selection and existing data contract

The source is the [official Hypersim release](https://github.com/apple-aiml-research/ml-hypersim/tree/3463c5c4a75f3cbfc65ed31cfd6e87204b3a2254).
The split CSV is pinned by revision and SHA256. E1's published download log
identifies 56 Hypersim scenes, all in volumes 1–10. This supplement excludes
**all ten volumes**, uses only public frames in the official scene-level
training partition, and initially considers even frame numbers only.

Candidate order is deterministic: seed-42 SHA256 ordering within scenes and
round-robin selection across scenes. Require at least 50% valid depth pixels;
reject exact decoded-RGB duplicates and frames with mean quantized-depth
roundtrip error above 5 cm. Replace rejected frames with the next candidates
until exactly 20,000 are accepted. The complete release must still pass the
existing **mean encoded-depth error < 1 cm** gate; the threshold is not relaxed.
All rejections are recorded. Network failures abort/resume the same selection,
rather than changing the dataset according to download success.

The initial 20k preparation covers 271 scenes and 461 camera trajectories.
Of 22,227 processed candidates, 2,207 exceeded the per-frame **mean** 5 cm
error limit, 19 had too few valid depth pixels, and one duplicated decoded RGB.
Although its overall mean error was 7.19 mm, a distant tail reached 53.36 m
at a single pixel. That initial package is **not the published training release**.

The final filter rejects another **6,733 frames**, requiring the maximum
absolute encoding error over **every valid native pixel in each frame to be
at most 5 cm**. It retains **13,267 frames** and keeps the existing overall
mean <1 cm gate. Float32 metre-depth arrays are retained without clipping or
replacement. The release JSON records the resulting statistics and fingerprints.
The retained data covers **243 scenes and 406 camera trajectories**; encoding
MAE is **4.56 mm**, maximum error **4.994 cm**, and native metric depth ranges
from 0.0203 to 46.9769 m. The 39,801 native assets total 46,761,936,860 bytes.
The initial and final frame identities are recorded in
[`hypersim_depth20k_v1.frames.json`](../configs/datasets/hypersim_depth20k_v1.frames.json)
and [`hypersim_filtered_v1.frames.json`](../configs/datasets/hypersim_filtered_v1.frames.json).

“Independent frames” means distinct source `(scene, camera, frame)` identities,
distinct decoded RGB, and no adjacent frame numbers from the same camera.
It does not establish statistical independence or eliminate perceptually
similar views. Quantization filtering can reduce distant-scene coverage; the
release reports the actual scenes, depth statistics and error distribution.

Each row uses the existing `task/input/target/prompt/split` JSONL contract and
manifest-relative paths. Native JPEGs are retained without re-encoding. Raw
Hypersim HDF5 depth is **ray distance**, converted with the existing
`ray_distance_to_planar_depth` helper to planar z in metres. Native float32 NPYs
retain the metric depth and NaN invalid pixels. Targets use the existing
`DepthRGBMapper(metric_power, cube_edge, λ=-3, c=10/3)` and quantized RGB PNGs.
The merger now also rebases and checks the optional `metric_depth` path.

No alternate dataset, runtime loader, loss or referring protocol is introduced.
Reasoning supplements retain `binary_white_black_v1`, expressions, native masks
and their published source annotations. Answer/COT metadata is not appended to
the conditioning prompt. See [reasoning sources](res_reasoning_sources.md).

The initial 20,275 native referring masks are binary, nonempty and
dimension-matched; nevertheless, one seeded crop pass produced 936 empty targets.
The final filter checks **every integer crop position** allowed by the existing
512px aspect-preserving loader, after exactly matching its nearest-neighbour
resize and rounding. A sliding foreground sum along the long axis includes
thin resized targets, disconnected masks and extreme crops; it is not a random
sample of crop positions.

If any annotation can become empty, exclude the whole source image across both
reasoning sets, using decoded RGB and COCO identity propagation. This removes
3,463 Rea2Seg and 696 LLM-Seg rows, including other annotations of an unsafe image.
The pinned [filter index](../configs/datasets/reasoning_crop_filter_v1.json)
records every excluded sample ID and retained manifest fingerprint. The runtime
loader and crop distribution are unchanged. The final full-loader validation
requires **zero empty referring targets**. These guarantees apply to this
512px crop/resize policy; regenerate the filter if that policy changes.

## Download and merge

Install `pip install -e '.[data,training]'`. The
[published depth repository](https://huggingface.co/datasets/Melo111/open-vision-banana-depth)
uses the existing verified bundle downloader; allow approximately 90 GB for its
compressed bundle and restored depth assets, plus the reasoning sources.
Published Hub revision: [`9f56e9c8d6a05f4e9a512cdc9de5231bbda0628b`](https://huggingface.co/datasets/Melo111/open-vision-banana-depth/tree/9f56e9c8d6a05f4e9a512cdc9de5231bbda0628b).
The full bundle and publication metadata were verified before anonymous access
was enabled. Use this immutable revision from the release record:

```bash
DEPTH_REV="$(python -c "import json; print(json.load(open('docs/main_depth_reasoning_release.json'))['publication']['revision'])")
python scripts/download_res_dataset.py --repo-id Melo111/open-vision-banana-depth \
  --revision "$DEPTH_REV" --subfolder hypersim-filtered-v1 \
  --output-dir data/prep/hypersim_filtered_v1

python scripts/download_res_dataset.py --repo-id Melo111/open-vision-banana-res \
  --revision e479f7013cc72ff355d60fdcbeb18b78e395bee5 \
  --subfolder supplements/reasoning-v1/rea2seg --output-dir data/prep/rea2seg
python scripts/download_res_dataset.py --repo-id Melo111/open-vision-banana-res \
  --revision e479f7013cc72ff355d60fdcbeb18b78e395bee5 \
  --subfolder supplements/reasoning-v1/llmseg-coco5k --output-dir data/prep/llmseg

hf download Melo111/open-vision-banana-res evaluation_image_ids.json \
  --repo-type dataset --revision e479f7013cc72ff355d60fdcbeb18b78e395bee5 \
  --local-dir data/res_binary_v1

PYTHONPATH=src python scripts/merge_main_depth_reasoning.py \
  --hypersim data/prep/hypersim_filtered_v1/pairs.jsonl \
  --rea2seg data/prep/rea2seg/training.jsonl \
  --llmseg data/prep/llmseg/training.jsonl \
  --exclude-image-ids data/res_binary_v1/evaluation_image_ids.json \
  --output data/depth_reasoning_supplement/pairs.jsonl
```

The last command applies the pinned referring exclusions to the original
published manifests and produces the **29,383-row train-only increment**.
It records deliberate quality exclusions separately from accidental missing
assets or merger drops, which remain errors. To build the
actual six-task training manifest, add
`--base-manifest data/qwen_baseline/pairs.jsonl` and change the output to
`data/qwen_main_depth_reasoning/pairs.jsonl`. Use the actual E1 base, with its
original assets and held-out splits. The script verifies pinned supplement
identities, counts and files, forbids source benchmark rows in the increment,
checks known held-out COCO IDs, and, for a main join, checks exact decoded RGB
and COCO IDs across **all tasks** for train/held-out overlap. It also checks the
new Hypersim scenes against that actual base. It never silently drops rows,
caps sources or reassigns validation; the original held-out order is retained.

The recorded E1 base audit contains **262 internal train/validation COCO
overlaps**. Resolve these in the base first and evaluate the control on the
same corrected splits; do not bypass the merge failure or compare against a
newly reshuffled validation set. Exact hashes/IDs do not detect arbitrary crops
or lossy re-encodings. Reasoning-v1's original full official held-out exclusion
audit remains part of the release provenance.

The existing path-only benchmark detector does not inspect `source_dataset`.
Consequently its historical `zero_shot_claim_allowed=true` is insufficient
evidence for the complete E1 mixture. This extension is explicitly recorded as
**supervised main training**, with no strict zero-shot claim. Source-aware
benchmark exclusion in the new increment and image-level held-out checks are
still enforced. Keep the existing unit/metric contract checks and trainer
preflight enabled. The increment itself has no validation split and is not a
standalone six-task training manifest.

For an audited six-task main join, run the existing gate with the supervised
protocol explicitly selected; this retains the metric, split and loader checks:

```bash
PYTHONPATH=src python scripts/check_experiment_gates.py data \
  --merged-audit data/qwen_main_depth_reasoning/pairs.audit.json \
  --final-manifest data/qwen_main_depth_reasoning/pairs.jsonl \
  --hypersim-manifest data/prep/hypersim_filtered_v1/manifest.json \
  --model-name qwen-image-2.1 --allow-in-domain
```

Do not use `--allow-in-domain` to relabel this as strict zero-shot training.
The train-only increment intentionally cannot pass the main training split gate.
The historical E1 assets were unavailable on the validation server; the release
therefore validates the full increment and the join logic, not an actual rebuilt
six-task E1 manifest. Reproduce the full increment's actual loader check with:

```bash
PYTHONPATH=src python scripts/validate_depth_reasoning_loader.py \
  --manifest data/depth_reasoning_supplement/pairs.jsonl \
  --output data/depth_reasoning_supplement/loader_validation.json
```

## Rebuild from the original source

For an independent source rebuild, reserve space for the initial 20k preparation
as well as the filtered output. The downloader reads selected RGB/depth members
using HTTP ranges, without fetching complete multi-terabyte scene ZIPs. After
downloading the pinned reasoning sources above, run:

```bash
PYTHONPATH=src python scripts/prepare_hypersim_supplement.py \
  --output data/prep/hypersim_initial20k --count 20000 --workers 8 \
  --verify-release docs/main_depth_reasoning_release.json

PYTHONPATH=src python scripts/filter_depth_reasoning_supplement.py \
  --hypersim data/prep/hypersim_initial20k \
  --depth-output data/prep/hypersim_filtered_v1 \
  --rea2seg data/prep/rea2seg/training.jsonl --llmseg data/prep/llmseg/training.jsonl \
  --ref-filter configs/datasets/reasoning_crop_filter_v1.json \
  --ref-ledger data/prep/reasoning_crop_filter_ledger.jsonl

```

The checked-in initial/final frame indexes, source and manifest fingerprints,
and referring filter checksum make the resulting selection auditable. Packaging
recomputes native asset hashes, depth encoding and quality gates before creating
the portable bundle; the initial 20k output remains separate.

## Verified scope

All **39,801 native assets** passed checksum, shape, dtype and depth-encoding
checks; a complete local bundle restoration passed. The actual training loader
processed all **29,383 rows** with the configured 512px augmentations: zero empty
referring targets and zero nonfinite tensors. The referring crop audit also
exhaustively covers every permitted crop, beyond that single seeded loader pass.
Six retained depth examples were visually inspected for RGB/geometry alignment.

The 35 focused regression tests pass on Linux and Windows. The complete Linux
suite has 1,725 passes and four failures, matching the four failures on the
unmodified base (1,705 passes). Those failures are in the existing generation
shard test double's `guidance_scale` interface. Windows has 1,712 passes and the
same 32 failures as its base comparison, including platform/dependency limits.
Changed-file lint/format and generic pre-commit hooks pass; mypy still reports
33 errors in existing imported modules. Global Windows coverage is 71.14%, below
the repository's 80% gate; focused affected-source coverage is 86.60%. These
results do not claim that whole-repository CI is green.

GitHub Actions did not start for the initial PR commit because of the repository
account's billing/spending limit, as recorded in the
[runner annotation](https://github.com/isLinXu/open-vision-banana/actions/runs/40560961309/job/122389515118).
The local and server validation above is the available execution evidence.

The increment-only main gate reports **9 passing checks and 12 split failures**:
the original four other tasks and held-out splits must come from the audited
main base. This is not a ready-to-launch six-task dataset or a model evaluation.

## Training recommendation and evaluation

Use [`qwen_image_21_main_depth_reasoning.yaml`](../configs/experiments/qwen_image_21_main_depth_reasoning.yaml):
Qwen-Image-2.1, 512px, rsLoRA rank32/alpha64, six tasks, `vision_ratio=0.5`,
uniform sampling among the five vision tasks, CFG dropout .05 and EMA .9999.
Continue for an initial **10,000 optimizer steps at 3e-5**, cosine schedule with
500 warmup steps, global batch32, checkpoint/evaluate every2,500 steps. These
optimizer choices are proposals, not parameters recovered from the paper.

```bash
# E1_ONLINE_CHECKPOINT must be an identified, hashed, re-evaluated online adapter.
PYTHONPATH=src torchrun --nproc_per_node=8 scripts/train.py \
  --config configs/experiments/qwen_image_21_main_depth_reasoning.yaml \
  --grad-accum 4 --resume "$E1_ONLINE_CHECKPOINT" --resume-weights-only
```

The YAML defaults to accumulation32 for one GPU; with eight GPUs use4, keeping
`1 × accumulation × GPU count = 32`. A changed distribution/LR requires weights
only, not optimizer-state resume. Do not launch until the actual main manifest,
source adapter and existing contract checks have passed.

Before any necessary base decontamination, the recorded E1 training counts
would change as follows:

| Task | Recorded E1 | With increment |
| --- | ---: | ---: |
| depth | 47,585 | 60,852 |
| referring | 33,009 | 49,125 |
| normal | 8,680 | 8,680 |
| semantic | 39,420 | 39,420 |
| instance | 39,153 | 39,153 |
| generation | 16,492 | 16,492 |
| **Total** | **184,339** | **213,722** |

These are expected counts, not a claim that the full historical main manifest
has been reconstructed. Record post-audit counts in each run. Sampling remains
50% generation and 10% per vision task: new depth is approximately 21.80% of
the depth pool (2.18% of all draws), and new reasoning is 32.81% of referring
(3.28% of all draws). At 10k steps and global batch32, a new depth row receives
only about0.53 expected draws. Do not infer full coverage or convergence from
the enlarged row count.

For attribution, compare equal-budget continuations from the **same verified
adapter**: unchanged corrected base, +depth, +reasoning, and +both (the proposed
main run). Freeze evaluation inputs, query protocols and inference settings;
report NYU/DIODE/ETH3D AbsRel and δ1, RefCOCOg and diagnostic ReasonSeg, plus
Cityscapes, normal, instance and generation-preservation checks. Inspect depth
saturation and metric distributions as well as averages. Extend the budget
only after the combined run shows useful gains without unacceptable regressions;
add a second seed before a main experimental conclusion.

Source data retains the official Hypersim, COCO and annotation-source terms.
This repository's code license does not relicense those assets.
The Hypersim derivative is released under the source dataset's
[CC BY-SA 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/), with
attribution to Mike Roberts and the Hypersim authors (ICCV 2021).
