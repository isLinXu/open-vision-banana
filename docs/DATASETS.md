# Datasets: one-command download and the path to a first training run

`scripts/download_datasets.sh` fetches every public source the paper-aligned recipe
needs, driven by the registry `configs/datasets/sources.tsv`. Adding a dataset is
adding one line to the TSV; the script itself does not change.

```bash
bash scripts/download_datasets.sh --list                       # what is registered
bash scripts/download_datasets.sh --group train --dry-run      # print the commands only
bash scripts/download_datasets.sh --group train                # vision-task training sources (not replay)
bash scripts/download_datasets.sh --only omniedit              # generation replay, fetched by name
bash scripts/download_datasets.sh --group eval                 # paper benchmarks (eval only)
bash scripts/download_datasets.sh --only hypersim,omniedit     # a subset by name
bash scripts/download_datasets.sh --task depth                 # a subset by task
```

Requirements: `bash`, `curl`, `unzip`, `tar`, and for ModelScope sources
`pip install modelscope` (the script uses the CLI if present, otherwise the Python
API); for Hugging Face sources the `hf` / `huggingface-cli` tool. `HF_ENDPOINT`
defaults to `https://hf-mirror.com`. `OpenDataLab/*` datasets on ModelScope need a
logged-in account (`modelscope login` or `MODELSCOPE_TOKEN`).

Size knobs (environment variables): `HYPERSIM_SCENES` (default 8 scenes, ~25 GB),
`TARTANAIR_CHUNKS` (default 2, ~6 GB), `OMNIEDIT_SHARDS` (default 4, ~20 GB),
`DATA_ROOT` (default `data/raw`), `SKIP_EXTRACT=1` to keep archives only. Archives
are kept and marked `*.extracted`, so re-running is idempotent and resumable.

## Registry columns

| column | meaning |
|:--|:--|
| `group` | `train` / `eval` / `replay`. **`eval` rows are paper benchmarks and must never be trained on**; the zero-shot audit in `build_qwen_baseline_manifest.py` rejects their paths. `replay` rows are fetched by name (`--only omniedit`): `--group replay` would also pull ultraedit (1.4 TB) and blip3o-60k (18 GB). |
| `tasks` | subset of `depth, normal, semantic, instance, referring, generation` |
| `source` | `ms` (ModelScope), `hf`, `url`, `manual` (login wall: the script prints instructions and skips) |
| `include` | `;`-separated globs, or macros `@hypersim_scenes`, `@tartanair_chunks`, `@omniedit_shards` |
| `post` | `unzip`, `untar`, `*_each`, `coco2014_links`, `refer_umd` / `refer_unc`, chainable with `+` |

`coco2014_links` exposes COCO 2017 images under the `train2014/COCO_train2014_*.jpg`
names the RefCOCO `refer` files use (2017 is a re-split of 2014; image ids match).

## What each source is for, and the script that turns it into training rows

| name | tasks | preparation (all write manifest-relative `pairs.jsonl`) |
|:--|:--|:--|
| `hypersim` | depth, normal, semantic, instance | `python scripts/prepare_hypersim.py --source data/raw/hypersim --output data/prep/hypersim --tasks depth,normal,semantic,instance` — ray distance → planar z, `normal_cam` → paper frame (back-facing check), NYU40 → Fig. 2 open-vocab prompts, `semantic_instance` → one Fig. 3 target per class |
| `tartanair` | depth | `python scripts/prepare_tartanair.py --source data/raw/tartanair --output data/prep/tartanair` |
| `ade20k` | semantic | `python scripts/build_openvocab_semantic_manifest.py --images data/raw/ade20k/ADEChallengeData2016/images/training --labels .../annotations/training --classes configs/ade20k_150_classes.json --source-name ade20k --label-offset 1 --ignore-index -1 --out-dir data/prep/ade20k` |
| `coco-stuff` | semantic | same script with `--labels data/raw/coco_stuff/train2017 --classes configs/coco_stuff_182_labels.json --label-offset 0 --ignore-index 255 --source-name coco_stuff` (11 unused ids are `""` in the list and ignored) |
| `coco2017-images` + `coco2017-instances` | instance | `python scripts/build_instance_manifest.py --annotations data/raw/coco2017/annotations/instances_train2017.json --images-dir data/raw/coco2017/train2017 --targets-dir data/prep/instance_coco/targets --output data/prep/instance_coco/instance.jsonl --data-root data/prep/instance_coco --exclude-image-ids "data/raw/refer/refcocog/refs(umd).p"` |
| `lvis-v1` | instance | same with `--format lvis --annotations data/raw/lvis/lvis_v1_train.json --images-dir data/raw/coco2017` |
| `refcoco`, `refcoco+`, `refcocog` | referring | `python scripts/build_res_manifest.py --annotations-root data/raw/refer --image-root data/raw/coco2017/train2014 --output-dir data/prep/res` — RefCOCOg val/test images are excluded from every train pool; the builder also emits the official eval rows |
| `omniedit` | generation (replay) | `python scripts/build_generation_replay_manifest.py --kind edit --preset omniedit --records data/raw/omniedit/data --images-dir data/prep/replay/images --data-root data/prep/replay --output data/prep/replay/omniedit.jsonl` |
| `blip3o-60k` | generation (replay, T2I backbones only) | `... --kind t2i --records data/raw/blip3o_60k/journeyDB.tar --captions-file data/raw/blip3o_60k/journeyDB.txt --images-dir ...` |
| `nyu-depth-v2-splits` + `nyu-depth-v2-labeled` | eval | post step `nyu_mat` writes `test/images` + `test/depths` (Eigen 654) for `NYUDepthDataset` |
| `genai-bench` | eval | post step `genai_prompts` writes `prompts.jsonl` for `GenAIBenchDataset` |
| `refcocog`, `reasonseg` | eval | `RefCOCOgDataset(protocol="umd")`, `ReasonSegDataset` read the downloaded layouts directly |
| `kitti-depth`, `cityscapes` | eval, manual | `scripts/prepare_kitti_metric.py`, `scripts/setup_cityscapes_official.py` after a manual download |

Then merge (paths are rebased to the merged file; tasks without a val split get
`--val-percent`), run the zero-shot / metric-depth audit, and train:

```bash
python scripts/merge_multitask_manifest.py \
    --inputs data/prep/hypersim/pairs.jsonl data/prep/tartanair/pairs.jsonl data/prep/ade20k/pairs.jsonl \
             data/prep/instance_coco/instance.jsonl data/prep/res/pairs.jsonl data/prep/replay/omniedit.jsonl \
    --out data/merged/pairs.jsonl --val-percent 2
python scripts/build_qwen_baseline_manifest.py \
    --base-manifest data/merged/pairs.jsonl --output data/qwen_baseline/pairs.jsonl \
    --data-root data/merged \
    --exclude-image-ids "data/raw/refer/refcocog/refs(umd).p"
PYTHONPATH=src python scripts/train.py --config configs/experiments/qwen_image_21_paper_aligned.yaml \
    --pair-manifest data/qwen_baseline/pairs.jsonl --data-dir data/qwen_baseline
```

`tests/test_pipeline_smoke_all_tasks.py` runs this exact chain on tiny fixtures in
the downloaded layouts and asserts that all six paper_aligned tasks come out of the
Trainer's `ManifestVisionDataset` + `TaskWeightedSampler` as batches.

## Sources surveyed and rejected

| source | reason |
|:--|:--|
| `SenseNova/SenseNova-Vision-Corpus-50M` | 406 split tar volumes; relative-depth pseudo-labels mixed with metric; cleaning cost exceeds the value of the subsets we need. The tar indexer (`scripts/build_snvc_tar_index.py`) stays for anyone who wants it. |
| `shimin2023/NYU_Depth_V2`, `Damo_XR_Lab/KITTI_Depth_Dataset` (ModelScope) | Index JSON files only, no images. |
| `AI-ModelScope/HyperSim-Absolute-Camera` | Camera parameters only (27 MB), no images or depth. |
| `AI-ModelScope/laion-high-resolution` | URL + caption parquet, no images. |
| `AI-ModelScope/text-to-image-2M` | ~60 % of images are Flux-dev generations; replaying them pulls the Qwen prior towards Flux. |
| `AI-ModelScope/ImgEdit` (train) | It is also the paper's generation benchmark (Appendix D); training on it changes the meaning of ImgEdit-Bench scores. Listed nowhere on purpose. |
| `thomas/SA-1B` | Class-agnostic masks; no category names for the Fig. 3 prompt. |
| `OpenDataLab/Objects365` | Boxes only. |
| `SOTA507/Mapillary_Vistas` | Login wall and tiny mirror; fetch from mapillary.com manually if a street-scene semantic source other than Cityscapes is wanted. |

## SA-Co/Gold (paper Table 3) — annotations on ModelScope, images elsewhere

`facebook/SACo-Gold` is mirrored on ModelScope **without a gate** (verified 2026-10-23: 474 MB, 24 files,
anonymous download through `--only saco-gold`): 7 subsets × 3 annotator passes, COCO-style JSON with RLE
masks, 168,202 image-NP pairs per pass of which ~84 % are negatives — matching the paper's description.
Licence: SAM License (research use; redistribution terms apply to anything derived from it).

The **images are not in the repository**. `file_name` points into MetaCLIP (6 subsets) and SA-1B (1 subset)
folders that must be fetched separately (`saco-gold-images`, manual): MetaCLIP from Roboflow Universe
(`sa-co-gold/gold-metaclip-merged-a-release-test`, free account) and SA-1B from Roboflow or Meta's
`sa_co_gold.tar` dynamic link (form). Put them under `data/raw/saco_gold/images/` so the annotation paths
resolve, then `scripts/prepare_saco_gold_queries.py`.

Protocol note: the official evaluator scores each prediction against all three annotator passes and keeps
the most favourable (oracle); `prepare_saco_gold_queries.py` currently exports pass `a` only — see the
tracking issue before reporting cgF1 as protocol-faithful.

## Canary run (2026-10-22)

A smoke test against the live sources with a 10 GB disk budget. Findings folded into the registry:

| source | result | action |
|:--|:--|:--|
| ModelScope (`swift/refcocog`, `anyforge/lvis` val json) | OK, 2–5 MB/s, CLI and Python-API paths both exercised | — |
| `bvisionweb1.cs.unc.edu` (REFER zips) | DNS no longer resolves | replaced by the parquet mirrors + `convert_refer_parquet.py`; Wayback copy exists but downloads at ~10 KB/s from CN |
| `data.csail.mit.edu/.../ADEChallenge/` | 301 → 404 | path is now `.../ADEchallenge/` (lower-case c), https |
| `calvin.inf.ed.ac.uk` (COCO-Stuff maps) | no response | maintainer mirror `calvin-vision.net` (HTTP 200, 659 MB) |
| `images.cocodataset.org`, `horatio.cs.nyu.edu` | HTTP 200 | — |
| `OpenDataLab/*` on ModelScope | not exercised (needs `modelscope login`) | — |

`curl` now uses `--retry-all-errors --retry-connrefused` and aborts stalls below 1 KB/s for 60 s, so a dead mirror fails in about a minute instead of hanging.
