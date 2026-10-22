# Datasets: one-command download and the path to a first training run

`scripts/download_datasets.sh` fetches every public source the paper-aligned recipe
needs, driven by the registry `configs/datasets/sources.tsv`. Adding a dataset is
adding one line to the TSV; the script itself does not change.

```bash
bash scripts/download_datasets.sh --list                       # what is registered
bash scripts/download_datasets.sh --group train --dry-run      # print the commands only
bash scripts/download_datasets.sh --group train                # everything for training
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
| `group` | `train` / `eval` / `replay`. **`eval` rows are paper benchmarks and must never be trained on**; the zero-shot audit in `build_qwen_baseline_manifest.py` rejects their paths. |
| `tasks` | subset of `depth, normal, semantic, instance, referring, generation` |
| `source` | `ms` (ModelScope), `hf`, `url`, `manual` (login wall: the script prints instructions and skips) |
| `include` | `;`-separated globs, or macros `@hypersim_scenes`, `@tartanair_chunks`, `@omniedit_shards` |
| `post` | `unzip`, `untar`, `*_each`, `coco2014_links`, chainable with `+` |

`coco2014_links` exposes COCO 2017 images under the `train2014/COCO_train2014_*.jpg`
names the RefCOCO `refer` files use (2017 is a re-split of 2014; image ids match).

## What each source is for

| name | tasks | why this one | next step |
|:--|:--|:--|:--|
| `hypersim` | depth, normal, semantic, instance | Synthetic indoor; one source covers four tasks. `depth_meters.hdf5` is **ray distance** — `prepare_hypersim.py` converts to planar z by default. | `scripts/prepare_hypersim.py --source data/raw/hypersim --output data/hypersim` |
| `tartanair` | depth, semantic | Synthetic outdoor / long range (planar z `.npy`); covers the KITTI / nuScenes domain without touching those benchmarks. | `scripts/prepare_tartanair.py` |
| `ade20k` | semantic | 150 natural class names, not an evaluated benchmark. | `scripts/build_openvocab_semantic_manifest.py` |
| `coco2017-images` + `coco2017-instances` | instance | Fig. 3 one-class-at-a-time targets. | `scripts/build_instance_manifest.py --format coco --exclude-image-ids data/raw/refer/refcocog/refs(umd).p` |
| `lvis-v1` | instance | 1203-class long tail on the same images. | `scripts/build_instance_manifest.py --format lvis ...` |
| `coco-stuff` | semantic | 171 thing + stuff classes on COCO images. | `build_openvocab_semantic_manifest.py` + `--exclude-image-ids` |
| `refcoco`, `refcoco+` | referring | Training expressions; images from `coco2017-images`. | `scripts/build_res_manifest.py` |
| `omniedit` | generation (replay) | Edit triplets for the editing backbones (Section 2 mixing). | `scripts/build_generation_replay_manifest.py --kind edit` |
| `blip3o-60k` | generation (replay) | T2I pairs; only for text-to-image backbones (the trainer rejects identity replay on editors). | `--kind t2i` |
| `refcocog`, `reasonseg`, `nyu-depth-v2-labeled`, `genai-bench` | eval | Paper Tables 4–7 / Appendix D. `refs(umd).p` doubles as the exclusion list. | `scripts/evaluate_paper_benchmarks.py` |
| `kitti-depth`, `cityscapes` | eval, manual | Login walls. Cityscapes **train** in the mixture breaks the zero-shot claim for Table 2. | `scripts/setup_cityscapes_official.py` |

Then assemble the training manifest:

```bash
python scripts/build_qwen_baseline_manifest.py \
    --base-manifest data/combined_pairs.jsonl --output data/qwen_baseline/pairs.jsonl \
    --data-root data --exclude-image-ids "data/raw/refer/refcocog/refs(umd).p"
PYTHONPATH=src python scripts/train.py --config configs/experiments/qwen_image_21_paper_aligned.yaml
```

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
