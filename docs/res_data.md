# Reproducible referring expression segmentation data

This release converts the local official RefCOCO, RefCOCO+, and RefCOCOg annotations into expression-conditioned RES training pairs. Each record contains one image, one human expression, and that expression's native binary mask. Images and masks are shared across records when possible.

The public dataset is [Melo111/open-vision-banana-res](https://huggingface.co/datasets/Melo111/open-vision-banana-res). Use the immutable release commit `dce183e8441c55c8fcf58afd87f90c776024c828` with the download command below. The repository contains code and configuration; the Hub dataset contains manifests, audit reports, copied source images, and target masks in checksum-verified tar shards.

## Data sources and partitions

The build reads `refs(unc).p` for RefCOCO and RefCOCO+, `refs(umd).p` for RefCOCOg, and each source's `instances.json`. COCO `train2014` supplies the shared images. These are the official [REFER dataset/API formats and split protocols](https://github.com/lichengunc/refer). PLVL's [data preparation and REC/RES training instructions](https://github.com/jcwang0602/PLVL) informed the local data audit; its mixed `.pth` caches are not required by this converter.

| Source | Protocol | Original train expressions |
| --- | --- | ---: |
| RefCOCO | UNC | 120,624 |
| RefCOCO+ | UNC | 120,191 |
| RefCOCOg | UMD | 80,512 |
| Total | | 321,327 |

The original training sum is not the released training count. The builder first collects the union of **all three sources' official validation and test image IDs**. It excludes those images from every training source, then removes exact duplicate training records by `(image_id, annotation geometry, expression)`. Geometry includes the bounding box and segmentation JSON; expressions use `raw` verbatim, falling back to `sent` only when `raw` is absent. Case folding and paraphrasing are not applied. Duplicate provenance is retained in `train_duplicates.jsonl`.

The final release has **270,195 training expressions**, **26,488 validation expressions**, and **30,969 test expressions**. Validation and test retain every official expression, including repeated text. The common `split` field maps `testA` and `testB` to `test`, while `source_split` preserves the original subset for separate evaluation. No image-level or expression-level deduplication is applied to official evaluation rows.

Training accounting is `321,327 - 33,723 held-out-image expressions - 17,409 exact duplicates = 270,195`. The released training source counts are 102,698 RefCOCO, 95,501 RefCOCO+, and 71,996 RefCOCOg expressions. The full corpus shares 30,956 images and 70,327 native masks; the training pool contains 24,407 distinct images. Source attribution is assigned deterministically for duplicates, and the duplicate log retains the suppressed record's identity.

The global exclusion is necessary because an image can be in one source's training partition and another source's evaluation partition. Checking each source independently would miss that leakage. Validation checks both COCO image identity and exact image-file content across the final train/evaluation boundary; it does not perform perceptual near-duplicate detection.

Cross-source validation/test overlap also exists: 123 UNC validation images occur in UMD test, and 48 UMD validation images occur in UNC test. Exact byte comparison catches additional duplicates across distinct IDs. `training.jsonl` therefore includes all 270,195 released training rows plus 24,554 validation rows whose image IDs and exact image bytes occur in no source's test set, excluding 1,934 validation rows from this view. Use it for checkpoint selection. The official `pairs.jsonl`, `val.jsonl`, `test.jsonl` and per-source evaluation manifests remain complete; they are for reporting the specified official benchmark, not pooled checkpoint selection.

Flickr30K Entities and Visual Genome are not included in this release: the audited local material does not provide verified native RES masks for the intended additional records. A region box or an image caption is insufficient mask supervision. Adding box-conditioned pseudo-masks would require a separately identified, quality-controlled release.

## Mixed-seg / `instances_nogoogle.json` compatibility

The additional `instances_nogoogle.json` is valid referring segmentation supervision, but it is an alternative serialization of the same source data, not extra annotations. The [C3VG merge script](https://github.com/Dmmm1997/C3VG/blob/main/tools/data_process/merge_mask_dataset.py) combines RefCOCO UNC, RefCOCO+ UNC and RefCOCOg UMD, excluding the Google protocol. Its [loader](https://github.com/Dmmm1997/C3VG/blob/main/c3vg/datasets/pipelines/loading.py) resolves `image_id` to the same `COCO_train2014_<12-digit-id>.jpg` images and decodes the `mask` polygons/RLE.

The JSON's top-level keys are `train` and eight named official evaluation subsets. Each row contains `image_id`, `width`, `height`, a COCO XYWH `bbox`, a zero-based contiguous `category_id`, a `mask` and an `expressions` list. Expand the list conceptually to obtain one expression per mask; it is not a COCO `images`/`annotations` document and is not directly a training JSONL manifest.

The supplied file has SHA256 `b5760bc94c56840929129a3273f84638915c79f062ad242b47a64b87cc95f9a7`. All **378,784 expressions** (321,327 train, 26,488 validation, 30,969 test) exactly match the original official `(split, image, geometry, dimensions, category, sent)` multiset. Unlike the canonical manifest's verbatim `raw` text, these expressions use the official processed `sent` field. Punctuation, tokenization and case differences are not new human annotations. Training rows also omit source dataset/ref/sentence IDs, so source identity must not be guessed from row order.

The supplemental source, automated audit and nine inspected examples are available under [`source_annotations/mixed-seg/`](https://huggingface.co/datasets/Melo111/open-vision-banana-res/tree/244d89993e5e08ff63fc1647f0e7fc5bbd61b361/source_annotations/mixed-seg) on the same Hub dataset. All 30,956 original images were decoded and matched to released asset hashes; all 71,991 source masks decoded successfully, and all 70,327 released masks were pixel-identical. The 1,664 source masks outside the filtered release are all on official held-out images and provide no eligible additional training masks. Download the supplement separately:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "Melo111/open-vision-banana-res",
    repo_type="dataset",
    revision="244d89993e5e08ff63fc1647f0e7fc5bbd61b361",
    allow_patterns=["source_annotations/mixed-seg/*"],
    local_dir="data/res_source",
)
```

Reproduce the audit with the original official refs/instances and a restored canonical release:

```bash
python scripts/audit_res_mixed.py --annotations-json data/res_source/source_annotations/mixed-seg/instances_nogoogle.json --annotations-root /path/to/official/annotations --image-root /path/to/train2014 --dataset-root data/res_binary_v1 --output-json output/res/mixed_source_audit.json
```

The auditor verifies official source fingerprints, all expression multiplicities, image bytes/dimensions, all unique native masks and exact agreement with released target pixels. Missing, altered, duplicate or unmatched records fail reconciliation. It writes a new report only and does not edit training/evaluation files. Automated checks do not establish semantic correctness; the separate `visual_review.json` records the limited visual sample and observed coarse boundaries/occlusions.

**Train with the existing `training.jsonl` and SANA RES configuration. Do not concatenate this mixed file into it.** Its 33,723 training expressions on official held-out images must stay excluded, and remaining expressions already contribute to the canonical release. The supplement adds provenance and input-format verification, with **zero new training examples**; the 270,195-row training set, checkpoint-selection view, official benchmark splits and original immutable bundle remain unchanged. No training/inference/evaluation interface changes are needed to reuse the already converted supervision.

## Binary RES contract and compatibility

The versioned encoding is `binary_white_black_v1`:

- Foreground is RGB `(255, 255, 255)`; every other pixel is `(0, 0, 0)`.
- Native masks are lossless RGB PNGs. COCO polygons and RLE are decoded with `pycocotools`; all valid disconnected foreground components are retained.
- The canonical prompt is `Segment {expression} in solid white. Render all other pixels black.` The expression is stripped only when inserted into the prompt; the original annotation text remains in the record.
- Training resizes referring targets with nearest-neighbor interpolation. Source-image resizing is unchanged.
- Inference and evaluation share `decode_referring_mask`: mean RGB greater than `0.5` selects foreground by default. Binary decoding does not use connected-component filtering, so installing SciPy does not change this protocol.
- Evaluation preserves native GT and resizes decoded predictions with nearest-neighbor interpolation to the original image dimensions.

`ManifestVisionDataset` continues to accept existing fields `task`, `split`, `input`, `target`, and `prompt`. For RES it resolves the prompt per expression, preserving expression/encoding metadata during collation. A versioned binary row requires a nonempty expression and the canonical prompt, or an omitted prompt that can be generated from that expression. Missing conditioning or a conflicting versioned prompt raises an error. Legacy unversioned RES records containing a complete per-sample prompt remain readable.

The default white-target inference behavior now uses the binary contract. For existing checkpoints trained with the former colored-background protocol, explicitly select `referring_encoding="legacy_color_distance_v1"` in `VisionBanana.predict` or `VisionEvaluator.evaluate`; `VisualizationDecoder.decode_referring` accepts `encoding="legacy_color_distance_v1"`. Non-white target colors select legacy decoding when no encoding is given. In legacy mode, a threshold remains an RGB L2 distance and the old optional SciPy component filtering is preserved. In binary mode, a threshold is an intensity value in `[0, 1]`. These protocols must not be mixed when comparing checkpoint results.

The official RefCOCOg benchmark entry point now requests `protocol="umd", strict=True`. Strict loading requires the requested protocol and matching image/annotation metadata, uses native COCO mask decoding, and fails on missing or empty masks. It does not replace missing annotations with black masks or silently use a different split. The direct `RefCOCOgDataset` constructor retains `strict=False` for existing fixture/custom-loader compatibility; official results must use strict mode. The new manifest evaluator also rejects missing or invalid assets.

These changes target RES. Depth, normal, semantic, instance, and other benchmark encodings remain on their existing paths. The old Cityscapes-based `gen_detpoint_refer_targets.py` is not the converter for this corpus. RES checkpoint task metadata cannot substitute a cached expression for the current sample's expression.

## Build from local annotations

Use a Python environment with the appropriate PyTorch/CUDA setup for later training. The data builder itself does not import the training or inference stack.

```bash
python -m pip install -e ".[data,training,evaluation]"
```

Expected source layout:

```text
annotations_root/
  refcoco/refs(unc).p
  refcoco/instances.json
  refcoco+/refs(unc).p
  refcoco+/instances.json
  refcocog/refs(umd).p
  refcocog/instances.json
image_root/
  COCO_train2014_000000000009.jpg
  ...
```

The actual local-source command used for this release is:

```powershell
python scripts/build_res_manifest.py --annotations-root "E:\vg_code\VLTVG-master\data\other" --image-root "E:\vg_code\VLTVG-master\data\other\train2014" --output-dir "E:\Open-vision-banana\output\res-binary-v1-20261011"
python scripts/validate_res_manifest.py "E:\Open-vision-banana\output\res-binary-v1-20261011"
```

For another machine, point the same arguments at the layout above. The output directory must be new or empty. Images are copied once with `--image-mode copy` (the default); absolute source paths are not written into the manifests or reports. Pickle globals are forbidden by a restricted unpickler. `ijson` streams the COCO JSON into a temporary SQLite index, limiting memory usage. An interrupted or failed build retains `BUILD_INCOMPLETE` and must not be used as a complete dataset.

For a pilot, use a separate output directory and add `--max-train-samples 1000`. This limits exported training expressions **after** constructing the full global blacklist and deduplicating training candidates. It still exports the complete official evaluation partitions, so it is not a 1,000-record total build. Do not publish a pilot as the full release.

The validator fully decodes every distinct image and mask. It checks portable paths, exact dimensions, nonempty binary RGB masks, expression/prompt correspondence, unique sample IDs, train deduplication, train/evaluation separation, original evaluation multiplicity, accounting of all raw training expressions, and SHA256 integrity. Passing validation means the format and these invariants hold; it is not a claim about model accuracy.

## Manifest and audit artifacts

Paths in every emitted manifest are POSIX-style paths relative to the dataset root. All manifests therefore live at that root, including the per-source subsets.

```text
res_binary_v1/
  pairs.jsonl
  train.jsonl
  val.jsonl
  test.jsonl
  refcoco_unc_testA.jsonl
  refcoco+_unc_testB.jsonl
  refcocog_umd_val.jsonl
  ...
  assets.jsonl
  evaluation_image_ids.json
  train_duplicates.jsonl
  build_report.json
  validation_report.json
  images/coco2014/...
  targets/coco2014/...
```

`pairs.jsonl` contains all partitions; the training loader selects `train` and validation selects `val`. Do not train on `test.jsonl` or alter official evaluation rows. An illustrative row is:

```json
{
  "sample_id": "refcoco_unc_ref123_sent456",
  "task": "referring",
  "split": "train",
  "input": "images/coco2014/COCO_train2014_000000000001.jpg",
  "target": "targets/coco2014/000000000001_<geometry_sha256>.png",
  "expression": "the woman in the blue shirt",
  "prompt": "Segment the woman in the blue shirt in solid white. Render all other pixels black.",
  "encoding": "binary_white_black_v1",
  "source_dataset": "refcoco",
  "source_protocol": "unc",
  "source_split": "train",
  "image_id": 1,
  "ann_id": 789,
  "ref_id": 123,
  "sent_id": 456,
  "supervision": "human_polygon",
  "geometry_sha256": "<geometry_sha256>",
  "width": 640,
  "height": 480
}
```

`assets.jsonl` records each distinct image/mask path, dimensions, byte size, and SHA256. `build_report.json` records source-file fingerprints, counts, exclusions, duplicates, build policy, emitted manifest hashes, and the dataset fingerprint. `validation_report.json` records the verified counts and matching fingerprint. `train_duplicates.jsonl` associates each suppressed training record with its deterministic primary record; original source/ref/sentence identity is retained.

`image_attribution.jsonl` preserves source-provided license identifiers, names and links plus COCO/Flickr source URLs for every exported image. Missing source metadata remains null; conflicting attribution or missing image coverage fails the build. The `BUILD_INCOMPLETE` marker blocks RES training and manifest evaluation until construction or restoration succeeds.

## Package and download a fixed Hub release

After full validation, package the corpus into uncompressed tar shards and metadata files:

```powershell
python scripts/package_res_dataset.py --data-dir "E:\Open-vision-banana\output\res-binary-v1-20261011" --output-dir "E:\Open-vision-banana\output\res-binary-v1-20261011-hub" --shard-size-mb 256
```

The package contains `bundle.json`, portable manifests/reports, and `assets/part-*.tar`. The bundle index records payload hashes and member counts. Source pickles and source-machine configuration are not part of the release. The upload is a separate publishing operation; the packager itself makes no remote writes.

Download and restore the published data from the repository root:

```bash
python scripts/download_res_dataset.py --repo-id Melo111/open-vision-banana-res --revision "dce183e8441c55c8fcf58afd87f90c776024c828" --output-dir data/res_binary_v1
python scripts/validate_res_manifest.py data/res_binary_v1
```

The command pins this release's full 40-character lowercase hexadecimal commit SHA. The downloader rejects branch names such as `main`, downloads the indexed payloads, checks their SHA256 values, and restores regular files under the selected output directory. Keep sufficient disk space for both the downloaded bundle and the restored corpus. An identical existing restoration can be reused; differing existing files are not overwritten silently. To restore an already downloaded local bundle, use `--bundle-dir /path/to/bundle --output-dir data/res_binary_v1` instead of `--repo-id/--revision`.

## Train the RES baseline

The checked-in [SANA RES configuration](../configs/sana_res_binary_v1.yaml) reads `data/res_binary_v1/training.jsonl`, including the test-disjoint checkpoint-validation view. It selects only `referring`, uses 512-pixel training inputs, LoRA rank 32, and an effective batch size of 16 through gradient accumulation. Inspect the effective configuration before starting:

```bash
python scripts/train.py --config configs/sana_res_binary_v1.yaml --print-effective-config
python scripts/train.py --config configs/sana_res_binary_v1.yaml
```

For a different dataset location, update both `data.dir` and `data.pair_manifest` in a copy of the configuration. The default output is `output/sana_res_binary_v1`. Model-weight access, a suitable accelerator, and training dependencies are required for an actual run. This configuration is a reproducible starting point, not a tuned result or a promise of convergence. The periodic training validation is not a substitute for the full native-resolution evaluation below.

## Evaluate native-resolution masks

The manifest evaluator works from the restored release without the original pickles. It requires an explicit validation/test split and validates all selected assets before scoring. First perform a model-free RefCOCOg UMD validation preflight:

```bash
python scripts/evaluate_res_manifest.py --dataset-root data/res_binary_v1 --split val --source-dataset refcocog --source-protocol umd --preflight --output-json output/res/refcocog_umd_val_preflight.json
```

Evaluate an actual trained checkpoint with an explicitly recorded inference setup:

```bash
python scripts/evaluate_res_manifest.py --dataset-root data/res_binary_v1 --split val --source-dataset refcocog --source-protocol umd --model-name sana-1600m --lora-path "output/sana_res_binary_v1/<CHECKPOINT_DIRECTORY>" --device cuda --dtype bfloat16 --resolution 512 --num-inference-steps 20 --strength 0.85 --seed 42 --output-json output/res/refcocog_umd_val_metrics.json
```

Replace `<CHECKPOINT_DIRECTORY>` with the saved checkpoint to assess. The example inference settings must be kept fixed across compared runs; they are not a reported optimized recipe. To score RefCOCO UNC testA, use `--split test --source-dataset refcoco --source-protocol unc --source-split testA`. Use `--source-dataset "refcoco+"` for RefCOCO+. RefCOCOg UMD uses `--split test --source-dataset refcocog --source-protocol umd --source-split test`.

The evaluator uses batch size 1, one expression per sample, native GT dimensions, and the shared binary decoder. Its JSON includes filters, selected source counts, the manifest hash, encoding, inference settings, and RES metrics: `cIoU` (cumulative intersection/union), `gIoU`/`IoU` (mean per-expression IoU), `Precision@0.5`, pixel precision/recall, and F-score. Here `gIoU` is not generalized bounding-box IoU. Metric ratios are returned on a `[0, 1]` scale. `--max-samples N` is only a smoke-test score on the first N selected rows; it is not a full benchmark result. Report per-source and per-official-subset results, rather than presenting a combined cross-source average as an official benchmark score.

Inference metadata separates `inference.requested` from `inference.effective`. The SANA generator resolves the historical one-step default to its configured schedule (20 steps for the supplied model), uses guidance 1.0, and does not use the img2img `strength` argument. The report resolves those settings with the same helper used by the generator. Backends without a verified metadata resolver explicitly report `effective: null`; their requested settings must not be presented as observed execution parameters. Strict official RefCOCOg loading and portable manifest loading both declare their input image range to preserve identical pixel brightness through inference.

## Attribution and experimental scope

The images, human expressions, and derived annotation masks retain their upstream ownership and applicable terms. The repository's code license does not relicense those data. Consult the release dataset card, the [REFER project and source citations](https://github.com/lichengunc/refer), and the [COCO terms of use](https://cocodataset.org/#termsofuse); retain the source identifiers and attribution with downstream copies.

This is a **supervised RES** dataset extension using RefCOCO-family training annotations. Results from models trained on this corpus must be labeled accordingly and must not be presented as zero-shot transfer onto those benchmarks. The release establishes data/interface correctness and reproducible preparation; it does not claim an accuracy improvement, trained checkpoint, or reproduction of a paper's reported results. Such claims require controlled training and held-out evaluation.

## Validation and compatibility

This change is limited to Grounding/RES data preparation, training reads, inference conditioning and native-resolution evaluation. Shared trainer/evaluator changes are conditional on `referring`; the permissive RefCOCOg dataset default and explicit legacy color protocol remain available. A pure depth/normal/semantic/instance batch retains its historical collated fields. Existing general CI, performance drivers, smoke scripts and non-RES tests are unchanged from main. Whole-repository gate failures are recorded separately rather than repaired through unrelated changes.

The independent `.github/workflows/res.yml` runs only for paths used by RES or manual dispatch, on Python 3.10, 3.11 and 3.12. It checks the actual COCO/streaming dependencies, RES formatting/types and the source-build, training-conditioning, inference and native-evaluation tests, without model downloads. `.coveragerc-res` requires 80% statement coverage of the nine new protocol/data modules, including mixed-source auditing; this is explicitly not whole-package coverage. `.mypy-res.ini` checks the listed modules and their mutual types while skipping imported implementations/stubs. These checks do not change the repository's existing gates.

Reproduce the focused suite after installing `.[inference,evaluation,data,dev]`:

```bash
python -m pytest tests/test_res_training_contract.py tests/test_res_training_view.py tests/test_res_manifest_builder.py tests/test_res_bundle.py tests/test_res_manifest_dataset.py tests/test_res_inference_contract.py tests/test_referring_masks_strict.py tests/test_normal_referring_benchmarks.py tests/test_res_mixed.py --cov --cov-config=.coveragerc-res --cov-report=term-missing
```

Regression coverage includes per-expression prompts, binary resizing, disconnected masks, strict annotation errors, polygon/RLE decoding, legacy inference, identical bright-image pixels through official/portable adapters, bundle integrity, held-out exclusion, mixed-source reconciliation and native evaluation. The local Python 3.12 focused suite passed all 179 tests with 89.28% scoped coverage, including checks that non-RES collated fields remain unchanged. The 24 mixed-source tests cover processed-versus-raw expressions, categories, polygon/RLE masks, shared-image integrity, official multiplicity and failure on tampering. A separate adjacent compatibility suite passed 38 tests covering manifest portability, evaluator aggregation and depth/normal inference. Consult the [PR checks](https://github.com/isLinXu/open-vision-banana/pull/25/checks) for the latest remote run; these local results do not certify all repository gates.

## Data release verification

The publication validator fully decoded all 101,283 assets and matched the bundled report byte for byte. All remote payload hashes were verified; an anonymously downloaded shard decoded successfully. The complete bundle restored, and the actual training loader indexed all 270,195 training and 24,554 checkpoint-validation rows from that restoration. Samples from all three sources collated at 512 pixels with binary targets. All 4,896 restored RefCOCOg UMD validation expressions passed native evaluation preflight. The immutable Hub release and its format remain unchanged by later interface fixes.

At publication, a same-environment full-suite comparison against base `faa9e0f9132b12c5cd21363b75d0e4c31a74921c` found no introduced failure IDs. Existing optional-dependency, platform and unrelated metric-test failures are outside this RES change. The data extra requires Hugging Face Hub 0.23 or newer for portable local-directory downloads without cache symlinks. No accelerator training or benchmark accuracy is claimed.
