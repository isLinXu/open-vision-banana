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

## Validation performed for this change

### Review gates

The root `agent.md` requires tests, Ruff and conventional commits. `CONTRIBUTING.md` also requires public API exports, an Unreleased changelog entry, pre-commit and 80% whole-package coverage; the existing CI workflow currently enforces 70% whole-package coverage. This PR preserves that existing threshold and documents the difference rather than lowering either requirement.

The additional `.github/workflows/res.yml` runs independently on Python 3.10, 3.11 and 3.12. It installs and imports the actual COCO/streaming decoders, forbids model downloads, checks Ruff/format/types, runs the RES and adjacent normal benchmark tests, and requires at least 80% statement coverage of the eight new RES protocol/data modules in `.coveragerc-res`. Its coverage scope is explicit and is not whole-package coverage. JUnit and coverage XML are uploaded even when tests fail. Ruff and mypy versions match pre-commit.

The repository-wide CPU test job now installs its declared `ci` extra and includes previously omitted export/generation/training tests. The performance workflow compares a PR against its exact base SHA on the same runner using the same drivers, fixed inputs and thread count. Missing measurements or failed comparators fail the gate; the 15% regression threshold remains unchanged. Main-branch baseline initialization is explicit and does not claim a regression comparison. The training smoke uses a documented lightweight fixture and verifies finite LoRA weight updates and checkpoint restoration; it does not validate real diffusion-model quality.

These checks supplement existing CI. Full-repository lint, coverage, pre-commit and tests still need to pass before claiming all contributor gates are satisfied. Do not merge based on the RES check alone.

The scoped type check uses `.mypy-res.ini` and the explicit eight-module list: it checks those modules and their mutual types while skipping imported implementations/stubs. The existing full-source mypy job remains unchanged in scope. This keeps the focused gate independent of unrelated Torch/NumPy stub versions and does not claim whole-repository type safety.

The review follow-up passed 151 RES/adjacent tests on each of Python 3.10, 3.11 and 3.12 in GitHub Actions, with 88.88% coverage of the scoped modules; a local combined run including 36 performance-driver tests passed all 187 tests. The exact-base performance comparison also passed. With Ruff 0.11.4, main had 1,607 findings versus 1,595 on this branch and no introduced path/rule/message findings. The first complete Linux CPU run reached 68.15% whole-package coverage, below the retained 70% CI threshold and the contributor guide's 80% target. See the current [PR checks](https://github.com/isLinXu/open-vision-banana/pull/25/checks) for the latest full-suite status; this evidence does not certify all repository gates as satisfied.

### Data release evidence

At data publication, the RES and adjacent benchmark suite passed **146 tests** on Windows with Python 3.12. Coverage includes multi-expression conditioning, strict source loading, polygon/RLE decoding, disconnected masks, legacy inference, explicit bright-image normalization, bundle integrity and native-resolution evaluation. The final validator fully decoded all 101,283 assets; its output matched the bundled validation report byte for byte. The complete bundle restored successfully, and the actual training loader indexed all 270,195 training and 24,554 checkpoint-validation rows from that restoration. Samples from all three sources collated at 512 pixels with binary targets. All 4,896 restored RefCOCOg UMD validation expressions passed native evaluation preflight.

A full-suite comparison used the same environment and explicit source `PYTHONPATH`: base `faa9e0f9132b12c5cd21363b75d0e4c31a74921c` had **1,284 passed / 26 failed / 39 skipped**; the integration checkpoint had **1,408 passed / the same 26 failed / 39 skipped**. No new failure IDs appeared. The existing failures include unavailable optional dependencies, Windows symlink/shell assumptions, and pre-existing metric assertions. Later attribution, test-disjoint checkpoint-validation and incomplete-restore checks are included in the 146-test final targeted run. This is not a claim that the entire repository suite is green, nor a substitute for accelerator training or measured benchmark accuracy.

At publication, Ruff 0.17 checks and formatting checks passed for all newly added Python files. Its whole-repository comparison found 1,545 findings on the base and 1,536 on that integration checkpoint: no introduced findings when compared by relative path, rule and message. The review follow-up uses the pinned tools above. CI installs the data extra so source-conversion tests run instead of being skipped for a missing `ijson` dependency. The data extra requires Hugging Face Hub 0.23 or newer for portable local-directory downloads without cache symlinks.
