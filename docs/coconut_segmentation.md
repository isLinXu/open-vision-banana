# COCONut-B category-conditioned segmentation supplement

This supplement converts the SNVC-derived COCONut-B RGB panoptic masks into OVB
`semantic` and `instance` pairs. It does not train a model or claim segmentation
benchmark improvements. The released targets are deterministic reconstructions
of source annotations, not model predictions.

## Published release

[Melo111/open-vision-banana-segmentation](https://huggingface.co/datasets/Melo111/open-vision-banana-segmentation)
contains the `coconut-b-v1` supplement: **307,598 pairs from 109,083 unique
images**, covering 91 semantic categories. The portable bundle is approximately
21.05 GB, including 75 asset TARs and the manifests/reports.

| Task | Train | Validation | Total |
|---|---:|---:|---:|
| Semantic | 106,894 | 2,189 | 109,083 |
| Instance | 194,537 | 3,978 | 198,515 |
| Total | 301,431 | 6,167 | 307,598 |

Of the 118,200 input pairs, 6,545 were excluded by held-out identity, 1,518 had
invalid legends (1,470 duplicate labels/colors; 48 malformed legends), 1,041 had
no eligible category, and 13 were duplicate decoded images. Instance rows include
117,716 single-instance and 80,799 multi-instance examples. Category exclusions
are recorded separately from whole-image exclusions.

The dataset fingerprint is
`a155b8aecc798476da187b261376bfca1da3efd64012b4fcb419d1e7d19ba88a`.
The Hub card includes source notices, cleaning statistics, and deterministic
semantic/instance sample sheets with exact selected records. The extreme-black
example (COCO 455943, 99.75% source black) contains a small, valid airplane mask;
it is retained by the same policy as the other images.

Download and checksum-verify the immutable data release:

```bash
python scripts/download_res_dataset.py \
  --repo-id Melo111/open-vision-banana-segmentation \
  --revision b28483c48a57bb8338912da02d2e00c03d6da3d7 \
  --subfolder coconut-b-v1 --output-dir data/coconut_b_v1
```

The downloader's RES name is historical; it restores the shared bundle format
used here. Allow disk space for both the downloaded bundle and restored assets.
The published `source_inventory.json` and per-row source revisions preserve the
upstream identities independently of the Hub release revision.

## Sources and scope

- Immediate source: [isLinXu/open-vision-banana on ModelScope](https://modelscope.cn/datasets/isLinXu/open-vision-banana),
  revision `ab56be89855bd1414a4be2aa98fef35602adc892`.
- Source annotations: `training/cleaned_manifests/segmentation/coconut_b_total_unified_genseg_edit_regenerated.jsonl`.
- Pair membership: `image_text_image/manifests/coconut_b.jsonl` (118,200 pairs).
- Media: ten TARs in `image_text_image/archives/coconut_b`, 20,363,386,880 bytes,
  checked against their SHA256 index before extraction.
- Upstream conversion: [SenseNova-Vision-Corpus-50M](https://huggingface.co/datasets/sensenova/SenseNova-Vision-Corpus-50M).
- Original segmentation: [COCONut: Modernizing COCO Segmentation](https://github.com/bytedance/coconut_cvpr2024),
  Deng et al., CVPR 2024. Source images are COCO train2017 images.

The SNVC upstream card identifies CC BY-NC 4.0. COCONut's dataset terms restrict
commercial use; image copyrights and individual COCO source-image licenses
remain with their original owners. This derivative does not relicense the source
images or imply affiliation with SenseNova, ByteDance, or COCO. The Hub dataset
card retains these sources, restrictions, and links.

## Why the original pair manifest is insufficient

The unified pair records retain input and output images but replace the original
assistant answer with `<image>`. The original answer contains the mapping from
labels such as `bowl-0` to exact RGB values. Preparation joins the two manifests
by COCO image ID and requires both image paths to agree. It never joins by row
number. The original legend and fixed source revision accompany every output row.

## Cleaning policy

`coconut_b_conditioned_v1` is recorded in `policy.json`:

- Require unique source IDs, matching pair populations, matching COCO filenames,
  same-size RGB PNG masks, parseable complete legends, distinct nonblack colors,
  and no mask colors absent from the legend except black.
- Use the explicit category vocabulary and alias mapping in
  `src/open_vision_banana/coconut_preparation.py`. Ambiguous terms including
  `food`, `watch`, `container`, `plant`, `table`, and `seat` are not silently
  interpreted as broad semantic labels. Other eligible categories of that image
  remain usable. `policy.json` is the exact authoritative list.
- Require indexed labels for thing categories and non-indexed labels for stuff.
  This is a conservative source-format convention, not an independent recovery
  of official `isthing`/`iscrowd` metadata.
- If any region of a category is missing or smaller than
  `max(64, ceil(0.0002 * H * W))` pixels, reject the entire category. Do not train
  a partially retained set of the category's instances.
- Never reject an image just because its black fraction is large. Black and
  nonselected categories are background relative to the requested categories.
  Do not use unannotated classes as absent-class negatives.
- One semantic view per source image: merge instances of each selected class,
  choose at most 12 largest eligible classes, and render a matching per-sample
  RGB legend including background. No absent distractor classes are invented.
- Up to three instance views per source image: deterministically choose eligible
  thing categories, paint each instance a distinct color from the existing OVB
  palette, use black background, and use the existing class-conditioned prompt.
  Single-instance examples are valid. The cap limits expansion of crowded images.
  A category exceeding the palette's 64 distinct colors is omitted from instance
  views rather than merging its instances; its semantic union can still be used.
- Preserve native resolution and original JPEG bytes. Targets are lossless PNG.
- Exclude supplied evaluation COCO IDs and nontrain identities from reference
  manifests. Keep known reference-training IDs in train; split other images by a
  deterministic decoded-RGB hash (2% validation). All tasks from an image share
  the split. Suppress duplicate decoded images within this supplement.

The publication records the exact exclusion-list and reference-manifest hashes.
When combining it with a different training corpus, rerun the cross-corpus
identity audit. Within-supplement pixel deduplication is not a claim of
exhaustive cross-dataset near-duplicate detection.

## Rebuild and validate

```bash
python scripts/download_coconut_source.py --output-dir data/coconut_source --workers 4
python scripts/prepare_coconut.py \
  --source-root data/coconut_source --output-dir data/coconut_b_v1 \
  --exclude-image-ids data/res_binary_v1/evaluation_image_ids.json \
  --reference-manifest data/qwen_main_depth_reasoning/pairs.jsonl --workers 16
python scripts/validate_coconut.py data/coconut_b_v1 \
  --source-media data/coconut_source/media --workers 16
python scripts/visualize_coconut.py data/coconut_b_v1 --output-dir output/coconut_visuals
python scripts/package_res_dataset.py --data-dir data/coconut_b_v1 \
  --output-dir output/coconut_bundle --shard-size-mb 256
```

Preparation requires a new output directory and leaves `BUILD_INCOMPLETE` on
failure. `--max-images` is for development only and marks the build partial in
provenance. The publisher must reject partial builds. No token is needed for the
public source downloader; it makes no uploads.

The validator checks every published asset's SHA256, source/target dimensions,
held-out identity exclusion, task/split counts, decoded-image uniqueness,
semantic prompt roundtrip, exact source-to-semantic pixel unions, and exact
instance foreground and per-region colors. The native OVB loader should also be
smoke-tested with semantic color augmentation before admitting a release.

The full v1 pixel audit passed on all 109,083 images / 416,681 assets. The
release also passed native `ManifestVisionDataset` and batch-collation checks on
16 rows of each task/split (64 total), at resolution 512 with aspect-preserving
crops and semantic color remapping enabled.

The new regression tests cover missing instances, ambiguous categories, unknown colors,
black-heavy examples, alias unions, identity exclusions, pixel deduplication,
and a CLI build/validate/package/restore round trip. Together with the required
existing data/trainer checks, 19 tests passed:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
python -m pytest tests/test_coconut_preparation.py \
  tests/test_pipeline_smoke_all_tasks.py tests/test_trainer_system_all_tasks.py \
  tests/test_download_datasets_script.py -q
```

Outputs include `pairs.jsonl`, `assets.jsonl`, `source_audit.jsonl`,
`rejected.json`, `policy.json`, `provenance.json`, `evaluation_image_ids.json`,
`build_report.json`, and `validation_report.json`. Per-row `source_legend`, source
paths, COCO IDs, source revision, and decoded-RGB hash retain traceability.

## Training

Use `data.dir` for the restored dataset root, `data.pair_manifest` for its
`pairs.jsonl`, and `data.tasks: [semantic, instance]`. The existing loader
already consumes these contracts; no model or loss change is required for this
category-conditioned background policy. Semantic `mapping` supports the existing
prompt-style and color-remapping augmentation. Choose task/source mixture weights
explicitly when joining other OVB corpora; expanded instance rows should not
implicitly determine task balance.

## Limits

Automated cleaning cannot prove that all source instances are annotated or that
all crowd/occluded regions are exhaustive. No new masks or referring expressions
are hallucinated. Black is not asserted to be a globally verified negative class.
If a requested category is missing from a black region, that source omission can
remain; ambiguous cases require annotation review or exclusion. Samples and the
dataset card expose this limitation rather than relabeling it as perfect ground
truth. This supplement does not contain metric-depth or RES supervision.
