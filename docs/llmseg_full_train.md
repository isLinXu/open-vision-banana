# LLM-Seg full-training supplement (preparation support, not a published release)

## Verified scope

The official annotations pinned in `open_vision_banana.llmseg` contain:

| Original split | COCO images / questions | EgoObjects images / questions | Total questions |
| --- | ---: | ---: | ---: |
| train | 8,980 / 33,755 | 2,356 / 11,025 | 44,780 |
| validation | 792 / 2,941 | 208 / 964 | 3,905 |

Both complete annotation hashes were independently recomputed on 2026-11-08 and
match `TRAIN_SHA256` and `VALIDATION_SHA256`. Train and validation filenames have
zero intersection. The official Drive folder snapshot contains `train.json` and
`validation.json`; no public test annotation file has been verified. References
to `test` in upstream preparation code are not evidence of a released test set.

Sources: [official repository](https://github.com/wangjunchi/LLMSeg/tree/65147a98cc49c2dac9b7a3633d9d3d336f1ac023),
[official annotation folder](https://drive.google.com/drive/folders/19MyzJN9hkvTSUr2lUFlMwmS2TSRRKmUX),
[upstream loader](https://github.com/wangjunchi/LLMSeg/blob/65147a98cc49c2dac9b7a3633d9d3d336f1ac023/utils/llm_seg_dataset.py),
and [PR #79](https://github.com/isLinXu/open-vision-banana/pull/79).

## Native-mask audit

All 44,780 training RLEs were decoded on CPU with zero decode errors. The existing
foreground-area rule accepts 22,144 COCO and 9,126 EgoObjects questions (31,270
total), before held-out exclusion, source-image dimension checks or training crop
visibility. These are annotation-only counts, not a completed training manifest.
See [the machine-readable audit](llmseg_full_train_audit.json).

## Adapter contract

The original CLI remains COCO-only, at most 5,000 accepted questions, with one
SHA256-ranked question per image. Increasing its cap alone does not expose the
remaining questions or EgoObjects. `--full-train` explicitly selects all questions
from both original train sources and disables the cap. It requires a separate
EgoObjects image root and rejects a simultaneous `--max-samples` option.

The existing native-mask quality rules remain: valid nonempty RLE matching the
source image, 0.5% to 95% foreground, known held-out exclusions, and existing
manifest integrity checks. Therefore 44,780 is an upstream count, **not a promised
accepted count**. Full means no source/question sampling cap, not bypassing checks.
All exclusions and rejections must be accounted for before calling a build complete.
No new broad dataset deduplication project is introduced.

`question` becomes the existing binary referring prompt. `rle_seg` is decoded to
native-resolution white-foreground/black-background RGB PNG. `answer` stays
metadata and is never added to the prompt. No SAM extraction, new runtime dataset,
loss, or task name is required. Conversion is CPU-only.

Both COCO and EgoObjects official validation identities are excluded. EgoObjects
filenames are constrained to safe basenames; they are not interpreted as COCO IDs.
`--exclude-image-ids` and held-out manifests remain required/recommended as in
[the original preparation instructions](res_reasoning_sources.md). There is no
validation-to-train conversion or synthetic split reassignment.

## Reproduction after obtaining authorized source images

```bash
PYTHONPATH=src python scripts/prepare_llmseg.py \
  --annotations data/raw/llmseg/train.json \
  --validation-annotations data/raw/llmseg/validation.json \
  --image-root data/raw/coco2017/train2017 \
  --ego-image-root data/raw/egoobjects/images \
  --full-train --reject-invalid \
  --exclude-image-ids data/res_binary_v1/evaluation_image_ids.json \
  --exclude data/res_binary_v1/pairs.jsonl data/res_binary_v1 \
  --output-dir data/prep/llmseg-full-train
python scripts/validate_prepared_res.py data/prep/llmseg-full-train
```

Missing source images fail the build, including with `--reject-invalid`; installed
asset coverage never silently determines the subset. Run the existing full asset
validator, loader/collator checks, deterministic visual reviews, and package/restore
round-trip on real data before release. Fixture tests alone do not establish real
image coverage, image/mask alignment, question quality, or redistribution permission.

## Existing 4,304 rows and integration

PR #79 released 5,000 COCO image/question pairs. The later 512-pixel crop visibility
filter retained 4,304 in the current reasoning supplement. Those 4,304 are an older
subset, not an additional source. A future full supplement should **replace the old
LLMSeg component**, not be appended alongside it. This is source-version selection,
not a new cross-dataset deduplication requirement. Preserve non-LLMSeg rows and all
existing held-out rows. Apply the selected training geometry visibility policy to
the new component and record its actual retained count before calculating ratios.
This change does not alter #109 configurations or launch training.

The old base RES pool was not globally purged of LLM-Seg validation images. Even
with a clean new supplement, do not claim LLM-Seg validation is an independent
held-out evaluation unless the complete training pool is rebuilt/audited for it.

## Publication gate and current limitations

As checked on 2026-11-08, the LLMSeg repository has no LICENSE file and GitHub
returns `license: null`; its README links public annotations but does not give
explicit redistribution terms for those annotations. Public download availability
and PR #79's earlier upload are not substitutes for permission to republish a new
full release. Confirm the annotation redistribution terms with the rights holder
or an authoritative license before uploading derived masks/questions.

The [EgoObjects repository](https://github.com/facebookresearch/EgoObjects) README
states MIT and links the [official image download page](https://ai.meta.com/datasets/egoobjects-downloads)
for the approximately 40 GB `EgoObjectsV1_images.zip`. Verify the applicable image
download terms and actual 2,356-image coverage separately; the repository code
license alone must not silently be substituted for any additional image terms.
COCO source image terms also remain applicable. No full EgoObjects asset build,
full output release, or full real-data loader validation is claimed here.

Once these gates pass, use a **new** subfolder such as
`supplements/reasoning-v1/llmseg-full-train-v1` in
`Melo111/open-vision-banana-res`, publish a new immutable revision, and include
source attribution, terms, hashes, preparation report, rejection ledger and visual
review. Do not overwrite `llmseg-coco5k`, other datasets, benchmark manifests, or
attempt to mutate revision `e479f7013cc72ff355d60fdcbeb18b78e395bee5`.


## Staged local preparation

`--full-train` still defaults to both sources. When one source is not yet available,
`--full-train --source-categories coco` permits a clearly marked COCO-only stage
without an EgoObjects root. Source annotation hashes still cover the original
complete official files. The preparation report records `source_categories` and
`all_source_categories_selected: false`; a stage must not be presented as the
completed full LLMSeg release. Use separate output directories for each build.

The 2026-11-08 server preflight found 8,973 of 8,980 COCO images in an existing
shared cache. Seven missing images were downloaded from the official COCO image
server into the isolated preparation directory, giving complete 8,980-image
availability. The shared cache is unchanged. This does not establish the final
accepted row count or completion of the EgoObjects component.

The live EgoObjects download page was subsequently inspected in a browser. Its
dataset-specific agreement includes restrictions on incorporating the data into
another program/dataset/product, as well as a 100-image redistribution limit.
Downloading/accessing the archive is described as agreement acceptance. The
repository's MIT license does not resolve these separate dataset terms. A working
official archive link was located, but no EgoObjects archive download has been
performed as part of this preparation.
