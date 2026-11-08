# LLM-Seg COCO full-training supplement

Current release scope is **COCO training questions only**, with no per-image question cap.
EgoObjects has not been downloaded or included. The all-source upstream counts below
are provenance, not a claim that the excluded source is part of this release.

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
  --full-train --source-categories coco --reject-invalid \
  --exclude-image-ids data/res_binary_v1/evaluation_image_ids.json \
  --exclude data/res_binary_v1/pairs.jsonl data/res_binary_v1 \
  --output-dir data/prep/llmseg-full-train
python scripts/finalize_llmseg_crop.py \
  --data-dir data/prep/llmseg-full-train \
  --output-dir data/prep/llmseg-coco-full-v1 --resolution 512 --workers 4
python scripts/package_res_dataset.py \
  --data-dir data/prep/llmseg-coco-full-v1 --output-dir output/llmseg-coco-full-v1
```

Missing source images fail the build, including with `--reject-invalid`; installed
asset coverage never silently determines the subset. Run the existing full asset
validator, loader/collator checks, deterministic visual reviews, and package/restore
round-trip on real data before release. Fixture tests alone do not establish real
image coverage, image/mask alignment, question quality, or redistribution permission.

## Existing 4,304 rows and integration

PR #79 released 5,000 COCO image/question pairs. The later 512-pixel crop visibility
filter retained 4,304 in the current reasoning supplement. Those 4,304 are an older
subset, not an additional source. This supplement must **replace the old
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
full release. The annotation license remains unstated; this supplement preserves that uncertainty
and does not grant or invent rights in the original questions or masks.

The [EgoObjects repository](https://github.com/facebookresearch/EgoObjects) README
states MIT and links the [official image download page](https://ai.meta.com/datasets/egoobjects-downloads)
for the approximately 40 GB `EgoObjectsV1_images.zip`. Verify the applicable image
download terms and actual 2,356-image coverage separately; the repository code
license alone must not silently be substituted for any additional image terms.
COCO source image terms also remain applicable. No full EgoObjects asset build,
full output release, or full real-data loader validation is claimed here.

Once these gates pass, use a **new** subfolder such as
`supplements/reasoning-v1/llmseg-coco-full-v1` in
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



## Completed COCO build (2026-11-08)

The final user-selected scope excludes EgoObjects. All 8,980 COCO source images
were available. Of 33,755 source questions, known held-out identities exclude
2,116 questions on 521 images. Of 31,639 inspected questions, 10,925 fail the
existing area rule and one has an empty mask. Native preparation yields 20,713
rows. Every native asset was validated before cropping.

The existing 512-pixel crop-visibility rule finds 2,967 directly unsafe rows.
Whole-image closure **within this LLMSeg component only** removes 7,215 rows
on 2,214 decoded images, retaining **13,498 training pairs**. It is not propagated
to other source components. No train/validation/test split is relabeled.

Final `training.jsonl` SHA256:
`1d07e0902c032d56c9b92626447732ac63851da53fbf291e147227d38ab946a8`.
All 13,498 retained rows passed the actual training loader, with finite 512x512
RGB targets and nonempty sampled crops; collator checks passed. All final native
assets and a local `ovb_res_bundle_v1` restore were validated, with restored
manifest equality and loader smoke checks. No training was launched.

The public package preserves official per-image COCO source/license attribution,
source hashes, rejection/crop ledgers, and eight deterministic visual samples.
The visual samples align masks to objects, but generated questions can remain
ambiguous (for example COCO image 244476); this is not semantic certification of
all questions. Native annotations were not relabeled.

Accepted rows have zero exact decoded-RGB overlap with the 200 available official
ReasonSeg validation images. Known COCO evaluation IDs were excluded. The supplied
old supplement exclusion manifest itself contains only train rows, so it contributes
no held-out pixel exclusions. Full mixed-pool/baseline pixel exclusion remains
unverified; do not claim global leakage freedom from these component checks.

The release replaces the old 4,304-row LLMSeg component. It is now published at
[immutable HF revision 9ead7f4](https://huggingface.co/datasets/Melo111/open-vision-banana-res/tree/9ead7f4c8ea1713fb5a2193ad2958ea02123f774/supplements/reasoning-v1/llmseg-coco-full-v1).
The package contains 27 files, totaling 2,773,988,959 bytes including provenance
and visual QA. All 27 remote file sizes/hashes match: 11 tar shards were checked
against HF LFS SHA256 metadata, and the remaining files were anonymously downloaded
and hashed. This is not a claim that every remote tar was downloaded again.

## Download the pinned published release

From the repository containing merged PR #110 and its normal data dependencies:

```bash
python scripts/download_res_dataset.py \
  --repo-id Melo111/open-vision-banana-res \
  --revision 9ead7f4c8ea1713fb5a2193ad2958ea02123f774 \
  --subfolder supplements/reasoning-v1/llmseg-coco-full-v1 \
  --output-dir data/llmseg_coco_full
```

The downloader checks bundle hashes and restores native images, masks and
`training.jsonl`. Use that manifest with `data/llmseg_coco_full` as its data root.
Replace the old 4,304-row LLMSeg component; do not append both versions.

**Server network verification remains blocked.** The anonymous server attempt
could not connect to `huggingface.co:443` (connection timeout; the HF client also
reported network unreachable). Zero remote files were downloaded there, so no
remote-restored server loader result is claimed. The earlier successful local
bundle restore and all-13,498-row loader run are separate prepublication checks.
No training was launched and no network/security setting was changed.

See [machine-readable verification](llmseg_coco_release_verification.json) for
per-file hashes, exact verification scopes and the remaining server-network check.
Run the command on a node with working HTTPS access to Hugging Face before starting
training. The upstream annotation-license and mixed-pool leakage limitations above
still apply.
