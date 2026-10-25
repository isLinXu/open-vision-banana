# Supervised RES with Rea2Seg and LLM-Seg40K

This experiment extends the existing RefCOCO-family release with reasoning
questions and native masks. It ports the offline adapter from
[PR #26](https://github.com/isLinXu/open-vision-banana/pull/26) onto current main.
Both source adapters use `res_preparation.build_res_pairs`, the existing
`decode_coco_segmentation`, `build_binary_referring_prompt`, and
`binary_white_black_v1` contract. Training still uses
`ManifestVisionDataset -> TaskWeightedSampler -> _collate_fn`; there is no new
runtime Dataset, loss, prompt protocol or task name.

## Sources and subset policy

| Addition | Selection | Purpose |
| --- | --- | --- |
| [Rea2Seg-16K](https://huggingface.co/datasets/snowball521/Rea2Seg-16K/tree/1b2247ca67e50e76f6494065ae089be4bd6ccd19) | `gqa-seg`, `lisa-plus`, `spatial`; exclude the entire `reasonseg` category | Object relations, reasoning queries and spatial references, without adding the upstream ReasonSeg training subset |
| [LLM-Seg40K](https://github.com/wangjunchi/LLMSeg/tree/65147a98cc49c2dac9b7a3633d9d3d336f1ac023) | Official **COCO train** partition; at most 5,000 images and one question per image | A bounded addition of longer reasoning queries backed by published native RLE masks |

The actual official LLM-Seg annotation files contain 44,780 training questions
over 11,336 images: COCO 33,755 / 8,980 and EgoObjects 11,025 / 2,356. Its
validation file contains 3,905 questions over 1,000 images. These are source
counts, not the size of the cleaned subset. EgoObjects is excluded from this
experiment: COCO shares the established image identities and exclusion lists
with RES, and does not require another image corpus or SAM proposal extraction.

LLM-Seg `qa_pairs[].question` is the expression and `qa_pairs[].rle_seg` is the
target. `answer` remains metadata. SAM proposals, boxes, answer text and
chain-of-thought are never substituted for target masks or appended to prompts.
The upstream loader implementing that schema is
[`utils/llm_seg_dataset.py`](https://github.com/wangjunchi/LLMSeg/blob/65147a98cc49c2dac9b7a3633d9d3d336f1ac023/utils/llm_seg_dataset.py).

LLM-Seg selection removes all COCO identities from its official validation file
and the supplied RES evaluation image-id union. Remaining images and their
questions are SHA256-ranked with seed 42, independently of JSON key order.
One question is selected per image. Masks must be nonempty, valid native RLE,
match image dimensions, and cover 0.5%–95% of pixels. This is a reproducible
structural filter, not a guarantee that every generated question is semantically
correct. The cap applies after asset validation and rejection. Missing images
abort the build; the subset never changes according to which images happen to
be installed. Questions after the cap are not audited.

`--reject-invalid` explicitly enables cleaning. Every rejected incoming record
is recorded in `rejected.jsonl` with source position and reason. Without it,
invalid/duplicate/overlapping records fail as in PR #26. No mask is stretched,
resized, filled from a box or regenerated. Existing train/val rows passed with
`--merge` remain unchanged in split; invalid base rows still fail. Source
revision, shard SHA256, row/question identity, original split, image dimensions,
COCO id where available, and decoded-pixel hashes are retained.

## Reproduce preparation

Install `pip install -e '.[data]'`. The following Bash commands assume the
canonical RES release has been restored to `data/res_binary_v1` using the
[existing release instructions](res_data.md). Use a fresh output directory.
On PowerShell, expand the Parquet glob with `Get-ChildItem` as in
[the PR #26 guide](rea2seg.md).

```bash
hf download snowball521/Rea2Seg-16K --repo-type dataset \
  --revision 1b2247ca67e50e76f6494065ae089be4bd6ccd19 \
  --include 'data/train-*.parquet' --local-dir data/raw/rea2seg
PYTHONPATH=src python scripts/prepare_rea2seg.py \
  --parquet data/raw/rea2seg/data/train-*.parquet \
  --revision 1b2247ca67e50e76f6494065ae089be4bd6ccd19 \
  --categories gqa-seg lisa-plus spatial --reject-invalid \
  --exclude data/res_binary_v1/pairs.jsonl data/res_binary_v1 \
  --exclude-image-ids data/res_binary_v1/evaluation_image_ids.json \
  --output-dir data/prep/rea2seg
```

Download `train.json` and `validation.json` from the authors'
[official annotation folder](https://drive.google.com/drive/folders/19MyzJN9hkvTSUr2lUFlMwmS2TSRRKmUX).
The CLI verifies both complete file hashes before converting:

| File | Google Drive file id | SHA256 |
| --- | --- | --- |
| `train.json` | `1L7IXSLxADcACOEJ-DfjAc632IByNBNsw` | `d925dabf2465f066c43f5d14b1360cf97de281744af15c1d063fe7cc61508754` |
| `validation.json` | `1DBFMQcvC0zwAb8Vcw1sQDpRAA9Qp5Eg1` | `e25345607d94aea5f30b09ebdd6d9acf024f9a10ea3953d6300245a2cad27689` |

Use COCO train2017 images, or the equivalent 2014 names for the same image ids.
The adapter reads those existing images and requires no SAM model or GPU.

```bash
PYTHONPATH=src python scripts/prepare_llmseg.py \
  --annotations data/raw/llmseg/train.json \
  --validation-annotations data/raw/llmseg/validation.json \
  --image-root data/raw/coco2017/train2017 \
  --exclude-image-ids data/res_binary_v1/evaluation_image_ids.json \
  --exclude data/res_binary_v1/pairs.jsonl data/res_binary_v1 \
  --max-samples 5000 --seed 42 --reject-invalid \
  --output-dir data/prep/llmseg
```

Repeat `--exclude MANIFEST ROOT` for other known held-out image collections.
The exclusion checks compare COCO ids when recoverable and exact decoded RGB
pixels/dimensions across differently named or losslessly encoded files. They
do not establish disjointness for crops, resized images, lossy re-encodings or
unknown collections. This supervised experiment makes no strict zero-shot
claim: the base RES corpus includes RefCOCOg training annotations.

## Fixed experiment settings

| Run | Training manifest | Change |
| --- | --- | --- |
| R0 | Original `data/res_binary_v1/training.jsonl` | Reference |
| R1 | R0 + cleaned Rea2Seg | Add reasoning/spatial source |
| R2 | R1 + selected LLM-Seg COCO pairs | Add a second reasoning source |

All three YAMLs use Qwen-Image-2.1, 512px, rsLoRA r32/alpha64, 10,000 optimizer
steps, batch 1 x accumulation 32, AdamW at 3e-5, cosine schedule and seed 42.
Only manifest/data/output paths differ. The task is `referring` and
`vision_ratio=1.0`; this is a supervised RES data ablation, not a reproduction
of the paper or a generation-preservation experiment. Sampling within the
single task is proportional to rows; adding a source changes its frequency.
No additional source-balancing sampler or duplicated rows are introduced.

Use the existing merger with **no cap and no new validation split**. The base
`training.jsonl` already contains its test-disjoint checkpoint-validation view;
the supplements are train-only. The complete official benchmark manifests stay
separate. Do not use base `pairs.jsonl` as a training input.

```bash
python scripts/merge_multitask_manifest.py \
  --inputs data/res_binary_v1/training.jsonl data/prep/rea2seg/training.jsonl \
  --out data/res_reasoning/r1/training.jsonl --val-percent 0
python scripts/merge_multitask_manifest.py \
  --inputs data/res_binary_v1/training.jsonl data/prep/rea2seg/training.jsonl data/prep/llmseg/training.jsonl \
  --out data/res_reasoning/r2/training.jsonl --val-percent 0
PYTHONPATH=src python scripts/train.py --config configs/experiments/qwen_image_21_res_base.yaml
PYTHONPATH=src python scripts/train.py --config configs/experiments/qwen_image_21_res_rea2seg.yaml
PYTHONPATH=src python scripts/train.py --config configs/experiments/qwen_image_21_res_rea2seg_llmseg.yaml
```

Check every merge audit: `per_task_dropped` must be empty and all three runs
must retain exactly the base validation rows. Record manifest hashes and source
counts with each run. Compare the same official RefCOCO/+/g held-out protocols
using `evaluate_res_manifest.py`, plus ReasonSeg validation separately with its
query protocol recorded. LLM-Seg validation is excluded from the new subset,
but has not been globally removed from the original RES training pool; do not
report R0–R2 as held-out evaluation on LLM-Seg validation without an additional
global exclusion/rebuild. GPU training and accuracy conclusions are separate
from data/interface verification.

## Optional portable publication

Source-adapter output is validated with `validate_prepared_res.py`, not the
RefCOCO-specific source-accounting validator. It then reuses the existing
`ovb_res_bundle_v1` packager and restorer, including SHA256 checks and safe
extraction. Keep supplements in a new Hub subfolder; do not replace the
existing root release, official benchmark manifests or immutable revisions.

```bash
python scripts/validate_prepared_res.py data/prep/llmseg
python scripts/package_res_dataset.py --data-dir data/prep/llmseg --output-dir output/llmseg_bundle
# After publishing that bundle under supplements/reasoning-v1/llmseg-coco5k:
python scripts/download_res_dataset.py --repo-id Melo111/open-vision-banana-res \
  --revision <immutable-supplement-commit> --subfolder supplements/reasoning-v1/llmseg-coco5k \
  --output-dir data/prep/llmseg
```

Upstream image/annotation ownership and terms remain applicable. These adapters
do not relicense source data under the repository's code license. See each
source's original card, COCO image terms and authors' release. Publication must
include the preparation report, rejection ledger, source pins and attribution.
