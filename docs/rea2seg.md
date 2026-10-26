# Rea²Seg-16K training pairs

The adapter from PR #26 is now integrated with the current main branch.
Rea2Seg and LLM-Seg40K share the offline writer in `res_preparation.py`; the
runtime training loader remains unchanged. For the fixed cleaned subset,
portable supplement and R0/R1/R2 settings, follow
[res_reasoning_sources.md](res_reasoning_sources.md). The optional
`--reject-invalid` flag records and excludes invalid/duplicate/held-out incoming
rows; the default behavior below still fails explicitly. COCO image ids are
also checked when recoverable from the original image filename and supplied
with `--exclude-image-ids`. No resize repair is performed.

This offline adapter uses [PR #25](https://github.com/isLinXu/open-vision-banana/pull/25)'s
generic pair manifest, canonical expression prompt, strict COCO decoder and
`binary_white_black_v1` contract. It adds no runtime dataset branch or training
parameters and does not change evaluation, models, losses or existing defaults.
The RefCOCO-specific builder/validator remains specific to RefCOCO.

## Source and conversion

The [dataset card](https://huggingface.co/datasets/snowball521/Rea2Seg-16K/blob/1b2247ca67e50e76f6494065ae089be4bd6ccd19/README.md)
and viewer schema were checked at revision
`1b2247ca67e50e76f6494065ae089be4bd6ccd19`: one `train` split with
16,090 question/mask records, comprising `gqa-seg` 7,998, `lisa-plus` 7,052,
`spatial` 801 and `reasonseg` 239. These are upstream row counts, not unique
images or counts verified by a full local conversion.

The actual fields are `image` (Parquet bytes/path image), `question`,
`cot_answer`, `mask` (JSON string containing `{"rle":{"size":[H,W],"counts":...}}`),
and `category`. There is no upstream ID column. Each output sample gets a
deterministic ID from dataset/revision, shard name/SHA256 and zero-based row
index; an optional supplied `id`/`sample_id` is retained as `source_sample_id`.
Original split, category, native dimensions and source position are retained.
`expression` is the unchanged question, including whitespace; `prompt` comes
from `build_binary_referring_prompt`. `cot_answer` is metadata only.

Install only the preparation dependencies needed for the chosen input:

```bash
pip install -e . pyarrow pycocotools
# Local Parquet shards: this command performs no network access.
python scripts/prepare_rea2seg.py \
  --parquet downloads/rea2seg/data/train-*.parquet \
  --revision 1b2247ca67e50e76f6494065ae089be4bd6ccd19 \
  --output-dir data/rea2seg
```

The shell example uses Bash glob expansion; in PowerShell pass explicit shard
paths (e.g. `(Get-ChildItem downloads/rea2seg/data/train-*.parquet).FullName`).
Local files are explicitly declared to be source **train** data; do not pass
evaluation Parquet renamed as training data. Without `--revision`, local
provenance records a null revision plus each shard's exact SHA256.

HF input reads an **already populated cache** and always uses
`local_files_only=True`; it never downloads data. Download separately if needed:

```bash
pip install huggingface_hub
hf download snowball521/Rea2Seg-16K --repo-type dataset \
  --revision 1b2247ca67e50e76f6494065ae089be4bd6ccd19 --include 'data/train-*.parquet'
python scripts/prepare_rea2seg.py --dataset snowball521/Rea2Seg-16K \
  --revision 1b2247ca67e50e76f6494065ae089be4bd6ccd19 --output-dir data/rea2seg
```

Use either local or cached input, and a new output directory. Optional
`--categories gqa-seg spatial --max-samples 32` selects a deterministic pilot:
shards sorted by name, rows in source order, cap applied after category filtering.
Rows after the cap are not audited. The cache must contain the full numbered
train shard set, even for a pilot. Missing dependencies/cache, malformed RLE,
empty masks, size mismatches, duplicate IDs/queries and output conflicts fail
explicitly with source position. Nothing is silently substituted or dropped.
Failed builds retain `BUILD_INCOMPLETE`, which the existing loader rejects;
correct the source and build into a fresh directory.

Images and masks are native-resolution lossless RGB PNGs. Masks contain only
white foreground and black background. Asset paths in `training.jsonl` are
relative to the output root; source image filenames are not used as output
names. Same-image/different-question supervision is allowed.

## Independent and mixed training

Use the existing trainer after installing its normal training dependencies:

```bash
python scripts/train.py --config configs/sana_rea2seg.yaml
# Inspect configuration without loading models or running training:
python scripts/train.py --config configs/sana_rea2seg.yaml --print-effective-config
```

The independent manifest is train-only; it creates no validation set or new
evaluation protocol. For mixed training, prepare a new portable directory:

```bash
python scripts/prepare_rea2seg.py \
  --parquet downloads/rea2seg/data/train-*.parquet \
  --revision 1b2247ca67e50e76f6494065ae089be4bd6ccd19 \
  --merge data/res_binary_v1/training.jsonl data/res_binary_v1 \
  --exclude data/res_binary_v1/pairs.jsonl data/res_binary_v1 \
  --output-dir data/res_rea2seg
python scripts/train.py --config configs/sana_rea2seg.yaml \
  --pair-manifest data/res_rea2seg/training.jsonl --data-dir data/res_rea2seg \
  --tasks referring --output-dir output/sana_res_rea2seg
```

`--merge MANIFEST ROOT` accepts one existing paired train/val manifest, resolves
its assets against its own root, preserves splits/metadata, and copies assets
into the new root. Other tasks retain their target bytes; list those tasks in
the existing `--tasks` option and use their established training configuration
and loss weights. Do not use the legacy merger to reassign splits.

Repeat `--exclude MANIFEST ROOT` for **all known validation/test collections**,
including ReasonSeg evaluation pairs. Exclusion manifests need only `input`
and `split`; their train rows are ignored, val/test/testA/testB images are
indexed. Evaluation rows are never added to the training manifest. Training
overlap with any held-out image, or validation overlap with test, is an error.

Duplicates and cross-split overlap use SHA256 of decoded RGB pixels plus native
dimensions, not filenames or encoded-file hashes. This detects lossless
re-encoding across roots. Exact image/task/canonical-prompt duplicates fail;
incoming `reasonseg` images already in existing training data fail even if
their question differs. This guards against counting the upstream 239 ReasonSeg
training records twice. Resolve duplicates explicitly in source manifests, or
exclude the whole `reasonseg` category using `--categories gqa-seg lisa-plus spatial`.
Existing validation rows are never removed or reassigned automatically.
Different crops/resolutions/lossy encodings cannot be identified by this exact
pixel check; audit their source identities separately. No leakage guarantee is
made for evaluation collections not supplied to `--exclude`.

`report.json` separates scanned/category-filtered/added rows, existing rows,
output train/val counts, new rows per category, unique images/targets and
exclusion counts. Counts describe records, not accuracy or effective training
weights. Tests use tiny synthetic CPU fixtures only; no full download, GPU
training, trained weights or accuracy improvement is claimed.
