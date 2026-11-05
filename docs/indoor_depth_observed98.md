# Indoor depth supplement: observed pixels >=98%

This supplement adds real indoor RGB/depth training pairs for evaluating transfer
to NYU Depth V2 and Hypersim. Preparation selects images and converts their
metric targets. It does not modify the training implementation. There is no
fixed 500k quota: observed depth quality and visual diversity determine the count.

## Published release

| Source | Train | Validation |
|---|---:|---:|
| ARKitScenes | 290,391 | 8,602 |
| DIODE indoor training subset | 996 | 0 |
| SUN RGB-D non-NYU | 766 | 0 |
| **Total** | **292,153** | **8,602** |

The release contains **300,755 unique pairs**, 1,203,020 native PNG assets and
134 TAR shards (37.83 GB indexed payload). All 5,061 eligible Apple videos were
acquired successfully; final frames represent 1,664 distinct ARKit visit IDs.
The minimum retained native observed fraction is 98.000081%, and the frame-mean
fraction is 99.420076%. Independent validation measured 2.085 mm mean RGB-depth
encoding error and 49.759 mm maximum error across observed pixels. Full bundle
restoration passed. The unchanged training loader at 512 resolution passed on
64 samples covering all three sources and the ARKit validation split.

[Hugging Face component](https://huggingface.co/datasets/Melo111/open-vision-banana-depth/tree/b65093931678929d0e5c98b6cac153561865ced0/indoor-observed98-v1)
contains source metadata, original license notices, exclusion records, checksummed
asset inventories, native validation and transport reports, and deterministic
[ARKitScenes](https://huggingface.co/datasets/Melo111/open-vision-banana-depth/resolve/5a864bd118b5bca23e814917b62c0cb9f1f56da6/indoor-observed98-v1/visualizations/arkitscenes_samples.png),
[DIODE](https://huggingface.co/datasets/Melo111/open-vision-banana-depth/resolve/5a864bd118b5bca23e814917b62c0cb9f1f56da6/indoor-observed98-v1/visualizations/diode_indoor_samples.png),
[SUN RGB-D](https://huggingface.co/datasets/Melo111/open-vision-banana-depth/resolve/5a864bd118b5bca23e814917b62c0cb9f1f56da6/indoor-observed98-v1/visualizations/sunrgbd_nonnyu_samples.png)
sample sheets.

Download only this component at its immutable data revision:

```bash
python scripts/download_res_dataset.py \
  --repo-id Melo111/open-vision-banana-depth \
  --revision 5a864bd118b5bca23e814917b62c0cb9f1f56da6 \
  --subfolder indoor-observed98-v1 \
  --output-dir data/indoor-observed98-v1
```

Dataset fingerprint:
`b888c55be9c801fd0d1c723baa05df8de6dd71ec56089f4d53dd2fc65338575d`.
The data card is pinned separately so that its download command refers to the
immutable payload commit without altering the bundle checksums.

[Anonymous publication verification](https://huggingface.co/datasets/Melo111/open-vision-banana-depth/blob/b65093931678929d0e5c98b6cac153561865ced0/indoor-observed98-v1/publication_verification.json)
checks all uploaded file sizes and hashes, the complete downloaded manifest and
representative downloaded TARs for all four asset roles. Existing Hub payloads
are preserved.

## Sources and evaluation boundaries

| Source | Eligible data | Exclusions / interpretation |
|---|---|---|
| ARKitScenes | Official raw Training videos; Validation remains validation | Unknown visit IDs and training videos sharing a held-out visit are excluded. Native 256x192 RGB, depth and confidence are synchronized. |
| DIODE indoor | An immutable raw training subset, checked against official train IDs | Outdoor images and every official validation ID are excluded. Future DIODE validation is same-source held-out evaluation, not source-unseen evaluation. |
| SUN RGB-D | Official 5,285-capture training list from the original toolbox | Every NYU source path and all 5,050 official test captures are excluded. Original `depth/` observations are used. `depth_bfx/` refinement does not count as observed depth. |

This protocol concerns the new supplement. Other components in the existing Hub
repository retain their original protocols. Audit their source/split identities
separately when constructing a combined training mixture.

Source snapshots:

- [Apple ARKitScenes](https://github.com/apple-aiml-research/ARKitScenes/tree/474aaa83d7b1c9080619d8485de2d7c885c01fc5),
  raw v1 assets and `metadata.csv` from Apple's public distribution. Archive ETags,
  native ZIP member CRCs and selected member SHA256 hashes are recorded.
- [DIODE](https://diode-dataset.org/) and
  [official devkit](https://github.com/diode-dataset/diode-devkit/tree/8b1765b7d801a5f5e2877c434ffe164e62ce8c90).
  Download mirror: `sayakpaul/diode-subset-train` at
  `7b2a00c90ab7898d94ecca9300987379b1636fa7`;
  `train_subset.tar.gz` SHA256
  `393d93089aa47caa3e13e3f108077f4878ff0dbbeec7eea8bbc8f1d3af38a476`.
  Despite the suffix, this immutable object is an uncompressed TAR. The subset is
  not represented as the entire official DIODE training release.
- [SUN RGB-D](https://rgbd.cs.princeton.edu/), original toolbox
  `SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat`.
  Mirror: `ZHAO11564/SUNRGBD.zip` at `4fafeae5f5bad5764cb8f3e89c38a5389ad2b633`;
  archive SHA256 `1a6dbf2a1c9044c4805a35ee648d616ea39a231fd5bd6f77e84cd2b8287fe41c`.

Retain each source's original terms and attribution. Apple's custom license and
DIODE's MIT notice accompany the publication. SUN RGB-D combines original data
providers; this derivative does not grant a new blanket license to their images.

## Image selection

- Valid means a finite depth in [0.25, 20] metres plus the source validity marker.
  ARKitScenes requires medium/high confidence (confidence >=1). DIODE uses its
  official validity mask. SUN uses nonzero original sensor depth after decoding.
- At least 98% of native pixels must be valid; exactly 98% is accepted. Coverage
  is measured before any resize, crop, or display normalization.
- Reject flat/distant depth distributions (p95-p5 <0.3 m or median >10 m), badly
  exposed images (mean grayscale outside [8,247]), and blur/featureless images
  (native grayscale Laplacian variance <15). These are recorded preparation
  choices, not guarantees of perfect sensor annotations.
- Final selection also rejects views with little distributed visual structure.
  On a grayscale thumbnail bounded by 256x256, apply Gaussian sigma=1 and Sobel
  gradients divided by 8. Require at least 1% of pixels to exceed magnitude 5,
  and at least 20% of an 8x8 tile grid to contain >=1% such pixels. This reduces
  blank walls and ceilings whose native-resolution sensor noise passes the blur
  gate; thresholds are preparation choices, not benchmark-tuned metrics.
- ARKitScenes uses a minimum 0.5-second spacing, nearest depth timestamp within
  8 ms, and no reused depth frame. RGB, depth and confidence use the same upright
  rotation from Apple's `sky_direction` convention.
- Exact decoded RGB duplicates are removed. Within scene groups, reject near
  duplicates with 64-bit difference-hash Hamming distance <=4 and 16x16 grayscale
  MAE <5. Candidate order is hash-ranked with a fixed seed, interleaved across
  videos. Retain at most 128 views/video and 384 views/scene group to limit repeated
  exposure of long captures. ARKit groups are physical visits; SUN groups are
  source capture IDs, not an independently recovered building/room identity.
- Retain only frames whose RGB depth encoding has maximum error <=5 cm over
  observed pixels. The release-wide observed-pixel mean error must be <1 cm,
  matching the existing depth supplement quality gate.

## Native training format

`pairs.jsonl` uses the current `task/input/target/prompt/split` contract. Each row
has `task: depth` and the existing metric-depth prompt. Native RGB pixels are
stored losslessly as PNG. Depths are planar z-depth in metres, encoded through
the existing metric-power cube-edge mapper (lambda=-3, c=10/3).

`metric_depth/` contains uint16 millimetres (`metric_depth_scale: 1000`), with zero
only at unobserved pixels. ARKit and SUN millimetres are preserved; floating DIODE
measurements are rounded to the nearest millimetre. `valid_masks/` stores the
exact accepted observation mask as 0/255 PNG. Dimensions and rotation agree
across all four files.

There is no depth interpolation. Missing target pixels use the existing zero /
black placeholder, occupy at most 2% of a retained image, and are not claimed to
be valid measurements. Masks are provenance: the unchanged OVB trainer does not
apply them to the loss. This small amount of missing-target noise is an explicit
limitation of retaining the existing training contract.

Per-frame source paths, timestamps, input hashes, source ZIP hashes/ETags,
observed fractions, exposure/blur statistics and encoding errors accompany the
release. Train/validation groups stay disjoint. No NYU or Hypersim measurement
is added by this supplement.

DIODE raw depths follow the existing OVB DIODE evaluation convention, without a
ray-distance correction. A separate geometric consistency check on four large
planar views supports camera z-depth: a 3 mm plane fit explains 44.6–68.0% of
valid pixels using raw depth, versus 4.8–11.2% after treating it as ray distance.
This is an empirical check, not an explicit convention statement by the source
authors; its source IDs, procedure and results accompany the release.

## Merge with existing manifests

Use an already-audited base mixture. The existing merge helper preserves the
source validation rows when depth already has a validation partition. This PR
also teaches it to rebase optional `valid_mask` paths with the core assets:

```bash
python scripts/merge_multitask_manifest.py \
  --inputs data/base/pairs.jsonl data/indoor-observed98-v1/pairs.jsonl \
  --out data/combined/pairs.jsonl
```

The combined manifest can be consumed directly by the current trainer. If you
rerun `build_qwen_baseline_manifest.py`, its strict paper protocol deliberately
filters every DIODE path, including official training data. To retain DIODE,
use its existing `--protocol in_domain --allow-benchmark diode` options and keep
the other audit arguments for your base mixture. That mode also permits
Cityscapes in the current builder: exclude Cityscapes from the input if retaining
its zero-shot protocol. NYU and other unallowed benchmark guards remain active.
The release includes a real-pair compatibility check for both protocol modes.
That audit copies path strings: keep its output beside the combined manifest,
or retain the combined directory as the loader's data root. The existing
`--absolute` merge option is also available for a machine-local mixture.

## Rebuild and verify

Preparation dependencies are the existing OVB environment plus `remotezip`,
`requests`, Pillow, NumPy and SciPy. Source downloads require no Hub token.
Acquisition supports an optional `INDOOR_DATA_PROXY` HTTPS proxy or a comma-separated
`INDOOR_DATA_PROXIES` list, round-robin across video sessions. Large RGB/depth ZIPs
use bounded byte ranges; small confidence ZIPs may be read once in memory (<=1 MiB)
to reduce tiny HTTP requests. Both transports pin ETags and validate member CRCs.

```bash
python scripts/prepare_arkitscenes_indoor.py \
  --metadata source/arkit_metadata.csv --output-dir data/indoor_observed98 --workers 32
python scripts/prepare_diode_indoor.py \
  --archive source/train_subset.tar.gz --official-metadata source/diode_meta.json \
  --output-dir data/indoor_observed98
python scripts/prepare_sunrgbd_indoor.py \
  --archive source/SUNRGBD.zip --official-split source/allsplit.mat \
  --output-dir data/indoor_observed98
python scripts/finalize_indoor_depth.py --source data/indoor_observed98
python scripts/validate_package_indoor_depth.py data/indoor_observed98 \
  --workers 16 --bundle-dir output/indoor-observed98-v1
python scripts/visualize_indoor_depth.py data/indoor_observed98 \
  --output-dir output/indoor_visualizations
```

Use the official Apple raw-v1 metadata, the official DIODE devkit enumeration,
and the original SUN toolbox split file above. Hashes of the exact metadata
bytes accompany the release. Download the two mirrors at their immutable
revisions, not their moving main branches.

The image transport is the existing OVB bundle format. A source-specific
packager appends observation-mask TARs to its index, so the existing download /
restore script reconstructs every path without changing the shared loader or
training code. `assets.jsonl` inventories the three core training assets;
`valid_mask_assets.jsonl` inventories the masks. Both are independently checked.

The visualization sheets show six deterministic scene-diverse samples, a
validation example where available, and the lowest-coverage retained example.
RGB, observed metric depth, cube-edge training target and observation mask are
shown separately; these are ground-truth conversions, not model predictions.
