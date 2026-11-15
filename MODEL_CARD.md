# Model Card: Vision Banana

**Model Page**: https://github.com/isLinXu/open-vision-banana

## Model Details

### Model Description

Open-Vision-Banana is a method-aligned, open implementation of the Vision Banana paradigm. It instruction-tunes an open image generator (Qwen-Image-2.1 in the current runs; SANA-1600M and FLUX.1-dev in earlier ones) with shared LoRA adapters so that prompts select vision tasks and outputs remain RGB images. These are open substitutes for the paper's closed Nano Banana Pro base; the paper's numbers have not been reproduced.

The current public release line is Qwen-Image-2.1: a fused full-weight base, [`Melo111/Fusion-C1-KQ0r128-60-40`](https://huggingface.co/Melo111/Fusion-C1-KQ0r128-60-40), and two 100k-step rsLoRA arms trained on it (I108 A and B). All published weights are listed under [Published Checkpoints](#published-checkpoints).

- **Developed by**: Open Vision Banana Contributors
- **Model type**: Instruction-tuned diffusion model for vision understanding
- **Languages**: English (prompts), Visual (outputs)
- **License**: MIT (project code); base models and datasets follow their own licenses
- **Base model**: Qwen-Image-2.1 (current runs; the I108 adapters must be loaded on the fused base `Melo111/Fusion-C1-KQ0r128-60-40`); SANA-1600M and FLUX.1-dev (earlier runs). Each checkpoint's `metadata.json` records its `model_name`

### Intended Uses

| Task | Input | Output | Intended Users |
|:-----|:------|:-------|:---------------|
| Semantic Segmentation | RGB image + "segment" prompt | Per-pixel class labels | CV researchers, autonomous driving engineers |
| Instance Segmentation | RGB image + "instance segment" prompt | Per-instance masks | Robotics engineers, medical imaging researchers |
| Referring/Reasoning Segmentation | RGB image + expression | Binary region mask | Vision-language researchers |
| Depth Estimation | RGB image + "estimate depth" prompt | Dense depth map (meters) | 3D reconstruction, SLAM researchers |
| Surface Normal Estimation | RGB image + "estimate normals" prompt | Per-pixel normal vectors | Scene understanding, graphics researchers |

### Out-of-Scope Uses

- Real-time safety-critical applications (model has stochastic output)
- Medical diagnosis without expert validation
- Video understanding (single-frame only)
- Long-text reasoning beyond visual tasks

## Performance

### Paper-Reported Reference Results

The values below are reported by arXiv:2604.20329v3 for the closed Vision Banana system. They are reference targets, not measurements from the checkpoints in this repository. Project results must be published separately with checkpoint, config, dataset, and result hashes.

#### Semantic Segmentation (Cityscapes val)

| Method | mIoU (%) |
|:-------|:---------|
| SAM 3 | 65.2 |
| **Vision Banana (paper)** | **69.9** |

#### Depth Estimation (4-dataset average)

| Method | δ1 ↑ |
|:-------|:-----|
| Depth Anything 3 | 0.918 |
| **Vision Banana (paper)** | **0.929** |

#### Surface Normal Estimation (paper-reported aggregate)

| Method | Mean Angular Error (°) ↓ |
|:-------|:-------------------------|
| Lotus-2 | 19.642 |
| **Vision Banana (paper)** | **18.928** |

### Project Evaluation Status

No checkpoint in this repository reproduces the paper's benchmark numbers, and no
checkpoint beats its parent on every benchmark. Longer training raises metric depth
while Cityscapes and ReasonSeg fall. The best measured value per benchmark is listed
below. All rows use the README protocol: raw metric δ1 with no scale alignment,
scored at native ground-truth resolution. Per-run tables, sample counts and inference
settings are in the README section
[Results](README.md#-results).

| Benchmark | Best measured | Checkpoint | Paper |
|:----------|--------------:|:-----------|------:|
| NYUv2 δ1 (n=654) | 0.497 | I108 B, `checkpoint-100000_final` | 0.948 |
| ETH3D δ1 (n=454) | 0.292 | Fused base, no adapter | 0.935 |
| DIODE indoor δ1 (n=325) | 0.353 | I108 B, `checkpoint-100000_final` | 0.917 |
| Cityscapes val mIoU (n=500) | 0.386 | Fused base, no adapter | 0.699 |
| RefCOCOg UMD val cIoU (n=4896) | 0.547 | I102-i1 (`ovb-ft_bl_i102i1_t1_cube_20261105_2213`) | 0.738 |
| ReasonSeg val gIoU (n=344, `legacy_union`) | 0.440 | I108 B, `checkpoint-20000_best` | 0.793 |

I108 scores come from [`khazic/ovb-i120-external-eval`](https://huggingface.co/khazic/ovb-i120-external-eval/tree/88f43c92048111bd25b4483e0117494937acb27f).
That snapshot has no RefCOCOg results for I108. The I127 merge evaluation adds them: B `checkpoint-17500_best` 0.518, B `checkpoint-100000_best` 0.482. The paper's ReasonSeg number uses
queries rewritten by Gemini; ours uses the raw queries, so the two are not the same
protocol.

## Published Checkpoints

All repositories below are public on the Hugging Face Hub (checked 2026-11-14). For
reproducible results, pin the listed revision.

### Current release: fused Qwen-Image-2.1 base and I108 adapters

| Repository | Revision | Type | Contents |
|:-----------|:---------|:-----|:---------|
| [`Melo111/Fusion-C1-KQ0r128-60-40`](https://huggingface.co/Melo111/Fusion-C1-KQ0r128-60-40) | `9d5b9b8e` | Full diffusers pipeline, 33.2 GB | BF16 transformer plus text encoder, VAE, processor and scheduler. The transformer is 0.6 × C1 + 0.4 × K-Q0-r128 effective weights, averaged in FP32 and cast to BF16 once. Recipe and validation are under `provenance/`. This is the base for I108 |
| [`khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3`](https://huggingface.co/khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3) | `f5f44cbb` | rsLoRA r64 / α64, lr 3e-5, 100k steps | 37 `ckpt/checkpoint-N_best`, 20 periodic `ckpt/checkpoint-N` (every 5k) and `ckpt/checkpoint-100000_final`. Each has `lora_adapter/`, `lora.pt` and `metadata.json` |
| [`khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3`](https://huggingface.co/khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3) | `66b028c7` | rsLoRA r64 / α64, lr 1e-5, 100k steps | Same layout, with 38 `_best` checkpoints |
| [`khazic/ovb-i120-external-eval`](https://huggingface.co/khazic/ovb-i120-external-eval) | `88f43c92` | Evaluation results only | 205 `full.json` files covering the fused base and 40 I108 checkpoints on NYUv2, ETH3D, DIODE, Cityscapes and ReasonSeg |
| [`khazic/ovb-i127-merge-eval`](https://huggingface.co/khazic/ovb-i127-merge-eval) | `d218ff58` | Merged adapters + evaluation results | `merged/P1`–`P6` are exact ΔW-weighted merges of B `_best` adapters (rank 128, loader scale 1), and `merged/M0` is B 17.5k passed through the same path. `eval/` has the full six-benchmark results; archived in `docs/evaluation/i127-merge-20261115/` |

Both I108 arms use the training recipe in `configs/experiments/fusion_r64_large_scale.yaml`.
In `metadata.json`, `adapter_weight_source` records which weights the adapter holds; for
the `_best` checkpoints it is the EMA adapter.

Suggested starting points. Each one trades some tasks for others:

| Goal | Checkpoint | NYU δ1 | ETH3D δ1 | DIODE δ1 | Cityscapes mIoU | ReasonSeg gIoU |
|:-----|:-----------|-------:|---------:|---------:|----------------:|---------------:|
| Segmentation | Fused base, no adapter | 0.278 | **0.292** | 0.300 | **0.386** | 0.417 |
| Metric depth | B `checkpoint-100000_final` | **0.497** | 0.225 | **0.353** | 0.298 | 0.333 |
| Referring segmentation | B `checkpoint-20000_best` | 0.340 | 0.166 | 0.229 | 0.353 | **0.440** |

The A/B archive on branch `docs/i108-ab-eval-and-merge-20261114`
(`docs/evaluation/i108-ab-20261114/`) adds a second, independent evaluation that
includes RefCOCOg. There, B `checkpoint-17500_best` is the most balanced checkpoint:
NYU δ1 0.407, Cityscapes 0.357, ReasonSeg 0.434, RefCOCOg cIoU 0.518. The archive
also proposes six weight-merge recipes. All six have since been evaluated (I127,
[`docs/evaluation/i127-merge-20261115/`](docs/evaluation/i127-merge-20261115/results.md)). None passes every gate,
because each loses 3–6 pp of ReasonSeg, so B `checkpoint-17500_best` remains the balanced checkpoint. P5 and P6 recover
Cityscapes (0.364), and P6 raises NYU δ1 to 0.449.

### Earlier Qwen-Image-2.1 runs (README "Earlier Qwen-Image-2.1 runs" table)

| README row | Repository | What to load |
|:-----------|:-----------|:-------------|
| Parent (step 0) | [`khazic/ovb-i88_parent_e2b_final`](https://huggingface.co/khazic/ovb-i88_parent_e2b_final) | Evaluation and logs only; no weights |
| C1: depth 75 %, cube-edge only | [`khazic/ovb-i88_c1_depth_cube100_20261030_1602`](https://huggingface.co/khazic/ovb-i88_c1_depth_cube100_20261030_1602) | `checkpoint-10000_final` (also the 60 % parent of the fused base) |
| T0: all tasks + 10 % replay | [`khazic/ovb-i88_t0_full_replay10_20261029_2202`](https://huggingface.co/khazic/ovb-i88_t0_full_replay10_20261029_2202) | `checkpoint-10000_final` |
| T1: no normals | [`khazic/ovb-i88_t1_no_normal_20261030_1602`](https://huggingface.co/khazic/ovb-i88_t1_no_normal_20261030_1602) | `checkpoint-10000_final` |
| T2: depth 75 %, five colormaps | [`khazic/ovb-i88_t2_depth_replay_resume_20261030`](https://huggingface.co/khazic/ovb-i88_t2_depth_replay_resume_20261030) | `checkpoint-10000_final` (`ovb-i88_t2_depth_replay_20261030_0106` holds logs only) |
| T3: segmentation specialist | [`khazic/ovb-i88_t3_seg_replay_20261030_1602`](https://huggingface.co/khazic/ovb-i88_t3_seg_replay_20261030_1602) | `checkpoint-10000_final` |
| I101-h1: full fine-tune, lr 5e-6 | [`khazic/ovb-ft_mf_lr5e-6_i101h1_t1_cube_20261105_2213`](https://huggingface.co/khazic/ovb-ft_mf_lr5e-6_i101h1_t1_cube_20261105_2213) | `checkpoint-10000_final` (full transformer, 14.4 GB) |
| I102-i1: LoRA | [`khazic/ovb-ft_bl_i102i1_t1_cube_20261105_2213`](https://huggingface.co/khazic/ovb-ft_bl_i102i1_t1_cube_20261105_2213) | `checkpoint-10000_final` |

### Other published ablations

These repositories have only development-set metrics (`eval_dev/`, `eval_*_sampled200/`).
They are not in the README result tables.

| Family | Repositories | Notes |
|:-------|:-------------|:------|
| I97 fusion-parent sweep | `khazic/ovb-i97-{q0,q1,m0,m1}-s42_20261102_2007` (LoRA, 20k)<br>`khazic/ovb-i97-f-m1-lr{1e-6,2e-6,1e-5}-s42_20261103_queue` and `khazic/ovb-i97-f-m1-lr5e-6-s42_20261102_2007` (full fine-tune, 15k)<br>`khazic/ovb-i97-k-q{0,1}-r{64,128}-s42_20261103_queue` (LoRA, 20k)<br>`khazic/ovb-i97-r-m{0,1}-768-s42_20261103_queue` (768 px) | `k-q0-r128` is the 40 % parent of the fused base |
| Full fine-tune vs LoRA | `khazic/ovb-ft_{bl,ml,bf,mf}_t1_cube_*`, `khazic/ovb-ft_step0_base`, `khazic/ovb-ft_step0_merged`, `khazic/ovb-ft_mf_lr{5e-6,1e-5,2e-5}_t1_cube_*` | `b`/`m` = start from the base or the merged step-0 model; `l`/`f` = LoRA or full fine-tune. `ft_step0_merged/model` is a full merged model |
| E-series | `khazic/ovb-e0_qwen21_six_smoke_20261024_1401`, `khazic/ovb-e1_qwen21_paper_aligned_20261025_0250`, `khazic/ovb-e2_vr{02,08}_s{42,43}_*`, `khazic/ovb-e2b_cap10k_s42_20261026_1016` | Paper-aligned recipe, `vision_ratio` sweep and seeds |
| Shared-adapter sampling sweeps | `khazic/ovb-qwen_image_21_shared_{smoke,uniform,sqrt}`<br>`khazic/ovb-sana_1600m_shared_{uniform,sqrt,depth_heavy,semantic_heavy,rea2seg_balanced}` | Qwen-Image-2.1 and SANA-1600M task-weighting sweeps |

### Not publicly reachable

The README cites SANA-1600M and FLUX.1-dev results from `khazic/ovb-fix_ctrl_depth`,
`ovb-fix_ctrl_normal`, `ovb-fix_ctrl_seg`, `ovb-fix_seg_plus_gcg`,
`ovb-fix_best3_rank64_60k` and `ovb-flux3_ctrl_{depth,normal,seg}_nominsnr_15k`.
As of 2026-11-14 these repositories return 404 on the Hub, so their weights cannot be
downloaded. The README numbers for them are kept as a historical record.

### Published datasets

| Repository | Contents |
|:-----------|:---------|
| [`Melo111/open-vision-banana-depth`](https://huggingface.co/datasets/Melo111/open-vision-banana-depth) | Metric-depth training additions (e.g. DIODE indoor) |
| [`Melo111/open-vision-banana-res`](https://huggingface.co/datasets/Melo111/open-vision-banana-res) | Referring / reasoning segmentation data, see `docs/res_data.md` |
| [`Melo111/open-vision-banana-segmentation`](https://huggingface.co/datasets/Melo111/open-vision-banana-segmentation) | Segmentation data, see `docs/coconut_segmentation.md` |
| [`Melo111/open-vision-banana-external-eval`](https://huggingface.co/datasets/Melo111/open-vision-banana-external-eval) | Packaged external evaluation sets with acquisition notes and SHA256 sums |

### Loading a published checkpoint

The I108 repositories are about 200 GB each, so download only the base and the one
adapter folder you need:

```bash
huggingface-cli download Melo111/Fusion-C1-KQ0r128-60-40 \
  --revision 9d5b9b8ede04d7d9f4a4ff8a6a75a4b734ab1b4d --local-dir checkpoints/fusion-base
huggingface-cli download khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3 \
  --revision 66b028c792c3811ffc87941e0889f657765611a6 \
  --include "ckpt/checkpoint-20000_best/*" --local-dir checkpoints/i108-b

OPEN_VISION_BANANA_MODEL_PATH=checkpoints/fusion-base \
PYTHONPATH=src python scripts/evaluate_paper_benchmarks.py --model-name qwen-image-2.1 \
  --lora-path checkpoints/i108-b/ckpt/checkpoint-20000_best ...
```

Do not pass `hf:<repo>` for these multi-checkpoint repositories. It downloads every
`*.safetensors` file in the repo and returns the repo root, which is not an adapter
directory.

### Checkpoint Registry

The aliases in `configs/checkpoint_registry.yaml` predate the Hub releases above.
The `open-vision-banana/*` aliases are still placeholders, and the published `khazic/*`
and `Melo111/*` runs are not registered as aliases. Load them by local path after
downloading.

| Alias | Hugging Face Repo | Status |
|:------|:------------------|:-------|
| ``sana-depth-relative-v1`` | local adapter (see `local_path`) | **local** |
| ``vision-banana`` | ``open-vision-banana/vision-banana-lora`` | placeholder |
| ``vision-banana-depth`` | ``open-vision-banana/vision-banana-depth-lora`` | placeholder |
| ``vision-banana-normal`` | ``open-vision-banana/vision-banana-normal-lora`` | placeholder |

## Technical Specifications

### Architecture

| Component | Details |
|:----------|:--------|
| Base model | Qwen-Image-2.1, SANA-1600M or FLUX.1-dev, selected by `model_name` |
| Fine-tuning method | One shared rsLoRA adapter; rank and alpha are recorded per checkpoint |
| Trainable params | Computed per checkpoint; not assumed from rank alone |
| Training data mix | `vision_ratio` per config; project choice, not paper-disclosed |
| Depth mapping | Cube-edge traversal colormap + power transform (λ=-3, c=10/3) |
| Normal mapping | Camera-space XYZ → RGB linear mapping |
| Instance decoding | 5-stage: init → group → prune → boundary → merge |

### RGB Encoding Schemes

**Depth → RGB**: Figure 5 cube-edge traversal (black→red→yellow→green→cyan→blue→magenta→white) with power transform λ=-3, c=10/3. Near-depth precision prioritized. The `legacy_cube_edge` option supports checkpoints trained with the previous mapping.

**Normals → RGB**: Linear camera-space mapping: R=(1-x)/2, G=(1+y)/2, B=(1+z)/2. Exact invertibility.

**Segmentation → RGB**: Discrete color palette with 5-stage instance decoding algorithm.

## Limitations

1. **Stochastic output**: Diffusion sampling introduces variability between runs. Multiple samples recommended for robust results.
2. **Resolution sensitivity**: The canonical project configs train at 512x512; higher-resolution quality and cost require benchmark evidence.
3. **No temporal consistency**: Video frames processed independently.
4. **Prompt coverage**: Quality depends on the prompt styles and languages represented by each released training manifest.
5. **Quantization sensitivity**: INT8/INT4 quantization may degrade RGB mapping precision.

## Ethical Considerations

- **Surveillance risk**: Depth/segmentation capabilities could enable surveillance applications. Users should comply with local privacy regulations.
- **Bias**: Performance may vary across demographic groups due to training data composition. Evaluate thoroughly before deployment.
- **Environmental cost**: Large model inference consumes significant energy. Consider carbon footprint when scaling.

## Citation

```bibtex
@article{visionbanana2026,
  title={Image Generators are Generalist Vision Learners},
  author={Gabeur, Valentin and others},
  year={2026},
  eprint={2604.20329},
  archivePrefix={arXiv}
}
```
