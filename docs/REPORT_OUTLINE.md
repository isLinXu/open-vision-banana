# Technical Report Outline — Open-Vision-Banana

> Working title: *Open-Vision-Banana: an open reproduction study of "Image Generators are Generalist Vision Learners" on public backbones and public data*
> Every table below names the evaluation JSON(s) it is rendered from. Numbers are never typed by hand: `scripts/render_results_tables.py` (release item R1) reads `results/<run>/*.json` and emits the Markdown.
> Status markers: ⬜ not started · 🟡 data collected, not written · ✅ written and reviewed.

## 1. Abstract and claims  ⬜
One paragraph. Then five one-line answers, each pointing at the section that supports it:

| claim (paper) | our finding (fill after E1/E2) | evidence |
|:--|:--|:--|
| Q1 one weight, five vision tasks via prompt | … | §5.1 |
| Q2 tuning unlocks pre-existing ability | … | §5.5 (E-0 vs E1, E2b) |
| Q3 low-ratio mixing keeps generation | … | §5.6 (E2 trade-off curve) |
| Q4 encodings are decodable | … | §3, §5.3, §5.7 |
| Q5 strict zero-shot holds | … | §4.3 |

Explicitly: what this report **does not** show (absolute paper numbers; human pairwise generation study; effect of in-house annotations).

## 2. Relationship to the paper  ⬜
- Summary of `docs/PAPER_ALIGNMENT_AUDIT_2026-10-22.md`: 58 checkable statements → 49 identical / 6 partial (paper gaps) / 3 missing (closed-source) / 0 wrong.
- Table: the 6 partial items and the choice made for each (threshold 0.5, aggregation, depth caps, trunc/round, ratio value, "lightweight" = LoRA).
- Table: everything the paper does **not** state and what we chose (optimizer, lr, schedule, steps, batch, resolution, LoRA, inference steps, CFG). Source: `effective_config.json`, `metadata.inference_protocol`.

## 3. Method (brief; cite the audit, do not re-derive)  ⬜
- 3.1 RGB task interface: depth (Eq. 1, cube-edge path, Fig. 5 verified), normals, open-vocabulary semantic, instance (Appendix A), referring (Fig. 4 template).
- 3.2 Backbone and conditioning: Qwen-Image-2.1 / Qwen-Image-Edit (source image as VLM visual tokens + reference latents); why a natively image-conditioned editor is the closest open analogue.
- 3.3 Training: LoRA, flow matching, mixing with generation replay, augmentations (colormaps, prompt styles, colour remap). One table from `configs/experiments/qwen_image_21_paper_aligned.yaml` with P/R marks.
- 3.4 Decoders and the two Appendix A readings (→ §5.7).

## 4. Data  ⬜
- 4.1 Sources table (from `configs/datasets/sources.tsv`): source · tasks · rows after audit · images · licence · paper analogue (in-house web annotations / synthetic renders).
- 4.2 Preparation pipeline figure: download → prepare → merge → strict audit → gates.
- 4.3 **Zero-shot evidence**: audit JSON fields (`protocol: strict`, `rejected_benchmark_leak_rows`, `rejected_excluded_image_id_rows`, `evaluation_image_id_overlap_by_source` = 0, `manifest_sha256`). Table rendered from `data/qwen_baseline/pairs.audit.json`.
- 4.4 Differences from the paper's data (public human annotations vs in-house model annotations; OmniEdit/BLIP3o vs the base model's own mixture) and the expected direction of their effect.

## 5. Experiments
Protocol box at the top of the section: steps / CFG / seed / canvas / native scoring / Gemini usage, copied from `metadata.inference_protocol`.

### 5.1 Main results (E1 @ 30 k, with E-0 and specialists)  ⬜
Table rendered from `results/e1/eval_30000.json`, `results/e-0/*.json`, `results/e-1/*.json`:

| benchmark (paper table) | metric | E-0 base (no training) | E1 ours | SANA shared (E-1, post-fix) | specialist (ours, SNVC era) | paper VB | paper best counterpart |
|:--|:--|--:|--:|--:|--:|--:|--:|
| Cityscapes val (T2) | mIoU | | | | 0.266 | 69.9 | 65.2 SAM 3 |
| SA-Co/Gold (T3) | cgF1 / IL_MCC / pmF1 | | | | | 47.5 | 24.6 OWLv2 |
| RefCOCOg UMD val (T4) | cIoU | | | | | 73.8 | 73.4 |
| ReasonSeg val (T5) | gIoU | | | | | 79.3 | 77.0 |
| NYU / iBims-1 / ETH3D / DIODE-In (T6) | δ1 / AbsRel | | | | | 0.929 avg | 0.918 DA3 |
| NYUv2 / DIODE-In / ScanNet / VKitti normals (T7) | mean° / median° | | | | | 18.928 avg | 19.642 Lotus-2 |
| GenAI-Bench / ImgEdit (App. D) | **proxy** FID / LPIPS vs base | | | | | 53.5% / 47.8% human | — |

Footnotes: protocol; noise floor per metric (§7); which cells are diagnostic (no Gemini); native-resolution scoring.

### 5.2 Learning curves  ⬜
Fig: metrics at 5k/10k/15k/20k/25k/30k (from `results/e1/eval_*.json`) + train/val loss per task (from `train.log`).

### 5.3 Decodability  ⬜
Table: per task, `skipped_samples` / total (decoder returned empty or invalid), and the mean decode round-trip error on GT targets (from Hypersim `manifest.json` and `validate_manifest_contract`). This is the quantitative form of Q4.

### 5.4 Backbone comparison  ⬜
SANA-1600M (T2I, pre-fix / post-fix) vs Qwen-Image-2.1 (editor) on the same manifest and protocol. Source: E-1. Discuss attention LoRA, text length, native image conditioning.

### 5.5 "Unlocking" argument  ⬜
E-0 → E1 deltas per task; E2b (¼ data) retention; Fig: base vs tuned outputs on fixed sample ids (E-0 left column, E1 right column).

### 5.6 Ablations: mixing ratio and data scale  ⬜
Fig: dual-axis trade-off (vision score normalised to E1 vs generation proxy) over `vision_ratio` ∈ {0.2, 0.5, 0.8}, 2 seeds each, error bars = seed spread. Table for E2b. Decision rule and noise floor stated.

### 5.7 Appendix A decoder readings  ⬜
2×2 table (seed rule × area basis) on cached predictions; column "mean instances / image"; default cell marked.

### 5.8 Continued training on re-cleaned data (E4, if done)  ⬜

### 5.9 Qualitative results  ⬜
Figures mirroring the paper: open-vocabulary semantic in four prompt styles (Fig. 2), instance one-class-at-a-time incl. non-black backgrounds (Fig. 3), referring with appearance/action/unusual-use/multilingual (Fig. 4), depth point clouds (Fig. 6), in-the-wild depth with measured distances (Fig. 7), normals vs Lotus-2 demo (Fig. 8). Each figure: checkpoint, prompt, seed in caption; regeneration script in `output/figures/`.

### 5.10 Robustness and cross-task transfer (E6)  ⬜
Tables: prompt-style spread on Cityscapes; colour-remap invariance; depth-colormap invariance; referring phrases inside semantic prompts; negative-query behaviour.

## 6. Failure analysis and lessons  ⬜
- 6.1 Typical failures per task (E5), with the paper's Appendix C taxonomy (merging, splitting group instances, misses).
- 6.2 Historical root causes and fixes, with before/after numbers where available: relative-depth targets under a metric decoder; FFN-only LoRA; 77-token truncation; batch-wide flow source for replay; dropped instance background colour; Hypersim ray distance vs planar z; row-wise val split. (Sources: `docs/ABLATION_BASELINE.md`, review of 2026-10-22, PRs #34, #42, #44.)

## 7. Compute, reproducibility, statistics  ⬜
- GPU hours per experiment (from the TODO budget table, updated with measured numbers); peak VRAM; s/step.
- Environment lock (`requirements-lock.txt`, CUDA, commit, HF revisions).
- **Noise floor method**: spread across 5k/15k/30k `_best` and across seeds 42/43 at 30 k; per-metric values table; how it was applied in §5.6.
- One-command reproduction: `scripts/run_e0_smoke.sh`, `scripts/run_e1_full.sh`; gates.

## 8. Limitations and honest boundaries  ⬜
Base model gap (Nano Banana Pro closed); no human pairwise study (proxy only); Gemini dependence for T3/T5; single public data tier; LoRA vs the paper's unspecified tuning; licences restricting checkpoint redistribution (Hypersim research-only).

## Appendix  ⬜
- A. Full `effective_config.json` of E1.
- B. Per-benchmark sample counts, splits, crops/caps, native resolutions.
- C. Index of all evaluation JSONs (path, run, protocol hash).
- D. Prompt strings used (verbatim from `inference/prompt.py`, `referring_contract.py`).
- E. Decoder constants (τ, θ_size, θ_erosion, γ, connectivity) and the two readings.

## Review checklist before release
- [ ] Every number in §5 traces to a JSON path listed in Appendix C
- [ ] Every table states its inference protocol; no mixed-protocol tables
- [ ] Diagnostic cells (no Gemini, `_aligned`, proxy) are visibly marked
- [ ] E-0 row present in every results table
- [ ] Noise floor stated wherever two runs are compared
- [ ] Licence section matches the model card
- [ ] A reviewer who did not run the experiments has re-rendered all tables from the JSONs
