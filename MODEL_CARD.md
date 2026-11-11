# Model Card: Vision Banana

**Model Page**: https://github.com/isLinXu/open-vision-banana

## Model Details

### Model Description

Open-Vision-Banana is a method-aligned, open implementation of the Vision Banana paradigm. It instruction-tunes an open image generator (Qwen-Image-2.1 in the current runs; SANA-1600M and FLUX.1-dev in earlier ones) with shared LoRA adapters so that prompts select vision tasks and outputs remain RGB images. These are open substitutes for the paper's closed Nano Banana Pro base; the paper's numbers have not been reproduced.

- **Developed by**: Open Vision Banana Contributors
- **Model type**: Instruction-tuned diffusion model for vision understanding
- **Languages**: English (prompts), Visual (outputs)
- **License**: MIT (project code); base models and datasets follow their own licenses
- **Base model**: Qwen-Image-2.1 (current runs); SANA-1600M and FLUX.1-dev (earlier runs). Each checkpoint's `metadata.json` records its base

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

No checkpoint in this repository reproduces the paper's benchmark numbers. Measured
paper-protocol scores for the published Qwen-Image-2.1, SANA and FLUX.1-dev runs, with
sample counts and repo ids, are in the README section
[Measurements in this repository](README.md#measurements-in-this-repository).
The best metric-depth δ1 so far is 0.348 on NYUv2 (paper: 0.948) and the best
RefCOCOg cIoU is 0.547 (paper: 0.738).

### Checkpoint Registry

LoRA weights can be loaded via registry aliases defined in
``configs/checkpoint_registry.yaml``:

| Alias | Hugging Face Repo | Status |
|:------|:------------------|:-------|
| ``sana-depth-relative-v1`` | local adapter (see `local_path`) | **local** |
| ``vision-banana`` | ``open-vision-banana/vision-banana-lora`` | placeholder |
| ``vision-banana-depth`` | ``open-vision-banana/vision-banana-depth-lora`` | placeholder |
| ``vision-banana-normal`` | ``open-vision-banana/vision-banana-normal-lora`` | placeholder |

```bash
python scripts/download_checkpoint.py --list
python scripts/download_checkpoint.py --alias vision-banana
```

Usage in Python:

```python
model = VisionBanana.from_pretrained(lora_path="registry:sana-depth-relative-v1")
```

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
