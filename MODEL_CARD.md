# Model Card: Vision Banana

**Model Page**: https://github.com/user/open-vision-banana

## Model Details

### Model Description

Vision Banana demonstrates that image generators are generalist vision learners. By instruction-tuning FLUX.1 [schnell] — a 12B-parameter rectified flow diffusion model — on a mixture of its original training data alongside a small proportion of vision task data, the model achieves state-of-the-art zero-shot transfer on semantic segmentation, instance segmentation, depth estimation, and surface normal estimation.

- **Developed by**: Open Vision Banana Contributors
- **Model type**: Instruction-tuned diffusion model for vision understanding
- **Languages**: English (prompts), Visual (outputs)
- **License**: Apache 2.0 (code); FLUX.1 base model follows its own license
- **Base model**: FLUX.1 [schnell] by Black Forest Labs

### Intended Uses

| Task | Input | Output | Intended Users |
|:-----|:------|:-------|:---------------|
| Semantic Segmentation | RGB image + "segment" prompt | Per-pixel class labels | CV researchers, autonomous driving engineers |
| Instance Segmentation | RGB image + "instance segment" prompt | Per-instance masks | Robotics engineers, medical imaging researchers |
| Depth Estimation | RGB image + "estimate depth" prompt | Dense depth map (meters) | 3D reconstruction, SLAM researchers |
| Surface Normal Estimation | RGB image + "estimate normals" prompt | Per-pixel normal vectors | Scene understanding, graphics researchers |

### Out-of-Scope Uses

- Real-time safety-critical applications (model has stochastic output)
- Medical diagnosis without expert validation
- Video understanding (single-frame only)
- Long-text reasoning beyond visual tasks

## Performance

### Benchmark Results

#### Semantic Segmentation (Cityscapes val)

| Method | mIoU (%) |
|:-------|:---------|
| SAM 3 | 65.2 |
| **Vision Banana** | **69.9** |

#### Depth Estimation (4-dataset average)

| Method | δ1 ↑ |
|:-------|:-----|
| Depth Anything 3 | 0.918 |
| **Vision Banana** | **0.929** |

#### Surface Normal Estimation (4-dataset average)

| Method | Mean Angular Error (°) ↓ |
|:-------|:-------------------------|
| Lotus-2 | 19.642 |
| **Vision Banana** | **18.928** |

### Hardware Requirements

| Configuration | VRAM | Inference Time (1024×1024) |
|:-------------|:-----|:---------------------------|
| A100 80GB | ~40GB | ~3s |
| RTX 4090 | ~24GB | ~5s |
| RTX 3090 | ~24GB | ~8s |

## Technical Specifications

### Architecture

| Component | Details |
|:----------|:--------|
| Base model | FLUX.1 [schnell] (12B params, rectified flow) |
| Fine-tuning method | rsLoRA (rank=64, alpha=128) |
| Trainable params | 0.5%–1% of total |
| Training data mix | 9:1 (original : vision task) |
| Depth mapping | Continuous spectral color mapping + parabolic interpolation |
| Normal mapping | Camera-space XYZ → RGB linear mapping |
| Instance decoding | 5-stage: init → group → prune → boundary → merge |

### RGB Encoding Schemes

**Depth → RGB**: Continuous spectral mapping with λ=-3, c=10/3. Near-depth precision prioritized. Parabolic interpolation refinement resolves reverse-projection ambiguity. Round-trip error: 0.0042m.

**Normals → RGB**: Linear camera-space mapping: R=(1-x)/2, G=(1+y)/2, B=(1-z)/2. Exact invertibility.

**Segmentation → RGB**: Discrete color palette with 5-stage instance decoding algorithm.

## Limitations

1. **Stochastic output**: Diffusion sampling introduces variability between runs. Multiple samples recommended for robust results.
2. **Resolution constraint**: Optimal at 1024×1024; lower resolutions may degrade quality.
3. **No temporal consistency**: Video frames processed independently.
4. **English prompts only**: Instruction tuning performed on English task descriptions.
5. **Quantization sensitivity**: INT8/INT4 quantization may degrade RGB mapping precision.

## Ethical Considerations

- **Surveillance risk**: Depth/segmentation capabilities could enable surveillance applications. Users should comply with local privacy regulations.
- **Bias**: Performance may vary across demographic groups due to training data composition. Evaluate thoroughly before deployment.
- **Environmental cost**: Large model inference consumes significant energy. Consider carbon footprint when scaling.

## Citation

```bibtex
@article{visionbanana2025,
  title={Image Generators are Generalist Vision Learners},
  author={Google Research},
  year={2025}
}
```
