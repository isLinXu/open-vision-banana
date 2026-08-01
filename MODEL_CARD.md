# Model Card: Vision Banana

**Model Page**: https://github.com/user/open-vision-banana

> **Release status:** this repository does not currently publish a validated
> Open Vision Banana checkpoint. Registry entries are placeholders. Scores
> below are reference results from the original closed Nano Banana Pro paper,
> not measurements of a FLUX/SSD checkpoint from this repository.

## Model Details

### Model Description

The original Vision Banana paper instruction-tunes Google's closed Nano Banana Pro model. This independent project replaces that base with configurable open-weight diffusion models and implements the RGB task mappings, training paths, and evaluators needed to test the same paradigm. Benchmark parity has not yet been established.

- **Developed by**: Open Vision Banana Contributors
- **Model type**: Instruction-tuned diffusion model for vision understanding
- **Languages**: English (prompts), Visual (outputs)
- **License**: MIT (code); each base model and dataset follows its own license
- **Base model**: Configurable; FLUX.1 [schnell] is the default inference configuration

### Intended Uses

| Task | Input | Output | Intended Users |
|:-----|:------|:-------|:---------------|
| Semantic Segmentation | RGB image + "segment" prompt | Per-pixel class labels | CV researchers, autonomous driving engineers |
| Instance Segmentation | RGB image + "instance segment" prompt | Per-instance masks | Robotics engineers, medical imaging researchers |
| Depth Estimation | RGB image + "estimate depth" prompt | Dense relative or metric depth, depending on checkpoint encoding | 3D reconstruction, SLAM researchers |
| Surface Normal Estimation | RGB image + "estimate normals" prompt | Per-pixel normal vectors | Scene understanding, graphics researchers |

### Out-of-Scope Uses

- Real-time safety-critical applications (model has stochastic output)
- Medical diagnosis without expert validation
- Video understanding (single-frame only)
- Long-text reasoning beyond visual tasks

## Performance

### Original Paper Reference Results

These results belong to the paper's Nano Banana Pro model. They must not be
reported as results of an open-weight checkpoint without a fresh evaluation.

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

### Hardware Planning Estimates

No checkpoint-specific latency or memory benchmark is currently available.
Use the model registry's memory budgets only for planning, then record measured
hardware, precision, resolution, latency, and peak memory in each released
checkpoint's model card.

### Checkpoint Registry

LoRA weights can be loaded via registry aliases defined in
``configs/checkpoint_registry.yaml``:

| Alias | Hugging Face Repo | Status |
|:------|:------------------|:-------|
| ``vision-banana`` | ``open-vision-banana/vision-banana-lora`` | placeholder |
| ``vision-banana-depth`` | ``open-vision-banana/vision-banana-depth-lora`` | placeholder |
| ``vision-banana-normal`` | ``open-vision-banana/vision-banana-normal-lora`` | placeholder |

```bash
python scripts/download_checkpoint.py --list
```

Aliases with `placeholder` status cannot be downloaded. After training a
compatible checkpoint locally, load it by path:

Usage in Python:

```python
from open_vision_banana import VisionBanana

model = VisionBanana.from_pretrained(lora_path="path/to/checkpoint")
result = model.predict(input_tensor, task="depth")
```

## Technical Specifications

### Architecture

| Component | Details |
|:----------|:--------|
| Base model | Configurable open-weight diffusion backbone |
| Fine-tuning method | rsLoRA; rank and alpha are checkpoint-specific |
| Trainable params | Depends on the backbone and selected target modules |
| Training data mix | Configurable; 9:1 generation/vision is a repository experiment, not a disclosed paper hyperparameter |
| Depth mapping | Cube-edge traversal colormap + power transform (λ=-3, c=10/3) |
| Normal mapping | Camera-space XYZ → RGB linear mapping |
| Instance decoding | 5-stage: init → group → prune → boundary → merge |

### RGB Encoding Schemes

**Depth → RGB**: Cube-edge traversal colormap (black→blue→cyan→green→yellow→white) with power transform λ=-3, c=10/3. Near-depth precision is prioritized. Round-trip error depends on the selected colormap, quantization, and valid depth range.

**Normals → RGB**: Linear camera-space mapping: R=(1-x)/2, G=(1+y)/2, B=(1+z)/2. Exact invertibility before quantization.

**Segmentation → RGB**: Discrete color palette with 5-stage instance decoding algorithm.

## Limitations

1. **No validated release checkpoint**: Paper benchmark parity and open-backbone generalization remain unproven.
2. **Checkpoint-specific encoding**: Relative-depth data without metric scale cannot support metric-depth claims.
3. **Stochastic output**: Diffusion sampling introduces variability between runs. Multiple samples may be needed for robust results.
4. **No temporal consistency**: Video frames are processed independently.
5. **Prompt coverage**: Current training paths primarily use English task descriptions.
6. **Quantization sensitivity**: INT8/INT4 quantization may degrade RGB mapping precision.

## Ethical Considerations

- **Surveillance risk**: Depth/segmentation capabilities could enable surveillance applications. Users should comply with local privacy regulations.
- **Bias**: Performance may vary across demographic groups due to training data composition. Evaluate thoroughly before deployment.
- **Environmental cost**: Large model inference consumes significant energy. Consider carbon footprint when scaling.

## Citation

```bibtex
@article{gabeur2026image,
  title={Image Generators are Generalist Vision Learners},
  author={Gabeur, Valentin and Long, Shangbang and Peng, Songyou and Voigtlaender, Paul and Sun, Shuyang and Bao, Yanan and Truong, Karen and Wang, Zhicheng and Zhou, Wenlei and Barron, Jonathan T. and Genova, Kyle and Kannen, Nithish and Ben, Sherry and Li, Yandong and Guo, Mandy and Yogin, Suhas and Gu, Yiming and Chen, Huizhong and Wang, Oliver and Xie, Saining and Zhou, Howard and He, Kaiming and Funkhouser, Thomas and Alayrac, Jean-Baptiste and Soricut, Radu},
  journal={arXiv preprint arXiv:2604.20329},
  year={2026}
}
```
