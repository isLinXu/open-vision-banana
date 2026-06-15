# Contributing to Open Vision Banana

Thank you for your interest in contributing! This guide covers everything you need to get started.

## Quick Start

```bash
# Clone and install
git clone https://github.com/user/open-vision-banana.git
cd open-vision-banana
pip install -e ".[all]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Verify everything works
python -m pytest tests/ -v
```

## Development Setup

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended, CPU works for core modules)

### Dependency Groups

| Group | Install Command | Purpose |
|:------|:----------------|:--------|
| `inference` | `pip install -e ".[inference]"` | Model loading & generation |
| `training` | `pip install -e ".[training]"` | LoRA fine-tuning |
| `export` | `pip install -e ".[export]"` | ONNX / TorchScript export |
| `evaluation` | `pip install -e ".[evaluation]"` | Metric computation |
| `dev` | `pip install -e ".[dev]"` | Testing & linting |
| `all` | `pip install -e ".[all]"` | Everything |

## Project Structure

```
src/open_vision_banana/
├── __init__.py          # Package root with lazy imports
├── config.py            # VisionBananaConfig + 8 sub-configs
├── core/                # Reversible RGB mapping (no external deps)
│   ├── depth_mapper.py  # Continuous spectral depth ↔ RGB
│   ├── normal_mapper.py # Camera-space normal ↔ RGB
│   ├── rgb_mapper.py    # Abstract RGB mapper base
│   └── segmentation.py  # Instance decode (5-stage algorithm)
├── inference/           # Model loading, generation, decoding
│   ├── generator.py     # BaseGenerator (FLUX.1 wrapper)
│   ├── pipeline.py      # VisionBanana unified pipeline
│   ├── decoder.py       # VisualizationDecoder
│   └── prompt.py        # PromptBuilder
├── training/            # LoRA instruction tuning
│   ├── tuner.py         # InstructionTuner (rectified flow)
│   ├── checkpoint.py    # CheckpointManager
│   └── utils.py         # EMA, DDPWrapper, VisionAugmentation
├── evaluation/          # Task-specific metrics
│   ├── evaluator.py     # VisionEvaluator facade
│   ├── depth.py         # AbsRel, SqRel, RMSE, δ1/δ2/δ3
│   ├── normal.py        # Mean angular error
│   ├── semantic.py      # mIoU, pixel accuracy
│   ├── instance.py      # AP, AR
│   └── referring.py     # Precision@0.5/0.75, F1
├── export/              # Model export
│   └── onnx_export.py   # ONNX + TorchScript
└── data/                # Dataset loaders
    ├── datasets.py       # SACoInstanceDataset, build_dataset factory
    └── mixed.py          # MixedVisionDataset
```

## Making Changes

### Branch Naming

| Type | Prefix | Example |
|:-----|:-------|:--------|
| Feature | `feat/` | `feat/add-panoptic-seg` |
| Bug fix | `fix/` | `fix/depth-roundtrip-error` |
| Docs | `docs/` | `docs/add-api-reference` |
| Refactor | `refactor/` | `refactor/split-evaluator` |

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(depth): add inverse depth parameterization
fix(segmentation): resolve label 0 vs background collision
docs(readme): add quickstart examples
refactor(evaluation): vectorize confusion matrix
test(onnx): add numerical parity assertions
chore(ci): add Python 3.12 to test matrix
```

### Code Style

- **Formatter**: ruff format (compatible with Black)
- **Linter**: ruff check (replaces flake8 + isort)
- **Line length**: 100 characters
- **Type hints**: Encouraged for all public APIs
- **Docstrings**: Google style for all public classes/functions

### Quality Gates

Before submitting a PR, ensure:

1. **Tests pass**: `python -m pytest tests/ -v`
2. **Lint clean**: `ruff check src/`
3. **Format clean**: `ruff format --check src/ tests/ scripts/`
4. **Coverage ≥ 80%**: `python -m pytest --cov=open_vision_banana --cov-fail-under=80`
5. **Pre-commit passes**: `pre-commit run --all-files`

### Pull Request Checklist

- [ ] Tests added/updated for new functionality
- [ ] All CI checks pass
- [ ] CHANGELOG.md updated (under "Unreleased" section)
- [ ] Public API changes reflected in `__init__.py` lazy imports
- [ ] Breaking changes flagged in PR description

## Adding New Vision Tasks

To add a new task (e.g., panoptic segmentation):

1. **Core mapper** — Add `panoptic_mapper.py` in `core/` implementing reversible RGB encoding
2. **Evaluation metrics** — Add `panoptic.py` in `evaluation/` with PQ, SQ, RQ metrics
3. **Dataset loader** — Register new dataset class in `data/datasets.py` via `build_dataset`
4. **Config** — Add `PanopticConfig` sub-config in `config.py`
5. **Pipeline integration** — Extend `VisionBanana` and `PromptBuilder` for the new task
6. **Tests** — Add `tests/test_panoptic.py`
7. **Export** — Extend `onnx_export.py` if applicable
8. **Docs** — Update README, MODEL_CARD, and CHANGELOG

## Reporting Issues

- **Bug reports**: Include Python version, PyTorch version, CUDA version, minimal reproducible example
- **Feature requests**: Describe the use case and expected behavior
- **Performance issues**: Include hardware specs, input sizes, and timing measurements

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
