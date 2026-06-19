# Docker Environment

Open-Vision-Banana provides two Docker images for reproducible training and inference.

## Quick Start

### CPU Image (Core Algorithms + Inference)

```bash
docker build -t open-vision-banana:latest -f docker/Dockerfile .
docker run --rm open-vision-banana:latest \
  python -c "from open_vision_banana.core import DepthRGBMapper; print('ok')"
```

### GPU Image (Training + CUDA Inference)

```bash
docker build -t open-vision-banana:gpu -f docker/Dockerfile.gpu .
docker run --gpus all --rm open-vision-banana:gpu \
  python scripts/inference.py --image /data/input.jpg --task depth
```

## Docker Compose Services

### Development (Editable, CPU)

```bash
docker compose -f docker/docker-compose.yml up dev
# Inside container:
# python -m pytest tests/ -v
# python scripts/benchmark.py
```

### Training (GPU, FSDP)

```bash
# Mount your data and output directories
docker compose -f docker/docker-compose.yml up training
```

### Jupyter Notebook

```bash
docker compose -f docker/docker-compose.yml up notebook
# Open http://localhost:8888
```

## Multi-Stage Builds

| Stage | Base | Purpose | Size |
|-------|------|---------|------|
| `builder` | `nvidia/cuda:12.1.1-devel` | Compile & install deps | ~8 GB |
| `runtime` | `nvidia/cuda:12.1.1-runtime` | Run inference/training | ~3 GB |
| `dev` | `python:3.12-slim` | CPU development | ~1 GB |

## GPU Requirements

- **Inference**: 1× GPU with 12 GB VRAM (RTX 3090, A10)
- **Training (DDP)**: 2–4× GPU with 24 GB VRAM (RTX 4090, A5000)
- **Training (FSDP)**: 4–8× GPU with 40 GB VRAM (A100-40GB, H100)
- **CPU Offload**: Works on CPU but 3–4× slower

## Health Check

All containers include a health check that verifies core imports:

```bash
docker ps  # HEALTHY status indicates successful startup
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `nvidia-container-toolkit not found` | Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |
| `CUDA out of memory` | Enable CPU offload: `--enable-sequential-cpu-offload` |
| Slow first inference | `torch.compile` warm-up; subsequent runs are faster |
| HuggingFace cache miss | Mount a persistent cache volume: `-v ~/.cache/huggingface:/root/.cache/huggingface` |
