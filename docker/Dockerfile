# Open-Vision-Banana — training & inference environment
#
# Build:
#   docker build -t open-vision-banana:latest -f docker/Dockerfile .
#
# Run inference (CPU core algorithms):
#   docker run --rm open-vision-banana:latest \
#     python -c "from open_vision_banana.core import DepthRGBMapper; print('ok')"
#
# Run with GPU (requires nvidia-container-toolkit):
#   docker run --gpus all -v $(pwd)/data:/data open-vision-banana:latest \
#     python scripts/inference.py --image /data/input.jpg --task depth

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md CHANGELOG.md MODEL_CARD.md ./
COPY configs ./configs
COPY src ./src
COPY scripts ./scripts
COPY experiments ./experiments

RUN pip install -e ".[all]"

WORKDIR /app

CMD ["python", "-c", "import open_vision_banana; print('Open-Vision-Banana', open_vision_banana.__version__ if hasattr(open_vision_banana, '__version__') else 'ready')"]
