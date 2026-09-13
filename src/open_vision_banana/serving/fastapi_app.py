"""Minimal optional FastAPI adapter for :class:`VisionBanana`.

The adapter accepts JSON with a base64-encoded input image and returns decoded
metadata plus compact tensor summaries.  The generated RGB output is also
returned as a base64 PNG for clients that need a raster artifact.  Raw task
tensors are intentionally not expanded into JSON arrays.
"""

import base64
import io
from typing import Any

import torch
from PIL import Image

from ..inference.pipeline import VisionBanana


def _decode_image(value: str) -> torch.Tensor:
    try:
        raw = base64.b64decode(value, validate=True)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise ValueError("image_base64 must contain a valid base64-encoded image") from exc
    import numpy as np

    array = np.asarray(image, dtype="float32") / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def _encode_png(rgb: torch.Tensor) -> str:
    tensor = rgb.detach().float().cpu().clamp(0, 1)
    if tensor.dim() == 4:
        tensor = tensor[0]
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _summarize(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        summary: dict[str, Any] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if value.numel():
            finite = torch.isfinite(value)
            summary["finite"] = bool(finite.all().item())
            if finite.all():
                summary["min"] = float(value.min().item())
                summary["max"] = float(value.max().item())
        return summary
    if isinstance(value, dict):
        return {str(key): _summarize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summarize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def create_app(model: VisionBanana):
    """Create a FastAPI app around an already-loaded model.

    Raises a clear error when the optional ``fastapi`` dependency is absent.
    Model loading remains outside the web process so deployment code controls
    checkpoint paths, devices, and lifecycle explicitly.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI serving requires the optional dependency: pip install 'open-vision-banana[serving]'"
        ) from exc

    class PredictRequest(BaseModel):
        image_base64: str
        task: str
        class_color_map: dict[str, tuple[int, int, int]] | None = None
        object_name: str | None = None
        expression: str | None = None
        num_inference_steps: int = 1
        strength: float = 0.85
        seed: int | None = None

    app = FastAPI(title="Open Vision Banana", version="0.5.0")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    def predict(request: PredictRequest) -> dict[str, Any]:
        try:
            image = _decode_image(request.image_base64)
            result = model.predict(
                image=image,
                task=request.task,
                class_color_map=request.class_color_map,
                object_name=request.object_name,
                expression=request.expression,
                num_inference_steps=request.num_inference_steps,
                strength=request.strength,
                seed=request.seed,
            )
            response = {
                "task": request.task,
                "metadata": _summarize(result.get("metadata", {})),
                "outputs": _summarize({key: value for key, value in result.items() if key not in {"rgb_output", "prompt"}}),
            }
            rgb_output = result.get("rgb_output")
            if isinstance(rgb_output, torch.Tensor):
                response["rgb_output_png_base64"] = _encode_png(rgb_output)
            return response
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="inference failed") from exc

    return app


__all__ = ["create_app"]
