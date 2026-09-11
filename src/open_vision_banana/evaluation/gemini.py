"""Optional Gemini judge used by the paper's segmentation protocols.

The client uses the public REST API through the Python standard library, so
importing the evaluation package does not require a vendor SDK. Network access
only occurs when a judge method is called.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any, Union
from urllib.request import Request, urlopen

import numpy as np
import torch
from PIL import Image

DEFAULT_PRESENCE_MODEL = "gemini-3.1-flash-lite"
DEFAULT_REASONING_MODEL = "gemini-2.5-pro"

Transport = Callable[[Request, float], bytes]
ImageInput = Union[bytes, str, Path, Image.Image, torch.Tensor]


def _default_transport(request: Request, timeout: float) -> bytes:
    with urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


def _image_payload(image: ImageInput) -> tuple[str, str]:
    if isinstance(image, bytes):
        raw = image
        mime_type = "image/jpeg"
    elif isinstance(image, (str, Path)):
        path = Path(image)
        raw = path.read_bytes()
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    else:
        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu().float()
            if tensor.dim() == 4 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if tensor.dim() != 3 or tensor.shape[0] != 3:
                raise ValueError(f"Image tensor must have shape [3,H,W], got {tuple(tensor.shape)}")
            if tensor.min().item() < 0:
                tensor = (tensor + 1.0) / 2.0
            array = (
                tensor.clamp(0.0, 1.0)
                .mul(255)
                .round()
                .byte()
                .permute(1, 2, 0)
                .numpy()
            )
            image = Image.fromarray(np.ascontiguousarray(array), mode="RGB")
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        raw = buffer.getvalue()
        mime_type = "image/png"
    return mime_type, base64.b64encode(raw).decode("ascii")


class GeminiPaperJudge:
    """Gemini-backed SA-Co presence filter and ReasonSeg query rewriter."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        endpoint: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 60.0,
        transport: Transport | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required; pass api_key or set GEMINI_API_KEY")
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.transport = transport or _default_transport

    def _generate(self, image: ImageInput, prompt: str, model: str) -> str:
        mime_type, data = _image_payload(image)
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"inline_data": {"mime_type": mime_type, "data": data}},
                        {"text": prompt},
                    ],
                }
            ],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 128},
        }
        request = Request(
            f"{self.endpoint}/models/{model}:generateContent",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )
        raw = self.transport(request, self.timeout)
        payload: dict[str, Any] = json.loads(raw)
        try:
            parts = payload["candidates"][0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Gemini response did not contain generated text: {payload}") from exc
        if not text:
            raise RuntimeError("Gemini returned an empty response")
        return text

    def classify_noun_phrase(
        self,
        image: ImageInput,
        noun_phrase: str,
        *,
        model: str = DEFAULT_PRESENCE_MODEL,
    ) -> bool:
        """Apply the exact positive/negative query used for SA-Co in the paper."""
        response = self._generate(
            image,
            f"Is there an instance of <{noun_phrase}> visible in this image? "
            "Choose your answer from the following options: (A): Yes, (B): No.",
            model,
        )
        tokens = set(re.findall(r"[A-Z]+", response.upper()))
        positive = bool(tokens & {"A", "YES"})
        negative = bool(tokens & {"B", "NO"})
        if positive == negative:
            raise RuntimeError(f"Could not parse Gemini presence response: {response!r}")
        return positive

    def rewrite_reasoning_query(
        self,
        image: ImageInput,
        query: str,
        *,
        model: str = DEFAULT_REASONING_MODEL,
    ) -> str:
        """Translate a ReasonSeg reasoning query into a descriptive reference."""
        return self._generate(
            image,
            "Identify the single image region requested by the reasoning query below. "
            "Return only a concise descriptive referring expression for that region; "
            f"do not explain your reasoning. Query: {query}",
            model,
        )


__all__ = [
    "DEFAULT_PRESENCE_MODEL",
    "DEFAULT_REASONING_MODEL",
    "GeminiPaperJudge",
]
