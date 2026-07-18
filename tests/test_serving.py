"""Tests for the optional, dependency-isolated serving adapter."""

from __future__ import annotations

import base64
import io

import pytest
import torch
from PIL import Image


def _image_b64() -> str:
    image = Image.new("RGB", (8, 6), color=(128, 64, 32))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class _FakeModel:
    def predict(self, **kwargs):
        image = kwargs["image"]
        return {
            "rgb_output": torch.zeros(image.shape[0], 3, image.shape[-2], image.shape[-1]),
            "depth": torch.ones(image.shape[0], 1, image.shape[-2], image.shape[-1]),
            "metadata": {"task": kwargs["task"], "batch_size": image.shape[0]},
        }


def test_serving_app_is_optional_and_serializes_prediction():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from open_vision_banana.serving import create_app

    client = TestClient(create_app(_FakeModel()))
    assert client.get("/healthz").json() == {"status": "ok"}
    response = client.post(
        "/predict",
        json={"image_base64": _image_b64(), "task": "depth"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["task"] == "depth"
    assert payload["metadata"]["batch_size"] == 1
    assert payload["outputs"]["depth"]["shape"] == [1, 1, 6, 8]
    assert payload["rgb_output_png_base64"]
