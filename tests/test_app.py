from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Dict

import sys
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vm_service.app import app, service  # noqa: E402  (import after path injection)


def _encode_dummy_image() -> str:
    image = Image.new("RGB", (service.image_size, service.image_size), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _get_client() -> TestClient:
    # TestClient executes FastAPI startup/shutdown handlers, ensuring the model loads once.
    return TestClient(app)


def test_healthz_returns_ok() -> None:
    with _get_client() as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_predict_returns_expected_keys() -> None:
    with _get_client() as client:
        payload = {"instances": [{"image_base64": _encode_dummy_image()}]}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        body: Dict[str, object] = response.json()
        predictions = body.get("predictions")

        assert isinstance(predictions, list)
        assert len(predictions) == 1

        first = predictions[0]
        assert isinstance(first, dict)
        assert set(first.keys()) == set(service.target_columns)
