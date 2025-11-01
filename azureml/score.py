"""Azure ML entry script for the food nutrition regression model."""
from __future__ import annotations

import base64
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from PIL import Image

from vm_service.app import NutritionModelService

logger = logging.getLogger("azureml.score")
logger.setLevel(logging.INFO)

_service: NutritionModelService | None = None


def init() -> None:
    """Initialize the model once when the Azure ML deployment starts."""
    global _service
    if _service is not None:
        logger.warning("Model service already initialized; skipping re-load.")
        return

    model_dir = Path(os.getenv("FOOD_MODEL_DIR", "model")).resolve()
    logger.info("Loading nutrition model artifacts from %s", model_dir)

    service = NutritionModelService(model_dir)
    service.load()
    _service = service
    logger.info("Model service initialization complete.")


def run(raw_data: str) -> Dict[str, Any]:
    """Score the incoming request payload.

    Azure ML sends the raw request body as a JSON string. The payload must match the
    FastAPI schema used by the VM service:
        {"instances": [{"image_base64": "..."}]}
    """
    if _service is None:
        raise RuntimeError("Model service is not initialized. Did Azure ML call init()?")

    try:
        payload = json.loads(raw_data)
    except json.JSONDecodeError as exc:
        logger.exception("Failed to decode request JSON.")
        return {"error": f"Invalid JSON payload: {exc}"}

    instances = payload.get("instances")
    if not instances:
        return {"error": "Request must contain a non-empty 'instances' list."}

    try:
        images = [_decode_base64_image(item["image_base64"]) for item in instances]
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Failed to decode image payload.")
        return {"error": f"Invalid image payload: {exc}"}

    batch = _service.preprocess_images(images)
    predictions = _service.predict(batch)
    return {"predictions": predictions}


def _decode_base64_image(encoded: str) -> Image.Image:
    blob = base64.b64decode(encoded)
    return Image.open(io.BytesIO(blob)).convert("RGB")
