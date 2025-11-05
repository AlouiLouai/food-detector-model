"""FastAPI service for hosting the food nutrition model on a VM or container.

Launch locally with:
    FOOD_MODEL_DIR=model uvicorn vm_service.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from torchvision import transforms
from torchvision.models import (
    convnext_base,
    efficientnet_b0,
    efficientnet_b3,
    efficientnet_b4,
    mobilenet_v3_small,
    resnet50,
    vit_b_16,
    vit_b_32,
)

try:  # pragma: no cover - optional model availability
    from torchvision.models import efficientnet_lite0
except ImportError:  # pragma: no cover - when efficientnet_lite0 is missing
    efficientnet_lite0 = None

logger = logging.getLogger("vm_service")
logger.setLevel(logging.INFO)

MODEL_DIR = Path(os.getenv("FOOD_MODEL_DIR", "model")).resolve()
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TargetNormalizer:
    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1e-6

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=tensor.device, dtype=tensor.dtype)
        std = self.std.to(device=tensor.device, dtype=tensor.dtype).clamp_min(self.eps)
        return tensor * std + mean


class NutritionModelService:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model: torch.nn.Module | None = None
        self.device: torch.device | None = None
        self.transform: transforms.Compose | None = None
        self.target_columns: List[str] = []
        self.normalizer: TargetNormalizer | None = None
        self.normalize_targets: bool = True
        self.image_size: int = 224
        self.head_dropout: float = 0.0

    def load(self) -> None:
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory '{self.model_dir}' does not exist. "
                "Copy trained artifacts or set FOOD_MODEL_DIR."
            )

        metadata_path = self.model_dir / "training_metadata.json"
        weights_path = self.model_dir / "model_fp32.pt"
        normalizer_path = self.model_dir / "target_normalizer.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing model weights: {weights_path}")

        with metadata_path.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)

        args_payload = metadata.get("args", {})
        model_name = args_payload.get("model", metadata.get("model_name", "mobilenet_v3_small"))
        self.image_size = int(metadata.get("image_size", args_payload.get("image_size", 224)))
        self.head_dropout = float(
            args_payload.get("head_dropout", metadata.get("head_dropout", 0.0) or 0.0)
        )
        self.target_columns = metadata.get(
            "target_columns", args_payload.get("target_cols", [])
        )
        if not self.target_columns:
            raise ValueError(
                "Target columns are not specified in metadata; cannot map outputs to labels."
            )

        self.normalize_targets = bool(metadata.get("normalize_targets", True))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading model '%s' onto %s", model_name, self.device)
        self.model = self._build_model(
            model_name=model_name,
            num_outputs=len(self.target_columns),
            head_dropout=self.head_dropout,
        )
        state_dict = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(int(self.image_size * 1.1)),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

        if self.normalize_targets:
            if not normalizer_path.exists():
                raise FileNotFoundError(
                    "Target normalization requested but target_normalizer.json is missing."
                )
            with normalizer_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            mean = torch.tensor(payload.get("mean", []), dtype=torch.float32)
            std = torch.tensor(payload.get("std", []), dtype=torch.float32)
            if mean.numel() != len(self.target_columns) or std.numel() != len(self.target_columns):
                raise ValueError(
                    "Target normalizer dimensions do not match the number of target columns."
                )
            self.normalizer = TargetNormalizer(mean=mean, std=std)
        else:
            self.normalizer = None

        logger.info("Model loaded successfully with targets: %s", self.target_columns)

    @staticmethod
    def _build_head(in_features: int, num_outputs: int, dropout: float) -> torch.nn.Module:
        layers: List[torch.nn.Module] = []
        dropout = max(0.0, float(dropout))
        if dropout > 0:
            layers.append(torch.nn.Dropout(p=min(dropout, 0.95)))
        layers.append(torch.nn.Linear(in_features, num_outputs))
        return torch.nn.Sequential(*layers)

    @staticmethod
    def _build_model(model_name: str, num_outputs: int, head_dropout: float) -> torch.nn.Module:
        if model_name == "mobilenet_v3_small":
            model = mobilenet_v3_small(weights=None)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = NutritionModelService._build_head(
                in_features, num_outputs, head_dropout
            )
            return model
        if model_name == "efficientnet_lite0":
            if efficientnet_lite0 is None:
                raise ImportError(
                    "efficientnet_lite0 is unavailable in the installed torchvision build. "
                    "Upgrade torchvision or retrain/export with a supported backbone."
                )
            model = efficientnet_lite0(weights=None)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = NutritionModelService._build_head(
                in_features, num_outputs, head_dropout
            )
            return model
        if model_name == "efficientnet_b0":
            model = efficientnet_b0(weights=None)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = NutritionModelService._build_head(
                in_features, num_outputs, head_dropout
            )
            return model
        if model_name == "efficientnet_b3":
            model = efficientnet_b3(weights=None)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = NutritionModelService._build_head(
                in_features, num_outputs, head_dropout
            )
            return model
        if model_name == "efficientnet_b4":
            model = efficientnet_b4(weights=None)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = NutritionModelService._build_head(
                in_features, num_outputs, head_dropout
            )
            return model
        if model_name == "resnet50":
            model = resnet50(weights=None)
            in_features = model.fc.in_features
            model.fc = NutritionModelService._build_head(in_features, num_outputs, head_dropout)
            return model
        if model_name == "convnext_base":
            model = convnext_base(weights=None)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = NutritionModelService._build_head(
                in_features, num_outputs, head_dropout
            )
            return model
        if model_name == "vit_b_16":
            model = vit_b_16(weights=None)
            in_features = model.heads.head.in_features
            model.heads.head = NutritionModelService._build_head(
                in_features, num_outputs, head_dropout
            )
            return model
        if model_name == "vit_b_32":
            model = vit_b_32(weights=None)
            in_features = model.heads.head.in_features
            model.heads.head = NutritionModelService._build_head(
                in_features, num_outputs, head_dropout
            )
            return model
        raise ValueError(f"Unsupported model architecture '{model_name}'.")

    def predict(self, batch: torch.Tensor) -> List[Dict[str, float]]:
        if self.model is None or self.device is None:
            raise RuntimeError("Model has not been loaded.")

        batch = batch.to(self.device)
        with torch.no_grad():
            outputs = self.model(batch)
            if self.normalize_targets and self.normalizer is not None:
                outputs = self.normalizer.denormalize(outputs)

        predictions: List[Dict[str, float]] = []
        for row in outputs.cpu().tolist():
            predictions.append(
                {name: float(value) for name, value in zip(self.target_columns, row)}
            )
        return predictions

    def preprocess_images(self, images: List[Image.Image]) -> torch.Tensor:
        if self.transform is None:
            raise RuntimeError("Transforms are not initialized.")
        tensors = [self.transform(image.convert("RGB")) for image in images]
        return torch.stack(tensors, dim=0)


service = NutritionModelService(MODEL_DIR)


class InstancePayload(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded RGB image.")


class PredictBody(BaseModel):
    instances: List[InstancePayload]


allowed_origins = os.getenv("FOOD_API_ALLOWED_ORIGINS")
if allowed_origins:
    origins = [item.strip() for item in allowed_origins.split(",") if item.strip()]
else:
    origins = []

app = FastAPI(title="Food Nutrition Regressor", version="1.0.0")

if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
_startup_error: Exception | None = None


@app.on_event("startup")
def _startup() -> None:
    global _startup_error
    try:
        logger.info("Initializing model service from %s", MODEL_DIR)
        service.load()
        logger.info("Initialization complete.")
    except Exception as exc:  # pragma: no cover - runtime guard
        logger.exception("Model initialization failed.")
        _startup_error = exc


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    if _startup_error is not None:
        raise HTTPException(status_code=500, detail=str(_startup_error))
    return {"status": "ok"}


def _decode_instance(instance: InstancePayload) -> Image.Image:
    try:
        image_bytes = base64.b64decode(instance.image_base64)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}") from exc


@app.post("/predict")
def predict(body: PredictBody) -> Dict[str, Any]:
    if _startup_error is not None:
        raise HTTPException(status_code=500, detail=str(_startup_error))
    if not body.instances:
        raise HTTPException(status_code=400, detail="Request must contain at least one instance.")
    images = [_decode_instance(instance) for instance in body.instances]
    batch = service.preprocess_images(images)
    predictions = service.predict(batch)
    return {"predictions": predictions}


@app.post("/predict-file")
async def predict_file(image: UploadFile = File(...)) -> Dict[str, Any]:
    if _startup_error is not None:
        raise HTTPException(status_code=500, detail=str(_startup_error))
    try:
        blob = await image.read()
        encoded = base64.b64encode(blob).decode("utf-8")
        body = PredictBody(instances=[InstancePayload(image_base64=encoded)])
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return predict(body)
