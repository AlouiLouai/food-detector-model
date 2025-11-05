#!/usr/bin/env python3
"""Train and export the food nutrition regression model."""
from __future__ import annotations

import argparse
import copy
import io
import json
import math
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from datasets import DatasetDict, load_dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
import torchvision
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
from torchvision.models import (
    ConvNeXt_Base_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
    MobileNet_V3_Small_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    ViT_B_32_Weights,
)
from torchvision.models._api import WeightsEnum
from tqdm.auto import tqdm

from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_

try:  # pragma: no cover - optional model depending on torchvision build
    from torchvision.models import efficientnet_lite0, EfficientNet_Lite0_Weights
except ImportError:  # pragma: no cover - gracefully handle missing export
    efficientnet_lite0 = None
    EfficientNet_Lite0_Weights = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train food nutrition regression model.")
    parser.add_argument("--dataset-name", default="mmathys/food-nutrients", type=str)
    parser.add_argument("--dataset-config", default=None, type=str)
    parser.add_argument("--image-column", default="image", type=str)
    parser.add_argument(
        "--target-cols",
        nargs="+",
        default=[
            "total_calories",
            "total_fat",
            "total_carb",
            "total_protein",
            "total_mass",
        ],
        help="Regression targets to predict.",
    )
    parser.add_argument(
        "--train-split",
        default="train",
        type=str,
        help="Split name for training set (after optional split).",
    )
    parser.add_argument(
        "--validation-split",
        default="validation",
        type=str,
        help="Split name for validation set.",
    )
    parser.add_argument("--test-split", default=None, type=str)
    parser.add_argument(
        "--train-val-test-split",
        default=None,
        type=str,
        help="Ratios for creating train/validation/test splits when dataset has just one split. "
        "Example: 0.8,0.1,0.1",
    )
    parser.add_argument(
        "--model",
        choices=[
            "mobilenet_v3_small",
            "efficientnet_lite0",
            "efficientnet_b0",
            "efficientnet_b3",
            "efficientnet_b4",
            "resnet50",
            "convnext_base",
            "vit_b_16",
            "vit_b_32",
        ],
        default="efficientnet_b0",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--normalize-targets", action="store_true", default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("model"))
    parser.add_argument("--no-augment", action="store_true", help="Disable random data augmentation.")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights for the selected backbone.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the feature extractor and train only the regression head.",
    )
    parser.add_argument(
        "--head-dropout",
        type=float,
        default=0.25,
        help="Dropout probability inserted before the regression head.",
    )
    parser.add_argument(
        "--trainable-backbone-layers",
        type=int,
        default=0,
        help="Additional backbone parameter tensors to unfreeze (counted from the end) "
        "when --freeze-backbone is supplied.",
    )
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=3,
        help="When freezing the backbone, number of initial epochs to train the regression head "
        "before unfreezing additional backbone layers.",
    )
    parser.add_argument(
        "--full-unfreeze-epoch",
        type=int,
        default=None,
        help="Epoch (1-indexed) at which to unfreeze the entire backbone. "
        "Defaults to halfway through training when --freeze-backbone is set.",
    )
    parser.add_argument(
        "--unfreeze-lr-factor",
        type=float,
        default=0.5,
        help="When new backbone parameters are unfrozen, multiply existing learning rates by this "
        "factor to stabilise optimisation (set to 1.0 to disable adjustment).",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision training.",
    )
    parser.add_argument(
        "--no-lr-scheduler",
        action="store_true",
        help="Disable learning-rate scheduling (ReduceLROnPlateau).",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.0,
        help="If > 0, clip gradient norm to this value each optimization step.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop training early if validation loss fails to improve for this many epochs "
        "(0 disables early stopping).",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-3,
        help="Minimum reduction in validation loss to qualify as an improvement for early stopping.",
    )
    parser.add_argument(
        "--early-stop-warmup",
        type=int,
        default=None,
        help="Number of initial epochs to skip before tracking early stopping patience. "
        "Defaults to the head warmup plus a few epochs after the first unfreeze.",
    )
    args = parser.parse_args()
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_dataset(args: argparse.Namespace) -> DatasetDict:
    dataset = load_dataset(args.dataset_name, args.dataset_config)
    need_split = (
        args.train_val_test_split
        or args.train_split not in dataset
        or args.validation_split not in dataset
    )
    if need_split:
        ratio_str = args.train_val_test_split or "0.8,0.1,0.1"
        ratios = [float(x) for x in ratio_str.split(",")]
        if not math.isclose(sum(ratios), 1.0, rel_tol=1e-5):
            raise ValueError("train-val-test ratios must sum to 1.")
        base_split = next(iter(dataset.keys()))
        base_dataset = dataset[base_split]
        holdout_ratio = ratios[1] + ratios[2]
        if holdout_ratio <= 0.0 or holdout_ratio >= 1.0:
            raise ValueError("Validation + test ratios must be within (0, 1).")
        if ratios[1] <= 0.0:
            raise ValueError("Validation ratio must be greater than zero.")
        primary_split = base_dataset.train_test_split(test_size=holdout_ratio, seed=args.seed)
        val_test_ratio = ratios[2] / holdout_ratio if ratios[2] > 0 else 0.0
        if ratios[2] > 0:
            val_test = primary_split["test"].train_test_split(
                test_size=val_test_ratio,
                seed=args.seed,
            )
            validation_ds = val_test["train"]
            test_ds = val_test["test"]
        else:
            validation_ds = primary_split["test"]
            test_ds = None
        dataset_dict = {
            args.train_split: primary_split["train"],
            args.validation_split: validation_ds,
        }
        test_split_name = args.test_split or "test"
        if test_ds is not None:
            dataset_dict[test_split_name] = test_ds
        dataset = DatasetDict(dataset_dict)
        tqdm.write(
            f"Created splits {list(dataset.keys())} from base split '{base_split}' "
            f"using ratios {ratios}."
        )
    return dataset


class HuggingFaceImageDataset(Dataset):
    def __init__(
        self,
        dataset,
        image_column: str,
        target_cols: Sequence[str],
        transform: transforms.Compose,
        normalizer: Optional["TargetNormalizer"],
    ):
        self.dataset = dataset
        self.image_column = image_column
        self.target_cols = list(target_cols)
        self.transform = transform
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        example = self.dataset[idx]
        image = example[self.image_column]
        if isinstance(image, Image.Image):
            pil_image = image
        else:
            pil_image = Image.open(io.BytesIO(image["bytes"]))
        tensor = self.transform(pil_image.convert("RGB"))
        target_values = [float(example[col]) for col in self.target_cols]
        target = torch.tensor(target_values, dtype=torch.float32)
        if self.normalizer is not None:
            target = self.normalizer.normalize(target)
        return tensor, target


@dataclass
class TargetNormalizer:
    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1e-6

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std.clamp_min(self.eps)

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.std + self.mean


class ParameterGroup:
    """Lightweight container for a labeled set of parameters."""

    def __init__(self, label: str, parameters: Sequence[torch.nn.Parameter]):
        self.label = label
        self._parameters: List[torch.nn.Parameter] = [
            param for param in parameters if isinstance(param, torch.nn.Parameter)
        ]

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return iter(self._parameters)

    def size(self) -> int:
        return len(self._parameters)

    def __bool__(self) -> bool:
        return bool(self._parameters)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic helper
        return f"ParameterGroup(label={self.label!r}, size={self.size()})"


def compute_target_stats(dataset, target_cols: Sequence[str]) -> TargetNormalizer:
    values = np.array([[float(item[col]) for col in target_cols] for item in dataset])
    mean = torch.tensor(values.mean(axis=0), dtype=torch.float32)
    std = torch.tensor(values.std(axis=0), dtype=torch.float32)
    std = torch.clamp(std, min=1e-6)
    return TargetNormalizer(mean=mean, std=std)


def build_transforms(
    image_size: int,
    augment: bool,
    weights: Optional["WeightsEnum"] = None,
) -> Tuple[transforms.Compose, transforms.Compose]:
    mean = IMAGENET_MEAN
    std = IMAGENET_STD
    if weights is not None and hasattr(weights, "meta"):
        meta = weights.meta or {}
        mean = tuple(meta.get("mean", IMAGENET_MEAN))
        std = tuple(meta.get("std", IMAGENET_STD))

    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.75, 1.0),
                    ratio=(0.9, 1.1),
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(
                    degrees=12,
                    translate=(0.05, 0.05),
                    scale=(0.9, 1.1),
                    shear=5,
                ),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.1)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    eval_transforms = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, eval_transforms


def resolve_pretrained_weights(model_name: str, use_pretrained: bool) -> Optional[WeightsEnum]:
    if not use_pretrained:
        return None

    if model_name == "mobilenet_v3_small":
        return MobileNet_V3_Small_Weights.IMAGENET1K_V1
    if model_name == "efficientnet_b0":
        return EfficientNet_B0_Weights.IMAGENET1K_V1
    if model_name == "efficientnet_b3":
        return EfficientNet_B3_Weights.IMAGENET1K_V1
    if model_name == "efficientnet_b4":
        return EfficientNet_B4_Weights.IMAGENET1K_V1
    if model_name == "efficientnet_lite0":
        if EfficientNet_Lite0_Weights is None:
            raise ImportError(
                "efficientnet_lite0 pretrained weights are unavailable. "
                "Upgrade torchvision to a build that includes EfficientNet Lite weights or "
                "choose a different backbone."
            )
        return EfficientNet_Lite0_Weights.IMAGENET1K_V1
    if model_name == "resnet50":
        return ResNet50_Weights.IMAGENET1K_V2
    if model_name == "convnext_base":
        return ConvNeXt_Base_Weights.IMAGENET1K_V1
    if model_name == "vit_b_16":
        return ViT_B_16_Weights.IMAGENET1K_V1
    if model_name == "vit_b_32":
        return ViT_B_32_Weights.IMAGENET1K_V1
    return None


def build_regression_head(in_features: int, num_outputs: int, dropout: float) -> nn.Sequential:
    dropout = max(0.0, float(dropout))
    layers: List[nn.Module] = []
    if dropout > 0:
        layers.append(nn.Dropout(p=min(dropout, 0.95)))
    layers.append(nn.Linear(in_features, num_outputs))
    return nn.Sequential(*layers)


def build_model(
    model_name: str,
    num_outputs: int,
    weights: Optional[WeightsEnum],
    head_dropout: float,
) -> Tuple[nn.Module, nn.Module, List[ParameterGroup]]:
    head: nn.Module
    if model_name == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        head = build_regression_head(in_features, num_outputs, head_dropout)
        model.classifier[-1] = head
    elif model_name == "efficientnet_lite0":
        if efficientnet_lite0 is None:
            raise ImportError(
                "efficientnet_lite0 is unavailable in the installed torchvision build. "
                "Upgrade to torchvision>=0.19 or choose a different --model."
            )
        model = efficientnet_lite0(weights=weights)
        in_features = model.classifier[-1].in_features
        head = build_regression_head(in_features, num_outputs, head_dropout)
        model.classifier[-1] = head
    elif model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        head = build_regression_head(in_features, num_outputs, head_dropout)
        model.classifier[-1] = head
    elif model_name == "efficientnet_b3":
        model = efficientnet_b3(weights=weights)
        in_features = model.classifier[-1].in_features
        head = build_regression_head(in_features, num_outputs, head_dropout)
        model.classifier[-1] = head
    elif model_name == "efficientnet_b4":
        model = efficientnet_b4(weights=weights)
        in_features = model.classifier[-1].in_features
        head = build_regression_head(in_features, num_outputs, head_dropout)
        model.classifier[-1] = head
    elif model_name == "resnet50":
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        head = build_regression_head(in_features, num_outputs, head_dropout)
        model.fc = head
    elif model_name == "convnext_base":
        model = convnext_base(weights=weights)
        in_features = model.classifier[-1].in_features
        head = build_regression_head(in_features, num_outputs, head_dropout)
        model.classifier[-1] = head
    elif model_name == "vit_b_16":
        model = vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        head = build_regression_head(in_features, num_outputs, head_dropout)
        model.heads.head = head
    elif model_name == "vit_b_32":
        model = vit_b_32(weights=weights)
        in_features = model.heads.head.in_features
        head = build_regression_head(in_features, num_outputs, head_dropout)
        model.heads.head = head
    else:  # pragma: no cover - safeguarded by argparse choices
        raise ValueError(f"Unsupported model '{model_name}'")

    backbone_groups = collect_backbone_groups(model_name, model)
    return model, head, backbone_groups


def collect_backbone_groups(model_name: str, model: nn.Module) -> List[ParameterGroup]:
    """Derive ordered parameter groups for progressively unfreezing the backbone."""
    groups: List[ParameterGroup] = []

    def build_groups_from_modules(
        modules: Sequence[nn.Module], prefix: str
    ) -> List[ParameterGroup]:
        collected: List[ParameterGroup] = []
        for idx, module in enumerate(modules):
            params = list(module.parameters())
            if params:
                collected.append(ParameterGroup(f"{prefix}[{idx}]", params))
        return collected

    if model_name in {"mobilenet_v3_small"}:
        groups = build_groups_from_modules(list(model.features), "features")
    elif model_name.startswith("efficientnet"):
        groups = build_groups_from_modules(list(model.features), "features")
    elif model_name == "resnet50":
        modules = [model.layer1, model.layer2, model.layer3, model.layer4]
        groups = build_groups_from_modules(modules, "resnet_layer")
    elif model_name == "convnext_base":
        groups = build_groups_from_modules(list(model.features), "features")
    elif model_name.startswith("vit"):
        # Vision Transformer encoder layers (ModuleList)
        encoder_layers = getattr(model.encoder, "layers", [])
        groups = build_groups_from_modules(list(encoder_layers), "encoder_layer")

    if not groups:
        # Fallback: chunk remaining parameters into coarse groups to avoid empty schedules.
        remaining_params = list(model.parameters())
        chunk_size = max(1, len(remaining_params) // 8)
        fallback_groups: List[ParameterGroup] = []
        for idx in range(0, len(remaining_params), chunk_size):
            chunk = remaining_params[idx : idx + chunk_size]
            fallback_groups.append(ParameterGroup(f"param_chunk_{idx // chunk_size}", chunk))
        groups = fallback_groups

    return groups


class BackboneFreezer:
    """Manage backbone parameter freezing/unfreezing schedules."""

    def __init__(
        self,
        model: nn.Module,
        head: nn.Module,
        backbone_groups: Sequence[ParameterGroup],
    ):
        self.model = model
        self.head = head
        self.head_param_ids = {id(param) for param in head.parameters()}

        self.groups: List[ParameterGroup] = []
        for group in backbone_groups:
            filtered_params = [
                param for param in group.parameters() if id(param) not in self.head_param_ids
            ]
            if filtered_params:
                self.groups.append(ParameterGroup(group.label, filtered_params))

        self.non_head_parameters: List[torch.nn.Parameter] = [
            param for param in model.parameters() if id(param) not in self.head_param_ids
        ]

        if not self.groups and self.non_head_parameters:
            chunk_size = max(1, len(self.non_head_parameters) // 8)
            for idx in range(0, len(self.non_head_parameters), chunk_size):
                chunk = self.non_head_parameters[idx : idx + chunk_size]
                self.groups.append(ParameterGroup(f"backbone_chunk_{idx // chunk_size}", chunk))

    def freeze_head_only(self) -> None:
        for param in self.non_head_parameters:
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True

    def unfreeze_last_groups(self, count: int) -> int:
        if count <= 0:
            return 0
        available = len(self.groups)
        if available == 0:
            return 0
        actual = min(count, available)
        for group in self.groups[-actual:]:
            for param in group.parameters():
                param.requires_grad = True
        return actual

    def unfreeze_all(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        return [param for param in self.model.parameters() if param.requires_grad]

    def total_groups(self) -> int:
        return len(self.groups)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler],
    use_amp: bool,
    grad_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()
    for inputs, targets in tqdm(dataloader, desc="train", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0:
                clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    normalizer: Optional[TargetNormalizer],
    use_amp: bool,
    target_cols: Sequence[str],
) -> Dict[str, Any]:
    model.eval()
    criterion = nn.MSELoss(reduction="sum")
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="eval", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            with autocast(enabled=use_amp):
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)
            total_loss += batch_loss.item()
            outputs_detached = outputs.detach()
            targets_detached = targets.detach()
            if normalizer is not None:
                outputs_detached = normalizer.denormalize(outputs_detached)
                targets_detached = normalizer.denormalize(targets_detached)
            preds.append(outputs_detached.cpu().numpy())
            trues.append(targets_detached.cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)
    metrics = {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    metrics["loss"] = total_loss / len(dataloader.dataset)
    if target_cols and y_pred.shape[1] == len(target_cols):
        mae_per_target: Dict[str, float] = {}
        mse_per_target: Dict[str, float] = {}
        r2_per_target: Dict[str, float] = {}
        for idx, name in enumerate(target_cols):
            y_true_col = y_true[:, idx]
            y_pred_col = y_pred[:, idx]
            mae_per_target[name] = float(mean_absolute_error(y_true_col, y_pred_col))
            mse_per_target[name] = float(mean_squared_error(y_true_col, y_pred_col))
            try:
                r2_per_target[name] = float(r2_score(y_true_col, y_pred_col))
            except ValueError:  # pragma: no cover - single sample path
                r2_per_target[name] = float("nan")
        metrics["mae_per_target"] = mae_per_target
        metrics["mse_per_target"] = mse_per_target
        metrics["r2_per_target"] = r2_per_target
    return metrics


def save_artifacts(
    output_dir: Path,
    model: nn.Module,
    metadata: Dict[str, Any],
    normalizer: Optional[TargetNormalizer],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model_fp32.pt")
    with (output_dir / "training_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    if normalizer is not None:
        payload = {
            "columns": metadata.get("target_columns", []),
            "mean": normalizer.mean.tolist(),
            "std": normalizer.std.tolist(),
        }
        with (output_dir / "target_normalizer.json").open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = prepare_dataset(args)
    if args.train_split not in dataset or args.validation_split not in dataset:
        raise ValueError("Dataset must contain training and validation splits.")

    train_norm = compute_target_stats(dataset[args.train_split], args.target_cols)
    normalizer = train_norm if args.normalize_targets else None

    weights = resolve_pretrained_weights(args.model, use_pretrained=not args.no_pretrained)

    train_tf, eval_tf = build_transforms(
        args.image_size,
        augment=not args.no_augment,
        weights=weights,
    )
    train_ds = HuggingFaceImageDataset(
        dataset[args.train_split],
        image_column=args.image_column,
        target_cols=args.target_cols,
        transform=train_tf,
        normalizer=normalizer,
    )
    val_ds = HuggingFaceImageDataset(
        dataset[args.validation_split],
        image_column=args.image_column,
        target_cols=args.target_cols,
        transform=eval_tf,
        normalizer=normalizer,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    model, head, backbone_groups = build_model(
        args.model,
        num_outputs=len(args.target_cols),
        weights=weights,
        head_dropout=args.head_dropout,
    )
    model = model.to(device)

    freezer = BackboneFreezer(model, head, backbone_groups)
    freeze_epochs = max(0, args.freeze_epochs if args.freeze_backbone else 0)
    if args.freeze_backbone and freeze_epochs >= args.epochs:
        freeze_epochs = max(0, args.epochs - 1)
        tqdm.write(
            f"Adjusted --freeze-epochs to {freeze_epochs} so fine-tuning can occur within {args.epochs} epochs."
        )

    full_unfreeze_epoch = args.full_unfreeze_epoch if args.freeze_backbone else None
    if args.freeze_backbone:
        if full_unfreeze_epoch is None or full_unfreeze_epoch <= 0:
            full_unfreeze_epoch = max(freeze_epochs + 5, args.epochs // 2)
        full_unfreeze_epoch = min(max(full_unfreeze_epoch, freeze_epochs + 1), args.epochs)
    else:
        full_unfreeze_epoch = None

    args.freeze_epochs = freeze_epochs
    args.full_unfreeze_epoch = full_unfreeze_epoch

    current_stage_desc = "full_backbone"
    initial_partial_blocks = 0
    last_partial_blocks = 0

    if not args.freeze_backbone:
        freezer.unfreeze_all()
    else:
        freezer.freeze_head_only()
        current_stage_desc = "head_only"
        schedule_message = (
            f"Backbone fine-tuning schedule: detected {freezer.total_groups()} groups | "
            f"head-only warmup epochs={freeze_epochs} | "
            f"full unfreeze epoch="
            f"{'disabled' if full_unfreeze_epoch is None else full_unfreeze_epoch}."
        )
        tqdm.write(schedule_message)
        if freeze_epochs == 0:
            initial_partial_blocks = freezer.unfreeze_last_groups(
                max(0, args.trainable_backbone_layers)
            )
            last_partial_blocks = initial_partial_blocks
            if initial_partial_blocks > 0:
                current_stage_desc = f"head_plus_last_{initial_partial_blocks}_blocks"
                tqdm.write(
                    f"Enabled gradient updates for the last {initial_partial_blocks} backbone blocks from the start."
                )
            else:
                tqdm.write("Starting with head-only optimisation; no backbone blocks requested yet.")

    if args.early_stop_warmup is None:
        if args.freeze_backbone:
            default_warmup = max(freeze_epochs + 2, (full_unfreeze_epoch or freeze_epochs) // 2)
            default_warmup = max(default_warmup, freeze_epochs + 5)
        else:
            default_warmup = 5
        args.early_stop_warmup = default_warmup
    else:
        args.early_stop_warmup = max(0, args.early_stop_warmup)

    if not any(param.requires_grad for param in model.parameters()):
        raise RuntimeError("No trainable parameters found. Disable --freeze-backbone or adjust the head.")

    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    use_amp = torch.cuda.is_available() and not args.no_amp
    if use_amp:
        tqdm.write("Enabling automatic mixed precision (fp16) on CUDA.")
    scaler = GradScaler() if use_amp else None

    scheduler = None
    if not args.no_lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )

    best_val_loss = float("inf")
    best_state = None
    history: List[Dict[str, float]] = []
    early_stop_enabled = max(0, args.early_stop_patience) > 0
    early_stop_patience = max(0, args.early_stop_patience)
    early_stop_min_delta = max(0.0, args.early_stop_min_delta)
    early_stop_wait = 0
    early_stop_triggered = False
    early_stop_epoch = None
    if early_stop_enabled:
        tqdm.write(
            "Early stopping enabled: "
            f"patience={early_stop_patience}, min_delta={early_stop_min_delta:.3g}, "
            f"warmup={args.early_stop_warmup} epochs."
        )

    for epoch in range(1, args.epochs + 1):
        if args.freeze_backbone:
            stage_changed = False
            stage_descriptions: List[str] = []
            if freeze_epochs > 0 and epoch == freeze_epochs + 1:
                freezer.freeze_head_only()
                partial = freezer.unfreeze_last_groups(max(0, args.trainable_backbone_layers))
                if partial > 0:
                    current_stage_desc = f"head_plus_last_{partial}_blocks"
                    last_partial_blocks = partial
                    stage_descriptions.append(f"last {partial} backbone blocks")
                    tqdm.write(
                        f"Epoch {epoch:03d}: unfroze the last {partial} backbone blocks after warmup."
                    )
                else:
                    current_stage_desc = "head_only"
                    tqdm.write(
                        f"Epoch {epoch:03d}: warmup complete; remaining in head-only configuration."
                    )
                stage_changed = True

            if full_unfreeze_epoch is not None and epoch == full_unfreeze_epoch:
                freezer.unfreeze_all()
                current_stage_desc = "full_backbone"
                stage_descriptions.append("entire backbone")
                tqdm.write(f"Epoch {epoch:03d}: unfroze the entire backbone.")
                stage_changed = True

            if stage_changed:
                unfroze_parameters = bool(stage_descriptions)
                if unfroze_parameters and args.unfreeze_lr_factor not in (0.0, 1.0):
                    for group in optimizer.param_groups:
                        group["lr"] = max(group["lr"] * args.unfreeze_lr_factor, 1e-8)
                    tqdm.write(
                        f"Scaled learning rates by factor {args.unfreeze_lr_factor:.2f} "
                        f"after unfreezing {', '.join(stage_descriptions) or 'nothing'}."
                    )
                elif stage_descriptions:
                    tqdm.write(
                        "Backbone stage change retained existing learning rate "
                        f"(factor={args.unfreeze_lr_factor})."
                    )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            use_amp,
            grad_clip_norm=max(0.0, args.grad_clip_norm),
        )
        metrics = evaluate(
            model,
            val_loader,
            device,
            normalizer,
            use_amp,
            args.target_cols,
        )
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss
        metrics["stage"] = current_stage_desc
        prev_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(metrics["loss"])
            new_lr = optimizer.param_groups[0]["lr"]
            metrics["learning_rate"] = new_lr
            if not math.isclose(new_lr, prev_lr, rel_tol=1e-6):
                tqdm.write(
                    f"ReduceLROnPlateau triggered: lr {prev_lr:.2e} -> {new_lr:.2e}"
                )
        else:
            metrics["learning_rate"] = prev_lr
        history.append(metrics)
        tqdm.write(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} mse={metrics['mse']:.4f} "
            f"mae={metrics['mae']:.4f} lr={metrics['learning_rate']:.2e}"
        )

        improved = metrics["loss"] < (best_val_loss - early_stop_min_delta)
        if improved:
            best_val_loss = metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            early_stop_wait = 0
        elif early_stop_enabled and epoch > args.early_stop_warmup:
            early_stop_wait += 1
            if early_stop_wait >= early_stop_patience:
                early_stop_triggered = True
                early_stop_epoch = epoch
                tqdm.write(
                    f"Early stopping triggered at epoch {epoch:03d} "
                    f"(patience={early_stop_patience}, min_delta={early_stop_min_delta:.3g})."
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        tqdm.write(f"Restored best validation checkpoint with loss {best_val_loss:.4f}.")
    if early_stop_triggered:
        tqdm.write(
            f"Training stopped early after epoch {early_stop_epoch:03d}; "
            "exporting the best validation checkpoint observed so far."
        )

    final_metrics = evaluate(
        model,
        val_loader,
        device,
        normalizer,
        use_amp=False,
        target_cols=args.target_cols,
    )
    if "mae_per_target" in final_metrics:
        per_target_mae = ", ".join(
            f"{name}: {value:.2f}" for name, value in final_metrics["mae_per_target"].items()
        )
        tqdm.write(f"Validation MAE per target -> {per_target_mae}")

    args_payload = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    dataset_summary = {split: len(split_dataset) for split, split_dataset in dataset.items()}

    best_epoch_index = int(
        min(range(len(history)), key=lambda idx: history[idx]["loss"]) if history else -1
    )
    history_summary = {
        "epochs_ran": len(history),
        "best_epoch_zero_indexed": best_epoch_index,
        "best_epoch_one_indexed": best_epoch_index + 1 if best_epoch_index >= 0 else -1,
    }
    if history:
        history_summary["best_val_loss"] = history[best_epoch_index]["loss"]

    metadata = {
        "schema_version": "1.0",
        "export_timestamp": time.time(),
        "framework": {
            "python_version": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
        "training_script": Path(__file__).name,
        "args": args_payload,
        "target_columns": args.target_cols,
        "image_size": args.image_size,
        "normalize_targets": args.normalize_targets,
        "dataset_summary": dataset_summary,
        "history_summary": history_summary,
        "metrics": {"validation": final_metrics},
        "training_history": history,
        "backbone_schedule": {
            "freeze_backbone": bool(args.freeze_backbone),
            "freeze_epochs": freeze_epochs if args.freeze_backbone else 0,
            "trainable_backbone_layers_requested": max(0, args.trainable_backbone_layers),
            "available_backbone_groups": freezer.total_groups(),
            "full_unfreeze_epoch": full_unfreeze_epoch,
            "unfreeze_lr_factor": args.unfreeze_lr_factor,
            "initial_partial_blocks": initial_partial_blocks,
            "latest_partial_blocks": last_partial_blocks,
        },
        "early_stopping": {
            "enabled": early_stop_enabled,
            "patience": early_stop_patience,
            "min_delta": early_stop_min_delta,
            "warmup_epochs": args.early_stop_warmup,
            "triggered": early_stop_triggered,
            "trigger_epoch": early_stop_epoch,
        },
    }

    save_artifacts(args.output_dir, model, metadata, normalizer)
    tqdm.write(f"Artifacts saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
