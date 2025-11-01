#!/usr/bin/env python3
"""Train and export the food nutrition regression model."""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from torchvision.models import mobilenet_v3_small
from tqdm.auto import tqdm

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
    parser.add_argument("--model", choices=["mobilenet_v3_small", "efficientnet_lite0"], default="mobilenet_v3_small")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--normalize-targets", action="store_true", default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("model"))
    parser.add_argument("--no-augment", action="store_true", help="Disable random data augmentation.")
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
        dataset = dataset[base_split].train_test_split(test_size=ratios[2], seed=args.seed)
        if ratios[1] <= 0.0:
            raise ValueError("Validation ratio must be greater than zero.")
        val_ratio = ratios[2] / (ratios[1] + ratios[2]) if ratios[1] + ratios[2] > 0 else 0.0
        val_test = dataset["test"].train_test_split(test_size=val_ratio, seed=args.seed)
        dataset_dict = {
            args.train_split: dataset["train"],
            args.validation_split: val_test["train"],
        }
        test_split_name = args.test_split or "test"
        dataset_dict[test_split_name] = val_test["test"]
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


def compute_target_stats(dataset, target_cols: Sequence[str]) -> TargetNormalizer:
    values = np.array([[float(item[col]) for col in target_cols] for item in dataset])
    mean = torch.tensor(values.mean(axis=0), dtype=torch.float32)
    std = torch.tensor(values.std(axis=0), dtype=torch.float32)
    std = torch.clamp(std, min=1e-6)
    return TargetNormalizer(mean=mean, std=std)


def build_transforms(image_size: int, augment: bool) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transforms = [
        transforms.Resize(int(image_size * 1.1)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if augment:
        train_transforms.insert(0, transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)))
        train_transforms.insert(1, transforms.RandomHorizontalFlip())

    eval_transforms = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(train_transforms), eval_transforms


def build_model(model_name: str, num_outputs: int) -> nn.Module:
    if model_name == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights="DEFAULT")
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_outputs)
        return model
    if model_name == "efficientnet_lite0":
        try:
            from torchvision.models import efficientnet_lite0
        except ImportError as exc:  # pragma: no cover - depends on torchvision build
            raise ImportError(
                "efficientnet_lite0 is unavailable in the installed torchvision build. "
                "Upgrade to torchvision>=0.19 or choose --model mobilenet_v3_small."
            ) from exc
        model = efficientnet_lite0(weights="DEFAULT")
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_outputs)
        return model
    raise ValueError(f"Unsupported model '{model_name}'")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()
    for inputs, targets in tqdm(dataloader, desc="train", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    normalizer: Optional[TargetNormalizer],
) -> Dict[str, float]:
    model.eval()
    criterion = nn.MSELoss(reduction="sum")
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="eval", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item()
            if normalizer is not None:
                outputs = normalizer.denormalize(outputs.cpu())
                targets = normalizer.denormalize(targets.cpu())
            preds.append(outputs.cpu().numpy())
            trues.append(targets.cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)
    metrics = {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    metrics["loss"] = total_loss / len(dataloader.dataset)
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

    train_tf, eval_tf = build_transforms(args.image_size, augment=not args.no_augment)
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

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args.model, num_outputs=len(args.target_cols)).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_val_loss = float("inf")
    best_state = None
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device, normalizer)
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss
        history.append(metrics)
        tqdm.write(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} mse={metrics['mse']:.4f} mae={metrics['mae']:.4f}"
        )
        if metrics["loss"] < best_val_loss:
            best_val_loss = metrics["loss"]
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = evaluate(model, val_loader, device, normalizer)
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
    }

    save_artifacts(args.output_dir, model, metadata, normalizer)
    tqdm.write(f"Artifacts saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
