import argparse
from pathlib import Path
from typing import Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from esp_ppq.executor.torch import TorchExecutor
from const import (
    TEST_DIRNAME,
    TRAIN_DIRNAME,
    VALIDATE_DIRNAME,
    train_transform,
    eval_transform,
    calib_transform,
)


def existing_file(value: str) -> Path:
    """argparse type: a path that must already be an existing file."""
    path: Path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"{path} does not exist or is not a file")
    return path


def load_training_data(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_dataset: datasets.ImageFolder = datasets.ImageFolder(
        root=data_dir / TRAIN_DIRNAME,
        transform=train_transform,
    )

    val_dataset: datasets.ImageFolder = datasets.ImageFolder(
        root=data_dir / VALIDATE_DIRNAME,
        transform=eval_transform,
    )

    test_dataset: datasets.ImageFolder = datasets.ImageFolder(
        root=data_dir / TEST_DIRNAME,
        transform=eval_transform,
    )

    train_loader: DataLoader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader: DataLoader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader: DataLoader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def load_calibrating_data(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_dataset: datasets.ImageFolder = datasets.ImageFolder(
        root=data_dir / TRAIN_DIRNAME,
        transform=calib_transform,
    )

    val_dataset: datasets.ImageFolder = datasets.ImageFolder(
        root=data_dir / VALIDATE_DIRNAME,
        transform=calib_transform,
    )

    test_dataset: datasets.ImageFolder = datasets.ImageFolder(
        root=data_dir / TEST_DIRNAME,
        transform=calib_transform,
    )

    train_loader: DataLoader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader: DataLoader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader: DataLoader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(
    model: Union[nn.Module, TorchExecutor],
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, int, int]:

    if not isinstance(model, TorchExecutor):
        model.eval()

    running_loss: float = 0.0
    correct: int = 0
    total: int = 0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        outputs = model(images)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        loss = criterion(outputs, labels)

        running_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total, correct, total
