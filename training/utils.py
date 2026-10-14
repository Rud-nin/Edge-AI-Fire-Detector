import argparse
import inspect
from pathlib import Path
from typing import Any, Dict, Tuple, Type, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from esp_ppq.executor.torch import TorchExecutor
from const import (
    FLAG_IMAGENET_NORMALIZATION,
    TEST_DIRNAME,
    TRAIN_DIRNAME,
    VALIDATE_DIRNAME,
    train_transform,
    eval_transform,
    calib_transform,
    normalize_train_transform,
    normalize_eval_transform,
    normalize_calib_transform,
)


def existing_file(value: str) -> Path:
    """argparse type: a path that must already be an existing file."""
    path: Path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"{path} does not exist or is not a file")
    return path


def existing_dir(value: str) -> Path:
    """argparse type: a path that must already be an existing directory."""
    path: Path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"{path} does not exist or is not a directory")
    return path


def select_transforms(
    flags: int = 0,
) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    """Pick the (train, eval, calib) transforms requested by a model's flags.

    flags is the bitmask a model declares in its DATA_FLAGS attribute (const.py's
    FLAG_* constants); 0 keeps the default raw [0, 255] pipelines.
    """
    if flags & FLAG_IMAGENET_NORMALIZATION:
        return normalize_train_transform, normalize_eval_transform, normalize_calib_transform
    return train_transform, eval_transform, calib_transform


def instantiate_model(model_class: Type[nn.Module], **kwargs: Any) -> nn.Module:
    """Build model_class, dropping keyword arguments it does not accept.

    Lets checkpoint/quantization callers pass pretrained=False (the weights come
    from the file anyway) without special-casing the models that have no such
    parameter.
    """
    accepted: Dict[str, Any] = {
        name: value
        for name, value in kwargs.items()
        if name in inspect.signature(model_class.__init__).parameters
    }
    return model_class(**accepted)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Number of parameters, optionally restricted to the trainable ones."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad or not trainable_only
    )


def load_training_data(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    flags: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_t, eval_t, _ = select_transforms(flags)

    train_dataset: datasets.ImageFolder = datasets.ImageFolder(
        root=data_dir / TRAIN_DIRNAME,
        transform=train_t,
    )

    val_dataset: datasets.ImageFolder = datasets.ImageFolder(
        root=data_dir / VALIDATE_DIRNAME,
        transform=eval_t,
    )

    test_dataset: datasets.ImageFolder = datasets.ImageFolder(
        root=data_dir / TEST_DIRNAME,
        transform=eval_t,
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
    flags: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    _, _, calib_t = select_transforms(flags)

    train_dataset: datasets.ImageFolder = datasets.ImageFolder(
        root=data_dir / TRAIN_DIRNAME,
        transform=calib_t,
    )

    val_dataset: datasets.ImageFolder = datasets.ImageFolder(
        root=data_dir / VALIDATE_DIRNAME,
        transform=calib_t,
    )

    test_dataset: datasets.ImageFolder = datasets.ImageFolder(
        root=data_dir / TEST_DIRNAME,
        transform=calib_t,
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
