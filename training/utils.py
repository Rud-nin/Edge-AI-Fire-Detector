from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from esp_ppq.executor.torch import TorchExecutor
from const import TEST_DIR, TRAIN_DIR, VALIDATE_DIR


def load_training_data(
    batch_size: int,
    image_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_transform: transforms.Compose = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0),
    ])

    eval_transform: transforms.Compose = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0),
    ])

    train_dataset: datasets.ImageFolder = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform,
    )

    val_dataset: datasets.ImageFolder = datasets.ImageFolder(
        root=VALIDATE_DIR,
        transform=eval_transform,
    )

    test_dataset: datasets.ImageFolder = datasets.ImageFolder(
        root=TEST_DIR,
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
    batch_size: int,
    image_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    transform: transforms.Compose = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.PILToTensor(),
    ])

    train_dataset: datasets.ImageFolder = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=transform,
    )

    val_dataset: datasets.ImageFolder = datasets.ImageFolder(
        root=VALIDATE_DIR,
        transform=transform,
    )

    test_dataset: datasets.ImageFolder = datasets.ImageFolder(
        root=TEST_DIR,
        transform=transform,
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
    model: nn.Module | TorchExecutor,
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