import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import Model

SEED: int = 42
EPOCHS: int = 10
BATCH_SIZE: int = 32
IMAGE_SIZE: int = 224
LEARNING_RATE: float = 1e-3
NUM_WORKERS: int = 4

PROJECT_DIR: Path = Path(__file__).resolve().parent.parent
LOG_DIR: Path = PROJECT_DIR / "training" / "logs"
MODELS_DIR: Path = PROJECT_DIR / "models"

log: logging.Logger = logging.getLogger(__name__)


def load_data(
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_dir: Path = PROJECT_DIR / "data" / "train"
    val_dir: Path = PROJECT_DIR / "data" / "validate"
    test_dir: Path = PROJECT_DIR / "data" / "test"

    train_transform: transforms.Compose = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

    eval_transform: transforms.Compose = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

    train_dataset: datasets.ImageFolder = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform,
    )

    val_dataset: datasets.ImageFolder = datasets.ImageFolder(
        root=val_dir,
        transform=eval_transform,
    )

    test_dataset: datasets.ImageFolder = datasets.ImageFolder(
        root=test_dir,
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss: float = 0.0
    correct: int = 0
    total: int = 0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, int, int]:
    model.eval()
    running_loss: float = 0.0
    correct: int = 0
    total: int = 0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total, correct, total


def main() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file: Path = LOG_DIR / f"train_{timestamp}.log"
    model_path: Path = MODELS_DIR / f"model_{timestamp}.pt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w"),
        ],
        force=True,
    )

    # set_seed(SEED)

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s seed=%d epochs=%d batch_size=%d lr=%g", device, SEED, EPOCHS, BATCH_SIZE, LEARNING_RATE)
    if device.type == "cuda":
        log.info("gpu=%s", torch.cuda.get_device_name(device))
    log.info("logging to terminal and %s", log_file)

    train_loader, val_loader, test_loader = load_data(batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, num_workers=NUM_WORKERS)
    classes = train_loader.dataset.classes
    log.info(
        "classes=%s train=%d val=%d test=%d",
        classes,
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    model: nn.Module = Model(num_classes=len(classes)).to(device)
    criterion: nn.Module = nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc: float = 0.0
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    try:
        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
            log.info(
                "epoch %d/%d train_loss=%.4f train_acc=%.2f%% val_loss=%.4f val_acc=%.2f%%",
                epoch, EPOCHS, train_loss, train_acc * 100.0, val_loss, val_acc * 100.0,
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state_dict = {k: v.clone().cpu() for k, v in model.state_dict().items()}

        test_loss, test_acc, correct, total = evaluate(model, test_loader, criterion, device)
        log.info("test_loss=%.4f", test_loss)
        log.info("OVERALL TEST ACCURACY: %.2f%% (%d/%d)", test_acc * 100.0, correct, total)
    finally:
        if best_state_dict is not None:
            torch.save(best_state_dict, model_path)
            log.info("saved best model val_acc=%.2f%% -> %s", best_val_acc * 100.0, model_path)


if __name__ == "__main__":
    main()
