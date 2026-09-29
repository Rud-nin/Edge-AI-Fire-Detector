import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import Model
from const import LOG_DIR, MODELS_DIR, TRAIN_DIR, VALIDATE_DIR, TEST_DIR, IMAGE_SIZE, DEVICE
from utils import load_training_data, evaluate

SEED: int = 42
EPOCHS: int = 10
BATCH_SIZE: int = 32
LEARNING_RATE: float = 1e-3
NUM_WORKERS: int = 4

log: logging.Logger = logging.getLogger(__name__)


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


def main() -> None:
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

    log.info("device=%s seed=%d epochs=%d batch_size=%d lr=%g", DEVICE, SEED, EPOCHS, BATCH_SIZE, LEARNING_RATE)
    if DEVICE.type == "cuda":
        log.info("gpu=%s", torch.cuda.get_device_name(DEVICE))
    log.info("logging to terminal and %s", log_file)

    train_loader, val_loader, test_loader = load_training_data(batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, num_workers=NUM_WORKERS)
    classes = train_loader.dataset.classes
    log.info(
        "classes=%s train=%d val=%d test=%d",
        classes,
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    model: nn.Module = Model(num_classes=len(classes)).to(DEVICE)
    criterion: nn.Module = nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc: float = 0.0
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    try:
        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
            log.info(
                "epoch %d/%d train_loss=%.4f train_acc=%.2f%% val_loss=%.4f val_acc=%.2f%%",
                epoch, EPOCHS, train_loss, train_acc * 100.0, val_loss, val_acc * 100.0,
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state_dict = {k: v.clone().cpu() for k, v in model.state_dict().items()}

        test_loss, test_acc, correct, total = evaluate(model, test_loader, criterion, DEVICE)
        log.info("test_loss=%.4f", test_loss)
        log.info("OVERALL TEST ACCURACY: %.2f%% (%d/%d)", test_acc * 100.0, correct, total)
    finally:
        if best_state_dict is not None:
            torch.save(best_state_dict, model_path)
            log.info("saved best model val_acc=%.2f%% -> %s", best_val_acc * 100.0, model_path)


if __name__ == "__main__":
    main()
