# ---------------------------------------------------------------------------
# Train the fire-detection model (builds model.py's Model4).
#
# Dataset layout (torchvision ImageFolder): the data directory needs one folder
# per split, each holding one folder per class:
#
#     <data-dir>/
#         train/<class>/<images>
#         validate/<class>/<images>
#         test/<class>/<images>
#
# Images are resized to 224x224 and kept as raw RGB in [0, 255] (the transforms
# in const.py), matching what the ESP32 firmware feeds the quantized model.
# Training uses CrossEntropyLoss + Adam; the checkpoint with the best validation
# accuracy is saved.
#
# Usage:
#     python train.py [options]
#
# Options (all optional; defaults are the constants at the top of this file):
#     --data-dir PATH       dataset root containing train/validate/test.
#                           Used only if the path exists, otherwise DATA_DIR
#                           from const.py is used.
#     --log-dir PATH        directory for the run log (<name>_<timestamp>.log);
#                           created if missing. Defaults to LOG_DIR.
#     --model-dir PATH      directory for the saved checkpoint; created if
#                           missing. Defaults to MODELS_DIR.
#     --seed INT            seed python/torch RNGs. Omit for true randomness.
#                           The seed is applied before the model name is drawn,
#                           so a seeded run is fully reproducible.
#     --epoch INT           training epochs                    (default: 15)
#     --batch-size INT      DataLoader batch size              (default: 128)
#     --learning-rate FLOAT Adam learning rate                 (default: 1e-3)
#     --num-worker INT      DataLoader workers                 (default: 4)
#     --name STR            model name, saved as models/<name>.pt. Omit for a
#                           random "<First>_<Last>" name drawn from the
#                           FIRST_NAMES / LAST_NAMES lists below.
#
# Examples:
#     python train.py                                # defaults, random name
#     python train.py --epoch 30 --batch-size 64     # longer, smaller batches
#     python train.py --seed 42 --name firenet_v1    # reproducible + named
#     python train.py --data-dir /mnt/fire --log-dir /tmp/logs --model-dir /tmp/models
#
# Outputs:
#     <model-dir>/<name>.pt             best checkpoint (highest val accuracy)
#     <log-dir>/<name>_<timestamp>.log   full training log
# ---------------------------------------------------------------------------

import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import Model4
from const import DATA_DIR, LOG_DIR, MODELS_DIR, DEVICE, configure_logging, log
from utils import load_training_data, evaluate

EPOCHS: int = 15
BATCH_SIZE: int = 128
LEARNING_RATE: float = 1e-3
NUM_WORKERS: int = 4

FIRST_NAMES = [
    "Veyra", "Nexis", "Kairo", "Axiom", "Vanta",
    "Orin", "Zyra", "Kael", "Novi", "Riven"
]
LAST_NAMES = [
    "Sora", "Vexa", "Tarin", "Elix", "Nyx",
    "Arvo", "Zeno", "Ceryx", "Oryn", "Quell"
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fire-detection model.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"dataset root holding train/validate/test sub-dirs (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help=f"directory for the run log (default: {LOG_DIR})",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help=f"directory for the saved checkpoint (default: {MODELS_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed; omit for true randomness",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=EPOCHS,
        help=f"number of epochs (default: {EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help=f"learning rate (default: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--num-worker",
        type=int,
        default=NUM_WORKERS,
        help=f"DataLoader worker count (default: {NUM_WORKERS})",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="model name, saved as <name>.pt (default: a random name)",
    )
    return parser.parse_args()


def random_name() -> str:
    """Random '<first>_<last>' model name."""
    return f"{random.choice(FIRST_NAMES)}_{random.choice(LAST_NAMES)}"


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
    args = parse_args()

    # Seed first so the generated model name is reproducible too.
    if args.seed is not None:
        set_seed(args.seed)

    name: str = args.name if args.name is not None else random_name()

    log_dir: Path = args.log_dir if args.log_dir is not None else LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file: Path = log_dir / f"{name}_{timestamp}.log"
    configure_logging(log_file)

    data_dir: Path = DATA_DIR
    if args.data_dir is not None:
        if args.data_dir.is_dir():
            data_dir = args.data_dir
        else:
            log.warning("data dir %s does not exist, falling back to %s", args.data_dir, DATA_DIR)

    model_dir: Path = args.model_dir if args.model_dir is not None else MODELS_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path: Path = model_dir / f"{name}.pt"

    log.info(
        "device=%s seed=%s epochs=%d batch_size=%d lr=%g num_workers=%d",
        DEVICE,
        args.seed if args.seed is not None else "none",
        args.epoch,
        args.batch_size,
        args.learning_rate,
        args.num_worker,
    )
    if DEVICE.type == "cuda":
        log.info("gpu=%s", torch.cuda.get_device_name(DEVICE))
    log.info("data_dir=%s", data_dir)
    log.info("logging to terminal and %s", log_file)
    log.info("model=%s", model_path)

    train_loader, val_loader, test_loader = load_training_data(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_worker,
    )
    classes = train_loader.dataset.classes

    model: nn.Module = Model4(num_classes=len(classes)).to(DEVICE)
    criterion: nn.Module = nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    log.info(
        "parameters=%d classes=%s train=%d val=%d test=%d",
        sum(p.numel() for p in model.parameters()),
        classes,
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    best_val_acc: float = 0.0
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    try:
        for epoch in range(1, args.epoch + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
            log.info(
                "epoch %d/%d train_loss=%.4f train_acc=%.2f%% val_loss=%.4f val_acc=%.2f%%",
                epoch, args.epoch, train_loss, train_acc * 100.0, val_loss, val_acc * 100.0,
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
