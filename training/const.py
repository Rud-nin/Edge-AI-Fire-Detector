import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torchvision import transforms

# Global const
PROJECT_DIR: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = PROJECT_DIR / "models"
DATA_DIR: Path = PROJECT_DIR / "data"
LOG_DIR: Path = PROJECT_DIR / "training" / "logs"

# Dataset layout: sub-directory names under a data dir.
TRAIN_DIRNAME: str = "train"
VALIDATE_DIRNAME: str = "validate"
TEST_DIRNAME: str = "test"

TEST_DIR: Path = DATA_DIR / TEST_DIRNAME
TRAIN_DIR: Path = DATA_DIR / TRAIN_DIRNAME
VALIDATE_DIR: Path = DATA_DIR / VALIDATE_DIRNAME

LOG_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

IMAGE_SIZE: int = 224
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Detection indicator drawn on the preview window (BGR red).
INDICATOR_SIZE: int = 5
INDICATOR_COLOR: Tuple[int, int, int] = (0, 0, 255)

# Class labels, in ImageFolder (and therefore model output) order.
CLASS_NAMES: Tuple[str, str] = ("0", "1")

# Longest edge (px) the preview window is downscaled to fit.
DISPLAY_MAX_SIZE: int = 960

# Logging
LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

log: logging.Logger = logging.getLogger(__name__)


def configure_logging(log_file: Optional[Path] = None) -> None:
    """Log INFO to the terminal, and additionally to log_file when given."""
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, mode="w"))

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=handlers,
        force=True,
    )


# Image transforms shared by the training, eval and calibration pipelines.
# Raw RGB in [0, 255] (no mean/std shift).
NOISE_STD: float = 5.0

# Input-preparation requirements a model can request from the load_* helpers in
# utils.py. Bitmask: bit 0 applies the ImageNet mean/std shift during
# preprocessing instead of leaving it to the model (see Model5.DATA_FLAGS).
FLAG_IMAGENET_NORMALIZATION: int = 1

# ImageNet statistics in [0, 255] units (the usual mean/std x255), matching the
# [0, 255] pixels the transforms below work on. firmware.cpp's
# ENABLE_IMAGENET_NORMALIZATION mirrors these values.
IMAGENET_MEAN: Tuple[float, float, float] = (123.675, 116.28, 103.53)
IMAGENET_STD: Tuple[float, float, float] = (58.395, 57.12, 57.375)


def add_gaussian_noise(image: torch.Tensor) -> torch.Tensor:
    """Add zero-mean Gaussian noise (std NOISE_STD, in 0..255 units).

    Keeps the input dtype, so it works both on the float [0, 255] tensors and
    on the uint8 tensor produced by PILToTensor.
    """
    noisy: torch.Tensor = image.float() + torch.randn_like(image, dtype=torch.float32) * NOISE_STD
    return torch.round(noisy).clamp(0.0, 255.0).to(image.dtype)


train_transform: transforms.Compose = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255.0),
    transforms.Lambda(add_gaussian_noise),
])

eval_transform: transforms.Compose = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255.0),
    transforms.Lambda(add_gaussian_noise),
])

calib_transform: transforms.Compose = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.PILToTensor(),
    transforms.Lambda(add_gaussian_noise),
])


def normalize_imagenet(image: torch.Tensor) -> torch.Tensor:
    """Apply the ImageNet mean/std shift to a [0, 255] image.

    Casts to float first, so it also works on the uint8 tensor produced by
    PILToTensor in the calibration pipeline.
    """
    mean: torch.Tensor = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std: torch.Tensor = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (image.float() - mean) / std


# Normalised variants for models that request FLAG_IMAGENET_NORMALIZATION
# (model.py's Model5). They are built from the transforms above so their stages
# cannot drift apart; the shift is appended last, i.e. after the [0, 255] noise.
normalize_train_transform: transforms.Compose = transforms.Compose([
    *train_transform.transforms,
    transforms.Lambda(normalize_imagenet),
])

normalize_eval_transform: transforms.Compose = transforms.Compose([
    *eval_transform.transforms,
    transforms.Lambda(normalize_imagenet),
])

normalize_calib_transform: transforms.Compose = transforms.Compose([
    *calib_transform.transforms,
    transforms.Lambda(normalize_imagenet),
])
