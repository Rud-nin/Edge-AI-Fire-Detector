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
