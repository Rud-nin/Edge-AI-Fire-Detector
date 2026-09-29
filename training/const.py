from pathlib import Path
import torch

# Global const
PROJECT_DIR: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = PROJECT_DIR / "models"
DATA_DIR: Path = PROJECT_DIR / "data"
TEST_DIR: Path = DATA_DIR / "test"
TRAIN_DIR: Path = DATA_DIR / "train"
VALIDATE_DIR: Path = DATA_DIR / "validate"
LOG_DIR: Path = PROJECT_DIR / "training" / "logs"

LOG_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

IMAGE_SIZE: int = 224
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
