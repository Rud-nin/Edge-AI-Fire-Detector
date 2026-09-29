import torch
import torch.nn as nn
from esp_ppq.api import espdl_quantize_torch
from esp_ppq.executor.torch import TorchExecutor
from pathlib import Path

from model import Model
from const import MODELS_DIR, IMAGE_SIZE
from utils import load_training_data, load_calibrating_data, evaluate

# todo: get model name or path by argument
MODEL_NAME = "model_20260929_214232_465468"
MODEL_PATH: Path = MODELS_DIR / (MODEL_NAME + ".pt")
ESPDL_MODEL_PATH: Path = MODELS_DIR / (MODEL_NAME + ".espdl")

TARGET = "esp32s3"
QUANT_TYPE = "w8a8"
BATCH_SIZE = 32
NUM_WORKERS = 4


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model: Model = Model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Calibration and evaluation MUST use identical preprocessing (Resize +
    # ToTensor + Normalize to [-1, 1]). `load_training_data` already returns the correct
    # normalized train/eval/test loaders, so reuse them instead of hand-rolling
    # a transform that silently skips Normalize.
    train_loader, _, _ = load_training_data(BATCH_SIZE, IMAGE_SIZE, NUM_WORKERS)
    _, _, test_loader = load_calibrating_data(BATCH_SIZE, IMAGE_SIZE, NUM_WORKERS)

    quant_ppq_graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=ESPDL_MODEL_PATH,
        calib_dataloader=train_loader,
        calib_steps=32,
        input_shape=[1, 3, IMAGE_SIZE, IMAGE_SIZE],
        inputs=None,
        target=TARGET,
        quant_type=QUANT_TYPE,
        collate_fn=lambda batch: batch[0].to(device),
        device=device,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1
    )

    criterion: nn.Module = nn.CrossEntropyLoss()
    executor: TorchExecutor = TorchExecutor(graph=quant_ppq_graph, device=device)
    
    print("Quantize completed, quick testing:")
    _, test_acc, correct, total = evaluate(executor, test_loader, criterion, device)
    print("OVERALL TEST ACCURACY: %.2f%% (%d/%d)" % (test_acc * 100.0, correct, total))


if __name__ == "__main__":
    main()