# ---------------------------------------------------------------------------
# Quantize a trained checkpoint into a .espdl model (via ESP-PPQ / ESP-DL).
#
# The input must be a checkpoint produced by train.py for the model selected
# with --model-type (model.py). Calibration samples come from the training split
# through the same transforms train.py uses, so a model that requests extra
# input preparation (Model5's DATA_FLAGS) is calibrated on matching values. The
# exported .espdl is what firmware/CMakeLists.txt flashes into the "model"
# partition; the logged input_contract says which firmware preprocessing it
# expects.
#
# Usage:
#     python quantize.py --model-type N --input <model.pt> [options]
#
# Options:
#     --model-type INT        required: quantize ModelN from model.py
#     --input PATH            required: trained checkpoint (.pt); must exist
#     --output STR            output name, saved as models/<name>.espdl
#                             (default: the input file's name)
#     --calibration-step INT  calibration steps passed to espdl_quantize_torch
#                             (default: 32)
#     --target STR            esp-dl target (default: esp32s3)
#     --quant-type STR        quantization type (default: w8a8)
#     --batch-size INT        calibration/eval batch size (default: 32)
#     --num-worker INT        DataLoader workers (default: 4)
#     --data-dir PATH         dataset root holding train/validate/test; used
#                             only if it exists (default: DATA_DIR)
#     --export-test-values BOOL
#                             export test inputs/outputs into the .info file
#                             (default: true)
#
# Examples:
#     python quantize.py --model-type 2 --input ../models/Kairo_Quell.pt
#     python quantize.py --model-type 5 --input ../models/Veyra_Sora.pt --output firenet_v1
#     python quantize.py --model-type 2 --input ../models/Kairo_Quell.pt --quant-type w8a16 \
#         --target esp32s3 --export-test-values false
#
# Outputs (next to the other model files):
#     models/<output>.espdl   quantized model
#     models/<output>.info    model structure/values dump (debug)
#     models/<output>.json    quantization config dump
# ---------------------------------------------------------------------------

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from esp_ppq.api import espdl_quantize_torch
from esp_ppq.executor.torch import TorchExecutor

from typing import Type

import torch
import torch.nn as nn
from esp_ppq.api import espdl_quantize_torch
from esp_ppq.executor.torch import TorchExecutor

from model import MODEL_CLASSES
from const import (
    DATA_DIR,
    FLAG_IMAGENET_NORMALIZATION,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODELS_DIR,
    configure_logging,
    log,
)
from utils import existing_file, instantiate_model, load_training_data, load_calibrating_data, evaluate

CALIBRATION_STEPS: int = 32
TARGET: str = "esp32s3"
QUANT_TYPE: str = "w8a8"
BATCH_SIZE: int = 32
NUM_WORKERS: int = 4


def str_to_bool(value: str) -> bool:
    """argparse type: accept true/false, yes/no, 1/0."""
    normalized: str = value.strip().lower()
    if normalized in ("true", "1", "yes", "y"):
        return True
    if normalized in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize a trained model into a .espdl model.")
    parser.add_argument(
        "--model-type",
        type=int,
        required=True,
        choices=sorted(MODEL_CLASSES),
        help="model to quantize: N selects ModelN from model.py (e.g. 5 -> Model5)",
    )
    parser.add_argument(
        "--input",
        type=existing_file,
        required=True,
        help="trained checkpoint (.pt) to quantize",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="output name, saved as <name>.espdl (default: input file name)",
    )
    parser.add_argument(
        "--calibration-step",
        type=int,
        default=CALIBRATION_STEPS,
        help=f"calibration steps (default: {CALIBRATION_STEPS})",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=TARGET,
        help=f"esp-dl target (default: {TARGET})",
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        default=QUANT_TYPE,
        help=f"quantization type (default: {QUANT_TYPE})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"calibration/eval batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--num-worker",
        type=int,
        default=NUM_WORKERS,
        help=f"DataLoader worker count (default: {NUM_WORKERS})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"dataset root holding train/validate/test (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--export-test-values",
        type=str_to_bool,
        default=True,
        metavar="BOOL",
        help="export test values into the .info file (default: true)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    configure_logging()

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir: Path = DATA_DIR
    if args.data_dir is not None:
        if args.data_dir.is_dir():
            data_dir = args.data_dir
        else:
            log.warning("data dir %s does not exist, falling back to %s", args.data_dir, DATA_DIR)

    output_name: str = args.output if args.output is not None else args.input.stem
    espdl_path: Path = MODELS_DIR / f"{output_name}.espdl"

    log.info("input=%s", args.input)
    log.info("output=%s", espdl_path)
    log.info(
        "target=%s quant_type=%s calibration_steps=%d batch_size=%d num_workers=%d",
        args.target, args.quant_type, args.calibration_step, args.batch_size, args.num_worker,
    )
    log.info("data_dir=%s export_test_values=%s device=%s", data_dir, args.export_test_values, device)

    model_class: Type[nn.Module] = MODEL_CLASSES[args.model_type]
    data_flags: int = getattr(model_class, "DATA_FLAGS", 0)
    if data_flags & FLAG_IMAGENET_NORMALIZATION:
        log.info(
            "input_contract=imagenet mean=%s std=%s -> firmware must enable ENABLE_IMAGENET_NORMALIZATION",
            IMAGENET_MEAN,
            IMAGENET_STD,
        )
    else:
        log.info("input_contract=raw [0,255] -> firmware mean=0/std=1")

    # The checkpoint supplies every weight, so instantiating with pretrained
    # ImageNet weights would be wasted work (instantiate_model drops the argument
    # for models that do not accept it).
    model: nn.Module = instantiate_model(model_class, pretrained=False)
    model.load_state_dict(torch.load(args.input, map_location=device))
    model.to(device)
    model.eval()

    # Calibration and evaluation must feed the same value range the model was
    # trained on and the firmware will feed: raw RGB [0, 255] by default, or
    # ImageNet-normalised when the model requests it (DATA_FLAGS). The model
    # input exponent is calibrated from these samples (Q = round(value / 2^exp)).
    train_loader, _, _ = load_training_data(data_dir, args.batch_size, args.num_worker, flags=data_flags)
    _, _, test_loader = load_calibrating_data(data_dir, args.batch_size, args.num_worker, flags=data_flags)

    quant_ppq_graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=espdl_path,
        calib_dataloader=train_loader,
        calib_steps=args.calibration_step,
        input_shape=[1, 3, IMAGE_SIZE, IMAGE_SIZE],
        inputs=None,
        target=args.target,
        quant_type=args.quant_type,
        collate_fn=lambda batch: batch[0].to(device),
        device=device,
        error_report=True,
        skip_export=False,
        export_test_values=args.export_test_values,
        verbose=1
    )

    criterion: nn.Module = nn.CrossEntropyLoss()
    executor: TorchExecutor = TorchExecutor(graph=quant_ppq_graph, device=device)

    log.info("Quantize completed, quick testing:")
    _, test_acc, correct, total = evaluate(executor, test_loader, criterion, device)
    log.info("OVERALL TEST ACCURACY: %.2f%% (%d/%d)", test_acc * 100.0, correct, total)


if __name__ == "__main__":
    main()
